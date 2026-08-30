#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
study-coach 记忆系统 CLI（HCE 落地实现）
=====================================

存储体系（每个学习项目独立文件夹内）：
- tree.json  : AI 专用知识树（Semantic Tree 落地），承担知识结构/诊断记录/检索来源
- _review.md : 人类专用温故知新文件（AI 检索时按文件名过滤，不注入上下文）

核心机制（严格对齐 HCE 设计）：
- HRS 分层相关性检索：numpy 暴力余弦 + 分层阈值剪枝 + 访问上限（限制递归深入，避免整棵知识树的无界遍历）
- Context Budgeting：贪心背包（Utility / Token_Cost）
- Entity Graph 落地：诊断数据（mastery/error_count/last_tested/relations）挂载于 tree.json 节点扩展字段
- Focus Buffer 落地：跨会话焦点缓冲（最近 6 轮，_session_context.md 由 AI 直接读写、CLI 不涉及），经 --recent 传入；
  支持新格式 [用户原句]/[用户精简]/[AI精简] 与旧格式 user:/assistant:
- 来源与学习历史：节点 source（user/ai/mixed）、learning_history/first_learned 由 CLI 维护，
  search/recall 完整输出返回（含 --source 过滤），不注入 context_block

嵌入模型：BAAI/bge-small-zh-v1.5（512 维，本地推理）

用法：
  python3 memory_cli.py <子命令> [参数]

子命令：
  init    <project_dir> [--name 项目名] [--summary 项目描述]
  node    add <project_dir> --name X --summary S [--parent P] [--mastery M]
                [--error_count N] [--last_tested YYYY-MM-DD] [--learned]
                [--source user|ai|mixed]
                [--relations JSON] [--clear_relations] [--attachments JSON] [--clear_attachments]
                # --learned：标记为用户实际学习（新建时初始化 last_learned/review_due/学习历史）；
                # 不带 --learned 只创建课程结构/蓝图节点，不进入复习调度（planned）。
  node    update <project_dir> --name X [--parent P] [--summary S] [--mastery M]
                [--error_count N] [--last_tested D] [--source user|ai|mixed]
                [--relations JSON] [--clear_relations] [--attachments JSON] [--clear_attachments]
                # node update 只写数据：诊断字段（mastery/error_count/last_tested）与内容修正
                # 均不刷新 last_learned、不进入复习调度（前置诊断 ≠ 学习；
                # 进入调度只由 --learned 或 schedule update 负责）
  node    rm <project_dir> --name X
  search  <project_dir> --query Q [--threshold 0.45] [--top_k 10] [--no_instruction]
                [--source user|ai|mixed|all]   # 默认 all；user→user+mixed；ai→ai+mixed；mixed→仅 mixed
  recall  <project_dir> --query Q [--budget 6000] [--recent "…"] [--threshold 0.45]
                [--no_instruction] [--source user|ai|mixed|all]
                # --recent 新格式：[用户原句]/[用户精简]/[AI精简]；旧格式 user:/assistant: 兼容
  review  <project_dir> --date D --section S [--what T] [--insight T] [--practice T]
                [--weak T] [--next T]
  meta    get <project_dir> [--keys k1 k2 ...]
  meta    set <project_dir> --patch '<JSON对象>' | --key K --value V
  schedule due <project_dir> [--plan]            # 复习到期检查（review_due 判定；--plan 输出紧凑复习计划）
  schedule update <project_dir> --name X [--passed] [--mastery M]  # 复习结果回写（更新 review_due）
  status  <project_dir>

所有子命令输出 JSON（UTF-8），便于调用方解析。
（例外：schedule due --plan 输出纯文本紧凑复习计划，专供 AI 阅读。）
"""

import argparse
import json
import math
import os
import re
import sys
import tempfile
import uuid
from datetime import date, datetime, timedelta


def _force_utf8_stdio():
    """Windows 控制台默认编码（如 GBK）下，强制标准输出/错误流以 UTF-8 写出，避免中文乱码。

    - 平台无关：Linux/macOS 默认即为 UTF-8，reconfigure 幂等，行为不变。
    - Python 3.7+ 提供 io.TextIOWrapper.reconfigure；stdout/stderr 非 TextIOWrapper
      或 reconfigure 不可用时静默跳过，不影响原有输出路径。
    - 本模块被导入（如 batch_update.py）时同样生效，覆盖该进程内的所有 print/警告/错误输出。
    """
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError, OSError):
        pass


_force_utf8_stdio()

# 系统环境固化了失效镜像 HF_ENDPOINT（hf-mirror.com 已失效）；默认官方源，
# 但尊重用户已有配置（setdefault）：用户显式设置镜像时以其为准，不无条件覆盖。
os.environ.setdefault('HF_ENDPOINT', 'https://huggingface.co')


def _hf_cache_root():
    """Hugging Face hub 缓存根目录（平台无关，优先尊重用户环境变量）。

    huggingface_hub 的解析顺序：HF_HUB_CACHE > $HF_HOME/hub > ~/.cache/huggingface/hub
    （~ 在 Windows 下即 %USERPROFILE%，与 Linux 语义一致，不写死平台路径；
    用户自定义 HF_HOME / HF_HUB_CACHE 时也能正确识别，避免离线检测失效导致
    每次调用都走联网检查而超时）。
    """
    env = os.environ.get('HF_HUB_CACHE')
    if env:
        return env
    home = os.environ.get('HF_HOME')
    if home:
        return os.path.join(home, 'hub')
    return os.path.join(os.path.expanduser('~'), '.cache', 'huggingface', 'hub')


def _model_cache_ready():
    """模型缓存是否完整可用（必要文件齐全，而非仅目录存在）。

    huggingface_hub 下载中断/失败会残留目录（snapshots 部分或为空），
    仅 isdir 判断会把不完整缓存误判为已部署 → 错误进入离线模式 → 加载报错。
    校验必要文件：任一完整 snapshot 含 config.json + modules.json + 权重文件
    （model.safetensors 或 pytorch_model.bin）即视为可用（手动放置缓存同样适用）。
    """
    cache = os.path.join(_hf_cache_root(), 'models--BAAI--bge-small-zh-v1.5')
    snap_dir = os.path.join(cache, 'snapshots')
    if not os.path.isdir(snap_dir):
        return False
    for d in os.listdir(snap_dir):
        sd = os.path.join(snap_dir, d)
        if not os.path.isdir(sd):
            continue
        files = set(os.listdir(sd))
        if 'config.json' in files and 'modules.json' in files \
                and ('model.safetensors' in files or 'pytorch_model.bin' in files):
            return True
    return False


def _setup_offline():
    """内建离线模式：嵌入模型缓存完整时跳过 HuggingFace 联网检查。

    防止调用方遗漏 export HF_HUB_OFFLINE=1 导致模型加载卡死
    （实测未设离线时单次调用 >180 秒超时）。首次运行/缓存不完整时不设离线，
    允许（重新）联网下载；完整缓存后继续支持离线运行。
    """
    if _model_cache_ready():
        os.environ.setdefault('HF_HUB_OFFLINE', '1')
        os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')


_setup_offline()

import numpy as np

# ─────────────────────────── 常量与默认值 ───────────────────────────

SCHEMA_VERSION = 1
MODEL_NAME = 'BAAI/bge-small-zh-v1.5'
EMBED_DIM = 512

# 检索
DEFAULT_THRESHOLD = 0.45          # 相似度阈值默认值（bge 语义空间）
THRESHOLD_FLOOR = 0.50            # 动态上浮后的阈值下限
MAX_VISIT_RATIO = 0.25            # 全局访问节点上限：总节点数比例
MAX_VISIT_ABS = 100               # 全局访问节点上限：绝对数
DYNAMIC_PASS_RATIO = 0.70         # 单层通过率超过该值则触发阈值上浮
QUERY_INSTRUCTION = '为这个句子生成表示以用于检索相关文章：'  # bge 官方查询侧指令

# 预算
BUDGET_DEFAULT = 6000             # 上下文预算（token）。缓冲与树结果统一装入；超限先淘汰旧轮次，不截断任何一条
TOKEN_CN_WEIGHT = 1.0             # 中文字符≈1 token
TOKEN_EN_WEIGHT = 1.3             # 英文单词≈1.3 token

# summary 长度保护（防 tree.json 因错误写入不断膨胀；不静默截断——截断会丢失用户理解记忆）
SUMMARY_HARD_LIMIT = 1500         # 硬上限：超过拒绝，要求调用方压缩后重传（文档约定：目标"一句话级"）
SUMMARY_WARN_LIMIT = 1200         # 接近上限：接受但输出 warning

# Utility 权重
U_SIM, U_MASTERY, U_ERR, U_DECAY = 0.50, 0.20, 0.15, 0.15
MASTERY_NEUTRAL = 0.5
DECAY_HALF_DAYS = 30              # 时间衰减尺度（天）：decay = 1 - exp(-天数/30)
DECAY_NO_TESTED = 0.6             # 从未记录 last_tested 时的中性值（≈27 天前水平，照顾新节点）

# 焦点缓冲（混合策略，N=6 轮）：用户输入与 AI 回复均由 AI 理解后精简后传入（见《记忆系统.md》「Focus Buffer」节），
# CLI 仅做兜底截断（正常不应触发）。精简方式：自然缩句/转换说法/合并同类项，不套固定格式。
BUFFER_N_ROUNDS = 6
USER_MAX_CHARS = 300              # 用户输入精简版兜底上限（AI 已去冗余，超限仅极端情况）
AI_MAX_CHARS = 150                # AI 回复摘要兜底上限

# 父子关系判断规则（基于实测 doc-doc 相似度分布标定）：
# - 显式 --parent：AI 依据课程结构指定，最准确，优先
# - 未指定：HRS 找最相似节点，sim>=0.85 视为同一知识点（合并防重复）；
#   否则一律挂 root（不自动挂子节点，避免"色相 vs 白平衡 0.55"这类同主题误挂）
MERGE_SIM = 0.85

# 复习调度（间隔复习，简化 SM-2 / Leitner 思想）
INITIAL_INTERVAL_DAYS = 1     # 新知识点初始复习间隔（天）
MAX_INTERVAL_DAYS = 60        # 间隔上限（天）：1→2→4→8→16→32→60，通过翻倍、失败重置 1

# 学习历史（按需引用，不参与复习调度与 Utility 计算）
LEARNING_HISTORY_MAX = 10     # learning_history 上限条数，超限移除最早

_model = None
_model_loaded = False


def out(obj, code=0):
    """统一 JSON 输出。"""
    print(json.dumps(obj, ensure_ascii=False))
    sys.exit(code)


def fail(msg):
    out({'error': msg}, 1)


# ─────────────────────────── 嵌入模型封装 ───────────────────────────

def get_model():
    global _model, _model_loaded
    if not _model_loaded:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
        _model_loaded = True
    return _model


def embed(text, with_instruction=False):
    """生成文本向量；失败返回 None（调用方降级处理）。"""
    try:
        t = text or ''
        if with_instruction:
            t = QUERY_INSTRUCTION + t
        v = get_model().encode(t, normalize_embeddings=True)
        return [float(x) for x in v]
    except Exception:
        return None


# ─────────────────────────── 基础工具 ───────────────────────────

def cosine(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.size == 0 or b.size == 0:
        return 0.0
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def estimate_tokens(text):
    if not text:
        return 1
    cn = len(re.findall(r'[\u4e00-\u9fff]', text))
    en = len(re.findall(r'[A-Za-z0-9]+', text))
    return max(1, int(cn * TOKEN_CN_WEIGHT + en * TOKEN_EN_WEIGHT))


def truncate(text, limit):
    if len(text) <= limit:
        return text
    return text[:limit] + '…[截断]'


# ─────────────────────────── tree.json 读写 ───────────────────────────

# Windows 保留设备名（含扩展名形式如 CON.txt；CONFIG/NULL 等前缀不同不受影响）
_WIN_RESERVED_NAMES = {'CON', 'PRN', 'AUX', 'NUL'}
_WIN_RESERVED_RE = re.compile(r'^(COM|LPT)[1-9]$')


def safe_folder_name(name):
    """项目目录名跨平台安全化（确定性映射，不引入 UUID；Windows/Linux/macOS 均可创建）。

    - Windows 非法字符 \\ / : * ? " < > | 与控制字符 → '-'（'/' 在任何平台都会破坏路径结构）
    - Windows 保留设备名（CON/PRN/AUX/NUL/COM1-9/LPT1-9，含 'CON.txt' 形式）→ 加 '_' 前缀
    - 去除首尾空白与尾部句点（Windows 不允许目录名以空格/句点结尾）
    - 空结果 → 'project'（避免创建失败）
    映射稳定：同名输入永远得到同名输出；不同项目映射撞名时由 init 自动检测
    （仅显式 --name 与现有 tree.json.project 不一致时判定为另一项目）并加稳定序号（-2、-3…）
    新建独立目录，绝不静默复用；未传 --name 视为已定位现有项目，保持幂等复用；
    AI 按 SKILL.md §1 四阶梯定位仍在上层。
    """
    s = re.sub(r'[\\/:*?"<>|\x00-\x1f]', '-', name or '')
    s = s.strip().rstrip('.')
    if not s:
        s = 'project'
    stem = s.split('.')[0].upper()
    if stem in _WIN_RESERVED_NAMES or _WIN_RESERVED_RE.match(stem):
        s = '_' + s
    return s


def _check_summary(summary):
    """summary 长度保护：>硬上限拒绝（fail），接近上限返回 warning 文本。

    不静默截断——summary 是用户个性化理解记忆，截断会丢失内容；
    超限由调用方（AI）压缩后重传，防止 tree.json 因错误写入不断膨胀。
    """
    if summary is None:
        return None
    n = len(summary)
    if n > SUMMARY_HARD_LIMIT:
        fail(f'summary 超过硬上限 {SUMMARY_HARD_LIMIT}字（当前 {n}字）：'
             '请压缩后重新提交（正常目标为"一句话级"精炼要点，见文档）')
    if n > SUMMARY_WARN_LIMIT:
        return f'summary 接近上限（{n}/{SUMMARY_HARD_LIMIT}字），建议压缩'
    return None


def tree_path(project_dir):
    return os.path.join(project_dir, 'tree.json')


def review_path(project_dir):
    return os.path.join(project_dir, '_review.md')


def load_tree(project_dir):
    p = tree_path(project_dir)
    if not os.path.exists(p):
        return None
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def _atomic_write(path, content):
    """原子写：临时文件 + os.replace（跨平台，无锁；写入中途崩溃不损坏原文件）。

    用于 tree.json 与 _review.md 等正式记忆文件（直接 'w' 写在写入中途崩溃时可能损坏文件）。
    """
    d = os.path.dirname(path) or '.'
    fd, tmp = tempfile.mkstemp(dir=d, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(content)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def save_tree(project_dir, tree):
    """保存 tree.json（原子写）。统一刷新 updated 时间戳。"""
    tree['updated'] = datetime.now().isoformat(timespec='seconds')
    _atomic_write(tree_path(project_dir), json.dumps(tree, ensure_ascii=False, indent=2))


def count_nodes(node):
    n = 1
    for c in node.get('children', []):
        n += count_nodes(c)
    return n


def find_node(node, name=None, nid=None):
    """递归查找节点（按 id 或 name，精确匹配）。返回 (节点, 父节点)。"""
    if (name is not None and node.get('name') == name) or (nid is not None and node.get('id') == nid):
        return node, None
    for c in node.get('children', []):
        found, parent = find_node(c, name=name, nid=nid)
        if found is not None:
            if parent is None:
                return found, node
            return found, parent
    return None, None


def _is_descendant(ancestor, node):
    """判断 node 是否为 ancestor 的后代（含自身），用于移动节点时防成环。"""
    if node is ancestor:
        return True
    for c in ancestor.get('children', []):
        if _is_descendant(c, node):
            return True
    return False


def _collect_ids(node, pv):
    """收集子树内所有节点 id 并从 pending_vectors 中移除（删除节点时清理残留）。"""
    if node.get('id') in pv:
        pv.remove(node['id'])
    for c in node.get('children', []):
        _collect_ids(c, pv)


def new_node(name, summary):
    return {
        'id': 'n_' + uuid.uuid4().hex[:12],
        'name': name,
        'summary': summary or '',
        'vector': [],
        'children': [],
        'created': date.today().isoformat(),
        'source': 'user',
    }


def _append_learning_history(node):
    """学习历史维护：同日去重、时间正序、上限 LEARNING_HISTORY_MAX 条（超限移除最早）。"""
    today = date.today().isoformat()
    hist = node.get('learning_history') or []
    if hist and hist[-1] == today:
        return
    hist.append(today)
    if len(hist) > LEARNING_HISTORY_MAX:
        del hist[:len(hist) - LEARNING_HISTORY_MAX]
    node['learning_history'] = hist


def node_content(node):
    """节点注入 LLM 的文本形态（不含通用知识本体，仅名称+个性化摘要+诊断）。"""
    parts = [f"[{node['name']}]"]
    if node.get('summary'):
        parts.append(node['summary'])
    diag = []
    if node.get('mastery') is not None:
        diag.append(f"掌握度{node['mastery']:.2f}")
    if node.get('error_count'):
        diag.append(f"错误{node['error_count']}次")
    if node.get('last_tested'):
        diag.append(f"最近检验{node['last_tested']}")
    if diag:
        parts.append('（' + '，'.join(diag) + '）')
    return ' '.join(parts)


# ─────────────────────────── HRS 分层相关性检索 ───────────────────────────

def hr_search(tree, query_vec, threshold=DEFAULT_THRESHOLD, top_k=None, adaptive=True):
    """HRS：从根逐层计算相似度，低于阈值的子树剪掉（父节点相关性不足时旁路直查其子节点，
    防"亮度家族"这类合集节点掩盖其下高相关知识点），按当前层相似度排序后优先深入。

    返回：按 score（sim*0.7 + inherited*0.3）降序的节点候选列表。
    保护机制（adaptive=True 时生效）：
      1. 全局访问上限 min(总数×25%, 100)，超限停止深入
         （上限约束"进入结果并继续深入的节点数"；单层/旁路的批量相似度评估
         在深入前完成、不计入预算——防无限深入而非严格限制计算量，见设计文档 §三.1）
      2. 单层通过率 >70% 且阈值 <0.50 时，阈值上浮至 0.50（防低阈值下无界深入）
      3. 空向量节点视为不匹配，直接跳过（降级策略）
    """
    root = tree.get('root')
    if root is None:
        return []
    total = count_nodes(root)
    max_visits = max(10, min(int(total * MAX_VISIT_RATIO), MAX_VISIT_ABS))
    results = []
    visited = 0
    cur_threshold = threshold

    def walk(node, inherited, path, is_root=False):
        """is_root=True 时该节点仅作入口，不参与剪枝、不加入结果。
        root 是项目容器（向量=项目名摘要），query 与其相似度通常不高，
        若对 root 剪枝会导致整棵树被剪掉，因此 root 始终放行。"""
        nonlocal visited, cur_threshold
        if visited >= max_visits:
            return
        vec = node.get('vector') or []
        if not is_root and not vec:
            return  # 空向量节点跳过（降级策略）
        if not is_root:
            sim = cosine(query_vec, vec)
            if sim < threshold:
                return  # 整棵子树剪掉（阈值上浮只收紧子层深入，不拒本层已通过节点）
            visited += 1
        else:
            sim = 0.0
        score = sim * 0.7 + inherited * 0.3
        if not is_root:
            results.append({
                'id': node.get('id'),
                'name': node.get('name'),
                'summary': node.get('summary', ''),
                'sim': round(sim, 4),
                'score': round(score, 4),
                'mastery': node.get('mastery'),
                'error_count': node.get('error_count'),
                'last_tested': node.get('last_tested'),
                'relations': node.get('relations', []),
                'path': path,
                'source': node.get('source') or 'user',
                'learning_history': node.get('learning_history', []),
                'first_learned': node.get('first_learned'),
            })
        children = node.get('children', [])
        if not children:
            return
        # 子层相似度计算与动态阈值保护（adaptive 开启时生效）
        child_sims = [(c, cosine(query_vec, c.get('vector') or [])) for c in children]
        passed = [(c, s) for c, s in child_sims if s >= cur_threshold]
        if adaptive and children and len(passed) / len(children) > DYNAMIC_PASS_RATIO \
                and cur_threshold < THRESHOLD_FLOOR:
            # 阈值上浮只收紧后续层深入，不重过滤本层已通过节点
            # （防边缘误杀：如 sim=0.4994 的节点被 0.50 阈值剪掉）
            cur_threshold = THRESHOLD_FLOOR
        passed.sort(key=lambda x: x[1], reverse=True)
        for c, s in passed:
            walk(c, score, path + [c['name']])
        # 旁路：未通过的子节点（合集节点）相关性不足时直查其子节点
        # （防"亮度家族"这类合集节点掩盖其下高相关知识点，访问上限仍生效）
        passed_ids = {c['id'] for c, _ in passed}
        for c, _ in child_sims:
            if c['id'] in passed_ids:
                continue
            sub = [cc for cc in c.get('children', []) if cc.get('vector')]
            if not sub:
                continue
            sub_sims = [(cc, cosine(query_vec, cc.get('vector') or [])) for cc in sub]
            sub_sims = [(cc, s) for cc, s in sub_sims if s >= cur_threshold]
            sub_sims.sort(key=lambda x: x[1], reverse=True)
            for cc, s in sub_sims:
                walk(cc, score, path + [c['name'], cc['name']])

    walk(root, 0.0, [root.get('name', 'root')], is_root=True)
    results.sort(key=lambda r: r['score'], reverse=True)
    if top_k and top_k > 0:
        results = results[:top_k]
    return results


def node_utility(sim, mastery, error_count, last_tested):
    """Utility = 0.50×sim + 0.20×(1-mastery) + 0.15×min(err,5)/5 + 0.15×time_decay。

    mastery 项采用"弱项得分"语义：weakness = 1 - mastery，
    掌握度越低（越薄弱）Utility 越高，与 error_count 共同实现"薄弱点优先被召回"，
    避免"掌握越好越优先注入"的反向偏差（旧公式 +0.20×mastery 与设计意图冲突）。

    time_decay：1 - exp(-天数/30)，last_tested 距今越久 decay 越大，
    使"久未复习的知识"优先被注入（防遗忘）；未记录时取中性值 0.6（≈27 天前水平）。
    """
    m = mastery if mastery is not None else MASTERY_NEUTRAL
    m = max(0.0, min(1.0, m))
    err = min(error_count or 0, 5) / 5.0
    days = None
    if last_tested:
        try:
            d = datetime.strptime(last_tested, '%Y-%m-%d').date()
            days = max(0, (date.today() - d).days)
        except Exception:
            days = None
    decay = (1.0 - math.exp(-days / DECAY_HALF_DAYS)) if days is not None else DECAY_NO_TESTED
    return U_SIM * sim + U_MASTERY * (1.0 - m) + U_ERR * err + U_DECAY * decay


# ─────────────────────────── 贪心背包预算 ───────────────────────────

def knapsack(candidates, budget):
    """按 Utility/Token_Cost 降序贪心装入；返回选中列表与剩余预算。"""
    ranked = sorted(
        candidates,
        key=lambda c: (c['utility'] / max(1, c['token_cost']), c['utility']),
        reverse=True,
    )
    selected = []
    remaining = budget
    for c in ranked:
        if c['token_cost'] <= remaining:
            selected.append(c)
            remaining -= c['token_cost']
    return selected, remaining


ROLE_LABEL = {'user_quote': '用户原句', 'user_simplified': '用户精简', 'assistant': 'AI'}


def _parse_recent(line):
    """解析焦点缓冲行 → (role, text) 或 None。

    新格式（_session_context.md，AI 写入）：[用户原句]/[用户精简]/[AI精简]
    旧格式（兼容）：user:/assistant:
    映射：用户原句 → user_quote（不截断、不改字）；用户精简/user: → user_simplified；
          AI精简/assistant: → assistant。
    """
    # 行首容忍空白与列表符号（"- "），兼容历史文件/手写格式差异；规范要求不带
    m = re.match(r'^\s*-?\s*\[(用户原句|用户精简|AI精简)\]\s*[:：]?\s*(.*)$', line, re.S)
    if m:
        kind, text = m.group(1), m.group(2)
        role = {'用户原句': 'user_quote', '用户精简': 'user_simplified', 'AI精简': 'assistant'}[kind]
        return role, text
    m = re.match(r'^(user|assistant):\s*(.*)$', line, re.S)
    if m:
        role, text = m.group(1), m.group(2)
        return ('user_simplified' if role == 'user' else 'assistant'), text
    return None


def _filter_source(hits, source):
    """按来源过滤检索结果：user→user+mixed；ai→ai+mixed；mixed→仅 mixed；all/None→全部。"""
    if not source or source == 'all':
        return hits
    if source == 'mixed':
        return [h for h in hits if (h.get('source') or 'user') == 'mixed']
    return [h for h in hits if (h.get('source') or 'user') in (source, 'mixed')]


# ─────────────────────────── 子命令实现 ───────────────────────────

def cmd_init(args):
    project_dir = args.project_dir
    tp = tree_path(project_dir)
    if not os.path.exists(tp):
        # 新建项目：目录名与用户可见项目名（tree.json 顶层 project）分离——用户可见名可含任意字符，
        # 目录名必须是 Windows/Linux/macOS 均可创建的（非法字符→'-'、保留名加前缀、去尾部空格句点）。
        # 传入 basename 不安全时自动改为安全名（确定性映射，不引入 UUID），输出返回实际 project_dir，
        # 后续命令以返回路径为准；旧项目已存在则跳过映射，不破坏旧数据。
        base = os.path.basename(os.path.normpath(project_dir))
        safe = safe_folder_name(base)
        if safe != base:
            project_dir = os.path.join(os.path.dirname(os.path.normpath(project_dir)), safe)
            tp = tree_path(project_dir)
    # 碰撞消解（防静默复用/误合并）：安全目录已存在且调用者**显式**提供 --name 时，
    # 若 tree.json.project 与 --name 不一致 → 真正项目冲突，加稳定序号（-2、-3…）新建独立目录，
    # 绝不静默复用；未传 --name（AI 已按 SKILL.md §1 四阶梯定位到现有目录）或 project 一致
    # → 保持幂等复用，不因 project 与目录 basename 不同而新建目录。
    # AI 四阶梯定位仍在上层，此处仅为 CLI 兜底。
    if os.path.exists(tp):
        existing_project = (load_tree(project_dir) or {}).get('project')
        if args.name and existing_project and existing_project != args.name:
            base = os.path.basename(os.path.normpath(project_dir))
            safe = safe_folder_name(base)
            parent = os.path.dirname(os.path.normpath(project_dir))
            n = 2
            while True:
                cand = os.path.join(parent, f'{safe}-{n}')
                ctp = tree_path(cand)
                if not os.path.exists(ctp):
                    project_dir = cand      # 空位：新建独立目录
                    tp = ctp
                    break
                cproj = (load_tree(cand) or {}).get('project')
                if cproj == args.name:
                    project_dir = cand      # 该序号目录正是本项目 → 幂等复用
                    tp = ctp
                    break
                n += 1
    if not os.path.exists(tp):
        os.makedirs(project_dir, exist_ok=True)
        tree = {
            'schema_version': SCHEMA_VERSION,
            'project': args.name or os.path.basename(os.path.normpath(project_dir)),
            'updated': datetime.now().isoformat(timespec='seconds'),
            'root': {
                'id': 'root',
                'name': args.name or os.path.basename(os.path.normpath(project_dir)),
                'summary': args.summary or '',
                'vector': [],
                'children': [],
                'created': date.today().isoformat(),
            },
        }
        save_tree(project_dir, tree)
        vector_status = 'pending'
        if tree['root']['summary']:
            v = embed(tree['root']['name'] + '：' + tree['root']['summary'])
            if v:
                tree['root']['vector'] = v
                tree['root'].pop('vector_pending', None)
                save_tree(project_dir, tree)
                vector_status = 'ok'
    else:
        tree = load_tree(project_dir)
        vector_status = 'exists'
    rp = review_path(project_dir)
    if not os.path.exists(rp):
        _atomic_write(rp, f'# 温故知新 - {tree["root"]["name"]}\n\n'
                          '> 本文件由 study-coach 技能自动追加（AI 默认只写不读；'
                          '默认检索走 tree.json，用户明确查看进度/复习记录时可读取转述）。\n')
    out({'ok': True, 'action': 'init', 'project_dir': project_dir, 'tree': tp, 'review': rp,
         'root_vector': vector_status})


def _resolve_parent(tree, parent):
    if parent is None:
        return tree['root']
    node, _ = find_node(tree['root'], name=parent)
    if node is None:
        node, _ = find_node(tree['root'], nid=parent)
    if node is None:
        return None
    return node


def cmd_node_add(args):
    tree = load_tree(args.project_dir)
    if tree is None:
        fail('tree.json 不存在，请先执行 init')
    # summary 长度保护（>1500 拒绝 / >1200 warning），不静默截断
    summary_warning = _check_summary(args.summary)
    # 同名节点已存在 → 视为更新
    existing, existing_parent = find_node(tree['root'], name=args.name)
    if existing is not None:
        if args.summary is not None:
            existing['summary'] = args.summary
        if args.source is not None:
            existing['source'] = args.source
        _apply_diag(existing, args)
        _apply_relations(existing, args.relations, args.clear_relations)
        _apply_attachments(existing, args.attachments, args.clear_attachments)
        if args.learned:
            # 显式学习语义：刷新 last_learned、追加学习历史、确保进入复习调度；
            # 不带 --learned 的同名更新（复习摘要写回/结构修正）不碰学习字段与调度。
            _touch_learned(existing, init_review=True)
            _append_learning_history(existing)
            existing.setdefault('first_learned', date.today().isoformat())
        _regen_vector(existing, tree, args.project_dir)
        save_tree(args.project_dir, tree)
        out({'ok': True, 'action': 'updated', 'node_id': existing['id'],
             'parent_id': existing_parent['id'] if existing_parent else 'root',
             'vector_status': existing.get('vector_status', 'ok'),
             'summary_warning': summary_warning})

    # 新节点：确定父节点
    parent = _resolve_parent(tree, args.parent)
    if args.parent is not None and parent is None:
        fail(f"父节点不存在: {args.parent}")

    # 未显式指定 parent（挂 root 前）→ 自动合并判断（防重复节点）。
    # 注意：_resolve_parent 对 parent=None 返回 root，因此这里判断 args.parent 而非 parent，
    # 否则该分支永远不可达（历史 bug：0.85 相似度合并从未生效）。
    if args.parent is None:
        # 自动父子关系判断（仅防重复合并，不自动挂子节点）：
        # HRS 找最相似节点，sim>=MERGE_SIM(0.85) 视为同一知识点→合并更新；
        # 否则挂 root。子节点挂载由 AI 显式 --parent 指定（依据课程结构）。
        # 注意：这是"文档-文档"相似度，不加 bge 查询指令（指令仅用于用户查询侧），
        # 且关闭 adaptive 动态阈值（此处要的是全局最相似节点）。
        probe = args.name + '：' + (args.summary or '')
        qv = embed(probe, with_instruction=False)
        if qv:
            hits = hr_search(tree, qv, threshold=0.0, top_k=1, adaptive=False)
            if hits:
                best = hits[0]
                if best['sim'] >= MERGE_SIM:
                    # 视为同一知识点 → 合并更新
                    target, _ = find_node(tree['root'], nid=best['id'])
                    if target is not None:
                        if args.summary is not None:
                            target['summary'] = args.summary
                        if args.source is not None:
                            target['source'] = args.source
                        _apply_diag(target, args)
                        _apply_relations(target, args.relations, args.clear_relations)
                        _apply_attachments(target, args.attachments, args.clear_attachments)
                        if args.learned:
                            # 合并 + 显式学习：视为用户实际学习该知识点（刷新学习字段并确保进入调度）；
                            # 不带 --learned 的合并只是数据去重整理，不伪装成学习。
                            _touch_learned(target, init_review=True)
                            _append_learning_history(target)
                            target.setdefault('first_learned', date.today().isoformat())
                        _regen_vector(target, tree, args.project_dir)
                        save_tree(args.project_dir, tree)
                        out({'ok': True, 'action': 'merged', 'node_id': target['id'],
                             'merged_with': best['name'], 'sim': best['sim'],
                             'vector_status': target.get('vector_status', 'ok'),
                             'summary_warning': summary_warning})

    node = new_node(args.name, args.summary or '')
    if args.source is not None:
        node['source'] = args.source
    _apply_diag(node, args)
    _apply_relations(node, args.relations, args.clear_relations)
    _apply_attachments(node, args.attachments, args.clear_attachments)
    if args.learned:
        # 用户实际学习该知识点 → 初始化学习字段与复习调度（review_due=明天）；
        # 不带 --learned 只创建课程蓝图/结构节点（planned），不进入复习调度。
        _touch_learned(node, init_review=True)
        node['first_learned'] = date.today().isoformat()
        _append_learning_history(node)
    parent['children'].append(node)
    _regen_vector(node, tree, args.project_dir)
    save_tree(args.project_dir, tree)
    out({'ok': True, 'action': 'added', 'node_id': node['id'],
         'parent_id': parent['id'], 'vector_status': node.get('vector_status', 'ok'),
         'summary_warning': summary_warning})


def _apply_diag(node, args):
    if getattr(args, 'mastery', None) is not None:
        node['mastery'] = float(args.mastery)
    if getattr(args, 'error_count', None) is not None:
        node['error_count'] = int(args.error_count)
    if getattr(args, 'last_tested', None) is not None:
        node['last_tested'] = args.last_tested


def _apply_relations(node, relations_json, clear):
    if clear:
        node['relations'] = []
    if relations_json:
        try:
            rels = json.loads(relations_json)
            if isinstance(rels, list):
                node.setdefault('relations', [])
                existing = node['relations']
                for r in rels:
                    if r not in existing:
                        existing.append(r)
        except Exception:
            pass


def _apply_attachments(node, attachments_json, clear):
    """附件引用（隔离目录路径 + 描述，见《记忆系统.md》「非文本材料」节），JSON 数组，去重追加。"""
    if clear:
        node['attachments'] = []
    if attachments_json:
        try:
            atts = json.loads(attachments_json)
            if isinstance(atts, list):
                node.setdefault('attachments', [])
                existing = node['attachments']
                for a in atts:
                    if a not in existing:
                        existing.append(a)
        except Exception:
            pass


def _touch_learned(node, init_review=False):
    """记录学习时间：仅在代表真实学习行为的写入路径上调用（node add --learned）。

    该知识点今天被正式学习 → last_learned 置为今天；
    review_interval_days 首次创建时初始化为 1 天（间隔由 schedule update 管理）。
    init_review=True（node add --learned 新建/转正）→ 确保节点进入复习调度：
    **仅当节点尚无 review_due 时**初始化 review_due = 今天+1（interval=1）；
    已有 review_due 不覆盖（复习调度只由 schedule update 重算，避免"改了摘要就顺延复习"）。
    复习/检验的 last_learned 与调度推进由 schedule update 统一维护；
    不带 --learned 的 node add（蓝图/结构写入、复习摘要写回）与 node update
    （含诊断字段更新——前置诊断/数据记录）不调用本函数（结构写入/诊断 ≠ 学习）。
    """
    node['last_learned'] = date.today().isoformat()
    node.setdefault('review_interval_days', INITIAL_INTERVAL_DAYS)
    if init_review and not node.get('review_due'):
        node['review_due'] = (date.today() + timedelta(days=INITIAL_INTERVAL_DAYS)).isoformat()


def _regen_vector(node, tree, project_dir):
    """节点向量生成：成功→写入并清除 pending（含顶层 pending_vectors 登记）；失败→保留 pending 标记。"""
    text = node.get('name', '') + '：' + (node.get('summary') or '')
    v = embed(text)
    if v and len(v) == EMBED_DIM:
        node['vector'] = v
        node.pop('vector_pending', None)
        node['vector_status'] = 'ok'
        # 同步清理顶层 pending_vectors：节点已恢复成功，不得残留旧 id
        # （否则 status 会长期显示"待生成向量"的假象；删除节点的清理在 cmd_node_rm）
        pv = tree.get('pending_vectors')
        if pv and node['id'] in pv:
            pv.remove(node['id'])
    else:
        node['vector'] = []
        node['vector_pending'] = True
        node['vector_status'] = 'failed'
        pv = tree.setdefault('pending_vectors', [])
        if node['id'] not in pv:
            pv.append(node['id'])


def cmd_node_update(args):
    tree = load_tree(args.project_dir)
    if tree is None:
        fail('tree.json 不存在，请先执行 init')
    # summary 长度保护（>1500 拒绝 / >1200 warning），不静默截断
    summary_warning = _check_summary(args.summary)
    node, parent = find_node(tree['root'], name=args.name)
    if node is None:
        fail(f'节点不存在: {args.name}')
    # 移动节点（--parent 为父节点名或 id；root 不可移动）
    moved = False
    new_parent = None
    if args.parent is not None and parent is not None:
        new_parent = _resolve_parent(tree, args.parent)
        if new_parent is None:
            fail(f'父节点不存在: {args.parent}')
        if new_parent is node:
            fail('不能把节点移动到自己下面')
        if _is_descendant(node, new_parent):
            fail('不能把节点移动到自己的子节点下')
        if parent is not new_parent:
            parent['children'] = [c for c in parent['children'] if c.get('id') != node['id']]
            new_parent['children'].append(node)
            moved = True
    if args.summary is not None:
        node['summary'] = args.summary
    if args.source is not None:
        node['source'] = args.source
    _apply_diag(node, args)
    _apply_relations(node, args.relations, args.clear_relations)
    _apply_attachments(node, args.attachments, args.clear_attachments)
    changed = args.summary is not None or args.source is not None \
        or args.mastery is not None \
        or args.error_count is not None or args.last_tested is not None \
        or args.relations or args.clear_relations \
        or args.attachments or args.clear_attachments
    # 诊断字段（mastery/error_count/last_tested）只记录数据，不代表学习/复习/检验行为：
    # 不刷新 last_learned、不初始化复习调度——前置诊断不能把 planned 节点推进成 learned，
    # 也不能因诊断自动进入复习调度（"被诊断/测试 ≠ 已学习"）。
    # 进入调度只由 node add --learned（显式学习）或 schedule update（检验/复习结果回写）负责；
    # 技能三"学习后检验"的 last_learned 由伴随的 schedule update 统一维护（通过/失败均刷新）。
    if changed:
        _regen_vector(node, tree, args.project_dir)
    if changed or moved:
        save_tree(args.project_dir, tree)
    out({'ok': True, 'action': 'moved' if moved else 'updated', 'node_id': node['id'],
         'parent_id': (new_parent or parent)['id'] if (new_parent or parent) else 'root',
         'vector_status': node.get('vector_status', 'ok'),
         'summary_warning': summary_warning})


def cmd_node_rm(args):
    tree = load_tree(args.project_dir)
    if tree is None:
        fail('tree.json 不存在，请先执行 init')
    node, parent = find_node(tree['root'], name=args.name)
    if node is None:
        fail(f'节点不存在: {args.name}')
    if parent is None:
        fail('不能删除根节点')
    # 注意：连带删除该节点下所有子节点（无回收站），调用前请确认
    parent['children'] = [c for c in parent['children'] if c.get('id') != node['id']]
    _collect_ids(node, tree.setdefault('pending_vectors', []))
    save_tree(args.project_dir, tree)
    out({'ok': True, 'action': 'removed', 'node_id': node['id']})


def cmd_node_get(args):
    """按名精确读取节点（复习/回讲对照基准、写回前读旧摘要用；不走相似度检索，不会漏召）。"""
    tree = load_tree(args.project_dir)
    if tree is None:
        fail('tree.json 不存在，请先执行 init')
    node, parent = find_node(tree['root'], name=args.name)
    if node is None:
        fail(f'节点不存在: {args.name}')
    path = []

    def _collect(n, cur):
        cur = cur + [n['name']]
        if n is node:
            path.extend(cur)
            return True
        for c in n.get('children', []):
            if _collect(c, cur):
                return True
        return False

    _collect(tree['root'], [])
    out({
        'ok': True,
        'id': node.get('id'),
        'name': node.get('name'),
        'summary': node.get('summary', ''),
        'mastery': node.get('mastery'),
        'error_count': node.get('error_count'),
        'last_tested': node.get('last_tested'),
        'last_learned': node.get('last_learned'),
        'review_due': node.get('review_due'),
        'review_interval_days': node.get('review_interval_days'),
        'relations': node.get('relations', []),
        'source': node.get('source') or 'user',
        'attachments': node.get('attachments', []),
        'parent': parent['name'] if parent else None,
        'path': path,
    })


def cmd_search(args):
    tree = load_tree(args.project_dir)
    if tree is None:
        fail('tree.json 不存在，请先执行 init')
    qv = embed(args.query, with_instruction=not args.no_instruction)
    if qv is None:
        fail('查询向量生成失败（嵌入模型不可用）')
    hits = hr_search(tree, qv, threshold=args.threshold, top_k=args.top_k)
    hits = _filter_source(hits, args.source)
    for h in hits:
        h['utility'] = round(node_utility(h['sim'], h['mastery'],
                                          h['error_count'], h['last_tested']), 4)
    out({'ok': True, 'query': args.query, 'threshold': args.threshold,
         'source': args.source,
         'total_nodes': count_nodes(tree['root']), 'hits': hits})


def cmd_recall(args):
    """检索 + 焦点缓冲合并 + 贪心背包预算，输出注入上下文块。"""
    tree = load_tree(args.project_dir)
    if tree is None:
        fail('tree.json 不存在，请先执行 init')
    qv = embed(args.query, with_instruction=not args.no_instruction)
    if qv is None:
        fail('查询向量生成失败（嵌入模型不可用）')

    candidates = []

    # 1. 知识树候选（HRS，支持 --source 过滤）
    hits = hr_search(tree, qv, threshold=args.threshold, top_k=None)
    hits = _filter_source(hits, args.source)
    for h in hits:
        content = node_content(h)
        candidates.append({
            'source': 'tree', 'content': content,
            'utility': node_utility(h['sim'], h['mastery'], h['error_count'], h['last_tested']),
            'token_cost': estimate_tokens(content),
            'meta': {'node_id': h['id'], 'name': h['name'], 'sim': h['sim'],
                     'path': h['path']},
        })

    # 2. 焦点缓冲候选（AI 已理解后精简传入，CLI 仅兜底截断；用户原句不截断不改字）
    buffer_entries = []
    recent_skipped = 0
    if args.recent:
        for line in args.recent.split('\n'):
            line = line.rstrip('\r')
            if not line.strip():
                continue
            parsed = _parse_recent(line)
            if parsed:
                role, text = parsed
                if role == 'user_quote':
                    buffer_entries.append({'role': role, 'text': text})
                else:
                    limit = USER_MAX_CHARS if role == 'user_simplified' else AI_MAX_CHARS
                    buffer_entries.append({'role': role, 'text': truncate(text, limit)})
            else:
                recent_skipped += 1  # 无法解析的行静默丢弃会丢信息，计数供调用方检查
    n = len(buffer_entries)
    for i, e in enumerate(buffer_entries):
        recency = (i + 1) / n if n else 0.0
        content = f"{ROLE_LABEL.get(e['role'], e['role'])}: {e['text']}"
        candidates.append({
            'source': 'buffer', 'content': content,
            'utility': 0.5 + 0.5 * recency,
            'token_cost': estimate_tokens(content),
            'meta': {'role': e['role'], 'recency': round(recency, 3)},
        })

    # 3. 贪心背包
    selected, remaining = knapsack(candidates, args.budget)

    # 4. 保底：最近 1 轮（最后一条 assistant 行 + 其同轮（其前最近的）user 类行）强制入选，
    #    防对话衔接断裂；该轮无 user 类行则仅保底 assistant 行。允许小幅超出预算。
    if buffer_entries:
        buff_cands = [c for c in candidates if c['source'] == 'buffer']
        guard = set()
        last_a_idx = None
        for i, c in enumerate(buff_cands):
            if c['meta']['role'] == 'assistant':
                last_a_idx = i
        if last_a_idx is not None:
            guard.add(id(buff_cands[last_a_idx]))
            for c in reversed(buff_cands[:last_a_idx]):
                if c['meta']['role'] in ('user_simplified', 'user_quote'):
                    guard.add(id(c))
                else:
                    break
        for c in buff_cands:
            if id(c) in guard and c not in selected and c['token_cost'] <= args.budget:
                selected.append(c)
        selected.sort(key=lambda c: c['utility'] / max(1, c['token_cost']), reverse=True)

    # 5. 格式化注入块
    tree_sel = [c for c in selected if c['source'] == 'tree']
    buf_sel = [c for c in selected if c['source'] == 'buffer']
    sections = []
    if tree_sel:
        lines = [f"- {c['content']}" for c in tree_sel]
        sections.append('## 知识树检索（tree.json）\n' + '\n'.join(lines))
    if buf_sel:
        lines = [c['content'] for c in buf_sel]
        sections.append('## 近期对话（焦点缓冲）\n' + '\n'.join(lines))
    block = '\n\n'.join(sections)

    out({'ok': True, 'query': args.query, 'budget': args.budget,
         'source': args.source,
         'total_tokens': sum(c['token_cost'] for c in selected),
         'remaining': remaining,
         'tree_hits': len(hits), 'tree_selected': len(tree_sel),
         'buffer_selected': len(buf_sel), 'recent_skipped': recent_skipped,
         'hits': hits,
         'context_block': block,
         'selected': selected})


REVIEW_SECTIONS = [
    ('what', '学了什么'),
    ('insight', '关键理解'),
    ('practice', '练习/输出情况'),
    ('weak', '暴露的薄弱点'),
    ('next', '下一步建议'),
]


def _parse_review_block(block_text):
    """解析已有小节块，返回 {标题: [原文行...]}（行保留原格式，含 '- ' 前缀）。"""
    sections = {}
    current = None
    for line in block_text.splitlines():
        if line.startswith('### '):
            current = line[4:].strip()
            sections[current] = []
        elif current and line.strip():
            sections[current].append(line.strip())
    return sections


def cmd_review(args):
    rp = review_path(args.project_dir)
    if not os.path.exists(rp):
        os.makedirs(args.project_dir, exist_ok=True)
        _atomic_write(rp, f'# 温故知新 - {os.path.basename(os.path.normpath(args.project_dir))}\n\n')
    with open(rp, encoding='utf-8') as f:
        content = f.read()

    header = f'## {args.date} 小节：{args.section}'
    if header in content:
        # 更新已有小节块：只替换本次传入的字段，未传字段保留原文（防部分更新丢数据）。
        idx = content.index(header)
        nxt = content.find('\n## ', idx + 2)
        if nxt == -1:
            nxt = len(content)
        old_block = content[idx:nxt]
        old_sections = _parse_review_block(old_block)
        parts = [header, '']
        for attr, label in REVIEW_SECTIONS:
            val = getattr(args, attr)
            if val is not None:
                parts.append(f'### {label}')
                parts.append(_fmt_list(val))
            else:
                keep = old_sections.get(label)
                if keep:
                    parts.append(f'### {label}')
                    parts.append('\n'.join(keep))
                else:
                    parts.append(f'### {label}')
                    parts.append('- （暂无记录）')
            parts.append('')
        block = '\n'.join(parts).rstrip() + '\n'
        content = content[:idx] + block + content[nxt:]
        action = 'updated'
    else:
        block = (
            f'{header}\n\n'
            f'### 学了什么\n{_fmt_list(args.what)}\n\n'
            f'### 关键理解\n{_fmt_list(args.insight)}\n\n'
            f'### 练习/输出情况\n{_fmt_list(args.practice)}\n\n'
            f'### 暴露的薄弱点\n{_fmt_list(args.weak)}\n\n'
            f'### 下一步建议\n{_fmt_list(args.next)}\n'
        )
        if content and not content.endswith('\n\n'):
            content += '\n\n'
        content += block
        action = 'appended'
    _atomic_write(rp, content)
    out({'ok': True, 'action': action, 'review': rp})


def _fmt_list(text):
    if not text:
        return '- （暂无记录）'
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    return '\n'.join(f'- {l}' for l in lines)


META_OVERVIEW_FIELDS = [
    ('goal', '学习目标'),
    ('stage', '当前阶段'),
    ('stage_goal', '阶段目标'),
    ('checkpoint', '最近检查点'),
    ('weak_points', '已知薄弱点'),
    ('next_actions', '下一步行动'),
    ('outputs', '已完成输出'),
    ('notes', '备注'),
    ('last_session_date', '上次学习时间'),
]


def _sync_review_overview(tree, project_dir):
    """meta 写后同步 _review.md 头部项目概览（人类可读），保持两文件一致。"""
    rp = review_path(project_dir)
    if not os.path.exists(rp):
        return
    with open(rp, encoding='utf-8') as f:
        content = f.read()
    meta = tree.get('meta') or {}
    today = date.today().isoformat()
    lines = [f'## 项目概览（{today}）', '']
    for key, label in META_OVERVIEW_FIELDS:
        v = meta.get(key)
        if v is None:
            continue
        if isinstance(v, list):
            v = '；'.join(str(x) for x in v)
        lines.append(f'- **{label}**：{v}')
    block = '\n'.join(lines) + '\n'
    marker = '## 项目概览（'
    idx = content.find(marker)
    if idx != -1:
        nxt = content.find('\n## ', idx + 2)
        if nxt == -1:
            nxt = len(content)
        content = content[:idx] + block + content[nxt:]
    else:
        # 无概览块：插到文件头引用行之后
        anchor = '> 本文件'
        ai = content.find(anchor)
        if ai != -1:
            nl = content.find('\n', ai)
            insert_at = nl + 1 if nl != -1 else len(content)
        else:
            insert_at = content.find('\n\n')
            if insert_at == -1:
                insert_at = len(content)
        content = content[:insert_at] + block + content[insert_at:]
    _atomic_write(rp, content)


def cmd_meta(args):
    """项目元数据 meta 读写（存于 tree.json 顶层，记录项目级信息：目标/阶段/薄弱点/下一步等）。"""
    tree = load_tree(args.project_dir)
    if tree is None:
        fail('tree.json 不存在，请先执行 init')
        return
    meta = tree.get('meta') or {}
    if args.op == 'get':
        keys = args.keys or None
        res = {k: meta.get(k) for k in keys} if keys else dict(meta)
        out({'ok': True, 'meta': res})
        return
    # set
    if args.patch:
        try:
            patch = json.loads(args.patch)
        except json.JSONDecodeError as e:
            fail(f'--patch 不是合法 JSON: {e}')
            return
        if not isinstance(patch, dict):
            fail('--patch 必须是 JSON 对象')
            return
        meta.update(patch)
    elif args.key is not None:
        meta[args.key] = args.value
    else:
        fail('set 需要 --patch 或 --key/--value')
        return
    tree['meta'] = meta
    tree['updated'] = datetime.now().isoformat(timespec='seconds')
    save_tree(args.project_dir, tree)
    _sync_review_overview(tree, args.project_dir)
    out({'ok': True, 'meta_keys': sorted(meta.keys())})


def build_review_plan(due, per_unit=5, summary_limit=30):
    """schedule due --plan：按单元（直接父节点）分组压缩为紧凑复习计划（纯文本给 AI 读）。

    单元 = path 倒数第二级（顶层节点自成一单元）；组内按 review_due 升序（超期最久在前）；
    每组上限 per_unit 项（呼应 SKILL.md §5 "一次复习 3~5 项为上限"），超出标注顺延，
    不进入计划但顶部总览的到期总数 M 照实报。bge 无生成能力，压缩仅结构化逻辑。
    """
    units = {}
    for item in due:
        path = item.get('path') or []
        unit = path[-2] if len(path) >= 3 else item['name']
        units.setdefault(unit, []).append(item)
    lines = [f"复习计划（{len(units)}个单元 / {len(due)}项到期）："]
    for unit, items in units.items():
        items.sort(key=lambda x: x['review_due'])
        head, deferred = items[:per_unit], len(items) - per_unit
        parts = []
        for it in head:
            meta = []
            if it.get('mastery') is not None:
                meta.append(f"{it['mastery']:g}")
            if it.get('days_since') is not None:
                meta.append(f"隔{it['days_since']}天")
            m = f"({','.join(meta)})" if meta else ""
            parts.append(f"{it['name']}-{truncate(it.get('summary') or '', summary_limit)}{m}")
        suffix = f"，+{deferred}顺延" if deferred > 0 else ""
        lines.append(f"▸ {unit}（{len(head)}项{suffix}）：{'；'.join(parts)}")
    return '\n'.join(lines)


def cmd_schedule_due(args):
    """复习到期检查：review_due（下次复习日期）为唯一调度权威。

    到期判定：review_due <= 今天（由本命令计算，AI 不手工算）；无 review_due 的节点不调度。
    --plan 模式：按单元分组输出紧凑复习计划（文本），避免全量节点原文进上下文。
    """
    tree = load_tree(args.project_dir)
    if tree is None:
        fail('tree.json 不存在，请先执行 init')
    today = date.today()
    due = []

    def walk(node, path):
        if node.get('id') != 'root':
            rd = node.get('review_due')
            if rd:
                try:
                    d = datetime.strptime(rd, '%Y-%m-%d').date()
                except Exception:
                    d = None
                if d is not None and d <= today:
                    ll = node.get('last_learned')
                    days = None
                    if ll:
                        try:
                            days = max(0, (today - datetime.strptime(ll, '%Y-%m-%d').date()).days)
                        except Exception:
                            days = None
                    due.append({
                        'id': node.get('id'),
                        'name': node.get('name'),
                        'summary': node.get('summary', ''),
                        'mastery': node.get('mastery'),
                        'error_count': node.get('error_count'),
                        'last_learned': ll,
                        'review_due': rd,
                        'interval_days': node.get('review_interval_days'),
                        'days_since': days,
                        'path': path,
                    })
        for c in node.get('children', []):
            walk(c, path + [c.get('name', '')])

    walk(tree['root'], [tree['root'].get('name', 'root')])
    due.sort(key=lambda x: x['review_due'])
    if args.plan:
        print(build_review_plan(due))
        return
    for d in due:
        d.pop('summary', None)
    meta = tree.get('meta') or {}
    out({'ok': True, 'today': today.isoformat(),
         'last_session_date': meta.get('last_session_date'),
         'due_count': len(due), 'due': due})


def cmd_schedule_update(args):
    """复习结果回写：更新节点的复习间隔与学习时间。

    --passed true  → 复习通过，间隔翻倍（1→2→4→…→60 封顶），mastery 可选更新
    --passed false → 复习失败，间隔重置为 1 天，error_count +1

    生命周期保护：仅限"正式学习后的检验/复习结果"回写。节点必须已有学习字段
    （last_learned/review_due/learning_history/first_learned 任一），
    即已完成 node add --learned 正式学习转正；planned 节点（从未正式学习）被拒绝——
    单纯摸底/诊断请用 node update 记录 mastery/error_count/last_tested，
    不进入复习调度（"被诊断/测试 ≠ 已学习"）。
    """
    tree = load_tree(args.project_dir)
    if tree is None:
        fail('tree.json 不存在，请先执行 init')
    node, _ = find_node(tree['root'], name=args.name)
    if node is None:
        fail(f'节点不存在: {args.name}')
    if not (node.get('review_due') or node.get('last_learned')
            or node.get('learning_history') or node.get('first_learned')):
        fail('节点尚未正式学习（planned），不能调用 schedule update：'
             '请先 node add --learned 完成正式学习转正，再进入检验/复习；'
             '单纯摸底/诊断请用 node update 记录 mastery/error_count/last_tested，'
             '不进入复习调度')
    today = date.today()
    cur = node.get('review_interval_days') or INITIAL_INTERVAL_DAYS
    if args.passed:
        new_iv = min(max(cur, INITIAL_INTERVAL_DAYS) * 2, MAX_INTERVAL_DAYS)
    else:
        new_iv = INITIAL_INTERVAL_DAYS
    node['last_learned'] = today.isoformat()
    node['review_interval_days'] = new_iv
    node['review_due'] = (today + timedelta(days=new_iv)).isoformat()
    node['last_tested'] = today.isoformat()
    if args.mastery is not None:
        node['mastery'] = max(0.0, min(1.0, float(args.mastery)))
    if not args.passed:
        node['error_count'] = (node.get('error_count') or 0) + 1
    _append_learning_history(node)
    node.setdefault('first_learned', today.isoformat())
    save_tree(args.project_dir, tree)
    out({'ok': True, 'action': 'passed' if args.passed else 'failed',
         'node_id': node['id'], 'name': node['name'],
         'new_interval_days': new_iv,
         'review_due': node['review_due'],
         'last_learned': today.isoformat(),
         'mastery': node.get('mastery'),
         'error_count': node.get('error_count')})


def cmd_status(args):
    tree = load_tree(args.project_dir)
    if tree is None:
        out({'ok': True, 'exists': False, 'project_dir': args.project_dir})
    total = count_nodes(tree['root'])
    pending = tree.get('pending_vectors', [])

    def depth(n):
        return 1 + max((depth(c) for c in n.get('children', [])), default=0)

    review_scheduled = review_overdue = 0
    today = date.today()

    def sched_walk(n):
        nonlocal review_scheduled, review_overdue
        if n.get('id') != 'root':
            rd = n.get('review_due')
            if rd:
                review_scheduled += 1
                try:
                    if datetime.strptime(rd, '%Y-%m-%d').date() <= today:
                        review_overdue += 1
                except Exception:
                    pass
        for c in n.get('children', []):
            sched_walk(c)

    sched_walk(tree['root'])
    out({'ok': True, 'exists': True, 'project': tree.get('project'),
         'total_nodes': total, 'depth': depth(tree['root']),
         'review_scheduled': review_scheduled, 'review_overdue': review_overdue,
         'pending_vectors': pending,
         'tree_file': tree_path(args.project_dir),
         'review_file': os.path.exists(review_path(args.project_dir))})


# ─────────────────────────── CLI 入口 ───────────────────────────

def main():
    parser = argparse.ArgumentParser(description='study-coach 记忆系统 CLI')
    sub = parser.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('init', help='初始化项目记忆文件')
    p.add_argument('project_dir')
    p.add_argument('--name', default=None)
    p.add_argument('--summary', default=None)
    p.set_defaults(func=cmd_init)

    p = sub.add_parser('node', help='节点操作')
    ns = p.add_subparsers(dest='op', required=True)
    pa = ns.add_parser('add')
    pa.add_argument('project_dir')
    pa.add_argument('--name', required=True)
    pa.add_argument('--summary', default=None)
    pa.add_argument('--parent', default=None)
    pa.add_argument('--mastery', type=float, default=None)
    pa.add_argument('--error_count', type=int, default=None)
    pa.add_argument('--last_tested', default=None)
    pa.add_argument('--learned', action='store_true',
                    help='标记为用户实际学习（初始化 last_learned/review_due/学习历史；'
                         '不带此参数只创建课程结构节点，不进入复习调度）')
    pa.add_argument('--source', default=None, choices=['user', 'ai', 'mixed'])
    pa.add_argument('--relations', default=None)
    pa.add_argument('--clear_relations', action='store_true')
    pa.add_argument('--attachments', default=None)
    pa.add_argument('--clear_attachments', action='store_true')
    pa.set_defaults(func=cmd_node_add)
    pu = ns.add_parser('update')
    pu.add_argument('project_dir')
    pu.add_argument('--name', required=True)
    pu.add_argument('--parent', default=None)
    pu.add_argument('--summary', default=None)
    pu.add_argument('--mastery', type=float, default=None)
    pu.add_argument('--error_count', type=int, default=None)
    pu.add_argument('--last_tested', default=None)
    pu.add_argument('--source', default=None, choices=['user', 'ai', 'mixed'])
    pu.add_argument('--relations', default=None)
    pu.add_argument('--clear_relations', action='store_true')
    pu.add_argument('--attachments', default=None)
    pu.add_argument('--clear_attachments', action='store_true')
    pu.set_defaults(func=cmd_node_update)
    pr = ns.add_parser('rm')
    pr.add_argument('project_dir')
    pr.add_argument('--name', required=True)
    pr.set_defaults(func=cmd_node_rm)
    pg = ns.add_parser('get')
    pg.add_argument('project_dir')
    pg.add_argument('--name', required=True)
    pg.set_defaults(func=cmd_node_get)

    p = sub.add_parser('search', help='HRS 检索')
    p.add_argument('project_dir')
    p.add_argument('--query', required=True)
    p.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD)
    p.add_argument('--top_k', type=int, default=10)
    p.add_argument('--no_instruction', action='store_true')
    p.add_argument('--source', default='all', choices=['user', 'ai', 'mixed', 'all'])
    p.set_defaults(func=cmd_search)

    p = sub.add_parser('recall', help='检索+缓冲+预算，输出注入块')
    p.add_argument('project_dir')
    p.add_argument('--query', required=True)
    p.add_argument('--budget', type=int, default=BUDGET_DEFAULT)
    p.add_argument('--recent', default=None)
    p.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD)
    p.add_argument('--no_instruction', action='store_true')
    p.add_argument('--source', default='all', choices=['user', 'ai', 'mixed', 'all'])
    p.set_defaults(func=cmd_recall)

    p = sub.add_parser('review', help='温故知新文件追加')
    p.add_argument('project_dir')
    p.add_argument('--date', required=True)
    p.add_argument('--section', required=True)
    p.add_argument('--what', default=None)
    p.add_argument('--insight', default=None)
    p.add_argument('--practice', default=None)
    p.add_argument('--weak', default=None)
    p.add_argument('--next', default=None)
    p.set_defaults(func=cmd_review)

    p = sub.add_parser('meta', help='项目元数据 meta 读写')
    nm = p.add_subparsers(dest='op', required=True)
    ng = nm.add_parser('get')
    ng.add_argument('project_dir')
    ng.add_argument('--keys', nargs='*', default=None)
    ng.set_defaults(func=cmd_meta)
    ns2 = nm.add_parser('set')
    ns2.add_argument('project_dir')
    ns2.add_argument('--patch', default=None)
    ns2.add_argument('--key', default=None)
    ns2.add_argument('--value', default=None)
    ns2.set_defaults(func=cmd_meta)

    p = sub.add_parser('schedule', help='复习调度（间隔复习）')
    ns = p.add_subparsers(dest='op', required=True)
    nd = ns.add_parser('due')
    nd.add_argument('project_dir')
    nd.add_argument('--plan', action='store_true', help='输出紧凑复习计划（按单元分组，文本给 AI 读）')
    nd.set_defaults(func=cmd_schedule_due)
    nu = ns.add_parser('update')
    nu.add_argument('project_dir')
    nu.add_argument('--name', required=True)
    nu.add_argument('--passed', action='store_true', help='复习通过（未传视为失败）')
    nu.add_argument('--mastery', type=float, default=None)
    nu.set_defaults(func=cmd_schedule_update)

    p = sub.add_parser('status', help='记忆状态')
    p.add_argument('project_dir')
    p.set_defaults(func=cmd_status)

    args = parser.parse_args()
    try:
        args.func(args)
    except SystemExit:
        raise
    except Exception as e:
        fail(f'{type(e).__name__}: {e}')


if __name__ == '__main__':
    main()
