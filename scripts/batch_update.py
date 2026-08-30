#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
study-coach 记忆系统批量写入工具
============================
在单个 Python 进程内加载嵌入模型一次，连续执行多个记忆写入操作
（node add/update/rm、meta set/get、review、init、status），
结果与逐次调用 memory_cli.py 完全一致（复用其 cmd_* 函数），
但避免每次调用都重新加载嵌入模型（逐次 ~20s/次 → 批量共 ~20s）。

用法：
  python3 batch_update.py <project_dir> <ops.json>

ops.json 为 JSON 数组，每项一个操作，示例：
[
  {"op": "node_add", "name": "色相", "summary": "……", "parent": "色彩三要素", "mastery": 0.85},
  {"op": "node_update", "name": "高光", "summary": "……", "mastery": 0.9, "last_tested": "2026-08-21"},
  {"op": "meta_set", "patch": {"stage": "阶段2/8", "next_actions": ["……"]}},
  {"op": "review", "date": "2026-08-21", "section": "第2课", "what": "……", "insight": "……"},
  {"op": "node_rm", "name": "临时节点"},
  {"op": "init", "name": "项目名"},
  {"op": "meta_get"},
  {"op": "status"}
]

支持的操作字段与 memory_cli.py 参数一一对应：
  node_add / node_update : name, summary, mastery, error_count, last_tested, source,
                           relations, clear_relations, attachments, clear_attachments
                           parent: node_add 指定父节点；node_update 移动节点到新父节点
  node_add 额外支持     : learned(bool，标记为用户实际学习并进入复习调度；
                           不带则只创建课程结构节点，不进入复习调度)
  node_rm                 : name
  meta_set                : patch(对象) 或 key + value
  meta_get                : keys(数组，可选)
  review                  : date, section, what, insight, practice, weak, next
  init                    : name(可选)
  schedule_due            : 无（复习到期检查）
  schedule_update         : name, passed(bool，复习通过), mastery(可选)
source / learning_history / first_learned 的维护逻辑与单次 CLI 完全一致（复用 cmd_* 函数）：
node_add 带 learned=true 的新建/同名更新/合并追加学习历史并刷新 last_learned（真正学习语义）；
schedule_update 通过/失败均追加学习历史并刷新 last_learned（检验/复习回写；**仅限已正式学习节点，
planned 节点会被拒绝，单纯摸底/诊断请用 node_update，不进入调度**）；
node_update（含诊断字段更新）不追加学习历史、不刷新 last_learned（前置诊断/数据记录 ≠ 学习）；
不带 learned 的 node_add 不追加。
project_dir 由命令行传入并自动注入所有操作，无需在 ops.json 中重复填写。
若 init 将不安全目录名安全化（如 `C++:基础` → `C++-基础`），本次 batch 后续操作自动跟随 init 返回的实际 project_dir。
"""

import argparse
import json
import os
import sys
from argparse import Namespace

# 离线策略：嵌入模型已本地安装（缓存），强制离线运行，避免加载时联网检查超时
# （实测未设离线时单次调用 >180 秒超时，此前多次触发）。
# 仅空白环境（模型未安装/首次部署）需联网下载：此时请勿强制离线，
# 先用 memory_cli.py（_setup_offline() 无缓存时不设离线，允许下载）装好模型，再使用本脚本。
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import memory_cli  # noqa: E402

# Windows 控制台默认编码（如 GBK）下强制标准输出/错误流以 UTF-8 写出（memory_cli 导入时
# 已执行 _force_utf8_stdio，此处显式重申，保证本脚本独立运行时同样生效；幂等，Linux/macOS 行为不变）。
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except (AttributeError, ValueError, OSError):
    pass


class BatchExit(Exception):
    """替代 memory_cli.out 的 sys.exit：以异常形式结束单个操作，批量继续执行后续操作。"""

    def __init__(self, obj, code):
        super().__init__()
        self.obj = obj
        self.code = code


def _out(obj, code=0):
    raise BatchExit(obj, code)


def _fail(msg):
    _out({'error': msg}, 1)


# monkey-patch：语义与原版一致（成功/失败都终止当前操作），但不退出进程
memory_cli.out = _out
memory_cli.fail = _fail


def _to_relations(relations):
    """relations 兼容：list/dict → JSON 字符串（与原版 argparse 传入的字符串一致）。"""
    if relations is None:
        return None
    if isinstance(relations, str):
        return relations
    return json.dumps(relations, ensure_ascii=False)


def _to_attachments(attachments):
    """attachments 兼容：list → JSON 字符串（与原版 argparse 传入的字符串一致）。"""
    if attachments is None:
        return None
    if isinstance(attachments, str):
        return attachments
    return json.dumps(attachments, ensure_ascii=False)


def _make_args(params, project_dir):
    """构造与原版 argparse Namespace 等价的参数对象（字段缺省为默认值）。"""
    args = Namespace(**{
        'project_dir': project_dir,
        'name': None, 'summary': None, 'parent': None,
        'mastery': None, 'error_count': None, 'last_tested': None,
        'learned': False,
        'relations': None, 'clear_relations': False,
        'attachments': None, 'clear_attachments': False,
        'source': None,
        'patch': None, 'key': None, 'value': None, 'keys': None,
        'date': None, 'section': None, 'what': None, 'insight': None,
        'practice': None, 'weak': None, 'next': None,
        'passed': False,
        'plan': False,  # schedule_due 需要（缺省会 AttributeError，历史 bug）
    })
    for k, v in params.items():
        setattr(args, k, v)
    args.relations = _to_relations(args.relations)
    args.attachments = _to_attachments(args.attachments)
    return args


def _cmd_meta_set(args):
    args.op = 'set'
    if isinstance(args.patch, dict):
        args.patch = json.dumps(args.patch, ensure_ascii=False)
    memory_cli.cmd_meta(args)


def _cmd_meta_get(args):
    args.op = 'get'
    memory_cli.cmd_meta(args)


OP_MAP = {
    'init': lambda a: memory_cli.cmd_init(a),
    'node_add': memory_cli.cmd_node_add,
    'node_update': memory_cli.cmd_node_update,
    'node_rm': memory_cli.cmd_node_rm,
    'review': memory_cli.cmd_review,
    'meta_set': _cmd_meta_set,
    'meta_get': _cmd_meta_get,
    'schedule_due': memory_cli.cmd_schedule_due,
    'schedule_update': memory_cli.cmd_schedule_update,
    'status': lambda a: memory_cli.cmd_status(a),
}


def run_op(op, params, project_dir):
    """执行单个操作，返回结果 dict（与 CLI 输出 JSON 同构，附加 op 字段）。"""
    fn = OP_MAP.get(op)
    if fn is None:
        return {'ok': False, 'op': op, 'error': f'未知操作: {op}'}
    try:
        fn(_make_args(params, project_dir))
        return {'ok': False, 'op': op, 'error': '操作未返回结果（内部异常）'}
    except BatchExit as e:
        result = dict(e.obj)
        result['op'] = op
        result['ok'] = e.code == 0
        return result
    except Exception as e:
        # 兜底：意外异常（IO/ValueError 等）不中断批量，记录错误继续后续操作
        return {'ok': False, 'op': op, 'error': f'{type(e).__name__}: {e}'}


def main():
    parser = argparse.ArgumentParser(description='批量记忆写入（单进程复用嵌入模型）')
    parser.add_argument('project_dir')
    parser.add_argument('ops_file')
    args = parser.parse_args()

    with open(args.ops_file, encoding='utf-8') as f:
        ops = json.load(f)
    if not isinstance(ops, list):
        print(json.dumps({'ok': False, 'error': 'ops 文件必须是 JSON 数组'}, ensure_ascii=False))
        sys.exit(1)

    results = []
    current_dir = args.project_dir  # init 可能把不安全目录名改为安全名并返回实际 project_dir，后续操作跟随该路径
    for item in ops:
        if not isinstance(item, dict):
            results.append({'ok': False, 'op': '?', 'error': '操作项必须是 JSON 对象'})
            continue
        op = item.get('op')
        params = {k: v for k, v in item.items() if k != 'op'}
        res = run_op(op, params, current_dir)
        if op == 'init' and res.get('ok') and res.get('project_dir'):
            current_dir = res['project_dir']  # 本次 batch 后续操作一律使用 init 返回的实际路径
        results.append(res)

    print(json.dumps({'ok': all(r.get('ok') for r in results),
                      'count': len(results), 'results': results}, ensure_ascii=False))


if __name__ == '__main__':
    main()