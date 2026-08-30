# study-coach

AI 学习教练技能：诊断式学习规划、前置知识补齐、费曼式回讲检验、间隔复习调度。

## 核心特性

- **诊断式学习规划**：基于需求诊断生成阶段化学习路线与检查点
- **前置知识补齐**：支持有材料（模式A）与无材料（模式B）双路径
- **费曼式回讲检验**：五要素结构 + 审稿人标准反馈，定位理解漏洞
- **间隔复习调度**：简化 SM-2 调度，间隔 1→2→4→...→60 天翻倍
- **语义知识树**：BAAI/bge-small-zh-v1.5 向量检索 + HRS 分层剪枝
- **用户可读复习记录**：`_review.md` 温故知新，支持进度追踪

## 设计参考

记忆系统参考 [HCE（Holographic Context Engine）](https://github.com/nakurian/hce) 的三大结构设计：语义树（Semantic Tree）、实体图（Entity Graph）与焦点缓冲（Focus Buffer）。

## 快速开始

> Windows 使用 `python`，Linux/macOS/Termux 使用 `python3`（以下示例以 Linux 为准）。

```bash
# 安装依赖
pip install -r requirements.txt

# 初始化项目记忆
python scripts/memory_cli.py init ./my-project --name "我的学习项目" --summary "项目描述"

# 添加知识点（--learned 标记为已学习，进入复习调度）
python scripts/memory_cli.py node add ./my-project --name "知识点名称" --summary "个性化摘要" --learned

# 查看状态
python scripts/memory_cli.py status ./my-project
```

> 首次运行会自动下载嵌入模型（约 100MB），之后可离线使用。

## 文档导航

| 文件 | 内容 |
|---|---|
| `SKILL.md` | 核心技能流程、路由与命令速查 |
| `记忆系统.md` | CLI 完整参数、批量写入、Focus Buffer、非文本材料等细节 |
| `记忆系统-设计文档.md` | 架构设计依据、部署指南、跨平台说明 |
| `异常与附录.md` | 异常处理边界与 `_review.md` 格式 |

## 版本历史

见 [CHANGELOG.md](./CHANGELOG.md)。

## 许可证

[MIT](./LICENSE) © 2026 MingMaple
