# CoreFlow × Speculative Decoding

CoreFlow 的 speculative decoding 期刊扩展工作目录快照。

- `CoreFlow_/`：CoreFlow 源码、实验脚本、结构化结果与分析报告；其中包含项目原有的 vLLM 源码副本。
- `spec_decode_experiments/`：单实例 K 扫描和双 agent 并发回放脚本。
- `datasets/`：OWL-GAIA、OWL-HumanEval 数据说明、metadata 和参考长度 trace。
- `SC_26.pdf` 与根目录 Markdown 文件：已发表论文及后续设计文档。

这是代码与分析资料备份，不是完整的原始数据归档。完整语义对话日志位于本地
`datasets/*/raw_logs/`，因仓库公开且日志可能包含敏感文本，**未上传**。
运行实验前需要按各数据集 README 中的来源说明在本地准备这些日志。
废弃结果 `.trash/`、运行日志、环境文件、编译产物和 Python 缓存也未上传。

`CoreFlow_/` 原先是独立 Git 仓库；此仓库保存其当前文件快照，不包含其内部 `.git` 历史。
其中 vLLM 的模型注册表有一处模型类名采用等价的相邻字符串字面量写法，
以避免公开仓库的密钥扫描将该类名误判为 API key；本地原文件未修改。
