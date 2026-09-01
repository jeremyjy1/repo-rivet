RepoRivet

仓库地址：https://github.com/jeremyjy1/repo-rivet

项目简介

RepoRivet 是一个本地优先、事件驱动且可恢复的编程智能体。项目未使用任何 Agent 框架；模型只负责推理与原生 Tool Calling，文件读写、命令执行、上下文管理、审批、验证、错误恢复和终止判断均由本地 Agent Core 自行实现。

如何运行

环境：Python 3.12、uv。

1. 执行 uv sync。
2. 复制 reporivet.example.toml 为 reporivet.toml，填写 API Key、base_url、model 和 context_window_tokens；该文件已被 Git 忽略。
3. 启动 CLI：
   uv run reporivet chat --workspace ./examples/buggy_project
4. 启动本地 GUI：
   uv run reporivet gui --workspace ./examples/buggy_project

特色功能

1. 严谨状态机：Controller 强制隔离 Plan 与 Execute；计划绑定证据、快照和工作区修订，执行可审阅、恢复和打断转向。
2. 安全本地执行：RivetPatch 通过快照行号、预检 Diff 和原子写入拒绝陈旧编辑；路径策略、审批和硬规则共同约束工具。
3. 可验证完成：模型注册 Verification Plan，本地依据退出码、输出或产物判定；文件变化会使旧验证失效，必需检查通过后才能成功。
4. 可控上下文：完整请求保守估算 Token，并以服务端 usage 校准；支持自动/手动压缩、输出分层保存、超限恢复和会话续接。
5. 代码理解与协作：Tree-sitter 增量索引提供符号、定义、引用和语法诊断；只读子代理可收集证据，但不能写文件、运行命令或绕过审批。
6. 可审计交互：CLI 与 GUI 共用 Agent Core、会话和事件日志，实时展示计划、决策摘要、审批、Diff 与验证状态，不保存原始思维链。

安全说明

API Key 仅通过未入库配置提供，不会发送到浏览器。命令不经过 shell；请只在可信项目中运行。
