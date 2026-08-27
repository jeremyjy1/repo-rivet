RepoRivet

仓库地址：https://github.com/jeremyjy1/repo-rivet

RepoRivet 是一个本地优先的单智能体编程工具。它使用模型原生 Tool Calling，自行实现上下文管理、文件工具、命令执行、验证与终止循环，不依赖任何 Agent 框架或服务端代码执行工具。

运行环境：Python 3.12、uv。

1. 执行 uv sync。
2. 复制 reporivet.example.toml 为 reporivet.toml，填写 OpenAI 兼容 API 的 key、base_url、model、模型真实的 context_window_tokens 和 max_output_tokens。tokenizer_encoding 可选：能够识别时使用模型 tokenizer，否则自动采用保守近似估算；真实配置已被 Git 忽略。
3. 运行：
   uv run reporivet run --workspace ./examples/buggy_project "修复负数价格未被拒绝的问题，并运行测试"

交互对话：uv run reporivet chat --workspace ./examples/buggy_project。支持 /help、/history、/clear、/compact、/compact aggressive 和 /exit。手动压缩只处理近期原文并立即保存，固定任务和结构化状态不变。会话采用固定任务、近期工作记忆、结构化摘要和本地持久化四层记忆；原始任务不会被压缩覆盖，文件内容通过 SHA-256 判断是否需要重新注入，长命令只把头尾送入模型，完整脱敏输出保存在 .reporivet/sessions。

恢复会话：uv run reporivet chat --workspace ./examples/buggy_project --resume .reporivet/sessions/<session-id>。恢复时会核对工作区和已读文件哈希，外部变化的文件会被标记为失效并要求重新读取。/clear 只清除近期原文，保留固定任务和结构化状态。

核心能力：浏览、搜索和读取代码；安全创建、精确替换文件；限时执行本地命令；查看 Git Diff；修改后强制验证；带安全余量、服务端 usage 校准和超限恢复的上下文管理；会话恢复；最大步数、运行时间、重复调用与连续失败保护；JSONL 事件日志及凭据过滤。

安全边界：文件工具被限制在指定工作区，命令不经过 shell，并拒绝明显危险操作；普通 subprocess 不是完整操作系统沙箱，请仅在可信项目中运行。
