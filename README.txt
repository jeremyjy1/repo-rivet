RepoRivet

仓库地址：https://github.com/jeremyjy1/repo-rivet

RepoRivet 是一个本地优先的单智能体编程工具。它使用模型原生 Tool Calling，自行实现上下文管理、文件工具、命令执行、验证与终止循环，不依赖任何 Agent 框架或服务端代码执行工具。

运行环境：Python 3.12、uv。

1. 执行 uv sync。
2. 复制 reporivet.example.toml 为 reporivet.toml，填写 OpenAI 兼容 API 的 key、base_url、model、模型真实的 context_window_tokens 和 max_output_tokens。tokenizer_encoding 可选：能够识别时使用模型 tokenizer，否则自动采用保守近似估算；真实配置已被 Git 忽略。
3. 在 `[approval]` 中选择审批模式（默认 `safe-auto`），然后运行：
   uv run reporivet run --approval-mode safe-auto --workspace ./examples/buggy_project "修复负数价格未被拒绝的问题，并运行测试"

交互对话：uv run reporivet chat --workspace ./examples/buggy_project。支持 /help、/history、/clear、/compact、/compact aggressive 和 /exit。手动压缩只处理近期原文并立即保存，固定任务和结构化状态不变。会话采用固定任务、近期工作记忆、结构化摘要和本地持久化四层记忆；原始任务不会被压缩覆盖，文件内容通过 SHA-256 判断是否需要重新注入，长命令只把头尾送入模型，完整脱敏输出保存在 .reporivet/sessions。

恢复会话：uv run reporivet chat --workspace ./examples/buggy_project --resume .reporivet/sessions/<session-id>。恢复时会核对工作区和已读文件哈希，外部变化的文件会被标记为失效并要求重新读取。/clear 只清除近期原文，保留固定任务和结构化状态。

核心能力：浏览、搜索和读取代码；安全创建、精确替换文件；限时执行本地命令；查看 Git Diff；修改后强制验证；带安全余量、服务端 usage 校准和超限恢复的上下文管理；会话恢复；四模式工具审批；工具请求、审批风险、审批决定和执行结果的实时终端显示；最大步数、运行时间、重复调用与连续失败保护；JSONL 事件日志及凭据过滤。

审批模式：`allow-all` 在硬性安全规则外自动批准；`llm-auto` 仅允许审批模型高置信批准中低风险请求；`safe-auto` 自动放行窄范围只读工具并人工确认其余操作；`always-ask` 每次询问且不复用会话授权。人工界面使用数字 `1`～`5` 选择本次批准、当前会话精确批准、本次拒绝、当前会话精确拒绝或中止 Agent。单次 `run` 在无交互终端遇到询问时默认拒绝，可在配置中改为失败。

实时状态行只显示工具名、风险、审批来源、目标路径、命令程序名、参数数量、退出码和耗时；不显示文件内容、完整命令参数或原始输出。完整结构化事件仍写入会话的 `events.jsonl`。

安全边界：审批位于 Tool Registry 与本地执行器之间。所有工具声明能力，路径和命令先规范化并评估风险；工作区外写入、软链接逃逸、提权、设备与凭据访问由硬规则拒绝，任何审批模式均不能绕过。批准后会立即重新解析路径和指纹再执行。命令不经过 shell；普通 subprocess 不是完整操作系统沙箱，请仅在可信项目中运行。
