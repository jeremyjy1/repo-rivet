RepoRivet

仓库地址：https://github.com/jeremyjy1/repo-rivet

RepoRivet 是一个本地优先的单智能体编程工具。它使用模型原生 Tool Calling，自行实现上下文管理、文件工具、命令执行、验证与终止循环，不依赖任何 Agent 框架或服务端代码执行工具。

运行环境：Python 3.12、uv。

1. 执行 uv sync。
2. 复制 reporivet.example.toml 为 reporivet.toml，填写 OpenAI 兼容 API 的 key、base_url、model、模型真实的 context_window_tokens 和 max_output_tokens。tokenizer_encoding 可选：能够识别时使用模型 tokenizer，否则自动采用保守近似估算；真实配置已被 Git 忽略。
3. 在 `[approval]` 中选择审批模式（默认 `safe-auto`），然后运行：
   uv run reporivet run --approval-mode safe-auto --workspace ./examples/buggy_project "修复负数价格未被拒绝的问题，并运行测试"

交互对话：uv run reporivet chat --workspace ./examples/buggy_project。支持 /help、/history、/clear、/compact、/compact aggressive、/approval、/approval <mode> 和 /exit。手动压缩只处理近期原文并立即保存，固定任务和结构化状态不变。会话采用固定任务、近期工作记忆、结构化摘要和本地持久化四层记忆；原始任务不会被压缩覆盖，文件内容通过 SHA-256 判断是否需要重新注入，长命令只把头尾送入模型，完整脱敏输出保存在全局的 ~/.reporivet/sessions（可用 REPORIVET_HOME 修改根目录）。

多会话管理：使用 `reporivet session list` 查看会话，`session current` 查看当前工作区选择，`session use ID` 只切换选择但不运行模型，`session resume [ID]` 恢复交互执行并先显示已保存的对话历史。直接通过 `chat --session ID` 或工作区 active 会话加载已有会话时也会显示历史，新建会话不显示空历史。还支持 `session show/new/rename/fork/archive/delete/repair`；短 ID 必须唯一，完成或失败的会话需要先 fork。每个工作区的 active 指针、meta.json、state.json、summary.json、events.jsonl 和运行锁均由本地管理；恢复时会核对已读文件哈希，将外部变化和中断工具标为未知状态，并且不会自动重试写入或命令。

核心能力：浏览、搜索和读取代码；安全创建、精确替换文件；限时执行本地命令；查看 Git Diff；修改后强制验证；带安全余量、服务端 usage 校准和超限恢复的上下文管理；会话恢复；五模式工具审批；工具请求、审批风险、审批决定和执行结果的实时终端显示；最大步数、运行时间、重复调用与连续失败保护；JSONL 事件日志及凭据过滤。

可审计决策链：RepoRivet 不请求或保存模型的原始内部思维链，而是通过 `record_decision` 元工具记录有长度限制且可验证的 Plan、Decision、Reflection 和 Final Assessment。文件修改、命令及其他副作用操作必须声明匹配的 Decision；优先在同一模型响应中携带 Decision 和 Action，也支持一次性授权紧随其后的匹配 Action，每轮最多执行一个副作用工具。协议阻断不会伪装成工具 Observation，也不会占用真实工具失败额度；实际 Observation 和验证结论由 Controller 根据本地 Tool Result 生成，并带有可引用的 `obs-*` 证据 ID。`--reasoning off|summary|trace` 控制终端展示粒度，默认 `summary`；任何 Decision 都不能绕过独立审批系统。

审批模式：`allow-all` 在硬性安全规则外自动批准；`llm-auto` 仅允许审批模型高置信批准中低风险请求；`safe-auto` 自动放行窄范围只读工具并人工确认其余操作；`always-ask` 每次询问且不复用会话授权；`read-only` 只允许类型化文件检查，直接拒绝写入、替换和通用命令。交互对话中使用 `/approval` 查看当前模式，使用 `/approval <mode>` 即时切换并保存到会话。人工界面使用数字 `1`～`5` 选择本次批准、当前会话精确批准、本次拒绝、当前会话精确拒绝或中止 Agent；选择拒绝时可继续输入可选的正确方向，该指导会返回模型并进入审计记录，但不会视为授权。单次 `run` 在无交互终端遇到询问时默认拒绝，可在配置中改为失败。

CLI 默认省略工具请求 ID、步骤号、安全低风险审批和常规自动放行；重要审批单独显示一行，耗时命令保留运行提示，实际执行的工具各用一行显示最终成功、失败、退出码和耗时。状态行不显示文件内容、完整命令参数或原始输出，完整结构化事件仍写入会话的 `events.jsonl`。模型最终回答默认使用简洁纯文本；只有用户明确要求 Markdown，或内容确实需要相应结构时才使用 Markdown。

安全边界：审批位于 Tool Registry 与本地执行器之间。所有工具声明能力，路径和命令先规范化并评估风险；工作区外写入、软链接逃逸、提权、设备与凭据访问由硬规则拒绝，任何审批模式均不能绕过。批准后会立即重新解析路径和指纹再执行。命令不经过 shell；普通 subprocess 不是完整操作系统沙箱，请仅在可信项目中运行。
