RepoRivet

仓库地址：https://github.com/jeremyjy1/repo-rivet

RepoRivet 是一个本地优先的单智能体编程工具。它使用模型原生 Tool Calling，自行实现上下文管理、文件工具、命令执行、验证与终止循环，不依赖任何 Agent 框架或服务端代码执行工具。

运行环境：Python 3.12、uv。

1. 执行 uv sync。
2. 复制 reporivet.example.toml 为 reporivet.toml，填写 OpenAI 兼容 API 的 key、base_url、model 和模型真实的 context_window_tokens。RepoRivet 不向模型发送输出长度限制；`token.reserved_output_tokens` 只用于上下文预算预留。`token.active_prompt_limit` 是独立于模型窗口的单轮成本上限，默认 65536；达到上限后会压缩历史并开始新的缓存纪元。tokenizer_encoding 可选：能够识别时使用模型 tokenizer，否则自动采用保守近似估算；真实配置已被 Git 忽略。
3. 在 `[approval]` 中选择审批模式（默认 `safe-auto`），然后运行：
   uv run reporivet run --approval-mode safe-auto --workspace ./examples/buggy_project "修复负数价格未被拒绝的问题，并运行测试"

交互对话：uv run reporivet chat --workspace ./examples/buggy_project。支持 /help、/history、/clear、/compact、/compact aggressive、/approval、/approval <mode> 和 /exit。手动压缩只处理近期原文并立即保存，固定任务和结构化状态不变。会话采用固定任务、近期工作记忆、结构化摘要和本地持久化四层记忆；原始任务不会被压缩覆盖，文件内容通过 SHA-256 判断是否需要重新注入，长命令只把头尾送入模型，完整脱敏输出保存在全局的 ~/.reporivet/sessions（可用 REPORIVET_HOME 修改根目录）。

本地 GUI：运行 `uv run reporivet gui --workspace ./examples/buggy_project`，命令会在随机本机端口启动服务并打开一次性认证链接。GUI 与 CLI 共用 Agent Core、会话、审批、Plan、Skill 和验证状态，提供会话切换、实时 SSE 时间线、四选项审批、Markdown 回答、文件快照、Git Diff、Plan 审阅及执行。默认只监听回环地址；非回环监听必须显式传入 `--unsafe-network`。写请求同时校验 HttpOnly 会话 Cookie、CSRF、Origin 和 Host，模型 API Key 不会发送到浏览器。前端源码位于 `frontend/`，使用 `npm run build` 将静态资源打包到 Python wheel。

多会话管理：使用 `reporivet session list` 查看会话，`session current` 查看当前工作区选择，`session use ID` 只切换选择但不运行模型，`session resume [ID]` 恢复交互执行并先显示已保存的对话历史。直接通过 `chat --session ID` 或工作区 active 会话加载已有会话时也会显示历史，新建会话不显示空历史。还支持 `session show/new/rename/fork/archive/delete/repair`；短 ID 必须唯一，完成或失败的会话需要先 fork。每个工作区的 active 指针、meta.json、state.json、summary.json、events.jsonl 和运行锁均由本地管理；恢复时会核对已读文件哈希，将外部变化和中断工具标为未知状态，并且不会自动重试写入或命令。

自动规划通过 `[planning] auto_plan = "adaptive"` 配置，也可用 `--auto-plan off|adaptive|always` 覆盖。`adaptive` 对明确的项目级、多文件或长规格任务在首次调用前进入 Plan Mode，并允许模型在发现范围不确定时通过 `request_plan` 请求切换；简单且边界明确的修改继续直接执行。自动规划只会启用 Controller 强制的只读工具集，生成的 Plan Artifact 仍停在 `plan_ready` 等待用户审阅，不会自动批准计划、编辑或命令。

核心能力：浏览、搜索和读取代码；基于持久化内容快照和原始快照行号安全创建、修改文件；限时执行本地命令；查看 Git Diff；基于显式计划和确定性成功条件的修改后验证；带安全余量、服务端 usage 校准和超限恢复的上下文管理；会话恢复；四模式工具审批；工具请求、审批风险、审批决定和执行结果的实时终端显示；基于可观测进度的模型步骤检查点、运行时间、重复调用与连续失败保护；JSONL 事件日志及凭据过滤。

文件编辑使用 RivetPatch：`read_file` 返回带行号的 `snapshot_id`，`edit_file` 只接受针对该原始快照的结构化行替换、插入和删除操作。目标行必须已经展示；过期快照、路径不匹配、越界、重叠和无效编辑会在写入前被拒绝。预检完成后审批界面展示确定性 Diff，批准后再次核对磁盘版本并执行单文件原子替换；成功结果返回新快照并使旧验证失效。`write_file` 仅创建新文件，不覆盖现有路径。当前版本不提供结构块编辑、自动三方合并或多文件批量事务。

验证系统不再根据 `pytest`、`g++` 等命令名称猜测结果。模型先通过 `register_verification` 注册带类型、结构化命令、成功条件和必需 Claim 的 Verification Plan；普通 `run_command` 只产生 `[OBSERVE]` 进程事实，`run_verification CHECK_ID` 才会执行已注册检查并由本地规则产生 `[VERIFY]` 结果。Controller 在最终回答时自动调度尚未运行或已因文件修改而过期的必需检查，全部通过才完成；行为检查没有输出或产物 Oracle 时为 `inconclusive`。每次成功写入都会递增 workspace revision，使旧验证变为 `stale`。

对于要求回传思考模式状态的 OpenAI-compatible 网关，RepoRivet 会在当前调用链内原样重放 `reasoning_content`，并对 `finish_reason=length` 做最多三次有界续传；隐藏思考和内部续传提示不会写入会话存储。结果面板使用 `passed`、`failed`、`not run`、`not applicable` 表达验证状态，不再用布尔值把“未修改、未验证”显示成验证通过。已完成任务之后的新用户任务会建立独立的修改与验证证据范围。

可审计决策链：RepoRivet 不请求或保存模型的原始内部思维链，而是通过 `record_decision` 元工具记录有长度限制且可验证的 Plan、Decision、Reflection 和 Final Assessment。文件修改、命令及其他副作用操作必须声明匹配的 Decision；优先在同一模型响应中携带 Decision 和 Action，也支持一次性授权紧随其后的匹配 Action，每轮最多执行一个副作用工具。协议阻断不会伪装成工具 Observation，也不会占用真实工具失败额度；实际 Observation 由 Controller 根据本地 Tool Result 生成，并带有可引用的 `obs-*` 证据 ID。模型 Final Assessment 只显示为 `[ASSESS]`，不能修改验证状态；`[VERIFY]` 只来自本地 Verification Result。`--reasoning off|summary|trace` 控制终端展示粒度，默认 `summary`；任何 Decision 都不能绕过独立审批系统。

审批模式：`allow-all` 在硬性安全规则外自动批准；`llm-auto` 仅允许审批模型高置信批准中低风险请求；`safe-auto` 自动放行窄范围只读工具并人工确认其余操作；`always-ask` 对尚无匹配会话授权的请求逐次询问。交互对话中使用 `/approval` 查看当前模式，使用 `/approval <mode>` 即时切换并保存到会话。人工界面使用数字 `1`～`4` 选择本次允许、当前会话允许匹配请求、拒绝并继续或停止当前运行并保存会话；选择拒绝并继续后可选填原因或后续方向，提供的反馈会返回模型并进入审计记录，但不会视为授权。单次 `run` 在无交互终端遇到询问时默认拒绝，可在配置中改为失败。

Plan Mode 是 Controller 强制执行的只读规划工作流，而不是审批模式。使用 `reporivet plan <task>`，或在对话中输入 `:plan`、`/approval plan` 进入；后者是工作流快捷入口，不会把 `plan` 加入审批策略枚举。规划阶段只暴露项目浏览、搜索、Git 检查和结构化计划工具，写文件、运行命令与验证会被本地拒绝。计划会保存目标、证据、步骤、风险、验证要求、工作区修订和文件快照，经过本地验证后等待用户选择执行、修改、继续检查或取消。批准计划只切换到执行阶段，不会绕过任何工具审批。对话中使用 `:execute`、`:revise`、`:inspect` 和 `:cancel` 管理该流程。

Skill 是可复用、可版本化的声明式任务策略，不是可执行插件，也不会授予权限。RepoRivet 将 Skill 分为两层：随程序发布的系统 Skill 会在每个运行时按稳定顺序全部加载，但只在当前任务符合其目标或触发条件时提供指导，不收窄工具、也不强制任务级完成条件；用户安装在 `~/.reporivet/skills/<id>/SKILL.md` 的全局 Skill 对所有工作区可见，可为某个会话显式选择，并进一步收窄当前模式和全局策略已经允许的工具。所有实际调用仍经过参数验证、审批、路径策略与执行器，选中的全局 Skill 固定编辑前和完成条件由 Controller 本地判断。使用 `reporivet skill list/show/install/uninstall/use/clear` 管理系统与全局 Skill；`skill install --replace` 显式更新同 ID 的全局 Skill。运行时可传入 `--skill ID`、`--no-skills`，其中 `--no-skills` 只禁用会话选择的全局 Skill，系统 Skill 始终加载；对话中支持 `:skills`、`:skill current/show/use/clear`。选中的全局 Skill 会被会话和 Plan Artifact 固定 ID、版本与内容哈希，内容变化后必须重新选择，旧计划会变为 stale。

Skill 创作采用确定性工具链：`reporivet skill init ID` 在 `reporivet-skills/` 生成不会覆盖现有文件的原生草稿，`skill validate PATH` 检查完整 Schema、正文预算、工具名和固定条件，`skill convert SOURCE` 将原生、Codex 风格、Claude 风格或普通 Markdown 安全转换并报告被删除字段和无法映射的工具，`skill install PATH` 在再次校验后安装到用户全局目录且拒绝隐式覆盖，更新必须显式使用 `--replace`。转换永远剥离 Hook、脚本、回调、动态代码和未知权限声明；安装、更新和卸载只能由用户显式执行。系统 `skill-authoring` 可指导 Agent 在工作区设计或转换草稿，但不能自行安装。首版仍只支持一个会话选中的全局 Skill，不支持自动路由、项目 Skill、远程市场、依赖解析、Verification Profile 转换或可执行 Hook；系统 Skill 还包括 `repository-onboarding` 和 `test-failure-fix`。

CLI 默认省略工具请求 ID、步骤号、安全低风险审批和常规自动放行；重要审批单独显示一行，耗时命令保留运行提示，实际执行的工具各用一行显示最终成功、失败、退出码和耗时。状态行不显示文件内容、完整命令参数或原始输出，完整结构化事件仍写入会话的 `events.jsonl`。模型最终回答默认使用简洁纯文本；只有用户明确要求 Markdown，或内容确实需要相应结构时才使用 Markdown。

安全边界：审批位于 Tool Registry 与本地执行器之间。所有工具声明能力，路径和命令先规范化并评估风险；工作区外写入、软链接逃逸、提权、设备与凭据访问由硬规则拒绝，任何审批模式均不能绕过。批准后会立即重新解析路径和指纹再执行。命令不经过 shell；普通 subprocess 不是完整操作系统沙箱，请仅在可信项目中运行。
