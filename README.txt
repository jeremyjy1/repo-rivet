RepoRivet

仓库地址：公开仓库建立后填写。

RepoRivet 是一个本地优先的单智能体编程工具。它使用模型原生 Tool Calling，自行实现上下文管理、文件工具、命令执行、验证与终止循环，不依赖任何 Agent 框架或服务端代码执行工具。

运行环境：Python 3.12、uv。

1. 执行 uv sync。
2. 复制 reporivet.example.toml 为 reporivet.toml，填写 OpenAI 兼容 API 的 key、base_url 和 model。真实配置已被 Git 忽略。
3. 运行：
   uv run reporivet run --workspace ./examples/buggy_project "修复负数价格未被拒绝的问题，并运行测试"

核心能力：浏览、搜索和读取代码；安全创建、精确替换文件；限时执行本地命令；查看 Git Diff；修改后强制验证；最大步数、运行时间、重复调用与连续失败保护；JSONL 事件日志及凭据过滤。

安全边界：文件工具被限制在指定工作区，命令不经过 shell，并拒绝明显危险操作；普通 subprocess 不是完整操作系统沙箱，请仅在可信项目中运行。
