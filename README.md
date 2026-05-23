<p align="center">
  <h1 align="center">cold-agent-sdk</h1>
  <p align="center">LAMCLOD Agent 编排 SDK</p>
  <p align="center">
    <img src="https://img.shields.io/badge/language-Rust-orange?style=flat-square" alt="Rust">
    <img src="https://img.shields.io/badge/tests-88_pass-brightgreen?style=flat-square" alt="tests">
    <img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="MIT">
  </p>
</p>

---

## 简介

cold-agent-sdk 是 LAMCLOD 的 Agent 编排层，将 cold-sdk（传输）、cold-context（上下文）、cold-tools（工具）组合为一行代码即可运行的 AI Agent。

## 特性

| | |
|---|---|
| **Agent Loop** | 流式/非流式双路径，自动工具调度 |
| **Sub-Agent** | delegate_task 派发子 agent（General/Explore/Plan） |
| **上下文压缩** | 自动触发 + reactive compact 错误恢复 |
| **Model Fallback** | 主模型失败自动切换备用 |
| **Hooks** | pre/post tool call、pre/post compact、session start、stop |
| **Memory** | .cold/memory/*.md 持久记忆注入 |
| **Skills** | SKILL.md 触发匹配 + prompt 注入 |
| **Session** | JSON + JSONL 双重持久化 + metadata |
| **Plan Mode** | 模型可自主进入/退出只读模式 |
| **Prompt Cache** | 静态/动态边界标记 + 工具排序 |

## 安装

```toml
[dependencies]
cold-agent-sdk = "0.1"
```

## 用法

```rust
use cold_agent_sdk::{Agent, AgentConfig, PrintCallback};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = AgentConfig::new("gpt-4o", 128_000, "sk-xxx")
        .with_root_dir("./my-project")
        .with_system_prompt("You are a Rust expert.");

    let mut agent = Agent::new(config)?;
    let agent = agent.with_callback(PrintCallback);
    let result = agent.run("Read src/main.rs and add error handling").await?;

    println!("Turns: {}, Tokens: {}", result.turns_used, result.tokens.total_tokens);
    Ok(())
}
```

## Cold Stack

cold-cli 是 LAMCLOD 的 AI 编码助手 CLI，基于以下 4 个 Rust crate 构建：

```
cold-cli              CLI 入口
  |
cold-agent-sdk        Agent 编排 (loop + sub-agent + hooks + memory)
  |
  +-- cold-context    上下文管理 (压缩 + 安全 + 预算)
  +-- cold-tools      工具框架 + 20 内置工具 + MCP
  |
cold-sdk              API 传输层 (HTTP/2 + SSE + 重试)
```

| Crate | 描述 |
|-------|------|
| [cold-sdk](https://github.com/LamClod/cold-sdk) | API 通信层 |
| [cold-context](https://github.com/LamClod/cold-context) | 上下文窗口管理 |
| [cold-tools](https://github.com/LamClod/cold-tools) | 工具协议框架 |
| [cold-agent-sdk](https://github.com/LamClod/cold-agent-sdk) | Agent 编排 SDK |
| [cold-cli](https://github.com/LamClod/cold-cli) | 命令行界面 |

## License

MIT
