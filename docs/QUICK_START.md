# Quick Start

## Minimal setup (shell.py only)

```bash
source venv/bin/activate
python agent-mcp.py
```

In a second terminal:
```bash
python shell.py
```

Type `!help` to see all commands.

## With llama proxy (external apps)

Enable the llama proxy plugin first:
```bash
python agentctl.py enable plugin_proxy_llama
```

Then start the server — it now listens on both ports:
- **8765** — shell.py (MCP protocol)
- **11434** — OpenAI/Ollama API for external apps

Point any OpenAI or Ollama-compatible app at `http://localhost:11434`.

## Essential commands

```
!model                  list available LLMs (current marked with *)
!model gemini25         switch to a different model
!autoAIdb read true     auto-allow all DB reads (no gate pop-ups)
!autogate search true   auto-allow all web searches
!tool_preview_length 0  show full tool results (no truncation)
!reset                  clear conversation history
!help                   full command reference
```

## Per-turn model switch

Prefix any message with `@ModelName` to use a different model for one turn (bypasses all gates):

```
@Win11Local extract https://www.example.com and summarize it
```

## Managing plugins and models

```bash
python agentctl.py           # interactive menu
python agentctl.py list      # plugin status overview
python agentctl.py models    # model list
```

See [ADMINISTRATION.md](ADMINISTRATION.md) for full details.
See [ARCHITECTURE.md](ARCHITECTURE.md) for system internals.
