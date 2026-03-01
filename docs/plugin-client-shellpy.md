# Plugin: plugin_client_shellpy

Shell.py terminal client interface. Always keep this enabled — it is the primary administration interface.

## What it provides

- SSE streaming endpoint at `POST /submit` and `GET /stream`
- Persistent session via `.aiops_session_id` file

## Port

**8765** (MCP protocol, not an HTTP API)

## Dependencies

```bash
pip install sse-starlette
```

## Configuration

In `plugins-enabled.json`:
```json
"plugin_client_shellpy": {
  "enabled": true,
  "mcp_port": 8765,
  "mcp_host": "0.0.0.0"
}
```

## Running shell.py

```bash
python shell.py                        # connects to localhost:8765
python shell.py --host <ip> --port <n> # remote server
```
