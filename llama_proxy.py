#!/usr/bin/env python3
"""
DEPRECATED: This standalone llama proxy is deprecated.

Use main.py with --llama-proxy instead:
    python main.py --llama-proxy

The llama proxy functionality has been merged into main.py to provide:
- Unified server for both shell.py clients and external chat apps
- Shared command processing logic
- Single session management
- No redundant configuration (LLM backends already in LLM_REGISTRY)

Key changes from the old standalone proxy:
- No need for --llama-host argument (backends defined in LLM_REGISTRY)
- Listens on port 11434 by default when --llama-proxy is enabled
- All requests routed through MCP's dispatch_llm()
- Use !model <name> to switch between LLMs (Win11, gemini-2.5-flash, etc.)

See LLAMA_PROXY_USAGE.md for detailed usage instructions.

---

Original Documentation:
Llama Server Proxy with Special Command Interception
Proxies requests from local clients to a remote llama-server on port 11434
with support for intercepting special commands (!command format)

# Debug mode (verbose)
python3 llama_proxy.py --remote-host 192.168.1.100 --debug DEBUG

# Quiet mode (warnings/errors only)
python3 llama_proxy.py --remote-host 192.168.1.101 --remote-port 11434 --listen-port 11434  --debug WARNING


# Test with a simple completion request
curl http://localhost:11434/api/generate -d '{
  "model": "Qwen2.5-Coder-7B-Instruct-abliterated-Q4_K_M.gguf",
  "prompt": "Hello, world!"
}'
 curl http://YOUR_OLLAMA_HOST:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2.5-Coder-7B-Instruct-abliterated-Q4_K_M.gguf",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "Hello, world!"
      }
    ],
    "temperature": 0.7
  }'

"""

import asyncio
import argparse
import json
from aiohttp import web, ClientSession, ClientTimeout
import logging
from typing import Optional, Dict, Any

# Logging setup will be configured based on CLI args
logger = logging.getLogger(__name__)


class LlamaProxy:
    def __init__(self, remote_host: str, remote_port: int = 11434):
        self.remote_host = remote_host
        self.remote_port = remote_port
        self.remote_url = f"http://{remote_host}:{remote_port}"
        self.app = web.Application()
        self.setup_routes()
        
    def setup_routes(self):
        """Set up route handlers for all requests"""
        # Catch-all route handler
        self.app.router.add_route('*', '/{path:.*}', self.proxy_handler)
    
    def _extract_prompt_from_request(self, request_data: Dict[str, Any]) -> Optional[str]:
        """
        Extract prompt text from various request formats
        Handles both /api/generate (prompt field) and /v1/chat/completions (messages array)
        """
        # Check for direct prompt field (Ollama /api/generate format)
        if 'prompt' in request_data:
            return request_data['prompt']
        
        # Check for messages array (OpenAI /v1/chat/completions format)
        if 'messages' in request_data and isinstance(request_data['messages'], list):
            messages = request_data['messages']
            if messages:
                # Get the last user message
                for msg in reversed(messages):
                    if isinstance(msg, dict) and msg.get('role') == 'user':
                        content = msg.get('content', '')
                        if isinstance(content, str):
                            return content
                        elif isinstance(content, list):
                            # Handle content array (multimodal)
                            for item in content:
                                if isinstance(item, dict) and item.get('type') == 'text':
                                    return item.get('text', '')
        
        return None
    
    async def handle_special_command(self, request_data: Dict[str, Any]) -> Optional[web.Response]:
        """
        Handle special commands starting with !
        Returns None if not a special command, otherwise returns a Response
        """
        prompt = self._extract_prompt_from_request(request_data)
        
        if not prompt or not prompt.strip().startswith('!'):
            logger.debug("Not a special command")
            return None
        
        # Extract the command (first word after !)
        command_line = prompt.strip()
        parts = command_line.split(None, 1)
        cmd = parts[0]
        args = parts[1] if len(parts) > 1 else ""
        
        logger.info(f"Intercepted special command: {cmd} with args: {args}")
        
        # Handle different special commands
        if cmd == '!specialcommand':
            response_text = self._handle_specialcommand(args)
        else:
            # Unknown special command
            response_text = f"Unknown command: {cmd}\nAvailable commands: !specialcommand"
            logger.warning(f"Unknown special command: {cmd}")
        
        # Return response in appropriate format
        return response_text
    
    def _handle_specialcommand(self, args: str) -> str:
        """
        Handle !specialcommand - acknowledgement stub
        """
        logger.info(f"Executing !specialcommand with args: '{args}'")
        
        if args:
            return f"Special command acknowledged with args: {args}"
        else:
            return "Special command acknowledged"
    
    def _create_llama_response(self, text: str, original_request: Dict[str, Any], path: str) -> web.Response:
        """
        Create a response in the appropriate format based on the endpoint
        """
        # Check if streaming was requested
        stream = original_request.get('stream', False)
        
        # Determine response format based on path
        if path.startswith('v1/'):
            # OpenAI-compatible format
            if stream:
                return None  # Will be handled by streaming method
            else:
                response_data = {
                    "id": "chatcmpl-proxy",
                    "object": "chat.completion",
                    "created": 0,
                    "model": original_request.get("model", "proxy"),
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": text
                        },
                        "finish_reason": "stop"
                    }]
                }
        else:
            # Ollama format
            response_data = {
                "model": original_request.get("model", "proxy"),
                "created_at": "",
                "response": text,
                "done": True
            }
        
        return web.Response(
            text=json.dumps(response_data),
            content_type='application/json',
            status=200
        )
        
    async def proxy_handler(self, request: web.Request) -> web.Response:
        """
        Main proxy handler - intercepts special commands or forwards to remote llama-server
        """
        path = request.match_info.get('path', '')
        method = request.method
        target_url = f"{self.remote_url}/{path}"
        
        logger.debug(f"Received {method} request for path: /{path}")
        
        try:
            # Read request body
            body = await request.read()
            
            # Check for special commands (only for generate/chat endpoints)
            if path in ['api/generate', 'api/chat', 'v1/chat/completions'] and method == 'POST':
                try:
                    request_data = json.loads(body) if body else {}
                    logger.debug(f"Request data: {json.dumps(request_data, indent=2)}")
                    
                    # Check if this is a special command
                    special_response_text = await self.handle_special_command(request_data)
                    if special_response_text:
                        logger.info("Special command handled, returning direct response")
                        
                        # Handle streaming special commands
                        if request_data.get('stream', False):
                            return await self._handle_streaming_special_command(
                                special_response_text, 
                                request_data, 
                                request,
                                path
                            )
                        else:
                            return self._create_llama_response(special_response_text, request_data, path)
                    
                except json.JSONDecodeError:
                    logger.warning("Failed to parse request body as JSON")
            
            # Not a special command, proxy to remote server
            logger.info(f"Proxying {method} request to: {target_url}")
            
            # Prepare headers (exclude host-specific headers)
            headers = {
                k: v for k, v in request.headers.items()
                if k.lower() not in ['host', 'content-length']
            }
            
            # Create client session with appropriate timeout
            timeout = ClientTimeout(total=300)  # 5 minute timeout for LLM responses
            
            async with ClientSession(timeout=timeout) as session:
                # Forward request to remote server
                async with session.request(
                    method=method,
                    url=target_url,
                    headers=headers,
                    data=body,
                    params=request.query
                ) as response:
                    
                    # Handle streaming responses (SSE/streaming completions)
                    if response.headers.get('Content-Type', '').startswith('text/event-stream'):
                        return await self._handle_streaming_response(response, request)
                    else:
                        return await self._handle_regular_response(response)
                    
        except asyncio.TimeoutError:
            logger.error(f"Timeout while proxying to {target_url}")
            return web.Response(
                text=json.dumps({"error": "Gateway timeout"}),
                status=504,
                content_type='application/json'
            )
        except Exception as e:
            logger.error(f"Error proxying request: {e}", exc_info=True)
            return web.Response(
                text=json.dumps({"error": str(e)}),
                status=502,
                content_type='application/json'
            )
    
    async def _handle_streaming_special_command(
        self, 
        response_text: str, 
        request_data: Dict[str, Any], 
        request: web.Request,
        path: str
    ) -> web.StreamResponse:
        """Handle streaming response for special commands"""
        
        # Create streaming response
        stream_response = web.StreamResponse(
            status=200,
            headers={
                'Content-Type': 'text/event-stream',
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive'
            }
        )
        
        await stream_response.prepare(request)
        
        # Determine format based on path
        if path.startswith('v1/'):
            # OpenAI streaming format
            chunk = {
                "id": "chatcmpl-proxy",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": request_data.get("model", "proxy"),
                "choices": [{
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": response_text
                    },
                    "finish_reason": None
                }]
            }
            await stream_response.write(f"data: {json.dumps(chunk)}\n\n".encode('utf-8'))
            
            # Send done signal
            done_chunk = {
                "id": "chatcmpl-proxy",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": request_data.get("model", "proxy"),
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            }
            await stream_response.write(f"data: {json.dumps(done_chunk)}\n\n".encode('utf-8'))
            await stream_response.write(b"data: [DONE]\n\n")
        else:
            # Ollama streaming format
            chunk = {
                "model": request_data.get("model", "proxy"),
                "created_at": "",
                "response": response_text,
                "done": False
            }
            await stream_response.write(f"{json.dumps(chunk)}\n".encode('utf-8'))
            
            # Send done signal
            done_chunk = {
                "model": request_data.get("model", "proxy"),
                "created_at": "",
                "response": "",
                "done": True
            }
            await stream_response.write(f"{json.dumps(done_chunk)}\n".encode('utf-8'))
        
        await stream_response.write_eof()
        logger.info("Streaming special command response complete")
        return stream_response
    
    async def _handle_streaming_response(self, response, request: web.Request) -> web.StreamResponse:
        """Handle Server-Sent Events (SSE) streaming responses from remote server"""
        logger.info("Handling streaming response from remote server")
        
        # Create streaming response
        stream_response = web.StreamResponse(
            status=response.status,
            headers={
                'Content-Type': 'text/event-stream',
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive'
            }
        )
        
        await stream_response.prepare(request)
        
        # Stream data chunks from remote server to client
        async for chunk in response.content.iter_any():
            if chunk:
                logger.debug(f"Streaming chunk: {len(chunk)} bytes")
                await stream_response.write(chunk)
        
        await stream_response.write_eof()
        logger.info("Streaming response complete")
        return stream_response
    
    async def _handle_regular_response(self, response) -> web.Response:
        """Handle regular non-streaming responses from remote server"""
        content = await response.read()
        
        logger.info(f"Response status: {response.status}, size: {len(content)} bytes")
        logger.debug(f"Response content: {content[:500]}")  # Log first 500 chars
        
        # Copy relevant headers
        headers = {
            k: v for k, v in response.headers.items()
            if k.lower() not in ['transfer-encoding', 'connection']
        }
        
        return web.Response(
            body=content,
            status=response.status,
            headers=headers
        )
    
    async def start(self, host: str = '0.0.0.0', port: int = 11434):
        """Start the proxy server"""
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, host, port)
        await site.start()
        
        logger.info(f"Llama proxy server started on {host}:{port}")
        logger.info(f"Forwarding requests to {self.remote_url}")
        logger.info(f"Special commands enabled: !specialcommand")
        logger.info("Press Ctrl+C to stop")
        
        # Keep running
        try:
            await asyncio.Event().wait()
        except KeyboardInterrupt:
            logger.info("Shutting down proxy server...")
        finally:
            await runner.cleanup()


def setup_logging(debug_level: str):
    """Configure logging based on debug level"""
    levels = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    level = levels.get(debug_level.upper(), logging.INFO)
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Set aiohttp logging to WARNING unless DEBUG is specified
    if level != logging.DEBUG:
        logging.getLogger('aiohttp').setLevel(logging.WARNING)


async def main():
    parser = argparse.ArgumentParser(
        description='Llama Server Proxy - Forward requests to remote llama-server with special command interception'
    )
    parser.add_argument(
        '--remote-host',
        required=True,
        help='Remote llama-server hostname or IP address'
    )
    parser.add_argument(
        '--remote-port',
        type=int,
        default=11434,
        help='Remote llama-server port (default: 11434)'
    )
    parser.add_argument(
        '--listen-host',
        default='0.0.0.0',
        help='Local host to listen on (default: 0.0.0.0)'
    )
    parser.add_argument(
        '--listen-port',
        type=int,
        default=11434,
        help='Local port to listen on (default: 11434)'
    )
    parser.add_argument(
        '--debug',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Set debug level (default: INFO)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.debug)
    
    logger.info(f"Starting Llama Proxy with debug level: {args.debug}")
    
    proxy = LlamaProxy(args.remote_host, args.remote_port)
    await proxy.start(args.listen_host, args.listen_port)


if __name__ == '__main__':
    asyncio.run(main())