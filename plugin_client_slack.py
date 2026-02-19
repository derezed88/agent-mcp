"""
Slack Client Interface Plugin for MCP Agent

Provides bidirectional Slack integration with asymmetric transport:
- Receives messages from Slack via Socket Mode (WebSocket, no public endpoint needed)
- Routes to LLM agent for processing
- Sends responses back to Slack via the Web API (chat.postMessage, authenticated via SLACK_BOT_TOKEN)
- Thread-aware conversations (Slack threads map to agent sessions)
- Full support for !commands like other clients
"""

import os
import re
import json
import asyncio
from typing import List, Dict, Optional
from starlette.routing import Route
from starlette.requests import Request
from starlette.responses import JSONResponse
from plugin_loader import BasePlugin

# Slack SDK imports
try:
    from slack_sdk.web.async_client import AsyncWebClient
    from slack_sdk.socket_mode.aiohttp import SocketModeClient
    from slack_sdk.socket_mode.request import SocketModeRequest
    from slack_sdk.socket_mode.response import SocketModeResponse
    SLACK_SDK_AVAILABLE = True
except ImportError:
    SLACK_SDK_AVAILABLE = False

# Import agent infrastructure
from config import log
from state import sessions, get_queue, push_tok, push_done
from routes import process_request


class SlackClientPlugin(BasePlugin):
    """Slack bidirectional client interface plugin."""

    PLUGIN_NAME = "plugin_client_slack"
    PLUGIN_VERSION = "1.0.0"
    PLUGIN_TYPE = "client_interface"
    DESCRIPTION = "Slack bidirectional client with Socket Mode and webhook support"
    DEPENDENCIES = ["slack-sdk>=3.0", "aiohttp>=3.0"]
    ENV_VARS = ["SLACK_BOT_TOKEN", "SLACK_APP_TOKEN"]

    def __init__(self):
        self.enabled = False
        self.slack_port = 8766
        self.slack_host = "0.0.0.0"
        self.slack_client: Optional[AsyncWebClient] = None
        self.socket_client: Optional[SocketModeClient] = None

        # Map Slack thread_ts to agent session_id and channel
        # Format: {"thread_ts_or_channel_ts": {"session_id": "sess_xyz", "channel_id": "C123"}}
        self.slack_sessions: Dict[str, Dict[str, str]] = {}

        # Background task for Socket Mode listener
        self._socket_task: Optional[asyncio.Task] = None

    def init(self, config: dict) -> bool:
        """Initialize Slack client plugin."""
        if not SLACK_SDK_AVAILABLE:
            log.error("Slack SDK not available. Install with: pip install slack-sdk aiohttp")
            return False

        try:
            # Get configuration
            self.slack_port = config.get('slack_port', 8766)
            self.slack_host = config.get('slack_host', '0.0.0.0')

            # Get Slack credentials from environment
            bot_token = os.getenv("SLACK_BOT_TOKEN")
            app_token = os.getenv("SLACK_APP_TOKEN")

            if not bot_token:
                log.error("SLACK_BOT_TOKEN not found in .env")
                return False

            if not app_token:
                log.error("SLACK_APP_TOKEN not found in .env")
                return False

            # Initialize Slack Web API client (for metadata, not used for sending)
            self.slack_client = AsyncWebClient(token=bot_token)

            # Initialize Socket Mode client for receiving events
            self.socket_client = SocketModeClient(
                app_token=app_token,
                web_client=self.slack_client
            )

            # Register Socket Mode event handlers
            self.socket_client.socket_mode_request_listeners.append(
                self._handle_socket_mode_request
            )

            self.enabled = True

            log.info("Slack client plugin initialized (Socket Mode)")
            log.info(f"  Bot token: {bot_token[:10]}...")
            log.info(f"  App token: {app_token[:10]}...")

            # Start Socket Mode listener in background
            self._socket_task = asyncio.create_task(self._run_socket_mode())

            return True

        except Exception as e:
            log.error(f"Slack client plugin init failed: {e}", exc_info=True)
            return False

    def shutdown(self) -> None:
        """Cleanup Slack client resources."""
        self.enabled = False

        # Stop Socket Mode listener
        if self._socket_task:
            self._socket_task.cancel()
            self._socket_task = None

        # Disconnect Socket Mode client
        if self.socket_client:
            try:
                asyncio.create_task(self.socket_client.close())
            except Exception as e:
                log.error(f"Error closing Socket Mode client: {e}")
            self.socket_client = None

        self.slack_client = None
        self.slack_sessions.clear()
        log.info("Slack client plugin shutdown")

    def get_routes(self) -> List[Route]:
        """Return Starlette routes for Slack client (health/status only - events via Socket Mode)."""
        return [
            Route("/slack/health", self._handle_health, methods=["GET"]),
            Route("/slack/status", self._handle_status, methods=["GET"]),
        ]

    def get_config(self) -> dict:
        """Return plugin configuration for server startup."""
        return {
            "port": self.slack_port,
            "host": self.slack_host,
            "name": "Slack client"
        }

    # =========================================================================
    # Socket Mode listener
    # =========================================================================

    async def _run_socket_mode(self) -> None:
        """Run Socket Mode client to listen for Slack events."""
        try:
            log.info("Starting Slack Socket Mode listener...")
            await self.socket_client.connect()
            log.info("Slack Socket Mode connected successfully!")
        except Exception as e:
            log.error(f"Socket Mode connection error: {e}", exc_info=True)

    async def _handle_socket_mode_request(
        self,
        client: SocketModeClient,
        request: SocketModeRequest
    ) -> None:
        """
        Handle incoming Socket Mode requests (events from Slack).

        This is called automatically by the Socket Mode client when events arrive.
        """
        try:
            # Acknowledge the request immediately
            response = SocketModeResponse(envelope_id=request.envelope_id)
            await client.send_socket_mode_response(response)

            # Process the event payload
            if request.type == "events_api":
                event = request.payload.get("event", {})
                await self._process_slack_event(event)
            elif request.type == "slash_commands":
                # Future: handle slash commands if needed
                log.debug("Received slash command (not yet implemented)")
            else:
                log.debug(f"Unhandled Socket Mode request type: {request.type}")

        except Exception as e:
            log.error(f"Error handling Socket Mode request: {e}", exc_info=True)

    # =========================================================================
    # Event processing
    # =========================================================================

    async def _process_slack_event(self, event: dict) -> None:
        """Process individual Slack event (message, app_mention, etc.)."""
        event_type = event.get("type")

        # Ignore bot messages to prevent loops
        if event.get("bot_id") or event.get("subtype") == "bot_message":
            log.debug("Ignoring bot message")
            return

        # Handle message events
        if event_type == "message":
            await self._handle_message_event(event)
        elif event_type == "app_mention":
            await self._handle_app_mention_event(event)
        else:
            log.debug(f"Ignoring event type: {event_type}")

    async def _handle_message_event(self, event: dict) -> None:
        """Handle message event from Slack."""
        channel_id = event.get("channel")
        user_id = event.get("user")
        text = event.get("text", "").strip()
        thread_ts = event.get("thread_ts") or event.get("ts")  # Use thread or message ts
        message_ts = event.get("ts")

        if not text or not channel_id:
            log.debug("Ignoring empty message or missing channel")
            return

        log.info(f"Slack message: channel={channel_id}, user={user_id}, thread={thread_ts}")
        log.debug(f"Message text: {text}")

        # Process the message through the agent
        await self._process_user_message(channel_id, thread_ts, user_id, text, message_ts)

    async def _handle_app_mention_event(self, event: dict) -> None:
        """Handle app_mention event (when bot is @mentioned)."""
        channel_id = event.get("channel")
        user_id = event.get("user")
        text = event.get("text", "").strip()
        thread_ts = event.get("thread_ts") or event.get("ts")
        message_ts = event.get("ts")

        if not text or not channel_id:
            return

        # Remove the bot mention from text (e.g., "<@U12345> hello" -> "hello")
        # Slack mentions are in format <@U12345>
        words = text.split()
        cleaned_words = [w for w in words if not (w.startswith("<@") and w.endswith(">"))]
        cleaned_text = " ".join(cleaned_words).strip()

        log.info(f"Slack app_mention: channel={channel_id}, user={user_id}, thread={thread_ts}")
        log.debug(f"Cleaned text: {cleaned_text}")

        # Process through agent
        await self._process_user_message(channel_id, thread_ts, user_id, cleaned_text, message_ts)

    # =========================================================================
    # Agent integration
    # =========================================================================

    async def _process_user_message(
        self,
        channel_id: str,
        thread_ts: str,
        user_id: str,
        text: str,
        message_ts: str
    ) -> None:
        """
        Process user message through the agent.

        Maps Slack thread to agent session and routes message.
        """
        # Create unique client_id for this Slack thread
        client_id = f"slack-{channel_id}-{thread_ts}"

        # Get or create session mapping
        if thread_ts not in self.slack_sessions:
            self.slack_sessions[thread_ts] = {
                "session_id": client_id,
                "channel_id": channel_id,
                "user_id": user_id,
                "thread_ts": thread_ts
            }
            log.info(f"Created new Slack session: {client_id}")

        session_info = self.slack_sessions[thread_ts]

        # Create task to consume agent responses and send to Slack
        asyncio.create_task(self._consume_agent_responses(client_id, channel_id, thread_ts))

        # Submit message to agent (same as shell.py does via /submit)
        # This will trigger process_request which handles !commands and LLM
        payload = {
            "client_id": client_id,
            "text": text,
            "user_id": user_id,
            "channel_id": channel_id,
            "thread_ts": thread_ts
        }

        # Process through agent (imported from routes.py)
        asyncio.create_task(process_request(client_id, text, payload))

    # Regex for push_tok status lines emitted by the agent framework.
    # These bracket tokens (e.g. "[agent_call ▶ gemini25]", "[agent_call ◀]") are
    # informational noise that should not appear in Slack posts.
    _STATUS_LINE_RE = re.compile(
        r'^\[(?:agent_call|tool_call|llm_call)[^\]]*\]\s*$',
        re.MULTILINE,
    )

    @classmethod
    def _filter_status_lines(cls, text: str) -> str:
        """Remove push_tok status bracket lines from agent output."""
        filtered = cls._STATUS_LINE_RE.sub('', text)
        # Collapse runs of blank lines that may be left behind
        filtered = re.sub(r'\n{3,}', '\n\n', filtered)
        return filtered.strip()

    async def _consume_agent_responses(
        self,
        client_id: str,
        channel_id: str,
        thread_ts: str
    ) -> None:
        """
        Consume responses from agent queue and send to Slack.

        Similar to how shell.py consumes via SSE stream.

        Multi-turn behaviour: after each agent turn completes (done event) the
        accumulated text is posted to Slack immediately so the user sees progress.
        Accumulation then resets for the next turn.  The final turn's post is the
        last message the user sees — no silent 2-minute wait.
        """
        if not self.slack_client:
            log.error("Slack client not initialized")
            return

        # Get the queue for this client
        queue = await get_queue(client_id)

        # Accumulate tokens for the current turn
        response_parts: List[str] = []
        turn_index = 0  # which agent turn we are on (0-based)

        # Multi-turn agents (tool call → LLM response) emit multiple "done" signals.
        # After each "done" we post the turn's output immediately, then wait a short
        # grace period for the next turn to start.
        # If nothing arrives within the grace period, the conversation is truly finished.
        FIRST_TIMEOUT = 120.0   # max wait for first token (LLM can be slow)
        INTER_TURN_TIMEOUT = 5.0  # grace period between turns after a "done"

        timeout = FIRST_TIMEOUT
        received_done = False

        async def _flush_turn(label: str) -> None:
            """Post the current turn's accumulated text to Slack and reset."""
            nonlocal response_parts, turn_index
            if response_parts:
                turn_text = self._filter_status_lines("".join(response_parts))
                if turn_text:
                    if turn_index > 0:
                        # Prefix intermediate turns so the user knows more is coming
                        turn_text = f"_(turn {turn_index + 1})_\n{turn_text}"
                    await self._send_slack_message(channel_id, thread_ts, turn_text)
                    log.info(f"Slack: posted {label} for {client_id} (turn {turn_index + 1})")
            response_parts = []
            turn_index += 1

        try:
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=timeout)
                    received_done = False  # reset: activity means we're still in a turn
                except asyncio.TimeoutError:
                    if received_done:
                        # Grace period expired after a "done" — conversation finished.
                        # The turn was already flushed on the done event; nothing left to do.
                        break
                    else:
                        # No response at all within FIRST_TIMEOUT
                        log.warning(f"Slack consumer timeout waiting for response from {client_id}")
                        # Flush whatever we have (may be empty)
                        await _flush_turn("timeout flush")
                        break

                item_type = item.get("t")

                if item_type == "tok":
                    # Token/text data
                    response_parts.append(item["d"])
                    timeout = FIRST_TIMEOUT  # reset to long timeout while active

                elif item_type == "done":
                    # One turn complete — post immediately, then wait for next turn
                    await _flush_turn("turn complete")
                    received_done = True
                    timeout = INTER_TURN_TIMEOUT

                elif item_type == "err":
                    # Error occurred — append and flush immediately
                    error_msg = item.get("d", "Unknown error")
                    response_parts.append(f"\n\n⚠️ Error: {error_msg}")
                    await _flush_turn("error")
                    break

                elif item_type == "gate":
                    # Gate request - inform user that approval is needed
                    gate_data = item.get("d", {})
                    tool_name = gate_data.get("tool_name", "unknown")
                    gate_notice = (
                        f"🔒 Gate approval required for tool: `{tool_name}`\n"
                        f"(Approval must be provided via shell.py client)"
                    )
                    await self._send_slack_message(channel_id, thread_ts, gate_notice)
                    timeout = FIRST_TIMEOUT  # gate may take a while
                    received_done = False
                    # Continue listening for more responses

        except Exception as e:
            log.error(f"Error consuming agent responses for {client_id}: {e}", exc_info=True)

    @staticmethod
    def _close_open_backticks(text: str) -> str:
        """
        Ensure backtick spans are closed at the end of a message chunk.

        Slack truncates rendering when a backtick code span is left open.
        Handles triple-backtick (```) code blocks and single-backtick spans.
        """
        # Check for open triple-backtick code block first
        triple_count = text.count('```')
        if triple_count % 2 != 0:
            return text + '\n```'

        # Check for open single-backtick span (not inside a triple block).
        # Walk the string tracking state so embedded backticks don't count.
        in_triple = False
        in_single = False
        i = 0
        while i < len(text):
            if text[i:i+3] == '```':
                in_triple = not in_triple
                i += 3
                continue
            if not in_triple and text[i] == '`':
                in_single = not in_single
            i += 1

        if in_single:
            return text + '`'
        return text

    async def _send_slack_message(self, channel_id: str, thread_ts: str, text: str) -> None:
        """
        Send message to Slack channel/thread via Web API (chat.postMessage).

        Authenticated via SLACK_BOT_TOKEN. Supports threaded replies and
        automatic chunking for messages exceeding Slack's ~4000 char limit.
        """
        if not self.slack_client:
            log.error("Slack client not initialized")
            return

        try:
            # Clean up text for Slack rendering
            # Slack uses actual newlines, not \n literals
            cleaned_text = text.replace('\\n', '\n')

            # Slack has a ~4000 character limit for messages
            # Split into chunks if needed
            max_chunk_size = 3500

            if len(cleaned_text) <= max_chunk_size:
                # Close any unclosed backtick spans before sending
                cleaned_text = self._close_open_backticks(cleaned_text)
                await self.slack_client.chat_postMessage(
                    channel=channel_id,
                    thread_ts=thread_ts,
                    text=cleaned_text
                )
                log.info(f"Sent Slack message to {channel_id}/{thread_ts} ({len(cleaned_text)} chars)")
            else:
                # Split into chunks
                chunks = [cleaned_text[i:i+max_chunk_size] for i in range(0, len(cleaned_text), max_chunk_size)]
                log.info(f"Splitting message into {len(chunks)} chunks")

                for i, chunk in enumerate(chunks):
                    prefix = f"(Part {i+1}/{len(chunks)})\n" if len(chunks) > 1 else ""
                    # Close any unclosed backtick spans at chunk boundary
                    chunk = self._close_open_backticks(chunk)
                    await self.slack_client.chat_postMessage(
                        channel=channel_id,
                        thread_ts=thread_ts,
                        text=prefix + chunk
                    )
                    # Small delay between chunks to avoid rate limits
                    if i < len(chunks) - 1:
                        await asyncio.sleep(0.5)

        except Exception as e:
            log.error(f"Error sending Slack message: {e}", exc_info=True)

    # =========================================================================
    # Status endpoints
    # =========================================================================

    async def _handle_health(self, request: Request) -> JSONResponse:
        """Health check endpoint."""
        return JSONResponse({
            "status": "healthy",
            "plugin": self.PLUGIN_NAME,
            "version": self.PLUGIN_VERSION,
            "enabled": self.enabled,
            "active_sessions": len(self.slack_sessions)
        })

    async def _handle_status(self, request: Request) -> JSONResponse:
        """Status endpoint showing active Slack sessions."""
        sessions_list = [
            {
                "thread_ts": thread_ts,
                "session_id": info["session_id"],
                "channel_id": info["channel_id"],
                "user_id": info.get("user_id", "unknown")
            }
            for thread_ts, info in self.slack_sessions.items()
        ]

        return JSONResponse({
            "plugin": self.PLUGIN_NAME,
            "active_sessions": len(self.slack_sessions),
            "sessions": sessions_list
        })
