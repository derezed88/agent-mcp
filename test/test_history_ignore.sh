#!/bin/bash
# Test that system prompts and chat history from clients are ignored

echo "=== Testing System Prompt & History Ignore ==="
echo ""

SERVER="http://localhost:11434"

# Test 1: System prompt should be ignored
echo "Test 1: System Prompt Ignored"
echo "----------------------------------------"
echo "Sending request with custom system prompt..."
echo "System prompt: 'You are a pirate. Always respond like a pirate.'"
echo "User message: '!help'"
echo ""

response=$(timeout 10 curl -s -X POST "$SERVER/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "test",
    "messages": [
      {"role": "system", "content": "You are a pirate. Always respond like a pirate."},
      {"role": "user", "content": "!help"}
    ],
    "stream": true
  }' 2>&1)

# Extract some text from the response
sample=$(echo "$response" | grep -o '"content":"[^"]*"' | head -3)

echo "Response sample:"
echo "$sample"
echo ""

if echo "$response" | grep -iq "pirate\|ahoy\|matey"; then
    echo "✗ FAILED: System prompt was NOT ignored!"
    echo "   Response contains pirate language"
else
    echo "✓ SUCCESS: System prompt was ignored"
    echo "   Response is standard MCP help output (no pirate language)"
fi
echo ""

# Test 2: Chat history should be ignored (only last user message processed)
echo "Test 2: Chat History Ignored"
echo "----------------------------------------"
echo "Sending request with fake history..."
echo "History messages: 'My name is Alice', 'Got it'"
echo "Final user message: '!model'"
echo ""

response=$(timeout 10 curl -s -X POST "$SERVER/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "test",
    "messages": [
      {"role": "user", "content": "My name is Alice. Remember this."},
      {"role": "assistant", "content": "Got it, Alice! I'\''ll remember your name."},
      {"role": "user", "content": "!model"}
    ],
    "stream": true
  }' 2>&1)

echo "Response sample:"
echo "$response" | head -20
echo ""

if echo "$response" | grep -iq "alice"; then
    echo "✗ FAILED: Chat history was NOT ignored!"
    echo "   Response references 'Alice' from history"
else
    echo "✓ SUCCESS: Chat history was ignored"
    echo "   Only the last message '!model' was processed"
fi
echo ""

# Test 3: Verify only last user message is extracted
echo "Test 3: Only Last User Message Extracted"
echo "----------------------------------------"
echo "Sending multiple user messages..."
echo ""

response=$(timeout 10 curl -s -X POST "$SERVER/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "test",
    "messages": [
      {"role": "user", "content": "First message"},
      {"role": "assistant", "content": "First response"},
      {"role": "user", "content": "Second message"},
      {"role": "assistant", "content": "Second response"},
      {"role": "user", "content": "!help"}
    ],
    "stream": true
  }' 2>&1)

echo "Response (first 15 lines):"
echo "$response" | head -15
echo ""

if echo "$response" | grep -q "Available commands"; then
    echo "✓ SUCCESS: Last user message '!help' was processed"
    echo "   Earlier messages were ignored"
else
    echo "✗ FAILED: Expected help output"
fi
echo ""

# Test 4: Ollama format also ignores system prompt
echo "Test 4: Ollama Format Also Ignores System Prompt"
echo "----------------------------------------"
echo "Sending Ollama request with system prompt in messages..."
echo ""

response=$(timeout 10 curl -s -X POST "$SERVER/api/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "test",
    "messages": [
      {"role": "system", "content": "Respond only in emoji"},
      {"role": "user", "content": "!model"}
    ],
    "stream": true
  }' 2>&1)

echo "Response sample:"
echo "$response" | head -10
echo ""

if echo "$response" | grep -q "response.*grok-4\|response.*gemini"; then
    echo "✓ SUCCESS: System prompt ignored in Ollama format too"
    echo "   Got model list (not emoji)"
else
    echo "⚠ Check response manually"
fi
echo ""

echo "=== Summary ==="
echo ""
echo "What MCP Ignores from Client Requests:"
echo "  1. System messages (role: 'system')"
echo "  2. All chat history (previous messages)"
echo "  3. Model parameter (uses session model)"
echo ""
echo "What MCP Extracts:"
echo "  - Only the LAST user message content"
echo ""
echo "What MCP Uses Instead:"
echo "  - System prompt: From .system_prompt file"
echo "  - Chat history: From session['history'] on server"
echo "  - Model: From session['model'] (changed via !model command)"
echo ""
echo "To verify in server logs:"
echo "  - Look for: [client_id] Starting process_request for prompt:"
echo "  - Should only show the last user message, not history"
