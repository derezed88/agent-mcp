#!/bin/bash
# Test Win11 model response streaming to debug open-webui issue

echo "=== Testing Win11 Model Response Streaming ==="
echo ""

SERVER="http://localhost:11434"

# First switch to Win11 model
echo "Step 1: Switching to Win11 model"
echo "----------------------------------------"
response=$(timeout 10 curl -s -X POST "$SERVER/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "test",
    "messages": [{"role": "user", "content": "!model Win11"}],
    "stream": true
  }' | grep -o '"content":"[^"]*"' | head -5)

echo "Response: $response"
echo ""

# Now test actual prompt
echo "Step 2: Testing simple prompt with Win11"
echo "----------------------------------------"
echo "Sending: 'Hello'"
echo ""
echo "Streaming response:"
echo ""

timeout 40 curl -s -X POST "$SERVER/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "test",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }' > /tmp/win11_response.txt

echo "Full response saved to /tmp/win11_response.txt"
echo ""
echo "First 30 lines:"
cat /tmp/win11_response.txt | head -30
echo ""

# Analyze the response
echo "Step 3: Analysis"
echo "----------------------------------------"

chunk_count=$(grep -c '"content":' /tmp/win11_response.txt || echo "0")
echo "Number of content chunks: $chunk_count"

has_done=$(grep -c '\[DONE\]' /tmp/win11_response.txt || echo "0")
echo "Has [DONE] marker: $has_done"

has_finish=$(grep -c '"finish_reason":"stop"' /tmp/win11_response.txt || echo "0")
echo "Has finish_reason: $has_finish"

echo ""
echo "Content chunks:"
grep -o '"content":"[^"]*"' /tmp/win11_response.txt | head -20

echo ""
echo "If you see many small chunks (5-10 chars each), chunking is working"
echo "If you see one big chunk, Win11 is buffering and chunking didn't help"
