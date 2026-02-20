#!/bin/bash
# Test that streaming responses include role on first chunk (OpenAI spec compliance)

echo "=== Testing Streaming Role Fix for open-webui ==="
echo ""

SERVER="http://localhost:11434"

# Test 1: Verify role on first chunk
echo "Test 1: First Chunk Has Role Field"
echo "----------------------------------------"
echo "Sending: Hello"
echo ""

first_chunk=$(echo '{"model":"test","messages":[{"role":"user","content":"Hello"}],"stream":true}' | \
  curl -s -X POST "$SERVER/v1/chat/completions" -H "Content-Type: application/json" -d @- | \
  head -1)

echo "First chunk:"
echo "$first_chunk"
echo ""

if echo "$first_chunk" | grep -q '"role":"assistant"'; then
    echo "✓ SUCCESS: First chunk includes role field"
else
    echo "✗ FAILED: First chunk missing role field"
fi

if echo "$first_chunk" | grep -q '"content"'; then
    echo "✓ SUCCESS: First chunk includes content field"
else
    echo "✗ FAILED: First chunk missing content field"
fi

echo ""

# Test 2: Verify subsequent chunks don't have role
echo "Test 2: Subsequent Chunks Have No Role"
echo "----------------------------------------"

second_chunk=$(echo '{"model":"test","messages":[{"role":"user","content":"Hello"}],"stream":true}' | \
  curl -s -X POST "$SERVER/v1/chat/completions" -H "Content-Type: application/json" -d @- | \
  sed -n '2p')

echo "Second chunk:"
echo "$second_chunk"
echo ""

if echo "$second_chunk" | grep -q '"role"'; then
    echo "✗ FAILED: Second chunk should NOT have role field"
else
    echo "✓ SUCCESS: Second chunk has no role (correct)"
fi

if echo "$second_chunk" | grep -q '"content"'; then
    echo "✓ SUCCESS: Second chunk has content field"
else
    echo "⚠ Second chunk may be final chunk (no content)"
fi

echo ""

# Test 3: Full streaming response structure
echo "Test 3: Full Streaming Response Structure"
echo "----------------------------------------"

echo '{"model":"test","messages":[{"role":"user","content":"Say just hello"}],"stream":true}' | \
  curl -s -X POST "$SERVER/v1/chat/completions" -H "Content-Type: application/json" -d @- > /tmp/stream_test.txt

echo "Full response saved to /tmp/stream_test.txt"
echo ""

chunk_count=$(grep -c '^data:' /tmp/stream_test.txt)
done_marker=$(grep -c '\[DONE\]' /tmp/stream_test.txt)
role_count=$(grep -c '"role":"assistant"' /tmp/stream_test.txt)

echo "Statistics:"
echo "  Total chunks: $chunk_count"
echo "  Role fields: $role_count"
echo "  [DONE] marker: $done_marker"
echo ""

if [ "$role_count" = "1" ]; then
    echo "✓ SUCCESS: Exactly one chunk has role field (first chunk)"
else
    echo "✗ FAILED: Expected 1 role field, found $role_count"
fi

if [ "$done_marker" = "1" ]; then
    echo "✓ SUCCESS: [DONE] marker present"
else
    echo "✗ FAILED: [DONE] marker missing or duplicated"
fi

echo ""

# Test 4: Non-streaming still works
echo "Test 4: Non-Streaming Mode Still Works"
echo "----------------------------------------"

non_stream=$(echo '{"model":"test","messages":[{"role":"user","content":"Hello"}],"stream":false}' | \
  curl -s -X POST "$SERVER/v1/chat/completions" -H "Content-Type: application/json" -d @-)

echo "$non_stream" | python3 -m json.tool | head -20
echo ""

if echo "$non_stream" | grep -q '"object":"chat.completion"'; then
    echo "✓ SUCCESS: Non-streaming uses chat.completion (not chunk)"
else
    echo "✗ FAILED: Wrong object type"
fi

if echo "$non_stream" | grep -q '"message"'; then
    echo "✓ SUCCESS: Non-streaming uses message (not delta)"
else
    echo "✗ FAILED: Should use message field"
fi

if echo "$non_stream" | grep -q '"role":"assistant"'; then
    echo "✓ SUCCESS: Non-streaming includes role in message"
else
    echo "✗ FAILED: Missing role in message"
fi

echo ""

# Test 5: Detailed first chunk inspection
echo "Test 5: Detailed First Chunk Structure"
echo "----------------------------------------"

echo '{"model":"test","messages":[{"role":"user","content":"Hello"}],"stream":true}' | \
  curl -s -X POST "$SERVER/v1/chat/completions" -H "Content-Type: application/json" -d @- | \
  head -1 | python3 -c "
import sys, json
line = sys.stdin.read()
if line.startswith('data: '):
    data = json.loads(line[6:])
    print(json.dumps(data, indent=2))

    delta = data['choices'][0]['delta']
    print('\nDelta fields:')
    for key in delta:
        print(f'  - {key}: {delta[key][:50] if isinstance(delta[key], str) else delta[key]}')

    if 'role' in delta and 'content' in delta:
        print('\n✓ First chunk is OpenAI spec compliant!')
    else:
        print('\n✗ First chunk missing required fields')
"

echo ""

# Summary
echo "=== Summary ==="
echo ""
echo "OpenAI Streaming Specification:"
echo "  - First chunk: delta.role='assistant' + delta.content"
echo "  - Middle chunks: delta.content only"
echo "  - Final chunk: delta={} + finish_reason='stop'"
echo "  - End marker: data: [DONE]"
echo ""
echo "This fix ensures open-webui can properly parse and render streaming responses."
echo ""
echo "To test in open-webui:"
echo "  1. Send message: Hello"
echo "  2. Response should appear: 'Hello! How can I help you today?'"
echo "  3. Follow-up questions should also appear below"
