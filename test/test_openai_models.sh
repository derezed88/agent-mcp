#!/bin/bash
# Test OpenAI-compatible /v1/models endpoint

echo "=== Testing /v1/models Endpoint Format ==="
echo ""

SERVER="http://localhost:11434"

echo "OpenAI format (v1/models):"
echo "Expected: {\"object\": \"list\", \"data\": [{\"id\": \"...\", \"object\": \"model\", ...}]}"
echo ""

response=$(curl -s "$SERVER/v1/models")
echo "Actual response:"
echo "$response" | python3 -m json.tool 2>&1 || echo "$response"
echo ""

# Validate structure
if echo "$response" | grep -q '"object": "list"' && echo "$response" | grep -q '"data":'; then
    echo "✓ OpenAI format: CORRECT"
    echo "  - Has 'object' field set to 'list'"
    echo "  - Has 'data' array with model objects"
else
    echo "✗ OpenAI format: INCORRECT"
    echo "  - Should have 'object' and 'data' fields"
    echo "  - Current format doesn't match OpenAI spec"
fi

echo ""
echo "---"
echo ""

echo "Ollama format (api/tags):"
echo "Expected: {\"models\": [{\"name\": \"...\"}]}"
echo ""

response=$(curl -s "$SERVER/api/tags")
echo "Actual response:"
echo "$response" | python3 -m json.tool 2>&1 || echo "$response"
echo ""

if echo "$response" | grep -q '"models":'; then
    echo "✓ Ollama format: CORRECT"
else
    echo "✗ Ollama format: INCORRECT"
fi

echo ""
echo "---"
echo ""
echo "NOTE: If formats are incorrect, restart the server:"
echo "  pkill -f 'python.*main.py'"
echo "  python main.py --llama-proxy"
