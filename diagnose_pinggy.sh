#!/bin/bash
# Diagnose Pinggy connection issues

echo "=== Pinggy Connection Diagnostics ==="
echo ""

# 1. Check if main.py is running
echo "1. Checking if main.py is running..."
if pgrep -f "python.*main.py" > /dev/null; then
    echo "   ✓ main.py is running"
    ps aux | grep "python.*main.py" | grep -v grep
else
    echo "   ✗ main.py is NOT running!"
    echo "   Start with: python main.py --llama-proxy"
fi
echo ""

# 2. Check local endpoint
echo "2. Checking local endpoint (localhost:11434)..."
if curl -s --max-time 5 http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "   ✓ Port 11434 is responding"
    curl -s http://localhost:11434/api/tags | jq -r '.models[].name' 2>/dev/null || echo "   (Models list available)"
else
    echo "   ✗ Port 11434 NOT responding!"
fi
echo ""

# 3. Check network connectivity
echo "3. Checking network connectivity to pinggy.io..."
if ping -c 3 pinggy.io > /dev/null 2>&1; then
    echo "   ✓ Can reach pinggy.io"
else
    echo "   ✗ Cannot reach pinggy.io - check network/DNS"
fi
echo ""

# 4. Test SSH connection (without tunnel)
echo "4. Testing SSH authentication to pinggy.io..."
timeout 10 ssh -p 443 -o StrictHostKeyChecking=no -o ConnectTimeout=5 a.pinggy.io exit 2>&1 | head -5
echo ""

# 5. Check if tunnel is already running
echo "5. Checking for existing SSH tunnel..."
if pgrep -f "ssh.*pinggy.io" > /dev/null; then
    echo "   ⚠ SSH tunnel process found:"
    ps aux | grep "ssh.*pinggy.io" | grep -v grep
    echo "   You may need to kill it first: pkill -f 'ssh.*pinggy.io'"
else
    echo "   ✓ No existing tunnel found"
fi
echo ""

# 6. Check firewall
echo "6. Checking firewall status..."
if command -v ufw > /dev/null; then
    sudo ufw status | grep -E "(11434|Status:)"
else
    echo "   (ufw not installed)"
fi
echo ""

# 7. Check recent tunnel logs
echo "7. Recent tunnel logs (last 10 lines)..."
if [ -f /tmp/pinggy_tunnel.log ]; then
    tail -10 /tmp/pinggy_tunnel.log
else
    echo "   (No log file found at /tmp/pinggy_tunnel.log)"
fi
echo ""

# 8. Recommendations
echo "=== Recommendations ==="
echo ""
echo "If you see exit code 255 repeatedly:"
echo "  - Your Pinggy auth token may have expired"
echo "  - Visit https://pinggy.io to get a fresh token"
echo "  - Update start_pinggy_tunnel.sh with new credentials"
echo ""
echo "If main.py is not running:"
echo "  - Start it with: python main.py --llama-proxy"
echo "  - Check logs for errors"
echo ""
echo "For connection testing:"
echo "  - Try a simple tunnel first: ssh -p 443 -R0:localhost:11434 a.pinggy.io"
echo "  - Check if you can access the public URL from your phone"
