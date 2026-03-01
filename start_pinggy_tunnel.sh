#!/bin/bash
# Robust Pinggy tunnel with auto-reconnect and better error handling
#
# Usage: edit LOCAL_PORT and PINGGY_TOKEN below, then run:
#   chmod +x start_pinggy_tunnel.sh
#   ./start_pinggy_tunnel.sh
#
# Get your Pinggy token from: https://pinggy.io
# Set LOCAL_PORT to the port your agent-mcp server listens on (default: 8765).

LOCAL_PORT=8765          # Port your local server listens on
PINGGY_TOKEN="<YOUR_PINGGY_TOKEN>"   # Replace with your Pinggy token from pinggy.io

LOG_FILE="/tmp/pinggy_tunnel.log"
MAX_RETRIES=999999  # Effectively infinite

echo "[$(date)] Starting Pinggy tunnel..." | tee -a "$LOG_FILE"

# Test local endpoint first
if ! curl -s --max-time 5 http://localhost:${LOCAL_PORT} > /dev/null 2>&1; then
    echo "[$(date)] WARNING: Local port ${LOCAL_PORT} not responding! Is agent-mcp.py running?" | tee -a "$LOG_FILE"
    echo "Start the server with: python agent-mcp.py" | tee -a "$LOG_FILE"
    exit 1
fi

retry_count=0
consecutive_failures=0

while [ $retry_count -lt $MAX_RETRIES ]; do
    echo "[$(date)] Attempt $((retry_count + 1)): Connecting to Pinggy..." | tee -a "$LOG_FILE"

    # Check if we can reach pinggy.io first
    if ! ping -c 1 -W 5 pinggy.io > /dev/null 2>&1; then
        echo "[$(date)] WARNING: Cannot reach pinggy.io - network issue?" | tee -a "$LOG_FILE"
        sleep 10
        continue
    fi

    # Run SSH with improved keepalive and connection settings
    ssh -p 443 \
        -R0:localhost:${LOCAL_PORT} \
        -L4300:localhost:4300 \
        -o StrictHostKeyChecking=no \
        -o ServerAliveInterval=10 \
        -o ServerAliveCountMax=6 \
        -o TCPKeepAlive=yes \
        -o ExitOnForwardFailure=yes \
        -o ConnectTimeout=30 \
        -o ConnectionAttempts=3 \
        -o LogLevel=ERROR \
        -t "${PINGGY_TOKEN}@pro.pinggy.io" x:https

    exit_code=$?
    echo "[$(date)] SSH tunnel exited with code $exit_code" | tee -a "$LOG_FILE"

    # Track consecutive failures for diagnosis
    if [ $exit_code -eq 255 ]; then
        consecutive_failures=$((consecutive_failures + 1))
        if [ $consecutive_failures -ge 3 ]; then
            echo "[$(date)] WARNING: 3 consecutive connection failures (exit 255)" | tee -a "$LOG_FILE"
            echo "[$(date)] Possible causes: auth key expired, network issues, pinggy.io down" | tee -a "$LOG_FILE"
            # Longer wait on repeated failures
            echo "[$(date)] Waiting 30 seconds before retry..." | tee -a "$LOG_FILE"
            sleep 30
            consecutive_failures=0  # Reset after long wait
            continue
        fi
    else
        consecutive_failures=0  # Reset on successful connection or different error
    fi

    # Adaptive backoff: quick retry on first few, then settle at 5 seconds
    if [ $retry_count -eq 0 ]; then
        wait_time=1
    elif [ $retry_count -lt 3 ]; then
        wait_time=3
    else
        wait_time=5
    fi

    echo "[$(date)] Waiting $wait_time seconds before reconnecting..." | tee -a "$LOG_FILE"
    sleep $wait_time

    retry_count=$((retry_count + 1))
done

echo "[$(date)] Max retries reached, exiting" | tee -a "$LOG_FILE"
