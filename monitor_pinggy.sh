#!/bin/bash
# Monitor Pinggy tunnel health and auto-reconnect

LOG_FILE="/tmp/pinggy_monitor.log"
CHECK_INTERVAL=30  # seconds

echo "[$(date)] Pinggy monitor started" >> "$LOG_FILE"

while true; do
    # Check if main.py is running
    if ! pgrep -f "python.*main.py" > /dev/null; then
        echo "[$(date)] ERROR: main.py is not running!" >> "$LOG_FILE"
    fi

    # Check if SSH tunnel is running
    if ! pgrep -f "ssh.*pinggy.io" > /dev/null; then
        echo "[$(date)] ERROR: Pinggy SSH tunnel is not running!" >> "$LOG_FILE"
    fi

    # Test local endpoint
    if ! curl -s --max-time 5 http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "[$(date)] WARNING: Local port 11434 not responding" >> "$LOG_FILE"
    fi

    # Test through tunnel (if you know your pinggy URL)
    # PINGGY_URL="https://your-url.a.free.pinggy.link"
    # if ! curl -s --max-time 10 "$PINGGY_URL/api/tags" > /dev/null 2>&1; then
    #     echo "[$(date)] ERROR: Pinggy tunnel not responding" >> "$LOG_FILE"
    # fi

    sleep $CHECK_INTERVAL
done
