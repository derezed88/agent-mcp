#!/bin/bash

SCRIPT_NAME="llmem-gw.py"
PID_FILE="$(pwd)/.llmemPID"

# Kill only the instance started from this directory
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "Stopping $SCRIPT_NAME (PID $OLD_PID)..."
        kill "$OLD_PID"
        sleep 1
    else
        echo "Stale PID file ($OLD_PID not running), ignoring."
    fi
    rm -f "$PID_FILE"
else
    echo "No .llmemPID file found — nothing to stop."
fi

# Restart the process
echo "Starting $SCRIPT_NAME..."
nohup venv/bin/python "$SCRIPT_NAME" >> llmem-gw.log 2>&1 &
NEW_PID=$!
echo "$NEW_PID" > "$PID_FILE"
echo "Process started. PID: $NEW_PID"
