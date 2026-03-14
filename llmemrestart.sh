#!/bin/bash

SCRIPT_NAME="llmem-gw.py"
PID_FILE="$(pwd)/.llmemPID"

# Kill only the instance started from this directory
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "Stopping $SCRIPT_NAME (PID $OLD_PID)..."
        kill "$OLD_PID"
        # Wait up to 10s for graceful shutdown
        for i in $(seq 1 20); do
            if ! kill -0 "$OLD_PID" 2>/dev/null; then
                echo "Process $OLD_PID confirmed dead."
                break
            fi
            sleep 0.5
        done
        # If still alive, force kill
        if kill -0 "$OLD_PID" 2>/dev/null; then
            echo "Process $OLD_PID still alive after 10s, sending SIGKILL..."
            kill -9 "$OLD_PID"
            sleep 1
            if kill -0 "$OLD_PID" 2>/dev/null; then
                echo "ERROR: Failed to kill PID $OLD_PID. Aborting restart."
                exit 1
            fi
            echo "Process $OLD_PID force-killed."
        fi
    else
        echo "Stale PID file ($OLD_PID not running), ignoring."
    fi
    rm -f "$PID_FILE"
else
    echo "No .llmemPID file found — nothing to stop."
fi

# Restart the process
echo "Starting $SCRIPT_NAME..."
nohup venv/bin/python -u "$SCRIPT_NAME" >> llmem-gw.log 2>&1 </dev/null &
disown
NEW_PID=$!
echo "$NEW_PID" > "$PID_FILE"
echo "Process started. PID: $NEW_PID"
