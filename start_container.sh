#!/bin/bash
set -e

echo "================================================================="
echo "  🚀 Starting RMLTP Distributed Platform in Container..."
echo "================================================================="

# Trap signals for graceful shutdown of all background daemons
cleanup() {
    echo "Terminating cluster daemons..."
    kill $(jobs -p) 2>/dev/null || true
    wait
    echo "Cluster shutdown complete."
    exit 0
}
trap cleanup SIGINT SIGTERM

PORT="${PORT:-7860}"
CONTROLLER_PORT=8000

echo "[1/4] Starting Controller Daemon on 0.0.0.0:${CONTROLLER_PORT}..."
python src/controller/main.py --host 0.0.0.0 --port ${CONTROLLER_PORT} --mode serve &
CTRL_PID=$!

# Wait for controller socket to settle
sleep 1.5

echo "[2/4] Starting Worker 1 Daemon..."
python src/node/main.py --id worker-1 --port 8001 --controller 127.0.0.1:${CONTROLLER_PORT} &
W1_PID=$!

echo "[3/4] Starting Worker 2 Daemon..."
python src/node/main.py --id worker-2 --port 8002 --controller 127.0.0.1:${CONTROLLER_PORT} &
W2_PID=$!

sleep 1.5

echo "[4/4] Starting Web Dashboard on 0.0.0.0:${PORT}..."
python dashboard.py --host 0.0.0.0 --port ${PORT} &
DASH_PID=$!

echo "================================================================="
echo "  ✅ RMLTP Cluster is LIVE!"
echo "  Dashboard:  http://0.0.0.0:${PORT}"
echo "  Controller: tcp://0.0.0.0:${CONTROLLER_PORT}"
echo "================================================================="

# Keep container running and wait for any process exit
wait -n $CTRL_PID $W1_PID $W2_PID $DASH_PID
