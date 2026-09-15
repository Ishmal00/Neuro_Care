#!/bin/bash
set -m

echo "=================================================="
echo " Starting NeuroCare-AI Dual Service Container"
echo "=================================================="

# 1. Start FastAPI Backend on port 8000
echo "[1/2] Launching FastAPI backend on http://0.0.0.0:8000..."
uvicorn app.api.main:app --host 0.0.0.0 --port 8000 &
FASTAPI_PID=$!

# 2. Start Streamlit UI on port 8501
echo "[2/2] Launching Streamlit dashboard on http://0.0.0.0:8501..."
streamlit run app/ui/dashboard.py --server.port 8501 --server.address 0.0.0.0 &
STREAMLIT_PID=$!

# Trap termination signals to gracefully stop both services
cleanup() {
    echo "Stopping NeuroCare-AI services..."
    kill -TERM "$FASTAPI_PID" "$STREAMLIT_PID" 2>/dev/null
    wait "$FASTAPI_PID" "$STREAMLIT_PID" 2>/dev/null
    exit 0
}

trap cleanup SIGINT SIGTERM

echo "Both services are running:"
echo " - FastAPI PID: $FASTAPI_PID (Port 8000)"
echo " - Streamlit PID: $STREAMLIT_PID (Port 8501)"

# Wait for any process to exit
wait -n

# Exit with status of process that exited first
exit $?
