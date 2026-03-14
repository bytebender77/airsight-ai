#!/bin/bash
# ─────────────────────────────────────────────────────────────
# AirSight AI — Demo Startup Script
# Usage: bash start_demo.sh [ngrok-url]
# Example: bash start_demo.sh https://abc123.ngrok-free.app
# ─────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="$SCRIPT_DIR/config.js"

# If ngrok URL passed as argument, update config.js automatically
if [ ! -z "$1" ]; then
  NGROK_URL="${1%/}"   # strip trailing slash
  sed -i '' "s|const API_URL = '.*'|const API_URL = '$NGROK_URL'|g" "$CONFIG"
  echo "✅ API URL set to: $NGROK_URL"
else
  echo "ℹ️  Using localhost:5050 (no ngrok URL provided)"
  sed -i '' "s|const API_URL = '.*'|const API_URL = 'http://localhost:5050'|g" "$CONFIG"
fi

echo ""
echo "🚀 Starting AirSight AI..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Kill old instances
pkill -f "api.py" 2>/dev/null
pkill -f "http.server 8502" 2>/dev/null
sleep 1

# Start Flask API
cd "$SCRIPT_DIR"
echo "⚡ Starting Flask API on port 5050..."
python3 api.py &
API_PID=$!
sleep 3

# Test API health
if curl -s http://localhost:5050/health > /dev/null 2>&1; then
  echo "✅ Flask API is running (PID: $API_PID)"
else
  echo "⚠️  Flask API may still be loading models..."
fi

# Start static file server
echo "🌐 Starting dashboard server on port 8502..."
python3 -m http.server 8502 --directory "$SCRIPT_DIR" &
HTTP_PID=$!
sleep 1

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🌍 Dashboard: http://localhost:8502/dashboard.html"
echo "🧪 Evaluate:  http://localhost:8502/evaluate.html"
echo "🔌 API:       http://localhost:5050/health"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📡 To expose via ngrok (in a new terminal):"
echo "   ngrok http 5050"
echo "   → then run: bash start_demo.sh https://xxxx.ngrok-free.app"
echo ""
echo "Press Ctrl+C to stop all servers"

# Wait for user to stop
wait $API_PID
