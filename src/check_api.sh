#!/bin/bash

# Ensure we are in the src directory
cd "$(dirname "$0")"

# Load environment variables from parent directory
if [ -f ../.env ]; then
    export $(grep -v '^#' ../.env | xargs)
fi

# Force LLM enabled for testing
export LLM_ENABLED=true

# Check for API Key
if [ -z "$GROQ_API_KEY" ]; then
    echo "⚠️  WARNING: GROQ_API_KEY is not set. LLM tests will likely fail."
fi

echo "=================================================="
echo "Checking UPI Fraud Detection API"
echo "=================================================="

# Check if port 5000 is already in use (API already running)
PORT_OPEN=$(python -c "import socket; s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); print(s.connect_ex(('127.0.0.1', 5000)))")

if [ "$PORT_OPEN" -eq "0" ]; then
    echo "⚠️  Port 5000 is already in use."
    echo "   Assuming API is already running."
    echo "   Running tests against existing instance..."
    python test_api.py
else
    # Start the API server in the background
    echo "🚀 Starting API server..."
    # Create logs directory if it doesn't exist
    mkdir -p ../logs
    
    python app.py > ../logs/api_test.log 2>&1 &
    API_PID=$!
    
    echo "⏳ Waiting 5 seconds for server to initialize..."
    sleep 5
    
    # Run the tests
    echo "🧪 Running API tests..."
    python test_api.py
    TEST_EXIT_CODE=$?
    
    # Stop the server
    echo "🛑 Stopping API server..."
    kill $API_PID
    
    exit $TEST_EXIT_CODE
fi