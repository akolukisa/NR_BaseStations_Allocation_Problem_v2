#!/bin/bash
# Quick test script for RIC algorithms

echo "=================================="
echo "5G-LENA RIC Algorithm Testing"
echo "=================================="
echo ""

# Test scenario parameters
BEAMS=8
UES=4

# Array of algorithms to test
ALGORITHMS=("Max-SINR" "GA" "HGA" "PBIG")

cd "$(dirname "$0")"

for algo in "${ALGORITHMS[@]}"; do
    echo "Testing $algo algorithm..."
    echo "---"
    
    # Start RIC server in background
    python3 ric_server.py --algorithm "$algo" --port 5555 > /dev/null 2>&1 &
    SERVER_PID=$!
    
    # Wait for server to start
    sleep 2
    
    # Run test client
    python3 test_ric_client.py --algorithm "$algo" --beams $BEAMS --ues $UES
    
    # Stop server
    kill $SERVER_PID 2>/dev/null
    wait $SERVER_PID 2>/dev/null
    
    echo ""
    echo "=================================="
    echo ""
done

echo "All tests completed!"
echo "See results above ☝️"
