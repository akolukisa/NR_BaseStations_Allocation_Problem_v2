#!/bin/bash
# Comprehensive benchmark runner for all RIC algorithms

echo "============================================================================"
echo "5G-LENA RIC COMPREHENSIVE BENCHMARK"
echo "============================================================================"
echo ""
echo "This script will:"
echo "  1. Test all algorithms (Max-SINR, GA, HGA, PBIG)"
echo "  2. Scale from 10 to 50 UEs (in steps of 5)"
echo "  3. Test with 8 and 16 beams"
echo "  4. Run 3 times per configuration for statistical significance"
echo "  5. Generate analysis plots"
echo ""
echo "Estimated time: 30-60 minutes"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."

cd "$(dirname "$0")"

# Array of algorithms to test
ALGORITHMS=("Max-SINR" "GA" "HGA" "PBIG")

# Output file
OUTPUT_FILE="comprehensive_benchmark_$(date +%Y%m%d_%H%M%S).csv"

echo ""
echo "============================================================================"
echo "RUNNING BENCHMARKS"
echo "============================================================================"
echo ""
echo "Results will be saved to: $OUTPUT_FILE"
echo ""

# Test each algorithm
for algo in "${ALGORITHMS[@]}"; do
    echo "------------------------------------------------------------------------"
    echo "Testing $algo algorithm..."
    echo "------------------------------------------------------------------------"
    
    # Start RIC server in background
    python3 ric_server.py --algorithm "$algo" --port 5555 > /dev/null 2>&1 &
    SERVER_PID=$!
    
    # Wait for server to start
    sleep 2
    
    # Check if server is running
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "ERROR: Failed to start RIC server for $algo"
        continue
    fi
    
    echo "✓ RIC server started (PID: $SERVER_PID)"
    
    # Run benchmark for this algorithm
    if [ "$algo" = "${ALGORITHMS[0]}" ]; then
        # First algorithm: create new file
        python3 benchmark_scalability.py \
            --algorithms "$algo" \
            --beams 8 16 \
            --ues 10 15 20 25 30 35 40 45 50 \
            --runs 3 \
            --output "$OUTPUT_FILE"
    else
        # Subsequent algorithms: append to existing file
        TEMP_FILE="temp_${algo}.csv"
        python3 benchmark_scalability.py \
            --algorithms "$algo" \
            --beams 8 16 \
            --ues 10 15 20 25 30 35 40 45 50 \
            --runs 3 \
            --output "$TEMP_FILE"
        
        # Append results (skip header)
        if [ -f "$TEMP_FILE" ]; then
            tail -n +2 "$TEMP_FILE" >> "$OUTPUT_FILE"
            rm "$TEMP_FILE"
        fi
    fi
    
    # Stop server
    kill $SERVER_PID 2>/dev/null
    wait $SERVER_PID 2>/dev/null
    
    echo "✓ Completed $algo benchmark"
    echo ""
    sleep 1
done

echo ""
echo "============================================================================"
echo "GENERATING ANALYSIS"
echo "============================================================================"
echo ""

# Analyze results
python3 analyze_results.py "$OUTPUT_FILE"

echo ""
echo "============================================================================"
echo "BENCHMARK COMPLETE!"
echo "============================================================================"
echo ""
echo "Results saved to:"
echo "  - CSV data: $OUTPUT_FILE"
echo "  - Plots: scalability_analysis.png, algorithm_comparison.png"
echo ""
echo "You can now:"
echo "  1. View the plots"
echo "  2. Import CSV into Excel/spreadsheet for further analysis"
echo "  3. Use results in your thesis"
echo ""
