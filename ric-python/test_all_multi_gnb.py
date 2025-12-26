#!/usr/bin/env python3
"""
Test all algorithms with multi-gNB scenario
"""
import subprocess
import socket
import json
import numpy as np
import time
import sys

ALGORITHMS = ['Max-SINR', 'GA', 'HGA', 'PBIG']
PORT = 5557


def build_test_sinr(num_gnbs=3, num_beams=8, num_ues=20, seed=42):
    """Create realistic multi-gNB SINR matrix"""
    np.random.seed(seed)
    
    # Her UE'yi bir gNB'ye ata
    ue_gnb = np.random.randint(0, num_gnbs, num_ues)
    
    sinr = np.zeros((num_gnbs, num_beams, num_ues))
    
    for ue in range(num_ues):
        primary_gnb = ue_gnb[ue]
        for gnb in range(num_gnbs):
            if gnb == primary_gnb:
                # Primary gNB: yüksek SINR, 2-3 iyi beam
                sinr[gnb, :, ue] = np.random.uniform(-5, 10, num_beams)
                good_beams = np.random.choice(num_beams, np.random.randint(2, 4), replace=False)
                for b in good_beams:
                    sinr[gnb, b, ue] = np.random.uniform(15, 30)
            else:
                # Diğer gNB'ler: düşük SINR
                sinr[gnb, :, ue] = np.random.uniform(-20, 0, num_beams)
    
    return sinr.tolist()


def test_algorithm(algorithm, num_gnbs=3, num_beams=8, num_ues=20):
    """Test a single algorithm"""
    # Start server
    server_proc = subprocess.Popen(
        ['python3', 'ric_server.py', '--algorithm', algorithm, '--port', str(PORT)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    time.sleep(1)  # Wait for server to start
    
    try:
        sinr_matrix = build_test_sinr(num_gnbs, num_beams, num_ues)
        
        request = {
            "scenario_id": 1,
            "num_gnbs": num_gnbs,
            "num_ues": num_ues,
            "sinr_matrix_dB": sinr_matrix,
        }
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(30)
        sock.connect(('127.0.0.1', PORT))
        
        start_time = time.time()
        sock.sendall(json.dumps(request).encode('utf-8'))
        
        response_data = sock.recv(131072)
        elapsed = (time.time() - start_time) * 1000
        
        sock.close()
        
        response = json.loads(response_data.decode('utf-8'))
        
        return {
            'algorithm': algorithm,
            'objective': response.get('objective_value', 0),
            'runtime_ms': elapsed,
            'gnb_for_ue': response.get('gnb_for_ue', []),
            'beam_for_ue': response.get('beam_for_ue', []),
            'success': True
        }
        
    except Exception as e:
        return {
            'algorithm': algorithm,
            'error': str(e),
            'success': False
        }
    finally:
        server_proc.terminate()
        server_proc.wait()


def main():
    print("=" * 70)
    print("MULTI-GNB ALGORITHM TEST (3 gNB, 8 beams/gNB, 20 UEs)")
    print("=" * 70)
    
    results = []
    
    for algo in ALGORITHMS:
        print(f"\nTesting {algo}...", end=' ')
        sys.stdout.flush()
        
        result = test_algorithm(algo)
        results.append(result)
        
        if result['success']:
            print(f"✓")
            print(f"   Objective: {result['objective']:.2f}")
            print(f"   Runtime: {result['runtime_ms']:.2f} ms")
        else:
            print(f"✗ Error: {result.get('error', 'Unknown')}")
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Algorithm':<15} {'Objective':<15} {'Runtime (ms)':<15} {'Status':<10}")
    print("-" * 55)
    
    for r in results:
        if r['success']:
            print(f"{r['algorithm']:<15} {r['objective']:>10.2f}     {r['runtime_ms']:>10.2f}     ✓")
        else:
            print(f"{r['algorithm']:<15} {'N/A':<15} {'N/A':<15} ✗")
    
    print("\n" + "=" * 70)
    
    # Check if all successful
    all_success = all(r['success'] for r in results)
    if all_success:
        print("All algorithms tested successfully!")
    else:
        print("Some algorithms failed. Check errors above.")
        sys.exit(1)


if __name__ == '__main__':
    main()
