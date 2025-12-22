#!/usr/bin/env python3
"""
Test client for RIC Server - simulates ns-3 requests
"""

import socket
import json
import numpy as np


def test_ric_server(algorithm='GA', num_beams=8, num_ues=4):
    """
    Test the RIC server with a synthetic scenario
    """
    print(f"Testing RIC Server with {algorithm} algorithm")
    print(f"Scenario: {num_beams} beams, {num_ues} UEs")
    
    # Create synthetic SINR matrix (beams x UEs)
    # Simulate a realistic scenario where different UEs have different best beams
    np.random.seed(42)
    sinr_matrix = np.random.uniform(-10, 20, (num_beams, num_ues))
    
    # Make it more realistic: each UE has 2-3 good beams
    for ue in range(num_ues):
        best_beams = np.random.choice(num_beams, 2, replace=False)
        for beam in best_beams:
            sinr_matrix[beam, ue] += np.random.uniform(5, 15)
    
    print("\nSINR Matrix (dB):")
    print(sinr_matrix)
    
    # Create request
    request = {
        "scenario_id": 1,
        "gNB_id": 0,
        "num_beams": num_beams,
        "num_ues": num_ues,
        "sinr_matrix_dB": sinr_matrix.tolist()
    }
    
    # Connect to RIC server
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(('127.0.0.1', 5555))
        print("\nConnected to RIC server")
        
        # Send request
        request_json = json.dumps(request)
        sock.sendall(request_json.encode('utf-8'))
        print("Request sent")
        
        # Receive response
        response_data = sock.recv(65536)
        response = json.loads(response_data.decode('utf-8'))
        
        print("\n=== RIC Response ===")
        print(f"Algorithm: {response['algorithm']}")
        print(f"Beam Assignment: {response['beam_for_ue']}")
        if 'objective_value' in response:
            print(f"Objective Value: {response['objective_value']:.2f}")
        
        # Verify assignment
        print("\n=== Assignment Details ===")
        for ue_idx, beam_idx in enumerate(response['beam_for_ue']):
            sinr = sinr_matrix[beam_idx, ue_idx]
            print(f"UE {ue_idx} -> Beam {beam_idx} (SINR: {sinr:.2f} dB)")
        
        sock.close()
        print("\nTest completed successfully!")
        
    except ConnectionRefusedError:
        print("\nERROR: Could not connect to RIC server.")
        print("Make sure the server is running:")
        print("  python3 ric_server.py --algorithm", algorithm)
    except Exception as e:
        print(f"\nERROR: {e}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test RIC Server')
    parser.add_argument('--algorithm', choices=['Max-SINR', 'Exhaustive', 'GA', 'HGA', 'PBIG'],
                       default='GA', help='Algorithm to test')
    parser.add_argument('--beams', type=int, default=8, help='Number of beams')
    parser.add_argument('--ues', type=int, default=4, help='Number of UEs')
    
    args = parser.parse_args()
    
    test_ric_server(args.algorithm, args.beams, args.ues)
