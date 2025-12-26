#!/usr/bin/env python3
"""
5G-LENA SINR ile Tüm Algoritmaları Test Et
"""

import json
import socket
import subprocess
import numpy as np
from pathlib import Path
import time
import sys
import signal
import os

ALGORITHMS = ['Max-SINR', 'GA', 'HGA', 'PBIG']

def load_lena_sinr(sinr_dir):
    sinr_dir = Path(sinr_dir)
    gnb_data = {}
    json_files = list(sinr_dir.glob('sinr_gnb*_*.json'))
    
    latest_files = {}
    for f in json_files:
        parts = f.stem.split('_')
        if len(parts) >= 3:
            gnb_id = int(parts[1].replace('gnb', ''))
            timestamp = int(parts[2].replace('ms', ''))
            if gnb_id not in latest_files or timestamp > latest_files[gnb_id][0]:
                latest_files[gnb_id] = (timestamp, f)
    
    for gnb_id, (ts, filepath) in sorted(latest_files.items()):
        with open(filepath, 'r') as f:
            data = json.load(f)
        gnb_data[gnb_id] = {
            'num_beams': data['num_beams'],
            'num_ues': data['num_ues'],
            'ue_ids': data['ue_ids'],
            'sinr_matrix': np.array(data['sinr_matrix_dB'])
        }
    return gnb_data

def normalize_sinr(sinr_db, min_sinr=-100):
    return max(sinr_db, min_sinr) - min_sinr

def create_combined(gnb_data):
    all_ue_ids = set()
    for gdata in gnb_data.values():
        all_ue_ids.update(gdata['ue_ids'])
    all_ue_ids = sorted(list(all_ue_ids))
    
    num_gnbs = len(gnb_data)
    num_beams = list(gnb_data.values())[0]['num_beams']
    num_ues = len(all_ue_ids)
    ue_id_to_idx = {ue_id: idx for idx, ue_id in enumerate(all_ue_ids)}
    
    sinr_matrix_3d = np.full((num_gnbs, num_beams, num_ues), 0.0)
    
    for gnb_id, gdata in gnb_data.items():
        for local_ue_idx, ue_id in enumerate(gdata['ue_ids']):
            global_ue_idx = ue_id_to_idx[ue_id]
            for beam in range(num_beams):
                sinr_val = gdata['sinr_matrix'][beam, local_ue_idx]
                sinr_matrix_3d[gnb_id, beam, global_ue_idx] = normalize_sinr(sinr_val)
    
    return sinr_matrix_3d, all_ue_ids, num_beams

def calculate_metrics(sinr_matrix, beam_assignment):
    """Sum-rate ve fairness hesapla"""
    num_beams, num_ues = sinr_matrix.shape
    
    rates = []
    for ue_idx in range(num_ues):
        beam_idx = beam_assignment[ue_idx]
        if beam_idx < num_beams:
            sinr_db = sinr_matrix[beam_idx, ue_idx]
            sinr_linear = 10 ** (sinr_db / 10)
            rate = np.log2(1 + sinr_linear)
        else:
            rate = 0
        rates.append(rate)
    
    rates = np.array(rates)
    sum_rate = np.sum(rates)
    
    if np.sum(rates) > 0 and np.sum(rates ** 2) > 0:
        fairness = (np.sum(rates) ** 2) / (num_ues * np.sum(rates ** 2))
    else:
        fairness = 0.0
    
    return sum_rate, fairness, rates

def test_algorithm(algorithm, sinr_matrix_flat, port=5555):
    """Tek algoritma test et"""
    num_beams, num_ues = sinr_matrix_flat.shape
    
    request = {
        'num_beams': num_beams,
        'num_ues': num_ues,
        'sinr_matrix_dB': sinr_matrix_flat.tolist(),
        'alpha': 1.0
    }
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(120)
        sock.connect(('127.0.0.1', port))
        sock.sendall(json.dumps(request).encode())
        
        response_data = b''
        while True:
            chunk = sock.recv(65536)
            if not chunk: break
            response_data += chunk
            try:
                json.loads(response_data.decode())
                break
            except: continue
        sock.close()
        
        if response_data:
            return json.loads(response_data.decode())
    except Exception as e:
        print(f"    Error: {e}")
    
    return None

def main():
    print("=" * 70)
    print("5G-LENA SINR BENCHMARK - TÜM ALGORİTMALAR")
    print("=" * 70)
    
    # SINR yükle
    gnb_data = load_lena_sinr('./lena_sinr')
    print(f"\nLoaded {len(gnb_data)} gNBs from 5G-LENA simulation")
    
    sinr_matrix_3d, ue_ids, num_beams_per_gnb = create_combined(gnb_data)
    num_gnbs, num_beams, num_ues = sinr_matrix_3d.shape
    
    # Flat matrix (RIC server için)
    sinr_matrix_flat = sinr_matrix_3d.reshape(-1, num_ues)
    
    print(f"Combined SINR: {num_gnbs} gNBs × {num_beams} beams × {num_ues} UEs")
    print(f"Total beams: {num_gnbs * num_beams}")
    print(f"SINR range: [{sinr_matrix_flat.min():.1f}, {sinr_matrix_flat.max():.1f}] dB")
    
    results = []
    
    for algorithm in ALGORITHMS:
        print(f"\n{'='*60}")
        print(f"Testing: {algorithm}")
        print(f"{'='*60}")
        
        # Server'ı başlat
        print(f"  Starting RIC server with {algorithm}...")
        
        # Eski process'i öldür
        os.system("pkill -f 'ric_server.py' 2>/dev/null")
        time.sleep(1)
        
        # Yeni server başlat
        server_proc = subprocess.Popen(
            ['python3.11', 'ric_server.py', '--algorithm', algorithm, '--port', '5555'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        time.sleep(2)  # Server'ın başlamasını bekle
        
        # Test et
        print(f"  Sending request...")
        start_time = time.time()
        response = test_algorithm(algorithm, sinr_matrix_flat, port=5555)
        elapsed_ms = (time.time() - start_time) * 1000
        
        # Server'ı kapat
        server_proc.terminate()
        
        if response and 'beam_for_ue' in response:
            beam_assignment = response['beam_for_ue']
            
            # Metrikleri hesapla
            sum_rate, fairness, rates = calculate_metrics(sinr_matrix_flat, beam_assignment)
            
            result = {
                'algorithm': algorithm,
                'sum_rate': sum_rate,
                'fairness': fairness,
                'runtime_ms': elapsed_ms,
                'min_rate': min(rates),
                'max_rate': max(rates),
                'avg_rate': np.mean(rates)
            }
            results.append(result)
            
            print(f"  ✓ Sum-Rate: {sum_rate:.2f} bps/Hz")
            print(f"  ✓ Fairness: {fairness:.4f}")
            print(f"  ✓ Runtime: {elapsed_ms:.2f} ms")
            print(f"  ✓ Rate range: [{min(rates):.2f}, {max(rates):.2f}]")
        else:
            print(f"  ✗ FAILED!")
    
    # Sonuç özeti
    if results:
        print(f"\n{'='*70}")
        print("BENCHMARK RESULTS - 5G-LENA Real SINR (3 gNB × 8 beam × 51 UE)")
        print(f"{'='*70}")
        print(f"{'Algorithm':<12} {'Sum-Rate':>12} {'Fairness':>12} {'Runtime(ms)':>14}")
        print("-" * 52)
        
        for r in sorted(results, key=lambda x: -x['sum_rate']):
            print(f"{r['algorithm']:<12} {r['sum_rate']:>12.2f} {r['fairness']:>12.4f} {r['runtime_ms']:>14.2f}")
        
        # En iyi algoritma
        best = max(results, key=lambda x: x['sum_rate'])
        print(f"\n🏆 Best Sum-Rate: {best['algorithm']} ({best['sum_rate']:.2f})")
        
        best_fair = max(results, key=lambda x: x['fairness'])
        print(f"🏆 Best Fairness: {best_fair['algorithm']} ({best_fair['fairness']:.4f})")
        
        fastest = min(results, key=lambda x: x['runtime_ms'])
        print(f"⚡ Fastest: {fastest['algorithm']} ({fastest['runtime_ms']:.2f} ms)")
        
        # CSV kaydet
        import csv
        with open('results_lena_real.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to: results_lena_real.csv")

if __name__ == "__main__":
    main()
