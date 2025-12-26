#!/usr/bin/env python3
"""
4 Beam Skalabilite Benchmark - Gerçek 5G-LENA Verileri
3 Metrik: Sum-rate, Jain Fairness, Amaç Fonksiyonu
"""

import numpy as np
import json
import os
import matplotlib.pyplot as plt
import time

# =====================================
# SINR Yükleme
# =====================================
def load_lena_sinr(base_dir, ue_target):
    """LENA SINR verilerini yükle"""
    ue_dir = f"{base_dir}/ue{ue_target}"
    if not os.path.exists(ue_dir):
        print(f"HATA: {ue_dir} bulunamadı")
        return None, 0
    
    # 300ms dosyasını tercih et
    all_sinr = []
    all_ues = []
    
    for gnb_id in range(3):
        for time_ms in ['300ms', '200ms']:
            filepath = os.path.join(ue_dir, f'sinr_gnb{gnb_id}_{time_ms}.json')
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    data = json.load(f)
                sinr = np.array(data['sinr_matrix_dB'])
                all_sinr.append(sinr)
                all_ues.extend(data['ue_ids'])
                break
    
    if not all_sinr:
        return None, 0
    
    # Tüm gNB'leri birleştir
    combined = np.hstack(all_sinr)
    return combined, len(set(all_ues))

# =====================================
# Rate Hesaplama (Interferans dahil)
# =====================================
def compute_rates(sinr_matrix, assignment, interference_factor=0.5):
    """Shannon rate hesapla (interferans dahil)"""
    num_beams, num_ues = sinr_matrix.shape
    rates = np.zeros(num_ues)
    
    # Beam yüklerini hesapla
    beam_loads = {}
    for ue, beam in enumerate(assignment):
        if beam not in beam_loads:
            beam_loads[beam] = []
        beam_loads[beam].append(ue)
    
    for ue, beam in enumerate(assignment):
        original_sinr_db = sinr_matrix[beam, ue]
        num_sharing = len(beam_loads.get(beam, [1]))
        
        # Interferans penalty
        if num_sharing > 1:
            interference_penalty = interference_factor * (num_sharing - 1)
            effective_sinr_db = original_sinr_db - interference_penalty
        else:
            effective_sinr_db = original_sinr_db
        
        sinr_linear = 10 ** (effective_sinr_db / 10)
        rate = np.log2(1 + max(0.001, sinr_linear))
        rates[ue] = max(0.01, rate)
    
    return rates

def jain_fairness(rates):
    """Jain Fairness Index"""
    if len(rates) == 0 or np.sum(rates) == 0:
        return 0.0
    return (np.sum(rates) ** 2) / (len(rates) * np.sum(rates ** 2))

def objective_function(rates, alpha=0.7):
    """Amaç fonksiyonu: F(X) = α⋅(ΣRijk) + (1-α)⋅Jain(X)"""
    sum_rate = np.sum(rates)
    jain = jain_fairness(rates)
    return alpha * sum_rate + (1 - alpha) * jain

# =====================================
# Algoritmalar
# =====================================
def max_sinr_assignment(sinr_matrix):
    """Her UE en yüksek SINR beam'e atanır"""
    return np.argmax(sinr_matrix, axis=0)

def hga_algorithm(sinr_matrix, pop_size=50, generations=100, alpha=0.7):
    """Hybrid Genetic Algorithm"""
    num_beams, num_ues = sinr_matrix.shape
    
    def evaluate(ind):
        rates = compute_rates(sinr_matrix, ind)
        return objective_function(rates, alpha)
    
    # Max-SINR ile başlat
    base = np.argmax(sinr_matrix, axis=0)
    population = [np.clip(base + np.random.randint(-1, 2, num_ues), 0, num_beams-1) for _ in range(pop_size)]
    
    for gen in range(generations):
        scores = [evaluate(ind) for ind in population]
        sorted_idx = np.argsort(scores)[::-1]
        population = [population[i] for i in sorted_idx[:pop_size//2]]
        
        while len(population) < pop_size:
            p1, p2 = population[np.random.randint(0, len(population)//2)], population[np.random.randint(0, len(population)//2)]
            cp = np.random.randint(1, num_ues)
            child = np.concatenate([p1[:cp], p2[cp:]])
            if np.random.random() < 0.1:
                child[np.random.randint(0, num_ues)] = np.random.randint(0, num_beams)
            population.append(child)
    
    return max(population, key=evaluate)

def ga_algorithm(sinr_matrix, pop_size=50, generations=100, alpha=0.7):
    """Standard Genetic Algorithm"""
    num_beams, num_ues = sinr_matrix.shape
    
    def evaluate(ind):
        rates = compute_rates(sinr_matrix, ind)
        return objective_function(rates, alpha)
    
    population = [np.random.randint(0, num_beams, num_ues) for _ in range(pop_size)]
    
    for gen in range(generations):
        scores = [evaluate(ind) for ind in population]
        sorted_idx = np.argsort(scores)[::-1]
        population = [population[i] for i in sorted_idx[:pop_size//2]]
        
        while len(population) < pop_size:
            p1, p2 = population[np.random.randint(0, len(population)//2)], population[np.random.randint(0, len(population)//2)]
            cp = np.random.randint(1, num_ues)
            child = np.concatenate([p1[:cp], p2[cp:]])
            if np.random.random() < 0.1:
                child[np.random.randint(0, num_ues)] = np.random.randint(0, num_beams)
            population.append(child)
    
    return max(population, key=evaluate)

def pbig_algorithm(sinr_matrix, max_iter=100, alpha=0.7):
    """Probability-Based Iterative Greedy"""
    num_beams, num_ues = sinr_matrix.shape
    prob_matrix = np.ones((num_ues, num_beams)) / num_beams
    
    def sample():
        return np.array([np.random.choice(num_beams, p=prob_matrix[ue]) for ue in range(num_ues)])
    
    def evaluate(ind):
        rates = compute_rates(sinr_matrix, ind)
        return objective_function(rates, alpha)
    
    best = sample()
    best_fitness = evaluate(best)
    
    for iteration in range(max_iter):
        samples = [sample() for _ in range(20)]
        scores = [evaluate(s) for s in samples]
        best_idx = np.argmax(scores)
        
        if scores[best_idx] > best_fitness:
            best_fitness = scores[best_idx]
            best = samples[best_idx]
        
        for ue in range(num_ues):
            prob_matrix[ue] *= 0.9
            prob_matrix[ue, best[ue]] += 0.1
            prob_matrix[ue] /= prob_matrix[ue].sum()
    
    return best

# =====================================
# Ana Benchmark
# =====================================
def run_benchmark():
    print("="*70)
    print("4 BEAM SKALABILITE BENCHMARK - GERÇEK 5G-LENA VERİLERİ")
    print("="*70)
    
    base_dir = "lena_scalability_4beam"
    ue_targets = [10, 15, 20, 25, 30, 35, 40, 45, 50]
    algorithms = ['GA', 'HGA', 'PBIG', 'Max-SINR']
    alpha = 0.7
    
    # Sonuç depolama
    results = {
        'ue_counts': [],
        'actual_ues': [],
        'sum_rate': {alg: [] for alg in algorithms},
        'jain': {alg: [] for alg in algorithms},
        'fitness': {alg: [] for alg in algorithms},
        'runtime': {alg: [] for alg in algorithms}
    }
    
    for ue_target in ue_targets:
        sinr_matrix, actual_ues = load_lena_sinr(base_dir, ue_target)
        
        if sinr_matrix is None:
            print(f"UYARI: {ue_target} UE verisi bulunamadı")
            continue
        
        print(f"\n{ue_target} UE (gerçek: {actual_ues} UE, {sinr_matrix.shape[0]} beam):")
        results['ue_counts'].append(ue_target)
        results['actual_ues'].append(actual_ues)
        
        for alg_name in algorithms:
            start = time.time()
            
            if alg_name == 'HGA':
                assignment = hga_algorithm(sinr_matrix, alpha=alpha)
            elif alg_name == 'PBIG':
                assignment = pbig_algorithm(sinr_matrix, alpha=alpha)
            elif alg_name == 'GA':
                assignment = ga_algorithm(sinr_matrix, alpha=alpha)
            else:  # Max-SINR
                assignment = max_sinr_assignment(sinr_matrix)
            
            runtime = time.time() - start
            rates = compute_rates(sinr_matrix, assignment)
            
            sum_rate = np.sum(rates)
            jain = jain_fairness(rates)
            fitness = objective_function(rates, alpha)
            
            results['sum_rate'][alg_name].append(sum_rate)
            results['jain'][alg_name].append(jain)
            results['fitness'][alg_name].append(fitness)
            results['runtime'][alg_name].append(runtime)
            
            print(f"  {alg_name:10s}: Sum-rate={sum_rate:7.2f} bps/Hz, Jain={jain:.4f}, Fitness={fitness:.2f}, Time={runtime:.3f}s")
    
    return results

# =====================================
# Grafik Oluşturma
# =====================================
def create_plots(results, output_dir="plots_lena_4beam"):
    os.makedirs(output_dir, exist_ok=True)
    
    ue_counts = results['ue_counts']
    
    # bps/Hz -> Mbps dönüşümü (100 MHz, %80 efficiency)
    bw_mhz = 100
    efficiency = 0.8
    
    styles = {
        'GA': {'color': 'orange', 'marker': 'o', 'linestyle': '-'},
        'HGA': {'color': 'red', 'marker': 's', 'linestyle': '-'},
        'PBIG': {'color': 'blue', 'marker': '^', 'linestyle': '-'},
        'Max-SINR': {'color': 'purple', 'marker': 'D', 'linestyle': '--'}
    }
    
    # 1. Sum-Rate Grafiği (Mbps)
    plt.figure(figsize=(12, 8))
    for alg in ['GA', 'HGA', 'PBIG', 'Max-SINR']:
        data_mbps = [r * bw_mhz * efficiency for r in results['sum_rate'][alg]]
        label = alg if alg != 'Max-SINR' else 'Max-SINR (Üst Sınır)'
        plt.plot(ue_counts, data_mbps, 
                 color=styles[alg]['color'],
                 marker=styles[alg]['marker'],
                 linestyle=styles[alg]['linestyle'],
                 linewidth=2.5, markersize=10, label=label)
    
    plt.xlabel('Kullanıcı Sayısı', fontsize=14, fontweight='bold')
    plt.ylabel('Sistem Toplam Veri Hızı (Mbps)', fontsize=14, fontweight='bold')
    plt.title('Sistem Toplam Veri Hızı (4 Beam)\n5G-LENA PHY, n78 3.5GHz, 100 MHz', fontsize=14, fontweight='bold')
    plt.legend(loc='upper left', fontsize=11)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xticks(ue_counts)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/sumrate_4beam_lena.png', dpi=150)
    plt.savefig(f'{output_dir}/sumrate_4beam_lena.pdf')
    print(f"Kaydedildi: {output_dir}/sumrate_4beam_lena.png")
    
    # 2. Jain Fairness Grafiği
    plt.figure(figsize=(12, 8))
    for alg in ['GA', 'HGA', 'PBIG', 'Max-SINR']:
        plt.plot(ue_counts, results['jain'][alg],
                 color=styles[alg]['color'],
                 marker=styles[alg]['marker'],
                 linestyle=styles[alg]['linestyle'],
                 linewidth=2.5, markersize=10, label=alg)
    
    plt.xlabel('Kullanıcı Sayısı', fontsize=14, fontweight='bold')
    plt.ylabel('Jain Adalet İndeksi', fontsize=14, fontweight='bold')
    plt.title('Jain Adalet İndeksi (4 Beam)\n5G-LENA PHY', fontsize=14, fontweight='bold')
    plt.legend(loc='lower left', fontsize=11)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xticks(ue_counts)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/jain_4beam_lena.png', dpi=150)
    plt.savefig(f'{output_dir}/jain_4beam_lena.pdf')
    print(f"Kaydedildi: {output_dir}/jain_4beam_lena.png")
    
    # 3. Amaç Fonksiyonu Grafiği
    plt.figure(figsize=(12, 8))
    for alg in ['GA', 'HGA', 'PBIG', 'Max-SINR']:
        plt.plot(ue_counts, results['fitness'][alg],
                 color=styles[alg]['color'],
                 marker=styles[alg]['marker'],
                 linestyle=styles[alg]['linestyle'],
                 linewidth=2.5, markersize=10, label=alg)
    
    plt.xlabel('Kullanıcı Sayısı', fontsize=14, fontweight='bold')
    plt.ylabel('Amaç Fonksiyonu Değeri', fontsize=14, fontweight='bold')
    plt.title('Amaç Fonksiyonu: F(X) = α⋅ΣR + (1-α)⋅Jain (α=0.7)\n5G-LENA PHY, 4 Beam', fontsize=14, fontweight='bold')
    plt.legend(loc='upper left', fontsize=11)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xticks(ue_counts)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fitness_4beam_lena.png', dpi=150)
    plt.savefig(f'{output_dir}/fitness_4beam_lena.pdf')
    print(f"Kaydedildi: {output_dir}/fitness_4beam_lena.png")
    
    # Tablo çıktısı
    print("\n" + "="*80)
    print("SONUÇ TABLOSU (4 Beam, Gerçek 5G-LENA)")
    print("="*80)
    print(f"{'UE':>5} | {'GA Sum-Rate':>12} | {'HGA Sum-Rate':>12} | {'PBIG Sum-Rate':>13} | {'Max-SINR':>12}")
    print("-"*70)
    for i, ue in enumerate(ue_counts):
        sr_mbps = {alg: results['sum_rate'][alg][i] * bw_mhz * efficiency for alg in ['GA', 'HGA', 'PBIG', 'Max-SINR']}
        print(f"{ue:>5} | {sr_mbps['GA']:>10.0f} Mbps | {sr_mbps['HGA']:>10.0f} Mbps | {sr_mbps['PBIG']:>11.0f} Mbps | {sr_mbps['Max-SINR']:>10.0f} Mbps")

# =====================================
# Main
# =====================================
if __name__ == '__main__':
    os.chdir('/Users/akolukisa/FinalThesis/ric-python')
    results = run_benchmark()
    create_plots(results)
    print("\nTamamlandı!")
