#!/usr/bin/env python3
"""
8 Beam Skalabilite Benchmark - 10 to 100 UE
Gerçek 5G-LENA Verileri
"""

import numpy as np
import json
import os
import matplotlib.pyplot as plt
import time

def load_lena_sinr_base(sinr_dir):
    """Load base SINR data from lena_sinr_8beam directory"""
    all_sinr = []
    for gnb_id in range(3):
        filepath = os.path.join(sinr_dir, f'sinr_gnb{gnb_id}_300ms.json')
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
            sinr = np.array(data['sinr_matrix_dB'])
            all_sinr.append(sinr)
    if not all_sinr:
        return None
    combined = np.hstack(all_sinr)
    return combined

def create_sinr_for_ues(base_sinr, num_beams, num_ues, seed=42):
    """Create SINR matrix for given UE count from base LENA data"""
    np.random.seed(seed)
    base_num_ues = base_sinr.shape[1]
    
    if num_ues <= base_num_ues:
        indices = np.random.choice(base_num_ues, num_ues, replace=False)
        return base_sinr[:, indices]
    else:
        sinr_matrix = np.zeros((num_beams, num_ues))
        for ue in range(num_ues):
            base_ue = ue % base_num_ues
            noise = np.random.uniform(-2, 2, num_beams)
            sinr_matrix[:, ue] = base_sinr[:, base_ue] + noise
        return sinr_matrix

def compute_rates(sinr_matrix, assignment, interference_factor=0.5):
    num_beams, num_ues = sinr_matrix.shape
    rates = np.zeros(num_ues)
    
    beam_loads = {}
    for ue, beam in enumerate(assignment):
        if beam not in beam_loads:
            beam_loads[beam] = []
        beam_loads[beam].append(ue)
    
    for ue, beam in enumerate(assignment):
        original_sinr_db = sinr_matrix[beam, ue]
        num_sharing = len(beam_loads.get(beam, [1]))
        
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
    if len(rates) == 0 or np.sum(rates) == 0:
        return 0.0
    return (np.sum(rates) ** 2) / (len(rates) * np.sum(rates ** 2))

def objective_function(rates, alpha=0.7):
    sum_rate = np.sum(rates)
    jain = jain_fairness(rates)
    return alpha * sum_rate + (1 - alpha) * jain

def max_sinr_assignment(sinr_matrix):
    return np.argmax(sinr_matrix, axis=0)

def hga_algorithm(sinr_matrix, pop_size=50, generations=100, alpha=0.7):
    num_beams, num_ues = sinr_matrix.shape
    
    def evaluate(ind):
        rates = compute_rates(sinr_matrix, ind)
        return objective_function(rates, alpha)
    
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
        
        # Local search
        for idx in np.random.choice(len(population), max(1, len(population)//3), replace=False):
            ind = population[idx]
            for _ in range(10):
                ue = np.random.randint(num_ues)
                neighbor = ind.copy()
                neighbor[ue] = np.random.randint(num_beams)
                if evaluate(neighbor) > evaluate(ind):
                    population[idx] = neighbor
    
    return max(population, key=evaluate)

def ga_algorithm(sinr_matrix, pop_size=50, generations=100, alpha=0.7):
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
    """Population-Based Iterated Greedy with Load-Aware Reconstruction"""
    num_beams, num_ues = sinr_matrix.shape
    
    def evaluate(ind):
        rates = compute_rates(sinr_matrix, ind)
        return objective_function(rates, alpha)
    
    def load_aware_greedy():
        """Load-aware greedy construction"""
        assignment = np.full(num_ues, -1, dtype=int)
        beam_loads = {b: 0 for b in range(num_beams)}
        
        # Sort UEs by max SINR
        ue_max_sinr = [(ue, np.max(sinr_matrix[:, ue])) for ue in range(num_ues)]
        ue_max_sinr.sort(key=lambda x: x[1], reverse=True)
        
        for ue, _ in ue_max_sinr:
            beam_sinrs = [(b, sinr_matrix[b, ue]) for b in range(num_beams)]
            beam_sinrs.sort(key=lambda x: x[1], reverse=True)
            top_beams = beam_sinrs[:min(3, num_beams)]
            
            best_beam = top_beams[0][0]
            best_sinr = top_beams[0][1]
            
            for beam, sinr in top_beams[1:]:
                if best_sinr - sinr < 5.0 and beam_loads[beam] < beam_loads[best_beam]:
                    best_beam = beam
                    best_sinr = sinr
            
            assignment[ue] = best_beam
            beam_loads[best_beam] += 1
        
        return assignment
    
    def destruct_reconstruct(ind, d_ratio=0.3):
        """Destruction and load-aware reconstruction"""
        new_ind = ind.copy()
        num_destroy = max(1, int(num_ues * d_ratio))
        destroy_indices = np.random.choice(num_ues, num_destroy, replace=False)
        
        beam_loads = {b: 0 for b in range(num_beams)}
        for ue, beam in enumerate(new_ind):
            if ue not in destroy_indices:
                beam_loads[beam] += 1
        
        for ue in destroy_indices:
            beam_sinrs = [(b, sinr_matrix[b, ue]) for b in range(num_beams)]
            beam_sinrs.sort(key=lambda x: x[1], reverse=True)
            
            best_beam = beam_sinrs[0][0]
            best_sinr = beam_sinrs[0][1]
            
            for beam, sinr in beam_sinrs[1:3]:
                if best_sinr - sinr < 5.0 and beam_loads[beam] < beam_loads[best_beam]:
                    best_beam = beam
            
            new_ind[ue] = best_beam
            beam_loads[best_beam] += 1
        
        return new_ind
    
    # Initialize population
    pop_size = 30
    population = [load_aware_greedy() for _ in range(pop_size // 2)]
    population += [np.random.randint(0, num_beams, num_ues) for _ in range(pop_size - len(population))]
    
    best = max(population, key=evaluate)
    best_fitness = evaluate(best)
    
    for iteration in range(max_iter):
        new_population = []
        for ind in population:
            new_ind = destruct_reconstruct(ind)
            if evaluate(new_ind) > evaluate(ind):
                new_population.append(new_ind)
            else:
                new_population.append(ind)
            
            # Track best
            ind_fitness = evaluate(new_population[-1])
            if ind_fitness > best_fitness:
                best = new_population[-1].copy()
                best_fitness = ind_fitness
        
        population = new_population
    
    return best

def run_benchmark():
    print("="*70)
    print("8 BEAM BENCHMARK - 10 to 100 UE (Gerçek 5G-LENA + İnterpolasyon)")
    print("="*70)
    
    base_sinr = load_lena_sinr_base("lena_sinr_8beam")
    if base_sinr is None:
        print("HATA: lena_sinr_8beam verisi bulunamadı!")
        return None
    
    num_beams = 8
    print(f"Base LENA verisi: {base_sinr.shape[0]} beams, {base_sinr.shape[1]} UEs")
    print(f"SINR range: {np.min(base_sinr):.1f} to {np.max(base_sinr):.1f} dB")
    
    ue_targets = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    algorithms = ['GA', 'HGA', 'PBIG', 'Max-SINR']
    alpha = 0.7
    bw_mhz = 100
    efficiency = 0.8
    
    results = {
        'ue_counts': [],
        'sum_rate': {alg: [] for alg in algorithms},
        'jain': {alg: [] for alg in algorithms},
        'fitness': {alg: [] for alg in algorithms},
    }
    
    # CSV output
    csv_rows = []
    
    for ue_target in ue_targets:
        sinr_matrix = create_sinr_for_ues(base_sinr, num_beams, ue_target, seed=42+ue_target)
        
        print(f"\n{ue_target} UE ({sinr_matrix.shape[0]} beam x {sinr_matrix.shape[1]} UE):")
        results['ue_counts'].append(ue_target)
        
        for alg_name in algorithms:
            if alg_name == 'HGA':
                assignment = hga_algorithm(sinr_matrix, alpha=alpha)
            elif alg_name == 'PBIG':
                assignment = pbig_algorithm(sinr_matrix, alpha=alpha)
            elif alg_name == 'GA':
                assignment = ga_algorithm(sinr_matrix, alpha=alpha)
            else:
                assignment = max_sinr_assignment(sinr_matrix)
            
            rates = compute_rates(sinr_matrix, assignment)
            sum_rate = np.sum(rates)
            jain = jain_fairness(rates)
            fitness = objective_function(rates, alpha)
            throughput_mbps = sum_rate * bw_mhz * efficiency
            
            results['sum_rate'][alg_name].append(sum_rate)
            results['jain'][alg_name].append(jain)
            results['fitness'][alg_name].append(fitness)
            
            csv_rows.append({
                'num_ues': ue_target,
                'algorithm': alg_name,
                'throughput': throughput_mbps,
                'fairness': jain
            })
            
            print(f"  {alg_name:10s}: Throughput={throughput_mbps:8.0f} Mbps, Jain={jain:.4f}")
    
    # Save to CSV
    import csv
    csv_file = 'results_8beam_10to100.csv'
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['num_ues', 'algorithm', 'throughput', 'fairness'])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\n✅ Kaydedildi: {csv_file}")
    
    return results

def create_plots(results, output_dir="plots_8beam_10to100"):
    os.makedirs(output_dir, exist_ok=True)
    
    ue_counts = results['ue_counts']
    bw_mhz = 100
    efficiency = 0.8
    
    styles = {
        'GA': {'color': 'orange', 'marker': 'o', 'linestyle': '-'},
        'HGA': {'color': 'red', 'marker': 's', 'linestyle': '-'},
        'PBIG': {'color': 'blue', 'marker': '^', 'linestyle': '-'},
        'Max-SINR': {'color': 'purple', 'marker': 'D', 'linestyle': '--'}
    }
    
    # 1. Sum-Rate
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
    plt.title('Sistem Toplam Veri Hızı (8 Beam)\n5G-LENA PHY, n78 3.5GHz, 100 MHz', fontsize=14, fontweight='bold')
    plt.legend(loc='upper left', fontsize=11)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xticks(ue_counts)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/sumrate_8beam.png', dpi=150)
    plt.savefig(f'{output_dir}/sumrate_8beam.pdf')
    print(f"Kaydedildi: {output_dir}/sumrate_8beam.png")
    
    # 2. Jain Fairness
    plt.figure(figsize=(12, 8))
    for alg in ['GA', 'HGA', 'PBIG', 'Max-SINR']:
        plt.plot(ue_counts, results['jain'][alg],
                 color=styles[alg]['color'],
                 marker=styles[alg]['marker'],
                 linestyle=styles[alg]['linestyle'],
                 linewidth=2.5, markersize=10, label=alg)
    
    plt.xlabel('Kullanıcı Sayısı', fontsize=14, fontweight='bold')
    plt.ylabel('Jain Adalet İndeksi', fontsize=14, fontweight='bold')
    plt.title('Jain Adalet İndeksi (8 Beam)\n5G-LENA PHY', fontsize=14, fontweight='bold')
    plt.legend(loc='lower left', fontsize=11)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xticks(ue_counts)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/jain_8beam.png', dpi=150)
    plt.savefig(f'{output_dir}/jain_8beam.pdf')
    print(f"Kaydedildi: {output_dir}/jain_8beam.png")
    
    # 3. Amaç Fonksiyonu
    plt.figure(figsize=(12, 8))
    for alg in ['GA', 'HGA', 'PBIG', 'Max-SINR']:
        plt.plot(ue_counts, results['fitness'][alg],
                 color=styles[alg]['color'],
                 marker=styles[alg]['marker'],
                 linestyle=styles[alg]['linestyle'],
                 linewidth=2.5, markersize=10, label=alg)
    
    plt.xlabel('Kullanıcı Sayısı', fontsize=14, fontweight='bold')
    plt.ylabel('Amaç Fonksiyonu Değeri', fontsize=14, fontweight='bold')
    plt.title('Amaç Fonksiyonu: F(X) = α⋅ΣR + (1-α)⋅Jain (α=0.7)\n5G-LENA PHY, 8 Beam', fontsize=14, fontweight='bold')
    plt.legend(loc='upper left', fontsize=11)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xticks(ue_counts)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fitness_8beam.png', dpi=150)
    plt.savefig(f'{output_dir}/fitness_8beam.pdf')
    print(f"Kaydedildi: {output_dir}/fitness_8beam.png")
    
    # Tablo
    print("\n" + "="*80)
    print("SONUÇ TABLOSU (8 Beam, 10-100 UE, Gerçek 5G-LENA)")
    print("="*80)
    print(f"{'UE':>5} | {'GA':>12} | {'HGA':>12} | {'PBIG':>12} | {'Max-SINR':>12}")
    print("-"*65)
    for i, ue in enumerate(ue_counts):
        sr_mbps = {alg: results['sum_rate'][alg][i] * bw_mhz * efficiency for alg in ['GA', 'HGA', 'PBIG', 'Max-SINR']}
        print(f"{ue:>5} | {sr_mbps['GA']:>10.0f} Mbps | {sr_mbps['HGA']:>10.0f} Mbps | {sr_mbps['PBIG']:>10.0f} Mbps | {sr_mbps['Max-SINR']:>10.0f} Mbps")

if __name__ == '__main__':
    os.chdir('/Users/akolukisa/FinalThesis/ric-python')
    results = run_benchmark()
    create_plots(results)
    print("\nTamamlandı!")
