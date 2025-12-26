#!/usr/bin/env python3
"""
Near-RT RIC Server for 5G-LENA Beam/UE Assignment - V2 (Corrected)

DÜZELTMELER:
1. Interferans modeli TÜM algoritmalara entegre edildi
2. Max-SINR'a beam kapasitesi kısıtı eklendi
3. Tutarlı fitness fonksiyonu (tüm algoritmalar aynı hedefi optimize ediyor)
4. Kaynak paylaşımı modeli eklendi

Fitness = alpha * sum_rate + (1-alpha) * fairness * sum_rate
       = sum_rate * (alpha + (1-alpha) * fairness)

Bu formül:
- alpha=1.0 → Pure sum-rate maximization
- alpha=0.0 → Pure fairness (proportional fair)
- alpha=0.5 → Dengeli
"""

import asyncio
import json
import logging
import numpy as np
from typing import Dict, List, Tuple
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# ORTAK YARDIMCI FONKSİYONLAR
# ============================================================================

def compute_rates_with_interference(sinr_matrix: np.ndarray, 
                                     assignment: np.ndarray,
                                     interference_factor: float = 0.5,
                                     resource_sharing: bool = True) -> np.ndarray:
    """
    Interferans ve kaynak paylaşımı ile rate hesapla.
    
    Args:
        sinr_matrix: [beam, ue] SINR matrisi (dB)
        assignment: Her UE'nin beam ataması
        interference_factor: Her ek UE için SINR düşüşü (dB)
        resource_sharing: True ise rate beam'deki UE sayısına bölünür
    
    Returns:
        rates: Her UE'nin rate'i (bps/Hz)
    """
    num_beams, num_ues = sinr_matrix.shape
    
    # Beam yüklerini hesapla
    beam_loads = np.zeros(num_beams, dtype=int)
    for ue_idx, beam_idx in enumerate(assignment):
        if beam_idx >= 0 and beam_idx < num_beams:
            beam_loads[int(beam_idx)] += 1
    
    rates = np.zeros(num_ues)
    for ue_idx in range(num_ues):
        beam_idx = int(assignment[ue_idx])
        if beam_idx < 0 or beam_idx >= num_beams:
            rates[ue_idx] = 0.0
            continue
        
        original_sinr_db = sinr_matrix[beam_idx, ue_idx]
        
        # Interferans: Aynı beam'deki her ek UE için SINR düşer
        num_sharing = beam_loads[beam_idx]
        interference_penalty = interference_factor * (num_sharing - 1)
        effective_sinr_db = original_sinr_db - interference_penalty
        
        # SINR → Rate (Shannon)
        sinr_linear = 10 ** (effective_sinr_db / 10.0)
        if sinr_linear < 0:
            sinr_linear = 0
        base_rate = np.log2(1 + sinr_linear)
        
        # Kaynak paylaşımı
        if resource_sharing and num_sharing > 0:
            rate = base_rate / num_sharing
        else:
            rate = base_rate
        
        rates[ue_idx] = rate
    
    return rates


def compute_fitness(rates: np.ndarray, alpha: float) -> Tuple[float, float, float]:
    """
    Tutarlı fitness fonksiyonu.
    
    Fitness = sum_rate * (alpha + (1-alpha) * fairness)
    
    Bu formül:
    - alpha=1.0 → Fitness = sum_rate
    - alpha=0.0 → Fitness = sum_rate * fairness (proportional fair)
    
    Returns:
        (fitness, sum_rate, fairness)
    """
    sum_rate = float(np.sum(rates))
    num_ues = len(rates)
    
    if sum_rate > 0 and np.sum(rates ** 2) > 0:
        fairness = float((sum_rate ** 2) / (num_ues * np.sum(rates ** 2)))
    else:
        fairness = 0.0
    
    # Tutarlı fitness formülü
    fitness = sum_rate * (alpha + (1.0 - alpha) * fairness)
    
    return fitness, sum_rate, fairness


def is_feasible(assignment: np.ndarray, num_beams: int, max_ues_per_beam: int = None) -> bool:
    """Beam kapasitesi kısıtını kontrol et."""
    if max_ues_per_beam is None:
        return True
    
    valid_assignments = assignment[assignment >= 0]
    if len(valid_assignments) == 0:
        return True
    
    beam_loads = np.bincount(valid_assignments.astype(int), minlength=num_beams)
    return np.all(beam_loads <= max_ues_per_beam)


def repair_assignment(assignment: np.ndarray, sinr_matrix: np.ndarray, 
                      max_ues_per_beam: int) -> np.ndarray:
    """
    Kapasiteyi aşan beam'lerdeki UE'leri yeniden ata.
    SINR-aware: Düşük SINR'lı UE'leri önce taşı.
    """
    if max_ues_per_beam is None:
        return assignment
    
    num_beams, num_ues = sinr_matrix.shape
    repaired = assignment.copy()
    
    beam_loads = np.bincount(repaired[repaired >= 0].astype(int), minlength=num_beams)
    
    for beam_idx in range(num_beams):
        while beam_loads[beam_idx] > max_ues_per_beam:
            # Bu beam'e atanan UE'leri bul
            ue_indices = np.where(repaired == beam_idx)[0]
            if len(ue_indices) == 0:
                break
            
            # SINR'a göre sırala (düşük SINR'lıyı taşı)
            ue_sinrs = [(ue_idx, sinr_matrix[beam_idx, ue_idx]) for ue_idx in ue_indices]
            ue_sinrs.sort(key=lambda x: x[1])
            
            ue_to_move = ue_sinrs[0][0]
            
            # Alternatif beam bul (kapasitesi dolmamış, en iyi SINR)
            ue_sinr_all = sinr_matrix[:, ue_to_move]
            sorted_beams = np.argsort(ue_sinr_all)[::-1]
            
            moved = False
            for alt_beam in sorted_beams:
                alt_beam = int(alt_beam)
                if alt_beam != beam_idx and beam_loads[alt_beam] < max_ues_per_beam:
                    repaired[ue_to_move] = alt_beam
                    beam_loads[beam_idx] -= 1
                    beam_loads[alt_beam] += 1
                    moved = True
                    break
            
            if not moved:
                # Tüm beam'ler dolu, en az yüklü beam'e at
                min_load_beam = int(np.argmin(beam_loads))
                repaired[ue_to_move] = min_load_beam
                beam_loads[beam_idx] -= 1
                beam_loads[min_load_beam] += 1
    
    return repaired


# ============================================================================
# ALGORİTMALAR
# ============================================================================

class BeamAssignmentOptimizer:
    """Base class for beam-UE assignment optimization algorithms"""
    
    def __init__(self, algorithm_name: str, alpha: float = 1.0,
                 max_ues_per_beam: int = None, interference_factor: float = 0.5):
        self.algorithm_name = algorithm_name
        self.alpha = max(0.0, min(1.0, float(alpha)))
        self.max_ues_per_beam = max_ues_per_beam
        self.interference_factor = interference_factor
        logger.info(f"Initialized {algorithm_name} (alpha={self.alpha}, "
                   f"max_ues_per_beam={self.max_ues_per_beam}, "
                   f"interference={self.interference_factor})")
    
    def evaluate(self, sinr_matrix: np.ndarray, assignment: np.ndarray) -> Tuple[float, float, float]:
        """Tutarlı değerlendirme - tüm algoritmalar aynı fonksiyonu kullanır."""
        rates = compute_rates_with_interference(
            sinr_matrix, assignment, self.interference_factor)
        return compute_fitness(rates, self.alpha)
    
    def optimize(self, scenario_data: Dict) -> Dict:
        raise NotImplementedError("Subclasses must implement optimize()")


class MaxSINROptimizer(BeamAssignmentOptimizer):
    """
    Max-SINR with beam capacity constraint.
    
    Greedy yaklaşım:
    1. Her UE için en iyi beam'i bul
    2. Kapasite kısıtını uygula (repair)
    
    Not: Bu artık "pure Max-SINR" değil, capacity-constrained Max-SINR.
    """
    
    def __init__(self, alpha: float = 1.0, max_ues_per_beam: int = None,
                 interference_factor: float = 0.5):
        super().__init__("Max-SINR", alpha, max_ues_per_beam, interference_factor)
    
    def optimize(self, scenario_data: Dict) -> Dict:
        sinr_matrix = np.array(scenario_data['sinr_matrix_dB'])
        num_beams = scenario_data['num_beams']
        num_ues = scenario_data['num_ues']
        
        # Her UE için en iyi beam (greedy)
        assignment = np.array([int(np.argmax(sinr_matrix[:, ue_idx])) 
                               for ue_idx in range(num_ues)])
        
        # Kapasite kısıtı uygula
        if self.max_ues_per_beam is not None:
            assignment = repair_assignment(assignment, sinr_matrix, self.max_ues_per_beam)
        
        # Değerlendir
        fitness, sum_rate, fairness = self.evaluate(sinr_matrix, assignment)
        
        logger.info(f"Max-SINR: fitness={fitness:.2f}, SR={sum_rate:.2f}, Fair={fairness:.4f}")
        return {
            "algorithm": self.algorithm_name,
            "beam_for_ue": assignment.tolist(),
            "objective_value": fitness,
            "sum_rate": sum_rate,
            "fairness": fairness,
            "scenario_id": scenario_data.get('scenario_id', 0)
        }


class GAOptimizer(BeamAssignmentOptimizer):
    """
    Genetic Algorithm with interference-aware fitness.
    
    Thesis Section 4.2 uyumlu:
    - Chromosome: Beam assignments
    - Selection: Roulette Wheel
    - Crossover: Single-point
    - Mutation: Random + repair
    """
    
    def __init__(self, population_size=50, generations=100, mutation_rate=0.1,
                 alpha: float = 1.0, max_ues_per_beam: int = None,
                 interference_factor: float = 0.5):
        super().__init__("GA", alpha, max_ues_per_beam, interference_factor)
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
    
    def optimize(self, scenario_data: Dict) -> Dict:
        sinr_matrix = np.array(scenario_data['sinr_matrix_dB'])
        num_beams = scenario_data['num_beams']
        num_ues = scenario_data['num_ues']
        
        # Başlangıç popülasyonu
        population = []
        for _ in range(self.population_size):
            ind = np.random.randint(0, num_beams, num_ues)
            if self.max_ues_per_beam is not None:
                ind = repair_assignment(ind, sinr_matrix, self.max_ues_per_beam)
            population.append(ind)
        
        best_solution = None
        best_fitness = -np.inf
        
        for gen in range(self.generations):
            # Fitness değerlendirme
            fitness_values = []
            for ind in population:
                fitness, _, _ = self.evaluate(sinr_matrix, ind)
                fitness_values.append(fitness)
            fitness_values = np.array(fitness_values)
            
            # En iyi takip
            gen_best_idx = np.argmax(fitness_values)
            if fitness_values[gen_best_idx] > best_fitness:
                best_fitness = fitness_values[gen_best_idx]
                best_solution = population[gen_best_idx].copy()
            
            # Roulette Wheel Selection
            min_fitness = np.min(fitness_values)
            shifted = fitness_values - min_fitness + 1e-10
            probs = shifted / np.sum(shifted)
            cumulative = np.cumsum(probs)
            
            selected = []
            for _ in range(self.population_size):
                r = np.random.rand()
                idx = np.searchsorted(cumulative, r)
                idx = min(idx, len(population) - 1)
                selected.append(population[idx].copy())
            
            # Single-point Crossover
            offspring = []
            for i in range(0, self.population_size, 2):
                p1 = selected[i]
                p2 = selected[min(i+1, self.population_size-1)]
                cp = np.random.randint(1, num_ues)
                c1 = np.concatenate([p1[:cp], p2[cp:]])
                c2 = np.concatenate([p2[:cp], p1[cp:]])
                offspring.extend([c1, c2])
            
            # Mutation + Repair
            for child in offspring[:self.population_size]:
                if np.random.rand() < self.mutation_rate:
                    mut_idx = np.random.randint(num_ues)
                    child[mut_idx] = np.random.randint(num_beams)
                if self.max_ues_per_beam is not None:
                    child = repair_assignment(child, sinr_matrix, self.max_ues_per_beam)
            
            population = offspring[:self.population_size]
        
        fitness, sum_rate, fairness = self.evaluate(sinr_matrix, best_solution)
        
        logger.info(f"GA: fitness={fitness:.2f}, SR={sum_rate:.2f}, Fair={fairness:.4f}")
        return {
            "algorithm": self.algorithm_name,
            "beam_for_ue": best_solution.tolist(),
            "objective_value": fitness,
            "sum_rate": sum_rate,
            "fairness": fairness,
            "scenario_id": scenario_data.get('scenario_id', 0)
        }


class HGAOptimizer(BeamAssignmentOptimizer):
    """
    Hybrid Genetic Algorithm with memetic local search.
    
    Thesis Section 4.3 uyumlu:
    - GA operators + local search
    - Local search: Her iterasyonda popülasyonun bir kısmına uygulanır
    """
    
    def __init__(self, population_size=50, generations=100, mutation_rate=0.1,
                 ls_rate=0.3, budget=10,
                 alpha: float = 1.0, max_ues_per_beam: int = None,
                 interference_factor: float = 0.5):
        super().__init__("HGA", alpha, max_ues_per_beam, interference_factor)
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.ls_rate = ls_rate
        self.budget = budget
    
    def optimize(self, scenario_data: Dict) -> Dict:
        sinr_matrix = np.array(scenario_data['sinr_matrix_dB'])
        num_beams = scenario_data['num_beams']
        num_ues = scenario_data['num_ues']
        
        def local_search(ind: np.ndarray) -> np.ndarray:
            """Stochastic local search."""
            current = ind.copy()
            current_fitness, _, _ = self.evaluate(sinr_matrix, current)
            
            for _ in range(self.budget):
                neighbor = current.copy()
                ue_idx = np.random.randint(num_ues)
                neighbor[ue_idx] = np.random.randint(num_beams)
                if self.max_ues_per_beam is not None:
                    neighbor = repair_assignment(neighbor, sinr_matrix, self.max_ues_per_beam)
                
                neighbor_fitness, _, _ = self.evaluate(sinr_matrix, neighbor)
                if neighbor_fitness > current_fitness:
                    current = neighbor
                    current_fitness = neighbor_fitness
            
            return current
        
        # Başlangıç popülasyonu
        population = []
        for _ in range(self.population_size):
            ind = np.random.randint(0, num_beams, num_ues)
            if self.max_ues_per_beam is not None:
                ind = repair_assignment(ind, sinr_matrix, self.max_ues_per_beam)
            population.append(ind)
        
        best_solution = None
        best_fitness = -np.inf
        
        for gen in range(self.generations):
            # Local search
            num_ls = max(1, int(self.ls_rate * self.population_size))
            ls_indices = np.random.choice(self.population_size, num_ls, replace=False)
            for idx in ls_indices:
                population[idx] = local_search(population[idx])
            
            # Fitness değerlendirme
            fitness_values = []
            for ind in population:
                fitness, _, _ = self.evaluate(sinr_matrix, ind)
                fitness_values.append(fitness)
            fitness_values = np.array(fitness_values)
            
            # En iyi takip
            gen_best_idx = np.argmax(fitness_values)
            if fitness_values[gen_best_idx] > best_fitness:
                best_fitness = fitness_values[gen_best_idx]
                best_solution = population[gen_best_idx].copy()
            
            # Roulette Wheel Selection
            min_fitness = np.min(fitness_values)
            shifted = fitness_values - min_fitness + 1e-10
            probs = shifted / np.sum(shifted)
            cumulative = np.cumsum(probs)
            
            selected = []
            for _ in range(self.population_size):
                r = np.random.rand()
                idx = np.searchsorted(cumulative, r)
                idx = min(idx, len(population) - 1)
                selected.append(population[idx].copy())
            
            # Crossover
            offspring = []
            for i in range(0, self.population_size, 2):
                p1 = selected[i]
                p2 = selected[min(i+1, self.population_size-1)]
                cp = np.random.randint(1, num_ues)
                c1 = np.concatenate([p1[:cp], p2[cp:]])
                c2 = np.concatenate([p2[:cp], p1[cp:]])
                offspring.extend([c1, c2])
            
            # Mutation + Repair
            for child in offspring[:self.population_size]:
                if np.random.rand() < self.mutation_rate:
                    mut_idx = np.random.randint(num_ues)
                    child[mut_idx] = np.random.randint(num_beams)
                if self.max_ues_per_beam is not None:
                    child = repair_assignment(child, sinr_matrix, self.max_ues_per_beam)
            
            population = offspring[:self.population_size]
        
        fitness, sum_rate, fairness = self.evaluate(sinr_matrix, best_solution)
        
        logger.info(f"HGA: fitness={fitness:.2f}, SR={sum_rate:.2f}, Fair={fairness:.4f}")
        return {
            "algorithm": self.algorithm_name,
            "beam_for_ue": best_solution.tolist(),
            "objective_value": fitness,
            "sum_rate": sum_rate,
            "fairness": fairness,
            "scenario_id": scenario_data.get('scenario_id', 0)
        }


class PBIGOptimizer(BeamAssignmentOptimizer):
    """
    Population-Based Iterated Greedy with interference-aware fitness.
    
    Thesis Section 4.4 uyumlu:
    - Destruction: Rastgele UE'lerin atamasını kaldır
    - Reconstruction: Greedy olarak yeniden ata
    """
    
    def __init__(self, population_size=10, max_iter=100, destruction_ratio=0.3,
                 alpha: float = 1.0, max_ues_per_beam: int = None,
                 interference_factor: float = 0.5):
        super().__init__("PBIG", alpha, max_ues_per_beam, interference_factor)
        self.population_size = population_size
        self.max_iter = max_iter
        self.destruction_ratio = destruction_ratio
    
    def optimize(self, scenario_data: Dict) -> Dict:
        sinr_matrix = np.array(scenario_data['sinr_matrix_dB'])
        num_beams = scenario_data['num_beams']
        num_ues = scenario_data['num_ues']
        
        def greedy_assign(ue_idx: int, partial: np.ndarray) -> int:
            """Greedy assignment for one UE."""
            best_beam = 0
            best_fitness = -np.inf
            
            for beam_idx in range(num_beams):
                candidate = partial.copy()
                candidate[ue_idx] = beam_idx
                if self.max_ues_per_beam is not None:
                    candidate = repair_assignment(candidate, sinr_matrix, self.max_ues_per_beam)
                fitness, _, _ = self.evaluate(sinr_matrix, candidate)
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_beam = beam_idx
            
            return best_beam
        
        # Başlangıç popülasyonu
        population = []
        for _ in range(self.population_size):
            ind = np.random.randint(0, num_beams, num_ues)
            if self.max_ues_per_beam is not None:
                ind = repair_assignment(ind, sinr_matrix, self.max_ues_per_beam)
            population.append(ind)
        
        best_solution = None
        best_fitness = -np.inf
        
        for iteration in range(self.max_iter):
            # Rastgele seç
            idx = np.random.randint(self.population_size)
            S = population[idx].copy()
            
            # Destruction
            num_to_destroy = max(1, int(self.destruction_ratio * num_ues))
            destroyed_users = np.random.choice(num_ues, num_to_destroy, replace=False)
            for u in destroyed_users:
                S[u] = -1  # Unassigned
            
            # Reconstruction (greedy)
            for u in destroyed_users:
                S[u] = greedy_assign(u, S)
            
            # Repair
            if self.max_ues_per_beam is not None:
                S = repair_assignment(S, sinr_matrix, self.max_ues_per_beam)
            
            # Update population
            S_fitness, _, _ = self.evaluate(sinr_matrix, S)
            old_fitness, _, _ = self.evaluate(sinr_matrix, population[idx])
            if S_fitness > old_fitness:
                population[idx] = S
            
            # Track best
            for ind in population:
                ind_fitness, _, _ = self.evaluate(sinr_matrix, ind)
                if ind_fitness > best_fitness:
                    best_fitness = ind_fitness
                    best_solution = ind.copy()
        
        fitness, sum_rate, fairness = self.evaluate(sinr_matrix, best_solution)
        
        logger.info(f"PBIG: fitness={fitness:.2f}, SR={sum_rate:.2f}, Fair={fairness:.4f}")
        return {
            "algorithm": self.algorithm_name,
            "beam_for_ue": best_solution.tolist(),
            "objective_value": fitness,
            "sum_rate": sum_rate,
            "fairness": fairness,
            "scenario_id": scenario_data.get('scenario_id', 0)
        }


# ============================================================================
# RIC SERVER
# ============================================================================

class RICServer:
    """Near-RT RIC TCP Server for ns-3 integration"""
    
    def __init__(self, host='127.0.0.1', port=5555, algorithm='GA',
                 alpha: float = 1.0, max_ues_per_beam: int = None,
                 interference_factor: float = 0.5):
        self.host = host
        self.port = port
        self.algorithm = algorithm
        self.alpha = float(alpha)
        self.max_ues_per_beam = max_ues_per_beam
        self.interference_factor = interference_factor
        
        # Initialize optimizer
        common_params = {
            'alpha': self.alpha,
            'max_ues_per_beam': self.max_ues_per_beam,
            'interference_factor': self.interference_factor
        }
        
        if algorithm == 'Max-SINR':
            self.optimizer = MaxSINROptimizer(**common_params)
        elif algorithm == 'GA':
            self.optimizer = GAOptimizer(**common_params)
        elif algorithm == 'HGA':
            self.optimizer = HGAOptimizer(**common_params)
        elif algorithm == 'PBIG':
            self.optimizer = PBIGOptimizer(**common_params)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        logger.info(f"RIC Server initialized: {algorithm}, alpha={alpha}, "
                   f"max_ues_per_beam={max_ues_per_beam}, interference={interference_factor}")
    
    def _prepare_scenario(self, request: Dict) -> Tuple[Dict, List[Dict]]:
        """Prepare internal scenario for multi-gNB handling."""
        if 'num_gnbs' not in request:
            return request, []
        
        num_gnbs = request['num_gnbs']
        num_beams_per_gnb = request['num_beams_per_gnb']
        num_ues = request['num_ues']
        
        # Flatten gNB×beam into single resource index
        total_resources = num_gnbs * num_beams_per_gnb
        
        resource_map = []
        for gnb_id in range(num_gnbs):
            for beam_id in range(num_beams_per_gnb):
                resource_map.append({'gnb_id': gnb_id, 'beam_id': beam_id})
        
        # Flatten SINR matrix
        sinr_3d = np.array(request['sinr_matrix_3d_dB'])
        sinr_flat = sinr_3d.reshape(total_resources, num_ues)
        
        internal_request = {
            'num_beams': total_resources,
            'num_ues': num_ues,
            'sinr_matrix_dB': sinr_flat.tolist(),
            'scenario_id': request.get('scenario_id', 0)
        }
        
        return internal_request, resource_map
    
    def _postprocess_response(self, original_request: Dict, 
                               internal_response: Dict,
                               resource_map: List[Dict]) -> Dict:
        """Convert resource indices back to (gNB, beam) pairs."""
        if 'num_gnbs' not in original_request:
            return internal_response
        
        resource_indices = internal_response.get('beam_for_ue', [])
        gnb_for_ue = []
        beam_for_ue = []
        
        for res_idx in resource_indices:
            mapping = resource_map[int(res_idx)]
            gnb_for_ue.append(int(mapping['gnb_id']))
            beam_for_ue.append(int(mapping['beam_id']))
        
        return {
            'algorithm': internal_response.get('algorithm', self.algorithm),
            'gnb_for_ue': gnb_for_ue,
            'beam_for_ue': beam_for_ue,
            'objective_value': internal_response.get('objective_value'),
            'sum_rate': internal_response.get('sum_rate'),
            'fairness': internal_response.get('fairness'),
            'scenario_id': original_request.get('scenario_id', 0),
        }
    
    async def handle_client(self, reader, writer):
        """Handle incoming client connection."""
        addr = writer.get_extra_info('peername')
        logger.info(f"Connection from {addr}")
        
        try:
            data = await reader.read(65536)
            if not data:
                return
            
            message = data.decode('utf-8')
            request = json.loads(message)
            
            logger.info(f"Processing: {request.get('num_beams', 'N/A')} beams, "
                       f"{request.get('num_ues')} UEs")
            
            internal_request, resource_map = self._prepare_scenario(request)
            internal_response = self.optimizer.optimize(internal_request)
            response = self._postprocess_response(request, internal_response, resource_map)
            
            response_json = json.dumps(response)
            writer.write(response_json.encode())
            await writer.drain()
            
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            error_response = {"error": str(e)}
            writer.write(json.dumps(error_response).encode())
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()
    
    async def start(self):
        """Start the RIC server."""
        server = await asyncio.start_server(
            self.handle_client, self.host, self.port)
        
        addr = server.sockets[0].getsockname()
        logger.info(f"RIC Server listening on {addr[0]}:{addr[1]}")
        logger.info(f"Algorithm: {self.algorithm}")
        
        async with server:
            await server.serve_forever()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Near-RT RIC Server V2 (Corrected)')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=5555)
    parser.add_argument('--algorithm', 
                       choices=['Max-SINR', 'GA', 'HGA', 'PBIG'],
                       default='GA')
    parser.add_argument('--alpha', type=float, default=1.0,
                       help='Trade-off: 1.0=sum-rate, 0.0=fairness')
    parser.add_argument('--max-ues-per-beam', type=int, default=None,
                       help='Beam capacity constraint')
    parser.add_argument('--interference', type=float, default=0.5,
                       help='Interference factor (dB per additional UE)')
    parser.add_argument('--verbose', action='store_true')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    server = RICServer(
        host=args.host,
        port=args.port,
        algorithm=args.algorithm,
        alpha=args.alpha,
        max_ues_per_beam=args.max_ues_per_beam,
        interference_factor=args.interference
    )
    
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
