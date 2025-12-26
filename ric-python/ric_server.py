#!/usr/bin/env python3
"""
Near-RT RIC Server for 5G-LENA Beam/UE Assignment
Implements GA, HGA, and PBIG algorithms for beam-user assignment optimization
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


class BeamAssignmentOptimizer:
    """Base class for beam-UE assignment optimization algorithms"""
    
    def __init__(self, algorithm_name: str):
        self.algorithm_name = algorithm_name
        logger.info(f"Initialized {algorithm_name} optimizer")
    
    def optimize(self, scenario_data: Dict) -> Dict:
        """
        Optimize beam-UE assignment based on scenario data
        
        Args:
            scenario_data: Dictionary containing:
                - num_beams: Number of available beams
                - num_ues: Number of UEs
                - sinr_matrix_dB: SINR matrix [beam][ue] in dB
                - scenario_id: Scenario identifier
                
        Returns:
            Dictionary with beam assignment for each UE
        """
        raise NotImplementedError("Subclasses must implement optimize()")


class MaxSINROptimizer(BeamAssignmentOptimizer):
    """Baseline: Max-SINR (Upper Bound) - assign best beam to each UE"""
    
    def __init__(self):
        super().__init__("Max-SINR")
    
    def optimize(self, scenario_data: Dict) -> Dict:
        sinr_matrix = np.array(scenario_data['sinr_matrix_dB'])  # [beam][ue]
        num_ues = scenario_data['num_ues']
        
        # For each UE, find the beam with maximum SINR
        beam_assignment = []
        for ue_idx in range(num_ues):
            ue_sinrs = sinr_matrix[:, ue_idx]  # All beams' SINR for this UE
            best_beam = int(np.argmax(ue_sinrs))
            beam_assignment.append(best_beam)
        
        # Calculate sum-rate (objective_value) for the assignment
        sum_rate = 0.0
        for ue_idx, beam_idx in enumerate(beam_assignment):
            sinr_db = sinr_matrix[beam_idx, ue_idx]
            sinr_linear = 10 ** (sinr_db / 10.0)
            rate = np.log2(1 + sinr_linear)  # Shannon capacity (bps/Hz)
            sum_rate += rate
        
        logger.info(f"Max-SINR assignment: {beam_assignment}, sum-rate: {sum_rate:.2f}")
        return {
            "algorithm": self.algorithm_name,
            "beam_for_ue": beam_assignment,
            "objective_value": sum_rate,
            "scenario_id": scenario_data.get('scenario_id', 0)
        }


class ExhaustiveSearchOptimizer(BeamAssignmentOptimizer):
    """Baseline: Exhaustive Search - try all combinations (limited scenarios only)"""
    
    def __init__(self):
        super().__init__("Exhaustive-Search")
    
    def optimize(self, scenario_data: Dict) -> Dict:
        sinr_matrix = np.array(scenario_data['sinr_matrix_dB'])  # [beam][ue]
        num_beams = scenario_data['num_beams']
        num_ues = scenario_data['num_ues']
        
        # Warning: Exhaustive search is O(num_beams^num_ues)
        if num_beams ** num_ues > 1e6:
            logger.warning(f"Exhaustive search space too large ({num_beams}^{num_ues}), falling back to Max-SINR")
            return MaxSINROptimizer().optimize(scenario_data)
        
        # Try all possible beam assignments
        best_assignment = None
        best_objective = -np.inf
        
        # Generate all possible assignments using recursion
        def evaluate_assignment(assignment: List[int]) -> float:
            # Sum-rate objective: sum of SINR (linear scale)
            total_rate = 0.0
            for ue_idx, beam_idx in enumerate(assignment):
                sinr_db = sinr_matrix[beam_idx, ue_idx]
                sinr_linear = 10 ** (sinr_db / 10.0)
                # Shannon capacity (simplified)
                rate = np.log2(1 + sinr_linear)
                total_rate += rate
            return total_rate
        
        from itertools import product
        for assignment in product(range(num_beams), repeat=num_ues):
            objective = evaluate_assignment(list(assignment))
            if objective > best_objective:
                best_objective = objective
                best_assignment = list(assignment)
        
        logger.info(f"Exhaustive Search best assignment: {best_assignment}, objective: {best_objective:.2f}")
        return {
            "algorithm": self.algorithm_name,
            "beam_for_ue": best_assignment,
            "objective_value": best_objective,
            "scenario_id": scenario_data.get('scenario_id', 0)
        }


class GAOptimizer(BeamAssignmentOptimizer):
    """Genetic Algorithm for beam-UE assignment (Thesis-compliant version)
    
    Implements GA as described in thesis Section 4.2:
    - Chromosome: List of beam assignments for each UE
    - Selection: Roulette Wheel (cumulative fitness-proportional)
    - Crossover: Single-point crossover
    - Mutation: Random beam assignment + repair (if needed)
    - Fitness: Weighted sum-rate + Jain fairness with penalty for constraint violations
    """
    
    def __init__(self, population_size=50, generations=100, mutation_rate=0.1,
                 constraint_penalty=0.5, max_ues_per_beam=None,
                 alpha: float = 1.0):
        super().__init__("GA")
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.constraint_penalty = constraint_penalty
        self.max_ues_per_beam = max_ues_per_beam  # None = no constraint
        # Alpha: trade-off between sum-rate and Jain fairness (0-1)
        self.alpha = float(alpha)
        if self.alpha < 0.0:
            self.alpha = 0.0
        if self.alpha > 1.0:
            self.alpha = 1.0
    
    def optimize(self, scenario_data: Dict) -> Dict:
        sinr_matrix = np.array(scenario_data['sinr_matrix_dB'])
        num_beams = scenario_data['num_beams']
        num_ues = scenario_data['num_ues']
        
        # Compute rates per UE
        def compute_rates(chromosome: np.ndarray) -> np.ndarray:
            rates = []
            for ue_idx, beam_idx in enumerate(chromosome):
                sinr_db = sinr_matrix[int(beam_idx), ue_idx]
                sinr_linear = 10 ** (sinr_db / 10.0)
                rate = np.log2(1 + sinr_linear)
                rates.append(rate)
            return np.array(rates, dtype=float)
        
        def compute_sumrate_and_jain(chromosome: np.ndarray) -> Tuple[float, float]:
            rates = compute_rates(chromosome)
            sum_rate = float(np.sum(rates))
            if np.all(rates == 0):
                jain = 0.0
            else:
                jain = float((np.sum(rates) ** 2) / (len(rates) * np.sum(rates ** 2)))
            return sum_rate, jain
        
        def is_feasible(chromosome: np.ndarray) -> bool:
            """Check if solution satisfies beam capacity constraints (if enabled)."""
            if self.max_ues_per_beam is None:
                return True  # No constraints
            beam_loads = np.bincount(chromosome.astype(int), minlength=num_beams)
            return np.all(beam_loads <= self.max_ues_per_beam)
        
        def repair_assignment(chromosome: np.ndarray) -> np.ndarray:
            """Repair infeasible solution by redistributing UEs from overloaded beams."""
            if is_feasible(chromosome):
                return chromosome
            repaired = chromosome.copy()
            beam_loads = np.bincount(repaired.astype(int), minlength=num_beams)
            overloaded = np.where(beam_loads > self.max_ues_per_beam)[0]
            for beam_idx in overloaded:
                ue_indices = np.where(repaired == beam_idx)[0]
                excess = int(beam_loads[beam_idx] - self.max_ues_per_beam)
                for ue_idx in ue_indices[:excess]:
                    # Find beam with lowest load
                    alternative_beam = int(np.argmin(beam_loads))
                    repaired[ue_idx] = alternative_beam
                    beam_loads[beam_idx] -= 1
                    beam_loads[alternative_beam] += 1
            return repaired
        
        def fitness(chromosome: np.ndarray) -> float:
            """Fitness: alpha*SumRate + (1-alpha)*Jain, with capacity penalty."""
            sum_rate, jain = compute_sumrate_and_jain(chromosome)
            core = self.alpha * sum_rate + (1.0 - self.alpha) * jain
            if not is_feasible(chromosome):
                return core * self.constraint_penalty
            return core
        
        # Initialize population (thesis: InitializePopulation)
        population = np.random.randint(0, num_beams, (self.population_size, num_ues))
        
        best_chromosome = None
        best_fitness = -np.inf
        
        for gen in range(self.generations):
            # Evaluate fitness (thesis: Evaluate(pop))
            fitness_values = np.array([fitness(ind) for ind in population])
            
            # Track best
            gen_best_idx = np.argmax(fitness_values)
            if fitness_values[gen_best_idx] > best_fitness:
                best_fitness = fitness_values[gen_best_idx]
                best_chromosome = population[gen_best_idx].copy()
            
            # Selection: Roulette Wheel (thesis requirement)
            min_fitness = np.min(fitness_values)
            if min_fitness < 0:
                shifted_fitness = fitness_values - min_fitness + 1e-10
            else:
                shifted_fitness = fitness_values + 1e-10
            fitness_sum = np.sum(shifted_fitness)
            probabilities = shifted_fitness / fitness_sum
            cumulative_prob = np.cumsum(probabilities)
            
            selected = []
            for _ in range(self.population_size):
                r = np.random.rand()
                idx = np.searchsorted(cumulative_prob, r)
                selected.append(population[idx].copy())
            
            # Crossover: Single-point (thesis: Tek Nokta Crossover)
            offspring = []
            for i in range(0, self.population_size, 2):
                parent1 = selected[i]
                parent2 = selected[i+1] if i+1 < self.population_size else selected[0]
                crossover_point = np.random.randint(1, num_ues)
                child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
                child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
                offspring.extend([child1, child2])
            
            # Mutation with repair
            for child in offspring:
                if np.random.rand() < self.mutation_rate:
                    mutate_idx = np.random.randint(num_ues)
                    child[mutate_idx] = np.random.randint(num_beams)
                    if self.max_ues_per_beam is not None:
                        child = repair_assignment(child)
            
            population = np.array(offspring[:self.population_size])
        
        logger.info(f"GA best assignment: {best_chromosome.tolist()}, fitness: {best_fitness:.2f}")
        return {
            "algorithm": self.algorithm_name,
            "beam_for_ue": best_chromosome.tolist(),
            "objective_value": float(best_fitness),
            "scenario_id": scenario_data.get('scenario_id', 0)
        }


class HGAOptimizer(BeamAssignmentOptimizer):
    """Hybrid Genetic Algorithm with local search (Thesis-compliant version)
    
    Implements HGA as described in thesis Section 4.3:
    - Inherits GA operators
    - Adds memetic local search: randomly select ls_rate × pop_size individuals
    - Local search: randomly change one user's assignment + repair
    - Budget: number of local search iterations per individual
    """
    
    def __init__(self, population_size=50, generations=100, mutation_rate=0.1,
                 ls_rate=0.3, budget=10, constraint_penalty=0.5, max_ues_per_beam=None,
                 alpha: float = 1.0):
        super().__init__("HGA")
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.ls_rate = ls_rate  # Fraction of population for local search
        self.budget = budget  # Local search iterations per individual
        self.constraint_penalty = constraint_penalty
        self.max_ues_per_beam = max_ues_per_beam
        self.alpha = float(alpha)
        if self.alpha < 0.0:
            self.alpha = 0.0
        if self.alpha > 1.0:
            self.alpha = 1.0
    
    def optimize(self, scenario_data: Dict) -> Dict:
        sinr_matrix = np.array(scenario_data['sinr_matrix_dB'])
        num_beams = scenario_data['num_beams']
        num_ues = scenario_data['num_ues']
        
        def compute_rates(chromosome: np.ndarray) -> np.ndarray:
            rates = []
            for ue_idx, beam_idx in enumerate(chromosome):
                sinr_db = sinr_matrix[int(beam_idx), ue_idx]
                sinr_linear = 10 ** (sinr_db / 10.0)
                rate = np.log2(1 + sinr_linear)
                rates.append(rate)
            return np.array(rates, dtype=float)
        
        def compute_sumrate_and_jain(chromosome: np.ndarray) -> Tuple[float, float]:
            rates = compute_rates(chromosome)
            sum_rate = float(np.sum(rates))
            if np.all(rates == 0):
                jain = 0.0
            else:
                jain = float((np.sum(rates) ** 2) / (len(rates) * np.sum(rates ** 2)))
            return sum_rate, jain
        
        def is_feasible(chromosome: np.ndarray) -> bool:
            if self.max_ues_per_beam is None:
                return True
            beam_loads = np.bincount(chromosome.astype(int), minlength=num_beams)
            return np.all(beam_loads <= self.max_ues_per_beam)
        
        def repair_assignment(chromosome: np.ndarray) -> np.ndarray:
            if is_feasible(chromosome):
                return chromosome
            repaired = chromosome.copy()
            beam_loads = np.bincount(repaired.astype(int), minlength=num_beams)
            overloaded = np.where(beam_loads > self.max_ues_per_beam)[0]
            for beam_idx in overloaded:
                ue_indices = np.where(repaired == beam_idx)[0]
                excess = int(beam_loads[beam_idx] - self.max_ues_per_beam)
                for ue_idx in ue_indices[:excess]:
                    alternative_beam = int(np.argmin(beam_loads))
                    repaired[ue_idx] = alternative_beam
                    beam_loads[beam_idx] -= 1
                    beam_loads[alternative_beam] += 1
            return repaired
        
        def fitness(chromosome: np.ndarray) -> float:
            sum_rate, jain = compute_sumrate_and_jain(chromosome)
            core = self.alpha * sum_rate + (1.0 - self.alpha) * jain
            if not is_feasible(chromosome):
                return core * self.constraint_penalty
            return core
        
        def local_search(individual: np.ndarray) -> np.ndarray:
            """Stochastic local search (thesis: LocalSearch with budget)
            
            Randomly change one user's assignment and apply repair.
            Repeat 'budget' times, accept if improvement.
            """
            current = individual.copy()
            current_fitness = fitness(current)
            
            for _ in range(self.budget):
                # Randomly select a user
                ue_idx = np.random.randint(num_ues)
                new_beam = np.random.randint(num_beams)
                neighbor = current.copy()
                neighbor[ue_idx] = new_beam
                if self.max_ues_per_beam is not None:
                    neighbor = repair_assignment(neighbor)
                neighbor_fitness = fitness(neighbor)
                if neighbor_fitness > current_fitness:
                    current = neighbor
                    current_fitness = neighbor_fitness
            return current
        
        # Initialize population
        population = np.random.randint(0, num_beams, (self.population_size, num_ues))
        
        best_chromosome = None
        best_fitness = -np.inf
        
        for gen in range(self.generations):
            # Apply GA operators
            fitness_values = np.array([fitness(ind) for ind in population])
            
            gen_best_idx = np.argmax(fitness_values)
            if fitness_values[gen_best_idx] > best_fitness:
                best_fitness = fitness_values[gen_best_idx]
                best_chromosome = population[gen_best_idx].copy()
            
            # Roulette Wheel Selection
            min_fitness = np.min(fitness_values)
            shifted_fitness = fitness_values - min_fitness + 1e-10 if min_fitness < 0 else fitness_values + 1e-10
            probabilities = shifted_fitness / np.sum(shifted_fitness)
            cumulative_prob = np.cumsum(probabilities)
            
            selected = []
            for _ in range(self.population_size):
                r = np.random.rand()
                idx = np.searchsorted(cumulative_prob, r)
                selected.append(population[idx].copy())
            
            # Crossover
            offspring = []
            for i in range(0, self.population_size, 2):
                parent1 = selected[i]
                parent2 = selected[i+1] if i+1 < self.population_size else selected[0]
                crossover_point = np.random.randint(1, num_ues)
                child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
                child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
                offspring.extend([child1, child2])
            
            # Mutation
            for child in offspring:
                if np.random.rand() < self.mutation_rate:
                    mutate_idx = np.random.randint(num_ues)
                    child[mutate_idx] = np.random.randint(num_beams)
                    if self.max_ues_per_beam is not None:
                        child = repair_assignment(child)
            
            population = np.array(offspring[:self.population_size])
            
            # Memetic step: Local search
            k = max(1, int(self.ls_rate * self.population_size))
            subset_indices = np.random.choice(self.population_size, k, replace=False)
            for idx in subset_indices:
                population[idx] = local_search(population[idx])
        
        logger.info(f"HGA best assignment: {best_chromosome.tolist()}, fitness: {best_fitness:.2f}")
        return {
            "algorithm": self.algorithm_name,
            "beam_for_ue": best_chromosome.tolist(),
            "objective_value": float(best_fitness),
            "scenario_id": scenario_data.get('scenario_id', 0)
        }


class PBIGOptimizer(BeamAssignmentOptimizer):
    """Population-Based Iterated Greedy algorithm (Thesis-compliant version)
    
    Implements PBIG as described in thesis Section 4.4:
    - Population of solutions (not single solution)
    - Destruction: Remove d_ratio of users from assignment
    - Reconstruction: Greedy reassignment of removed users
    - Iterate max_iter times
    """
    
    def __init__(self, population_size=20, max_iter=100, destruction_ratio=0.3,
                 constraint_penalty=0.5, max_ues_per_beam=None,
                 alpha: float = 1.0):
        super().__init__("PBIG")
        self.population_size = population_size
        self.max_iter = max_iter
        self.destruction_ratio = destruction_ratio  # d_ratio in thesis
        self.constraint_penalty = constraint_penalty
        self.max_ues_per_beam = max_ues_per_beam
        self.alpha = float(alpha)
        if self.alpha < 0.0:
            self.alpha = 0.0
        if self.alpha > 1.0:
            self.alpha = 1.0
    
    def optimize(self, scenario_data: Dict) -> Dict:
        sinr_matrix = np.array(scenario_data['sinr_matrix_dB'])
        num_beams = scenario_data['num_beams']
        num_ues = scenario_data['num_ues']
        
        def compute_rates(chromosome: np.ndarray) -> np.ndarray:
            rates = []
            for ue_idx, beam_idx in enumerate(chromosome):
                if beam_idx < 0:
                    rates.append(0.0)
                    continue
                sinr_db = sinr_matrix[int(beam_idx), ue_idx]
                sinr_linear = 10 ** (sinr_db / 10.0)
                rate = np.log2(1 + sinr_linear)
                rates.append(rate)
            return np.array(rates, dtype=float)
        
        def compute_sumrate_and_jain(chromosome: np.ndarray) -> Tuple[float, float]:
            rates = compute_rates(chromosome)
            sum_rate = float(np.sum(rates))
            if np.all(rates == 0):
                jain = 0.0
            else:
                jain = float((np.sum(rates) ** 2) / (len(rates) * np.sum(rates ** 2)))
            return sum_rate, jain
        
        def is_feasible(chromosome: np.ndarray) -> bool:
            if self.max_ues_per_beam is None:
                return True
            assigned = chromosome[chromosome >= 0]
            if len(assigned) == 0:
                return True
            beam_loads = np.bincount(assigned.astype(int), minlength=num_beams)
            return np.all(beam_loads <= self.max_ues_per_beam)
        
        def repair_assignment(chromosome: np.ndarray) -> np.ndarray:
            if is_feasible(chromosome):
                return chromosome
            repaired = chromosome.copy()
            assigned = repaired[repaired >= 0]
            beam_loads = np.bincount(assigned.astype(int), minlength=num_beams)
            overloaded = np.where(beam_loads > self.max_ues_per_beam)[0]
            for beam_idx in overloaded:
                ue_indices = np.where(repaired == beam_idx)[0]
                excess = int(beam_loads[beam_idx] - self.max_ues_per_beam)
                for ue_idx in ue_indices[:excess]:
                    alternative_beam = int(np.argmin(beam_loads))
                    repaired[ue_idx] = alternative_beam
                    beam_loads[beam_idx] -= 1
                    beam_loads[alternative_beam] += 1
            return repaired
        
        def fitness(chromosome: np.ndarray) -> float:
            sum_rate, jain = compute_sumrate_and_jain(chromosome)
            core = self.alpha * sum_rate + (1.0 - self.alpha) * jain
            if not is_feasible(chromosome):
                return core * self.constraint_penalty
            return core
        
        def greedy_assign(ue_idx: int, partial_solution: np.ndarray) -> int:
            """Greedy assignment for one user (thesis: GreedyAssign)
            
            Try all beams, select the one that maximizes objective.
            """
            best_beam = -1
            best_improvement = -np.inf
            
            for beam_idx in range(num_beams):
                candidate = partial_solution.copy()
                candidate[ue_idx] = beam_idx
                if self.max_ues_per_beam is not None:
                    candidate = repair_assignment(candidate)
                improvement = fitness(candidate)
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_beam = beam_idx
            return best_beam
        
        # Initialize population (thesis: InitializePopulation)
        population = []
        for _ in range(self.population_size):
            individual = np.random.randint(0, num_beams, num_ues)
            if self.max_ues_per_beam is not None:
                individual = repair_assignment(individual)
            population.append(individual)
        
        best_solution = None
        best_fitness = -np.inf
        
        for iteration in range(self.max_iter):
            # Select random solution from population (thesis: SelectRandom)
            idx = np.random.randint(self.population_size)
            S = population[idx].copy()
            
            # Destruction phase (thesis: Yıkım Fazı)
            num_to_destroy = max(1, int(self.destruction_ratio * num_ues))
            destroyed_users = np.random.choice(num_ues, num_to_destroy, replace=False)
            D = destroyed_users
            for u in D:
                S[u] = -1
            
            # Reconstruction phase (thesis: Yeniden İnşa Fazı)
            for u in D:
                best_beam = greedy_assign(u, S)
                S[u] = best_beam
            
            # Update population if improved
            if fitness(S) > fitness(population[idx]):
                population[idx] = S
            
            # Track global best
            for individual in population:
                ind_fitness = fitness(individual)
                if ind_fitness > best_fitness:
                    best_fitness = ind_fitness
                    best_solution = individual.copy()
        
        logger.info(f"PBIG assignment: {best_solution.tolist()}, objective: {best_fitness:.2f}")
        return {
            "algorithm": self.algorithm_name,
            "beam_for_ue": best_solution.tolist(),
            "objective_value": float(best_fitness),
            "scenario_id": scenario_data.get('scenario_id', 0)
        }


class RICServer:
    """Near-RT RIC TCP Server for ns-3 integration"""
    
    def __init__(self, host='127.0.0.1', port=5555, algorithm='GA',
                 alpha: float = 1.0,
                 max_ues_per_beam: int = None):
        self.host = host
        self.port = port
        self.algorithm = algorithm
        self.alpha = float(alpha)
        self.max_ues_per_beam = max_ues_per_beam
        
        # Initialize optimizer based on algorithm choice
        if algorithm == 'Max-SINR':
            self.optimizer = MaxSINROptimizer()
        elif algorithm == 'Exhaustive':
            self.optimizer = ExhaustiveSearchOptimizer()
        elif algorithm == 'GA':
            self.optimizer = GAOptimizer(max_ues_per_beam=self.max_ues_per_beam,
                                         alpha=self.alpha)
        elif algorithm == 'HGA':
            self.optimizer = HGAOptimizer(max_ues_per_beam=self.max_ues_per_beam,
                                          alpha=self.alpha)
        elif algorithm == 'PBIG':
            self.optimizer = PBIGOptimizer(max_ues_per_beam=self.max_ues_per_beam,
                                           alpha=self.alpha)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        logger.info(f"RIC Server initialized with {algorithm} algorithm (alpha={self.alpha}, max_ues_per_beam={self.max_ues_per_beam})")
    
    async def handle_client(self, reader, writer):
        """Handle incoming client connection from ns-3"""
        addr = writer.get_extra_info('peername')
        logger.info(f"Connection from {addr}")
        
        try:
            # Read request from ns-3
            data = await reader.read(65536)  # 64KB buffer
            if not data:
                logger.warning("Empty request received")
                return
            
            message = data.decode('utf-8')
            logger.debug(f"Received: {message[:200]}...")
            
            # Parse JSON request
            request = json.loads(message)
            logger.info(f"Processing scenario {request.get('scenario_id', 'unknown')} "
                       f"with {request.get('num_beams', 'N/A')} beams and {request.get('num_ues')} UEs")
            
            # Prepare internal scenario (single- or multi-gNB)
            internal_request, resource_map = self._prepare_scenario(request)
            
            # Optimize beam assignment (over resources)
            internal_response = self.optimizer.optimize(internal_request)
            
            # Post-process to map resources back to (gNB, beam)
            response = self._postprocess_response(request, internal_response, resource_map)
            
            # Send response back to ns-3
            response_json = json.dumps(response)
            writer.write(response_json.encode('utf-8'))
            await writer.drain()
            
            logger.info(f"Sent response with {len(response['beam_for_ue'])} assignments")
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
        except Exception as e:
            logger.error(f"Error handling client: {e}", exc_info=True)
        finally:
            writer.close()
            await writer.wait_closed()
    
    def _prepare_scenario(self, request: Dict) -> Tuple[Dict, List[Dict]]:
        """Convert incoming request to internal single-resource-index scenario.
        
        Supports both legacy single-gNB mode and multi-gNB mode.
        Returns (internal_request, resource_map) where resource_map is a list of
        dicts with keys {'gnb_id', 'beam_id'} for each resource index.
        """
        # Multi-gNB mode: expect sinr_matrix_dB as [gNB][beam][ue]
        if 'num_gnbs' in request:
            sinr = np.array(request['sinr_matrix_dB'])
            if sinr.ndim != 3:
                raise ValueError("For multi-gNB mode, sinr_matrix_dB must be 3D [gNB][beam][ue]")
            num_gnbs, num_beams_per_gnb, num_ues = sinr.shape
            resource_map: List[Dict] = []
            flat_rows = []
            for gnb_id in range(num_gnbs):
                for beam_id in range(num_beams_per_gnb):
                    resource_map.append({'gnb_id': int(gnb_id), 'beam_id': int(beam_id)})
                    flat_rows.append(sinr[gnb_id, beam_id, :])
            internal_request = {
                'scenario_id': request.get('scenario_id', 0),
                'num_beams': len(resource_map),  # now "beams" = resources
                'num_ues': int(num_ues),
                'sinr_matrix_dB': np.array(flat_rows).tolist(),
            }
            return internal_request, resource_map
        
        # Legacy single-gNB mode: pass through, but build a simple resource map
        num_beams = int(request.get('num_beams', 0))
        gnb_id = int(request.get('gNB_id', 0))
        resource_map = [{'gnb_id': gnb_id, 'beam_id': b} for b in range(num_beams)]
        return request, resource_map
    
    def _postprocess_response(self, original_request: Dict,
                              internal_response: Dict,
                              resource_map: List[Dict]) -> Dict:
        """Map optimizer response (over resources) back to (gNB, beam) per UE."""
        # Multi-gNB request: build gnb_for_ue + beam_for_ue using resource_map
        if 'num_gnbs' in original_request:
            resource_indices = internal_response.get('beam_for_ue', [])
            gnb_for_ue: List[int] = []
            beam_for_ue: List[int] = []
            for res_idx in resource_indices:
                mapping = resource_map[int(res_idx)]
                gnb_for_ue.append(int(mapping['gnb_id']))
                beam_for_ue.append(int(mapping['beam_id']))
            return {
                'algorithm': internal_response.get('algorithm', self.algorithm),
                'gnb_for_ue': gnb_for_ue,
                'beam_for_ue': beam_for_ue,
                'objective_value': internal_response.get('objective_value'),
                'scenario_id': original_request.get('scenario_id', 0),
            }
        
        # Single-gNB case: return optimizer response as-is
        return internal_response
    
    async def start(self):
        """Start the RIC server"""
        server = await asyncio.start_server(
            self.handle_client, self.host, self.port
        )
        
        addr = server.sockets[0].getsockname()
        logger.info(f"RIC Server listening on {addr[0]}:{addr[1]}")
        logger.info(f"Using algorithm: {self.algorithm}")
        logger.info("Waiting for ns-3 connections...")
        
        async with server:
            await server.serve_forever()


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Near-RT RIC Server for 5G-LENA')
    parser.add_argument('--host', default='127.0.0.1', help='Server host (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=5555, help='Server port (default: 5555)')
    parser.add_argument('--algorithm', choices=['Max-SINR', 'Exhaustive', 'GA', 'HGA', 'PBIG'],
                       default='GA', help='Optimization algorithm to use')
    parser.add_argument('--alpha', type=float, default=1.0,
                       help='Trade-off between sum-rate and Jain fairness (0-1)')
    parser.add_argument('--max-ues-per-beam', type=int, default=None,
                       help='Maximum number of UEs per beam (capacity constraint, optional)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run server
    server = RICServer(host=args.host,
                       port=args.port,
                       algorithm=args.algorithm,
                       alpha=args.alpha,
                       max_ues_per_beam=args.max_ues_per_beam)
    
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
