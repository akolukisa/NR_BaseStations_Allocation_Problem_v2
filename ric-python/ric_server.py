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
        
        logger.info(f"Max-SINR assignment: {beam_assignment}")
        return {
            "algorithm": self.algorithm_name,
            "beam_for_ue": beam_assignment,
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
    """Genetic Algorithm for beam-UE assignment"""
    
    def __init__(self, population_size=50, generations=100, mutation_rate=0.1):
        super().__init__("GA")
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
    
    def optimize(self, scenario_data: Dict) -> Dict:
        sinr_matrix = np.array(scenario_data['sinr_matrix_dB'])
        num_beams = scenario_data['num_beams']
        num_ues = scenario_data['num_ues']
        
        # Fitness function: sum-rate objective
        def fitness(chromosome: np.ndarray) -> float:
            total_rate = 0.0
            for ue_idx, beam_idx in enumerate(chromosome):
                sinr_db = sinr_matrix[int(beam_idx), ue_idx]
                sinr_linear = 10 ** (sinr_db / 10.0)
                rate = np.log2(1 + sinr_linear)
                total_rate += rate
            return total_rate
        
        # Initialize population
        population = np.random.randint(0, num_beams, (self.population_size, num_ues))
        
        best_chromosome = None
        best_fitness = -np.inf
        
        for gen in range(self.generations):
            # Evaluate fitness
            fitness_values = np.array([fitness(ind) for ind in population])
            
            # Track best
            gen_best_idx = np.argmax(fitness_values)
            if fitness_values[gen_best_idx] > best_fitness:
                best_fitness = fitness_values[gen_best_idx]
                best_chromosome = population[gen_best_idx].copy()
            
            # Selection (tournament)
            selected = []
            for _ in range(self.population_size):
                idx1, idx2 = np.random.choice(self.population_size, 2, replace=False)
                winner = idx1 if fitness_values[idx1] > fitness_values[idx2] else idx2
                selected.append(population[winner].copy())
            
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
            
            population = np.array(offspring[:self.population_size])
        
        logger.info(f"GA best assignment: {best_chromosome.tolist()}, fitness: {best_fitness:.2f}")
        return {
            "algorithm": self.algorithm_name,
            "beam_for_ue": best_chromosome.tolist(),
            "objective_value": best_fitness,
            "scenario_id": scenario_data.get('scenario_id', 0)
        }


class HGAOptimizer(BeamAssignmentOptimizer):
    """Hybrid Genetic Algorithm with local search"""
    
    def __init__(self, population_size=50, generations=100, mutation_rate=0.1, local_search_prob=0.3):
        super().__init__("HGA")
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.local_search_prob = local_search_prob
    
    def optimize(self, scenario_data: Dict) -> Dict:
        sinr_matrix = np.array(scenario_data['sinr_matrix_dB'])
        num_beams = scenario_data['num_beams']
        num_ues = scenario_data['num_ues']
        
        def fitness(chromosome: np.ndarray) -> float:
            total_rate = 0.0
            for ue_idx, beam_idx in enumerate(chromosome):
                sinr_db = sinr_matrix[int(beam_idx), ue_idx]
                sinr_linear = 10 ** (sinr_db / 10.0)
                rate = np.log2(1 + sinr_linear)
                total_rate += rate
            return total_rate
        
        def local_search(chromosome: np.ndarray) -> np.ndarray:
            """Try flipping each gene to improve fitness"""
            current_fitness = fitness(chromosome)
            improved = chromosome.copy()
            for ue_idx in range(num_ues):
                original_beam = improved[ue_idx]
                for beam_idx in range(num_beams):
                    if beam_idx == original_beam:
                        continue
                    improved[ue_idx] = beam_idx
                    new_fitness = fitness(improved)
                    if new_fitness > current_fitness:
                        current_fitness = new_fitness
                    else:
                        improved[ue_idx] = original_beam
            return improved
        
        # Initialize population (similar to GA)
        population = np.random.randint(0, num_beams, (self.population_size, num_ues))
        
        best_chromosome = None
        best_fitness = -np.inf
        
        for gen in range(self.generations):
            fitness_values = np.array([fitness(ind) for ind in population])
            
            gen_best_idx = np.argmax(fitness_values)
            if fitness_values[gen_best_idx] > best_fitness:
                best_fitness = fitness_values[gen_best_idx]
                best_chromosome = population[gen_best_idx].copy()
            
            # Selection
            selected = []
            for _ in range(self.population_size):
                idx1, idx2 = np.random.choice(self.population_size, 2, replace=False)
                winner = idx1 if fitness_values[idx1] > fitness_values[idx2] else idx2
                selected.append(population[winner].copy())
            
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
            
            # Local search (hybrid part)
            for i, child in enumerate(offspring):
                if np.random.rand() < self.local_search_prob:
                    offspring[i] = local_search(child)
            
            population = np.array(offspring[:self.population_size])
        
        logger.info(f"HGA best assignment: {best_chromosome.tolist()}, fitness: {best_fitness:.2f}")
        return {
            "algorithm": self.algorithm_name,
            "beam_for_ue": best_chromosome.tolist(),
            "objective_value": best_fitness,
            "scenario_id": scenario_data.get('scenario_id', 0)
        }


class PBIGOptimizer(BeamAssignmentOptimizer):
    """Priority-Based Iterative Greedy algorithm"""
    
    def __init__(self):
        super().__init__("PBIG")
    
    def optimize(self, scenario_data: Dict) -> Dict:
        sinr_matrix = np.array(scenario_data['sinr_matrix_dB'])
        num_beams = scenario_data['num_beams']
        num_ues = scenario_data['num_ues']
        
        # Priority-based: start with UEs that have the least beam options (highest variance)
        ue_sinr_variance = np.var(sinr_matrix, axis=0)  # Variance across beams for each UE
        ue_priorities = np.argsort(ue_sinr_variance)  # Ascending: least flexible first
        
        beam_assignment = [-1] * num_ues
        beam_load = np.zeros(num_beams)  # Track how many UEs per beam
        
        # Greedy assignment in priority order
        for ue_idx in ue_priorities:
            # Find the beam that maximizes SINR for this UE
            # (with optional penalty for overloaded beams)
            ue_sinrs = sinr_matrix[:, ue_idx]
            # Convert dB to linear for calculation
            ue_sinrs_linear = 10 ** (ue_sinrs / 10.0)
            
            # Simple greedy: pick max SINR beam
            # (Can add beam load balancing penalty here)
            best_beam = int(np.argmax(ue_sinrs_linear))
            
            beam_assignment[ue_idx] = best_beam
            beam_load[best_beam] += 1
        
        # Calculate objective
        total_rate = 0.0
        for ue_idx, beam_idx in enumerate(beam_assignment):
            sinr_db = sinr_matrix[beam_idx, ue_idx]
            sinr_linear = 10 ** (sinr_db / 10.0)
            rate = np.log2(1 + sinr_linear)
            total_rate += rate
        
        logger.info(f"PBIG assignment: {beam_assignment}, objective: {total_rate:.2f}")
        return {
            "algorithm": self.algorithm_name,
            "beam_for_ue": beam_assignment,
            "objective_value": total_rate,
            "scenario_id": scenario_data.get('scenario_id', 0)
        }


class RICServer:
    """Near-RT RIC TCP Server for ns-3 integration"""
    
    def __init__(self, host='127.0.0.1', port=5555, algorithm='GA'):
        self.host = host
        self.port = port
        self.algorithm = algorithm
        
        # Initialize optimizer based on algorithm choice
        if algorithm == 'Max-SINR':
            self.optimizer = MaxSINROptimizer()
        elif algorithm == 'Exhaustive':
            self.optimizer = ExhaustiveSearchOptimizer()
        elif algorithm == 'GA':
            self.optimizer = GAOptimizer()
        elif algorithm == 'HGA':
            self.optimizer = HGAOptimizer()
        elif algorithm == 'PBIG':
            self.optimizer = PBIGOptimizer()
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        logger.info(f"RIC Server initialized with {algorithm} algorithm")
    
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
                       f"with {request.get('num_beams')} beams and {request.get('num_ues')} UEs")
            
            # Optimize beam assignment
            response = self.optimizer.optimize(request)
            
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
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run server
    server = RICServer(host=args.host, port=args.port, algorithm=args.algorithm)
    
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
