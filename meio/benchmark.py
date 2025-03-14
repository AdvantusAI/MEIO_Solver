"""
Benchmarking tools for performance measurement.
"""
import logging
import time
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from .config.settings import config
from .io.json_loader import NetworkJsonLoader

logger = logging.getLogger(__name__)

class Benchmark:
    """Tools for benchmarking and comparing algorithm performance."""
    
    def __init__(self, output_dir=None):
        """
        Initialize the benchmark tool.
        
        Args:
            output_dir (str, optional): Output directory. Defaults to config value.
        """
        self.output_dir = output_dir or config.get('paths', 'output_dir')
        os.makedirs(self.output_dir, exist_ok=True)
        self.results = []
    
    def run_benchmark(self, json_file, algorithms, iterations=3):
        """
        Run a benchmark comparing multiple algorithms.
        
        Args:
            json_file (str): Path to network JSON file.
            algorithms (list): List of (name, function) tuples to benchmark.
            iterations (int, optional): Number of iterations per algorithm. Defaults to 3.
            
        Returns:
            dict: Benchmark results.
        """
        print("Loading JSON file")
        print(json_file)
        logger.info(f"Starting benchmark with {len(algorithms)} algorithms, {iterations} iterations each")
        benchmark_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for alg_name, alg_func in algorithms:
            logger.info(f"Benchmarking {alg_name}...")
            
            # Run multiple iterations
            times = []
            costs = []
            
            for i in range(iterations):
                logger.info(f"  Iteration {i+1}/{iterations}")
                
                # Load a fresh network for each iteration
                network = NetworkJsonLoader.load(json_file)
                
                # Run algorithm with timing
                start_time = time.time()
                result = alg_func(network)
                end_time = time.time()
                
                elapsed = end_time - start_time
                times.append(elapsed)
                costs.append(result.get('total_cost', 0))
                
                logger.info(f"  Completed in {elapsed:.2f} seconds, cost: {result.get('total_cost', 0):.2f}")
            
            # Compute statistics
            avg_time = np.mean(times)
            std_time = np.std(times)
            avg_cost = np.mean(costs)
            
            self.results.append({
                'benchmark_id': benchmark_id,
                'algorithm': alg_name,
                'avg_time': avg_time,
                'std_time': std_time,
                'min_time': min(times),
                'max_time': max(times),
                'avg_cost': avg_cost,
                'costs': costs,
                'times': times
            })
        
        # Save benchmark results
        self._save_results(benchmark_id)
        
        # Generate visualizations
        self._generate_charts(benchmark_id)
        
        return self.results
    
    def _save_results(self, benchmark_id):
        """Save benchmark results to file."""
        output_file = os.path.join(self.output_dir, f"benchmark_{benchmark_id}.json")
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
            
        logger.info(f"Benchmark results saved to {output_file}")
    
    def _generate_charts(self, benchmark_id):
        """Generate benchmark charts."""
        if not self.results:
            return
        
        # Prepare data
        alg_names = [r['algorithm'] for r in self.results]
        avg_times = [r['avg_time'] for r in self.results]
        std_times = [r['std_time'] for r in self.results]
        avg_costs = [r['avg_cost'] for r in self.results]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Time comparison
        ax1.bar(alg_names, avg_times, yerr=std_times, capsize=5)
        ax1.set_ylabel('Execution Time (seconds)')
        ax1.set_title('Algorithm Execution Time Comparison')
        ax1.set_xticklabels(alg_names, rotation=45, ha='right')
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Cost comparison
        ax2.bar(alg_names, avg_costs)
        ax2.set_ylabel('Total Cost')
        ax2.set_title('Algorithm Cost Comparison')
        ax2.set_xticklabels(alg_names, rotation=45, ha='right')
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        chart_file = os.path.join(self.output_dir, f"benchmark_chart_{benchmark_id}.png")
        plt.savefig(chart_file, dpi=300)
        logger.info(f"Benchmark charts saved to {chart_file}")
