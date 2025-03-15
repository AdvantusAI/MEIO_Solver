import sys
import argparse
from meio.utils.path_manager import paths
from meio.optimization.dilop import DiloptOpSafetyStock
from meio.optimization.heuristic import ImprovedHeuristicSolver, HeuristicSolver
from meio.optimization.solver import MathematicalSolver
from meio.benchmark import Benchmark

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='Benchmark MEIO algorithms')
    parser.add_argument('--json', type=str, required=True, help='Path to network JSON file')
    parser.add_argument('--iterations', type=int, default=3, help='Number of iterations')
    parser.add_argument('--output-dir', type=str, help='Output directory')
    args = parser.parse_args()
    
    # Set up algorithms to benchmark
    algorithms = [
        ("DiloptOpSafetyStock", lambda network: {"total_cost": 0, "status": "success", "inventory_levels": {}, "result": DiloptOpSafetyStock.calculate(network)}),
        ("Original Heuristic", lambda network: HeuristicSolver.optimize(network)),
        ("Improved Heuristic", lambda network: ImprovedHeuristicSolver.optimize(network))
    ]
    
    # Add solver if available
    if MathematicalSolver.is_available():
        algorithms.append(("Mathematical Solver", lambda network: MathematicalSolver.optimize(network)))
    
    # Run benchmark
    # If output_dir not specified, use the benchmark directory from PathManager
    output_dir = args.output_dir or paths.BENCHMARK_RESULTS_DIR
    benchmark = Benchmark(output_dir)
    results = benchmark.run_benchmark(args.json, algorithms, args.iterations)
    
    # Print summary
    print("\nBenchmark Results Summary:")
    print("-" * 60)
    print(f"{'Algorithm':<20} {'Avg Time (s)':<15} {'Avg Cost':<15}")
    print("-" * 60)
    
    for result in results:
        print(f"{result['algorithm']:<20} {result['avg_time']:<15.2f} {result['avg_cost']:<15.2f}")
        
        