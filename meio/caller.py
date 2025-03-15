python your_script.py --json 
# Later, reinitialize it to "clear" previous arguments
parser = argparse.ArgumentParser()




import sys
sys.path.append('D:\\Personal\\M8\\MEIO_solver\\MEIO_Solver\\meio\\results')
import argparse
from meio.optimization.dilop import DiloptOpSafetyStock 
from meio.optimization.heuristic import HeuristicSolver, ImprovedHeuristicSolver
from meio.optimization.solver import MathematicalSolver
# Import the Benchmark class
from meio.benchmark import Benchmark


# Parse arguments
parser = argparse.ArgumentParser(description='Benchmark MEIO algorithms')
parser.add_argument('--json', type=str, default='D:\\Personal\\M8\\MEIO_solver\\MEIO_Solver\meio\\config\\supply_chain_network.json')
parser.add_argument('--iterations', type=int, default=3, help='Number of iterations')
parser.add_argument('--output-dir', type=str, default='D:\\Personal\\M8\\MEIO_solver\\MEIO_Solver\\meio\\results')
args = parser.parse_args()
    
# Set up algorithms to benchmark
algorithms = [
    ("DiloptOpSafetyStock", lambda network: {"total_cost": 0, "result": DiloptOpSafetyStock.calculate(network)}),
    ("Original Heuristic", lambda network: HeuristicSolver.optimize(network)),
    ("Improved Heuristic", lambda network: ImprovedHeuristicSolver.optimize(network))
]
    
    
# Add solver if available
if MathematicalSolver.is_available():
    algorithms.append(("Mathematical Solver", lambda network: MathematicalSolver.optimize(network)))
    
# Run benchmark
benchmark = Benchmark(args.output_dir)
results = benchmark.run_benchmark(args.json, algorithms, args.iterations)
    
# Print summary
print("\nBenchmark Results Summary:")
print("-" * 60)
print(f"{'Algorithm':<20} {'Avg Time (s)':<15} {'Avg Cost':<15}")
print("-" * 60)
    
for result in results:
    print(f"{result['algorithm']:<20} {result['avg_time']:<15.2f} {result['avg_cost']:<15.2f}")
        
        
        
