"""
Updated main script that uses the results directory.
"""
import os
import argparse
import logging
from datetime import datetime, timedelta

# Import path manager for centralized path handling
from meio.utils.path_manager import paths

# Import components
from meio.models.network import MultiEchelonNetwork
from meio.io.json_loader import NetworkJsonLoader
from meio.io.csv_exporter import CSVExporter
from meio.optimization.solver import MathematicalSolver
from meio.optimization.heuristic import HeuristicSolver, ImprovedHeuristicSolver
from meio.optimization.dilop import DiloptOpSafetyStock
from meio.visualization.network_viz import NetworkVisualizer
from meio.visualization.charts import ChartVisualizer
from meio.config.settings import config

# Set up logging
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Multi-Echelon Inventory Optimization')
    parser.add_argument('--json', type=str, default='D:\\Personal\\M8\\MEIO_solver\\MEIO_Solver\\meio\\config\\supply_chain_network.json')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--date-interval', type=int, default=30, help='Days between periods')
    parser.add_argument('--service-level', type=float, default=config.get('optimization', 'default_service_level'), help='Service level (0-1)')
    parser.add_argument('--inflows', type=float, default=config.get('optimization', 'default_inflow'), help='Base inflow level')
    parser.add_argument('--no-solver', action='store_true', help='Skip mathematical solver')
    parser.add_argument('--no-heuristic', action='store_true', help='Skip heuristic solver')
    parser.add_argument('--improved-heuristic', action='store_true', help='Use improved heuristic')
    parser.add_argument('--no-viz', action='store_true', help='Skip visualizations')
    return parser.parse_args()


def analyze_stock_alerts(network, inventory_levels, method="Solver"):
    """
    Analyze potential stockouts and overstocks.
    
    Args:
        network (MultiEchelonNetwork): The network.
        inventory_levels (dict): Inventory levels by node, product, and period.
        method (str, optional): Method name for logging. Defaults to "Solver".
        
    Returns:
        tuple: (stockouts, overstocks) dictionaries.
    """
    stockouts = {}
    overstocks = {}
    overstock_threshold = 0.9  # Alert if inventory > 90% of capacity
    
    for node_id, node in network.nodes.items():
        stockouts[node_id] = {}
        overstocks[node_id] = {}
        for prod in node.products:
            stockouts[node_id][prod] = []
            overstocks[node_id][prod] = []
            for t in range(network.num_periods):
                inv = inventory_levels.get((node_id, prod, t), 0)
                demand = node.products[prod]['demand_by_date'][t]
                capacity = node.capacity
                date_str = network.dates[t].strftime('%Y-%m-%d')
                
                # Stockout check
                if inv < demand:
                    stockouts[node_id][prod].append({
                        'date': date_str,
                        'inventory': inv,
                        'demand': demand,
                        'shortfall': demand - inv
                    })
                
                # Overstock check
                total_inv = sum(inventory_levels.get((node_id, p, t), 0) 
                              for p in node.products)
                if total_inv > capacity * overstock_threshold:
                    overstocks[node_id][prod].append({
                        'date': date_str,
                        'inventory': inv,
                        'total_inventory': total_inv,
                        'capacity': capacity,
                        'excess': total_inv - capacity if total_inv > capacity else 0
                    })
    
    # Print alerts
    logger.info(f"\n{method} Stock Alerts:")
    logger.info("Potential Stockouts:")
    for node_id, prods in stockouts.items():
        for prod, alerts in prods.items():
            if alerts:
                logger.info(f"  {node_id} - {prod}:")
                for alert in alerts:
                    logger.info(f"    Date {alert['date']}: Inventory {alert['inventory']:.2f} < Demand {alert['demand']:.2f} (Shortfall: {alert['shortfall']:.2f})")
    
    logger.info("Potential Overstocks (above 90% capacity):")
    for node_id, prods in overstocks.items():
        for prod, alerts in prods.items():
            if alerts:
                logger.info(f"  {node_id} - {prod}:")
                for alert in alerts:
                    logger.info(f"    Date {alert['date']}: Inventory {alert['inventory']:.2f}, Total {alert['total_inventory']:.2f} > {overstock_threshold*100}% Capacity {alert['capacity']*overstock_threshold:.2f} (Excess: {alert['excess']:.2f})")
    
    return stockouts, overstocks


def calculate_receptions(network, inventory_levels):
    """
    Calculate inventory receptions based on inventory levels.
    
    Args:
        network (MultiEchelonNetwork): The network.
        inventory_levels (dict): Inventory levels by node, product, and period.
        
    Returns:
        dict: Receptions by node, product, and period.
    """
    receptions = {}
    for node_id, node in network.nodes.items():
        receptions[node_id] = {}
        for prod in node.products:
            receptions[node_id][prod] = []
            for t in range(network.num_periods):
                if t == 0:
                    reception = inventory_levels.get((node_id, prod, t), 0)
                else:
                    prev_inv = inventory_levels.get((node_id, prod, t-1), 0)
                    curr_inv = inventory_levels.get((node_id, prod, t), 0)
                    demand = node.products[prod]['demand_by_date'][t]
                    reception = max(0, curr_inv - prev_inv + demand)
                receptions[node_id][prod].append(reception)
    return receptions


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_args()
    
    # Ensure results directories exist
    os.makedirs(paths.RESULTS_DIR, exist_ok=True)
    os.makedirs(paths.VISUALIZATION_DIR, exist_ok=True)
    
    # Parse dates if provided
    start_date = None
    if args.start_date:
        try:
            start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
        except ValueError:
            logger.error(f"Invalid start date format: {args.start_date}. Use YYYY-MM-DD.")
            return 1
    
    end_date = None
    if args.end_date:
        try:
            end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
        except ValueError:
            logger.error(f"Invalid end date format: {args.end_date}. Use YYYY-MM-DD.")
            return 1
    
    # Load network from JSON
    try:
        logger.info(f"Loading network from {args.json}")
        network = NetworkJsonLoader.load(args.json, start_date, end_date, args.date_interval)
        logger.info(f"Loaded network: {len(network.nodes)} nodes, {network.num_periods} periods")
    except Exception as e:
        logger.error(f"Failed to load network: {str(e)}")
        return 1
    
    # Create a run-specific directory for this optimization
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(paths.RESULTS_DIR, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Initialize CSV exporter
    exporter = CSVExporter(run_dir)
    
    # Save parameters for reproducibility
    with open(os.path.join(run_dir, "parameters.txt"), "w") as f:
        f.write(f"Run timestamp: {timestamp}\n")
        f.write(f"JSON file: {args.json}\n")
        f.write(f"Start date: {start_date}\n")
        f.write(f"End date: {end_date}\n")
        f.write(f"Date interval: {args.date_interval}\n")
        f.write(f"Service level: {args.service_level}\n")
        f.write(f"Inflows: {args.inflows}\n")
        f.write(f"Skip solver: {args.no_solver}\n")
        f.write(f"Skip heuristic: {args.no_heuristic}\n")
        f.write(f"Use improved heuristic: {args.improved_heuristic}\n")
    
    # Calculate safety stock
    logger.info(f"Calculating safety stock with service level {args.service_level}")
    safety_recommendations = DiloptOpSafetyStock.calculate(network, args.service_level)
    
    # Run solver optimization if requested
    solver_results = {'status': 'skipped', 'inventory_levels': {}, 'total_cost': 0}
    if not args.no_solver and MathematicalSolver.is_available():
        try:
            logger.info("Optimizing with mathematical solver...")
            solver_results = MathematicalSolver.optimize(network, args.service_level)
            solver_opt_id = exporter.save_optimization_results(network, "SCIP Solver", solver_results)
            
            if solver_results['status'] == 'optimal':
                logger.info(f"Solver optimization complete. Total cost: {solver_results['total_cost']:.2f}")
                
                # Analyze stockouts and overstocks
                solver_stockouts, solver_overstocks = analyze_stock_alerts(
                    network, solver_results['inventory_levels'], method="SCIP Solver")
                exporter.save_stock_alerts(solver_opt_id, solver_stockouts, solver_overstocks, "SCIP Solver")
            else:
                logger.warning("Solver optimization failed or was infeasible.")
        except Exception as e:
            logger.error(f"Error in solver optimization: {str(e)}")
    
    # Run heuristic optimization if requested
    heuristic_results = {'status': 'skipped', 'inventory_levels': {}, 'total_cost': 0}
    if not args.no_heuristic:
        try:
            logger.info(f"Optimizing with heuristic (inflows={args.inflows})...")
            
            # Choose between original and improved heuristic
            if args.improved_heuristic:
                heuristic_results = ImprovedHeuristicSolver.optimize(network, args.service_level, args.inflows)
                heuristic_method = "Improved Heuristic"
            else:
                heuristic_results = HeuristicSolver.optimize(network, args.service_level, args.inflows)
                heuristic_method = "Heuristic"
                
            heuristic_opt_id = exporter.save_optimization_results(network, heuristic_method, heuristic_results)
            
            logger.info(f"Heuristic optimization complete. Total cost: {heuristic_results['total_cost']:.2f}")
            
            # Analyze stockouts and overstocks
            heuristic_stockouts, heuristic_overstocks = analyze_stock_alerts(
                network, heuristic_results['inventory_levels'], method=heuristic_method)
            exporter.save_stock_alerts(heuristic_opt_id, heuristic_stockouts, heuristic_overstocks, heuristic_method)
        except Exception as e:
            logger.error(f"Error in heuristic optimization: {str(e)}")
    
    # Skip visualizations if requested
    if args.no_viz:
        logger.info("Skipping visualizations as requested.")
        return 0
    
    # Create visualizations
    try:
        # Use the best available results
        results_to_use = solver_results if solver_results['status'] == 'optimal' else heuristic_results
        
        if results_to_use['status'] in ['optimal', 'heuristic']:
            # Network visualization
            logger.info("Generating network visualization...")
            NetworkVisualizer.visualize_network(
                network, results_to_use['inventory_levels'], date_idx=0,
                save_path=os.path.join(paths.VISUALIZATION_DIR, 'network_visualization.png')
            )
            
            # Comparison chart if both methods were used
            if solver_results['status'] == 'optimal' and heuristic_results['status'] == 'heuristic':
                logger.info("Generating comparison chart...")
                ChartVisualizer.plot_comparison_chart(
                    network, solver_results, heuristic_results,
                    save_path=os.path.join(paths.VISUALIZATION_DIR, 'comparison_chart.png')
                )
            
            # Node metrics charts
            logger.info("Generating node metrics charts...")
            receptions = calculate_receptions(network, results_to_use['inventory_levels'])
            ChartVisualizer.plot_node_metrics(
                network, results_to_use['inventory_levels'], receptions,
                save_path=os.path.join(paths.VISUALIZATION_DIR, 'node_metrics')
            )
            
            logger.info(f"All visualizations saved to {paths.VISUALIZATION_DIR}")
        else:
            logger.warning("No valid optimization results available for visualization.")
            
    except Exception as e:
        logger.error(f"Error generating visualizations: {str(e)}")
    
    logger.info(f"Run complete. All results saved to {paths.RESULTS_DIR}")
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())