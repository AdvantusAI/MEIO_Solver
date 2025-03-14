"""
Main entry point for the MEIO system.
"""
import logging
import argparse
from datetime import datetime, timedelta
from meio.config.settings import config
from meio.io.json_loader import NetworkJsonLoader
from meio.io.csv_exporter import CSVExporter
from meio.optimization.dilop import DiloptOpSafetyStock
from meio.optimization.solver import MathematicalSolver
from meio.optimization.heuristic import HeuristicSolver
from meio.visualization.network_viz import NetworkVisualizer
from meio.visualization.charts import ChartVisualizer

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Multi-Echelon Inventory Optimization')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--json', type=str, required=True, help='Path to network JSON file')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--date-interval', type=int, default=30, help='Days between periods')
    parser.add_argument('--service-level', type=float, help='Service level (0-1)')
    parser.add_argument('--inflows', type=float, help='Base inflow level')
    parser.add_argument('--output-dir', type=str, help='Output directory')
    parser.add_argument('--no-solver', action='store_true', help='Skip mathematical solver')
    parser.add_argument('--no-heuristic', action='store_true', help='Skip heuristic solver')
    parser.add_argument('--no-viz', action='store_true', help='Skip visualizations')
    return parser.parse_args()

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
    logging.info(f"\n{method} Stock Alerts:")
    logging.info("Potential Stockouts:")
    for node_id, prods in stockouts.items():
        for prod, alerts in prods.items():
            if alerts:
                logging.info(f"  {node_id} - {prod}:")
                for alert in alerts:
                    logging.info(f"    Date {alert['date']}: Inventory {alert['inventory']:.2f} < Demand {alert['demand']:.2f} (Shortfall: {alert['shortfall']:.2f})")
    
    logging.info("Potential Overstocks (above 90% capacity):")
    for node_id, prods in overstocks.items():
        for prod, alerts in prods.items():
            if alerts:
                logging.info(f"  {node_id} - {prod}:")
                for alert in alerts:
                    logging.info(f"    Date {alert['date']}: Inventory {alert['inventory']:.2f}, Total {alert['total_inventory']:.2f} > {overstock_threshold*100}% Capacity {alert['capacity']*overstock_threshold:.2f} (Excess: {alert['excess']:.2f})")
    
    return stockouts, overstocks

def main():
    """Main execution function."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration if specified
    if args.config:
        config.load_config(args.config)
    
    # Override configuration with command line arguments
    if args.output_dir:
        config.set('paths', 'output_dir', args.output_dir)
        
    if args.service_level:
        config.set('optimization', 'default_service_level', args.service_level)
        
    if args.inflows:
        config.set('optimization', 'default_inflow', args.inflows)
    
    # Parse dates if provided
    start_date = None
    if args.start_date:
        from meio.utils.validators import Validators
        valid, date_obj, error = Validators.validate_date_string(args.start_date)
        if valid:
            start_date = date_obj
        else:
            logging.error(f"Invalid start date: {error}")
            return 1
    
    end_date = None
    if args.end_date:
        from meio.utils.validators import Validators
        valid, date_obj, error = Validators.validate_date_string(args.end_date)
        if valid:
            end_date = date_obj
        else:
            logging.error(f"Invalid end date: {error}")
            return 1
    
    # Load network from JSON
    try:
        logging.info(f"Loading network from {args.json}")
        network = NetworkJsonLoader.load(args.json, start_date, end_date, args.date_interval)
        logging.info(f"Loaded network: {network}")
    except Exception as e:
        logging.error(f"Failed to load network: {str(e)}")
        return 1
    
    # Calculate safety stock
    service_level = config.get('optimization', 'default_service_level')
    logging.info(f"Calculating safety stock with service level {service_level}")
    safety_recommendations = DiloptOpSafetyStock.calculate(network, service_level)
    
    # Print safety stock recommendations
    logging.info("DILOP Safety Stock Recommendations (Average across dates):")
    for node_id, prods in safety_recommendations.items():
        logging.info(f"\n{node_id}:")
        for prod, details in prods.items():
            logging.info(f"  {prod}: {details['avg_safety_stock']:.2f}")
    
    # Initialize exporter
    exporter = CSVExporter()
    
    # Run solver optimization if requested
    solver_results = {'status': 'skipped', 'inventory_levels': {}, 'total_cost': 0}
    if not args.no_solver and MathematicalSolver.is_available():
        try:
            logging.info("Optimizing with mathematical solver...")
            solver_results = MathematicalSolver.optimize(network, service_level)
            solver_opt_id = exporter.save_optimization_results(network, "SCIP Solver", solver_results)
            
            if solver_results['status'] == 'optimal':
                logging.info(f"SCIP Solver Results (Date {network.dates[0].strftime('%Y-%m-%d')}):")
                for node_id in network.nodes:
                    for prod in network.nodes[node_id].products:
                        inv = solver_results['inventory_levels'].get((node_id, prod, 0), 0)
                        logging.info(f"{node_id} - {prod}: {inv:.2f}")
                logging.info(f"Total Cost: {solver_results['total_cost']:.2f}")
                
                # Analyze stockouts and overstocks
                solver_stockouts, solver_overstocks = analyze_stock_alerts(
                    network, solver_results['inventory_levels'], method="SCIP Solver")
                exporter.save_stock_alerts(solver_opt_id, solver_stockouts, solver_overstocks, "SCIP Solver")
            else:
                logging.warning("SCIP Solver optimization failed or was infeasible.")
        except Exception as e:
            logging.error(f"Error in solver optimization: {str(e)}")
    elif not args.no_solver and not MathematicalSolver.is_available():
        logging.warning("PySCIPOpt not available. Skipping mathematical solver.")
    
    # Run heuristic optimization if requested
    heuristic_results = {'status': 'skipped', 'inventory_levels': {}, 'total_cost': 0}
    if not args.no_heuristic:
        try:
            inflows = config.get('optimization', 'default_inflow')
            logging.info(f"Optimizing with heuristic (inflows={inflows})...")
            heuristic_results = HeuristicSolver.optimize(network, service_level, inflows)
            heuristic_opt_id = exporter.save_optimization_results(network, "Heuristic", heuristic_results)
            
            logging.info(f"Heuristic Results (Date {network.dates[0].strftime('%Y-%m-%d')}):")
            for node_id in network.nodes:
                for prod in network.nodes[node_id].products:
                    inv = heuristic_results['inventory_levels'].get((node_id, prod, 0), 0)
                    logging.info(f"{node_id} - {prod}: {inv:.2f}")
            logging.info(f"Total Cost: {heuristic_results['total_cost']:.2f}")
            
            # Analyze stockouts and overstocks
            heuristic_stockouts, heuristic_overstocks = analyze_stock_alerts(
                network, heuristic_results['inventory_levels'], method="Heuristic")
            exporter.save_stock_alerts(heuristic_opt_id, heuristic_stockouts, heuristic_overstocks, "Heuristic")
        except Exception as e:
            logging.error(f"Error in heuristic optimization: {str(e)}")
    
    # Skip visualizations if requested
    if args.no_viz:
        logging.info("Skipping visualizations as requested.")
        return 0
    
    # Create visualizations
    try:
        # Use the best available results
        results_to_use = solver_results if solver_results['status'] == 'optimal' else heuristic_results
        
        if results_to_use['status'] in ['optimal', 'heuristic']:
            # Network visualization
            logging.info("Generating network visualization...")
            NetworkVisualizer.visualize_network(
                network, results_to_use['inventory_levels'], date_idx=0,
                save_path=os.path.join(config.get('paths', 'output_dir'), 'network_visualization.png')
            )
            
            # Comparison chart if both methods were used
            if solver_results['status'] == 'optimal' and heuristic_results['status'] == 'heuristic':
                logging.info("Generating comparison chart...")
                ChartVisualizer.plot_comparison_chart(
                    network, solver_results, heuristic_results,
                    save_path=os.path.join(config.get('paths', 'output_dir'), 'comparison_chart.png')
                )
            
            # Node metrics charts
            logging.info("Generating node metrics charts...")
            receptions = calculate_receptions(network, results_to_use['inventory_levels'])
            ChartVisualizer.plot_node_metrics(
                network, results_to_use['inventory_levels'], receptions,
                save_path=os.path.join(config.get('paths', 'output_dir'), 'node_metrics')
            )
        else:
            logging.warning("No valid optimization results available for visualization.")
            
    except Exception as e:
        logging.error(f"Error generating visualizations: {str(e)}")
    
    return 0

if __name__ == "__main__":
    import os
    import sys
    sys.exit(main())