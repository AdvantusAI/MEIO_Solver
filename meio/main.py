"""
Main entry point for the MEIO system.
"""
import logging
import argparse
import os
from datetime import datetime, timedelta
from meio.utils.path_manager import paths
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
    
    # Sensitivity analysis options
    parser.add_argument('--sensitivity', action='store_true', help='Run sensitivity analysis')
    parser.add_argument('--service-levels', type=str, help='Comma-separated list of service levels for sensitivity analysis')
    parser.add_argument('--lead-time-factors', type=str, help='Comma-separated list of lead time factors for sensitivity analysis')
    parser.add_argument('--demand-factors', type=str, help='Comma-separated list of demand factors for sensitivity analysis')
    parser.add_argument('--inflow-levels', type=str, help='Comma-separated list of inflow levels for sensitivity analysis')
    parser.add_argument('--sensitivity-method', type=str, default='improved_heuristic',
                       choices=['heuristic', 'improved_heuristic', 'solver'],
                       help='Optimization method to use for sensitivity analysis')
    
    # Branch selection options
    parser.add_argument('--branch-selection', action='store_true', help='Run branch selection strategy analysis')
    parser.add_argument('--num-branches', type=int, default=5, help='Number of branches to generate')
    parser.add_argument('--branch-criteria', type=str, default='cost,service_level,robustness',
                      help='Comma-separated list of criteria for branch evaluation')
    parser.add_argument('--branch-weights', type=str, default='0.4,0.4,0.2',
                      help='Comma-separated list of weights for each criterion')
    parser.add_argument('--selection-criteria', type=str, default='balanced',
                      choices=['balanced', 'cost_focused', 'service_focused', 'robust'],
                      help='Strategy for selecting the best branch')
    parser.add_argument('--branch-method', type=str, default='improved_heuristic',
                      choices=['heuristic', 'improved_heuristic', 'solver'],
                      help='Optimization method to use for branch generation')
    
    # AI agent options
    parser.add_argument('--use-ai-agent', action='store_true', help='Use AI Parameter Evolution Agent')
    parser.add_argument('--import-history', type=str, help='Import historical data for AI agent from file or directory')
    parser.add_argument('--show-ai-trends', action='store_true', help='Show AI agent performance trends')
    
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

def run_sensitivity_analysis(args, network):
    """
    Run sensitivity analysis on the network.
    
    Args:
        args: Command line arguments
        network: The network to analyze
        
    Returns:
        int: Return code (0 for success)
    """
    from meio.analysis.sensitivity import run_sensitivity_analysis
    
    # Parse sensitivity parameters
    parameter_ranges = {}
    
    if args.service_levels:
        parameter_ranges['service_level'] = [float(x) for x in args.service_levels.split(',')]
    else:
        # Default service levels to test
        parameter_ranges['service_level'] = [0.90, 0.95, 0.98]
    
    if args.lead_time_factors:
        parameter_ranges['lead_time_factor'] = [float(x) for x in args.lead_time_factors.split(',')]
    
    if args.demand_factors:
        parameter_ranges['demand_factor'] = [float(x) for x in args.demand_factors.split(',')]
    
    if args.inflow_levels:
        parameter_ranges['inflows'] = [float(x) for x in args.inflow_levels.split(',')]
    
    # Base parameters
    base_params = {}
    if args.service_level:
        base_params['service_level'] = args.service_level
    if args.inflows:
        base_params['inflows'] = args.inflows
    
    logging.info(f"Running sensitivity analysis with parameters: {parameter_ranges}")
    
    try:
        result = run_sensitivity_analysis(
            network,
            parameter_ranges,
            base_params=base_params if base_params else None,
            method=args.sensitivity_method,
            output_dir=args.output_dir,
            visualize=not args.no_viz
        )
        
        logging.info(f"Sensitivity analysis complete")
        logging.info(f"Results saved to {result['output_dir']}")
        
        # Print summary of key insights
        if 'elasticity_metrics' in result and result['elasticity_metrics']:
            param_sensitivity = {}
            for metric in result['elasticity_metrics']:
                param = metric['parameter']
                elasticity = metric.get('abs_elasticity')
                if elasticity is not None:
                    if param not in param_sensitivity:
                        param_sensitivity[param] = []
                    param_sensitivity[param].append(elasticity)
            
            logging.info("Key insights from sensitivity analysis:")
            for param, values in param_sensitivity.items():
                avg_elasticity = sum(values) / len(values)
                
                if avg_elasticity > 1.0:
                    sensitivity = "HIGH"
                elif avg_elasticity > 0.5:
                    sensitivity = "MEDIUM"
                else:
                    sensitivity = "LOW"
                
                logging.info(f"  {param}: {sensitivity} sensitivity (elasticity = {avg_elasticity:.4f})")
            
            logging.info(f"Detailed report available at: {result['report_path']}")
        
        return 0
        
    except Exception as e:
        logging.error(f"Error during sensitivity analysis: {str(e)}")
        import traceback
        logging.debug(traceback.format_exc())
        return 1

def run_branch_selection(args, network):
    """
    Run branch selection on the network.
    
    Args:
        args: Command line arguments
        network: The network to analyze
        
    Returns:
        int: Return code (0 for success)
    """
    from meio.optimization.branch_selection import BranchManager
    from meio.visualization.branch_viz import BranchVisualizer
    
    # Parse criteria and weights
    criteria = args.branch_criteria.split(',')
    
    weights = {}
    if args.branch_weights:
        weight_values = [float(x) for x in args.branch_weights.split(',')]
        if len(weight_values) == len(criteria):
            weights = dict(zip(criteria, weight_values))
        else:
            logging.warning("Number of weights doesn't match number of criteria. Using default weights.")
    
    # Base parameters
    base_params = {}
    if args.service_level:
        base_params['service_level'] = args.service_level
    if args.inflows:
        base_params['inflows'] = args.inflows
    
    logging.info(f"Running branch selection with {args.num_branches} branches using criteria: {criteria}")
    
    try:
        # Initialize branch manager with AI agent if requested
        branch_manager = BranchManager(output_dir=args.output_dir, use_ai_agent=args.use_ai_agent)
        
        # Import historical data if provided
        if args.use_ai_agent and args.import_history:
            import_result = branch_manager.import_historical_data(args.import_history)
            logging.info(f"Import result: {import_result.get('message', 'Unknown')}")
        
        # Show AI agent trends if requested
        if args.use_ai_agent and args.show_ai_trends:
            trend_data = branch_manager.get_ai_agent_trend()
            if trend_data.get("status") == "success":
                logging.info("\nAI Agent Performance Trends:")
                logging.info(f"Data points: {trend_data.get('data_points', 0)}")
                logging.info(f"Overall trend: {trend_data.get('overall_score_trend', 'unknown')}")
                logging.info(f"Cost trend: {trend_data.get('cost_score_trend', 'unknown')}")
                logging.info(f"Service level trend: {trend_data.get('service_score_trend', 'unknown')}")
                logging.info(f"Robustness trend: {trend_data.get('robustness_score_trend', 'unknown')}")
                logging.info(f"Latest scores: {trend_data.get('latest_scores', {})}")
            else:
                logging.info(f"AI trend data: {trend_data.get('message', 'Insufficient data')}")
        
        # Run branch selection
        results = branch_manager.run_branch_selection(
            network,
            num_branches=args.num_branches,
            criteria=criteria,
            weights=weights,
            selection_criteria=args.selection_criteria,
            method=args.branch_method,
            base_params=base_params
        )
        
        # Create visualizations if not disabled
        if not args.no_viz:
            BranchVisualizer.visualize_branch_selection_summary(results)
        
        # Log selected branch information
        selected_branch = results['selection_results']['selected_branch']
        if selected_branch:
            selected_data = results['branch_results']['branches'][selected_branch]
            logging.info(f"\nSelected Branch: {selected_branch}")
            logging.info(f"Selection Criteria: {args.selection_criteria}")
            logging.info(f"Branch Parameters:")
            for k, v in selected_data['params'].items():
                logging.info(f"  {k}: {v}")
            
            logging.info(f"Selection Rationale: {results['selection_results']['rationale']}")
            
            # Implementation guidance
            logging.info("\nImplementation Guidance:")
            logging.info("To implement this inventory policy:")
            logging.info(f"1. Apply the service level settings shown above")
            
            params = selected_data['params']
            if 'lead_time_factor' in params:
                # Add type checking before formatting as float
                if isinstance(params['lead_time_factor'], (int, float)):
                    logging.info(f"2. Add a {(params['lead_time_factor']-1)*100:.0f}% buffer to lead time estimates")
                else:
                    logging.info(f"2. Add a buffer to lead time estimates ({params['lead_time_factor']})")
            if 'inflows' in params:
                # Add type checking before formatting as float
                if isinstance(params['inflows'], (int, float)):
                    logging.info(f"3. Set inflow levels to {params['inflows']:.2f}")
                else:
                    logging.info(f"3. Set inflow levels to {params['inflows']}")
            logging.info(f"4. Monitor performance and adjust if necessary")
            
            # AI agent information
            if args.use_ai_agent and 'ai_suggestion' in results:
                ai_suggestion = results['ai_suggestion']
                if ai_suggestion:
                    logging.info("\nAI Agent Information:")
                    logging.info(f"Confidence: {ai_suggestion.get('confidence', 'unknown')}")
                    logging.info(f"Data points: {ai_suggestion.get('data_points', 0)}")
                    logging.info(f"Similar networks: {ai_suggestion.get('similar_networks', 0)}")
                    logging.info(f"Rationale: {ai_suggestion.get('rationale', 'Not available')}")
            
            logging.info(f"\nBranch selection results saved to {results['output_dir']}")
        else:
            logging.warning("No valid branch could be selected")
        
        return 0
        
    except Exception as e:
        logging.error(f"Error during branch selection: {str(e)}")
        import traceback
        logging.debug(traceback.format_exc())
        return 1

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
    
    # If sensitivity analysis is requested, run it and exit
    if args.sensitivity:
        return run_sensitivity_analysis(args, network)
    
    # If branch selection is requested, run it and exit
    if args.branch_selection:
        return run_branch_selection(args, network)
    
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
    logging.info(f"CSV Output directory: {exporter.output_dir} (exists: {os.path.exists(exporter.output_dir)})")
    
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
                save_path=paths.get_visualization_path('network_visualization.png')
            )
            
            # Comparison chart if both methods were used
            if solver_results['status'] == 'optimal' and heuristic_results['status'] == 'heuristic':
                logging.info("Generating comparison chart...")
                ChartVisualizer.plot_comparison_chart(
                    network, solver_results, heuristic_results,
                    save_path=paths.get_visualization_path('comparison_chart.png')
                )
            
            # Node metrics charts
            logging.info("Generating node metrics charts...")
            receptions = calculate_receptions(network, results_to_use['inventory_levels'])
            ChartVisualizer.plot_node_metrics(
                network, results_to_use['inventory_levels'], receptions,
                save_path=paths.get_visualization_path('node_metrics')
            )
        else:
            logging.warning("No valid optimization results available for visualization.")
            
    except Exception as e:
        logging.error(f"Error generating visualizations: {str(e)}")
    
    # Save network statistics to CSV
    try:
        logging.info("Saving network statistics to CSV...")
        exporter.save_network_statistics(network)
    except Exception as e:
        logging.warning(f"Failed to save network statistics: {str(e)}")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())