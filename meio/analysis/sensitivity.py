"""
Sensitivity Analysis Module for MEIO System.

This module provides functionality to analyze how sensitive inventory strategies
are to changes in various parameters such as service levels, lead times, and demand patterns.
"""
import logging
import os
import copy
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional, Union, Callable

from meio.utils.path_manager import paths
from meio.optimization.heuristic import HeuristicSolver, ImprovedHeuristicSolver
from meio.optimization.solver import MathematicalSolver
from meio.io.csv_exporter import CSVExporter

logger = logging.getLogger(__name__)

def run_sensitivity_analysis(
    network,
    parameter_ranges: Dict[str, List[Any]],
    base_params: Optional[Dict[str, Any]] = None,
    method: str = 'improved_heuristic',
    output_dir: Optional[str] = None,
    visualize: bool = True
) -> Dict[str, Any]:
    """
    Run sensitivity analysis by varying parameters and measuring impact.
    
    Args:
        network: The network to optimize
        parameter_ranges: Dict mapping param names to lists of values to test
                         e.g., {'service_level': [0.9, 0.95, 0.99],
                                'lead_time_factor': [0.8, 1.0, 1.2]}
        base_params: Dict of baseline parameters (defaults used if None)
        method: Optimization method to use ('heuristic', 'improved_heuristic', or 'solver')
        output_dir: Directory to save results (uses default if None)
        visualize: Whether to generate visualizations
    
    Returns:
        Dict with sensitivity analysis results and paths to output files
    """
    # Start timer
    start_time = time.time()
    
    # Set up default base parameters if not provided
    if base_params is None:
        from meio.config.settings import config
        base_params = {
            'service_level': config.get('optimization', 'default_service_level'),
            'inflows': config.get('optimization', 'default_inflow')
        }
    
    # Create unique ID for this analysis
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    analysis_id = f"sensitivity_{timestamp}"
    
    # Set up output directory
    if output_dir is None:
        output_dir = os.path.join(paths.RESULTS_DIR, f"sensitivity_{timestamp}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize CSV exporter
    exporter = CSVExporter(output_dir)
    
    # Save parameters for reproducibility
    with open(os.path.join(output_dir, "sensitivity_parameters.txt"), "w", encoding='utf-8') as f:
        f.write(f"Sensitivity Analysis ID: {analysis_id}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Method: {method}\n")
        f.write(f"Base Parameters:\n")
        for key, value in base_params.items():
            f.write(f"  {key}: {value}\n")
        f.write(f"Parameter Ranges:\n")
        for key, values in parameter_ranges.items():
            f.write(f"  {key}: {values}\n")
    
    # Select optimizer based on method
    if method == 'improved_heuristic':
        optimizer = ImprovedHeuristicSolver.optimize
    elif method == 'heuristic':
        optimizer = HeuristicSolver.optimize
    elif method == 'solver' and MathematicalSolver.is_available():
        optimizer = MathematicalSolver.optimize
    else:
        logger.warning(f"Solver method '{method}' not available, falling back to improved heuristic")
        optimizer = ImprovedHeuristicSolver.optimize
    
    # Run baseline first
    logger.info(f"Running baseline optimization with parameters: {base_params}")
    baseline_network = copy.deepcopy(network)
    baseline_result = optimizer(baseline_network, **base_params)
    
    # Calculate total runs needed for single parameter analysis
    total_parameter_values = sum(len(values) for values in parameter_ranges.values())
    logger.info(f"Running {total_parameter_values} sensitivity analysis iterations...")
    
    # Store all results
    all_results = []
    
    # Run single parameter variation analysis
    for param_name, param_values in parameter_ranges.items():
        logger.info(f"Analyzing sensitivity to {param_name}")
        
        param_results = []
        for i, value in enumerate(param_values):
            logger.info(f"  Testing {param_name} = {value} ({i+1}/{len(param_values)})")
            
            # Create modified params for this run
            run_params = base_params.copy()
            
            # Special handling for different parameter types
            if param_name == 'service_level':
                run_params['service_level'] = value
            elif param_name == 'lead_time_factor':
                # Create a copy of network with modified lead times
                run_network = copy.deepcopy(network)
                _modify_network_lead_times(run_network, value)
            elif param_name == 'demand_factor':
                # Create a copy of network with modified demand
                run_network = copy.deepcopy(network)
                _modify_network_demand(run_network, value)
            elif param_name == 'inflows':
                run_params['inflows'] = value
            else:
                # Generic parameter, just add to run_params
                run_params[param_name] = value
            
            # Use appropriate network (modified or original)
            if param_name in ['lead_time_factor', 'demand_factor']:
                current_network = run_network
            else:
                current_network = copy.deepcopy(network)
            
            # Run optimization
            try:
                opt_result = optimizer(current_network, **run_params)
                
                # Extract key metrics
                result_data = {
                    'parameter': param_name,
                    'value': value,
                    'total_cost': opt_result.get('total_cost', 0),
                    'status': opt_result.get('status', 'unknown'),
                    'run_id': f"{param_name}_{value}".replace('.', '_')
                }
                
                # Analyze stockouts if available
                if 'inventory_levels' in opt_result:
                    stockouts, overstocks = _analyze_inventory_issues(current_network, opt_result['inventory_levels'])
                    result_data['stockout_count'] = _count_stockouts(stockouts)
                    result_data['overstock_count'] = _count_stockouts(overstocks)
                
                param_results.append(result_data)
                all_results.append(result_data)
                
                # Save detailed results for this run
                if 'inventory_levels' in opt_result:
                    _save_detailed_inventory(
                        current_network, 
                        opt_result['inventory_levels'], 
                        param_name, 
                        value, 
                        output_dir
                    )
                
            except Exception as e:
                logger.error(f"Error in optimization for {param_name}={value}: {str(e)}")
                result_data = {
                    'parameter': param_name,
                    'value': value,
                    'total_cost': None,
                    'status': 'error',
                    'error_message': str(e),
                    'run_id': f"{param_name}_{value}".replace('.', '_')
                }
                param_results.append(result_data)
                all_results.append(result_data)
        
        # Save parameter-specific results to CSV
        param_df = pd.DataFrame(param_results)
        param_csv_file = os.path.join(output_dir, f"sensitivity_{param_name}.csv")
        param_df.to_csv(param_csv_file, index=False)
        
        if visualize:
            _visualize_parameter_sensitivity(param_results, param_name, output_dir)
    
    # Save overall results
    results_df = pd.DataFrame(all_results)
    results_csv_file = os.path.join(output_dir, "sensitivity_results.csv")
    results_df.to_csv(results_csv_file, index=False)
    
    # Generate summary visualization if requested
    if visualize:
        figure_paths = _visualize_overall_sensitivity(results_df, output_dir)
    else:
        figure_paths = []
    
    # Calculate elasticity metrics
    elasticity_metrics = _calculate_elasticity(results_df, baseline_result)
    
    # Save elasticity metrics
    elasticity_df = pd.DataFrame(elasticity_metrics)
    elasticity_csv_file = os.path.join(output_dir, "elasticity_metrics.csv")
    elasticity_df.to_csv(elasticity_csv_file, index=False)
    
    # Generate report
    report_path = _generate_report(
        results_df, 
        elasticity_df, 
        baseline_result, 
        base_params, 
        parameter_ranges, 
        output_dir
    )
    
    # Calculate execution time
    execution_time = time.time() - start_time
    
    logger.info(f"Sensitivity analysis completed in {execution_time:.2f} seconds")
    logger.info(f"Results saved to {output_dir}")
    
    # Return results summary
    return {
        'analysis_id': analysis_id,
        'output_dir': output_dir,
        'results_csv': results_csv_file,
        'elasticity_csv': elasticity_csv_file,
        'report_path': report_path,
        'figure_paths': figure_paths,
        'execution_time': execution_time,
        'baseline_result': baseline_result,
        'parameter_ranges': parameter_ranges,
        'all_results': all_results,
        'elasticity_metrics': elasticity_metrics
    }

def _modify_network_lead_times(network, factor):
    """
    Modify lead times in the network by a given factor.
    
    Args:
        network: Network to modify
        factor: Factor to multiply lead times by
    """
    for node_id, node in network.nodes.items():
        if hasattr(node, 'lead_time'):
            for prod in node.lead_time:
                node.lead_time[prod] = max(1, int(node.lead_time[prod] * factor))

def _modify_network_demand(network, factor):
    """
    Modify demand in the network by a given factor.
    
    Args:
        network: Network to modify
        factor: Factor to multiply demand by
    """
    for node_id, node in network.nodes.items():
        for prod in node.products:
            if 'demand_by_date' in node.products[prod]:
                for t in range(len(node.products[prod]['demand_by_date'])):
                    node.products[prod]['demand_by_date'][t] *= factor

def _analyze_inventory_issues(network, inventory_levels):
    """
    Analyze potential stockouts and overstocks.
    
    Args:
        network: The network
        inventory_levels: Dict of inventory levels
        
    Returns:
        Tuple of (stockouts, overstocks) dictionaries
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
    
    return stockouts, overstocks

def _count_stockouts(stockouts):
    """
    Count total number of stockouts.
    
    Args:
        stockouts: Dict of stockouts by node and product
        
    Returns:
        Total number of stockouts
    """
    count = 0
    for node_id, prods in stockouts.items():
        for prod, alerts in prods.items():
            count += len(alerts)
    return count

def _save_detailed_inventory(network, inventory_levels, param_name, param_value, output_dir):
    """
    Save detailed inventory levels to CSV.
    
    Args:
        network: Network object
        inventory_levels: Dict of inventory levels
        param_name: Parameter name that was varied
        param_value: Value of the parameter
        output_dir: Directory to save the file
    """
    run_id = f"{param_name}_{param_value}".replace('.', '_')
    filename = os.path.join(output_dir, f"inventory_{run_id}.csv")
    
    # Prepare data
    data = []
    for (node_id, prod, t), inv in inventory_levels.items():
        if t < len(network.dates):
            date_str = network.dates[t].strftime('%Y-%m-%d')
        else:
            date_str = f"period_{t}"
            
        demand = network.nodes[node_id].products[prod]['demand_by_date'][t]
        
        data.append({
            'run_id': run_id,
            'parameter': param_name,
            'value': param_value,
            'node_id': node_id,
            'product_id': prod,
            'period': t,
            'date': date_str,
            'inventory': inv,
            'demand': demand,
            'coverage_ratio': inv / demand if demand > 0 else 0
        })
    
    # Save to CSV
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    return filename

def _visualize_parameter_sensitivity(param_results, param_name, output_dir):
    """
    Create visualization of sensitivity to a specific parameter.
    
    Args:
        param_results: List of result dictionaries for this parameter
        param_name: Name of the parameter
        output_dir: Directory to save the visualization
    
    Returns:
        Path to the saved figure
    """
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(param_results)
    
    # Skip if not enough data points
    if len(df) < 2:
        return None
    
    # Create figure
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot cost vs parameter value
    color = 'tab:blue'
    ax1.set_xlabel(param_name)
    ax1.set_ylabel('Total Cost', color=color)
    
    # Sort by parameter value for line plot
    df = df.sort_values('value')
    
    # Convert parameter values to strings for categorical plotting if needed
    if not all(isinstance(x, (int, float)) for x in df['value']):
        df['value'] = df['value'].astype(str)
    
    # Plot line
    ax1.plot(df['value'], df['total_cost'], 'o-', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Add stockout information if available
    if 'stockout_count' in df.columns:
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Stockout Count', color=color)
        ax2.plot(df['value'], df['stockout_count'], 'o--', color=color)
        ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title(f'Sensitivity to {param_name}')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save figure
    filename = os.path.join(output_dir, f"sensitivity_{param_name}.png")
    plt.savefig(filename, dpi=300)
    plt.close()
    
    return filename

def _visualize_overall_sensitivity(results_df, output_dir):
    """
    Create visualizations summarizing all sensitivity results.
    
    Args:
        results_df: DataFrame with all sensitivity results
        output_dir: Directory to save the visualizations
    
    Returns:
        List of paths to saved figures
    """
    figure_paths = []
    
    # Normalize costs for comparison
    parameters = results_df['parameter'].unique()
    
    # For each parameter, calculate baseline and normalized values
    normalized_data = []
    
    for param in parameters:
        param_data = results_df[results_df['parameter'] == param].copy()
        
        # Find the "middle" or default value as baseline
        if len(param_data) > 0:
            # Use the middle value as baseline
            middle_idx = len(param_data) // 2
            baseline_cost = param_data.iloc[middle_idx]['total_cost']
            baseline_value = param_data.iloc[middle_idx]['value']
            
            # Calculate normalized values
            param_data['normalized_cost'] = param_data['total_cost'] / baseline_cost
            param_data['normalized_value'] = param_data['value'] / baseline_value if baseline_value != 0 else param_data['value']
            
            normalized_data.append(param_data)
    
    if normalized_data:
        normalized_df = pd.concat(normalized_data)
        
        # Create comparative sensitivity plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for param in parameters:
            param_data = normalized_df[normalized_df['parameter'] == param]
            ax.plot(
                param_data['normalized_value'], 
                param_data['normalized_cost'], 
                'o-', 
                label=param
            )
        
        ax.set_xlabel('Normalized Parameter Value')
        ax.set_ylabel('Normalized Cost')
        ax.set_title('Comparative Sensitivity Analysis')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        # Add reference line at y=1
        ax.axhline(y=1, color='k', linestyle='--', alpha=0.5)
        
        # Add reference line at x=1
        ax.axvline(x=1, color='k', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        # Save figure
        filename = os.path.join(output_dir, "comparative_sensitivity.png")
        plt.savefig(filename, dpi=300)
        plt.close()
        figure_paths.append(filename)
        
        # Create heatmap of parameter sensitivities
        sensitivity_data = []
        
        for param in parameters:
            param_data = normalized_df[normalized_df['parameter'] == param]
            
            # Calculate slope of normalized cost vs normalized value
            if len(param_data) >= 2:
                # Use only the points around the baseline (normalized_value = 1)
                param_data = param_data.sort_values('normalized_value')
                
                # Find points closest to baseline
                idx_below = param_data[param_data['normalized_value'] <= 1]['normalized_value'].idxmax() \
                    if not param_data[param_data['normalized_value'] <= 1].empty else None
                idx_above = param_data[param_data['normalized_value'] > 1]['normalized_value'].idxmin() \
                    if not param_data[param_data['normalized_value'] > 1].empty else None
                
                if idx_below is not None and idx_above is not None:
                    point_below = param_data.loc[idx_below]
                    point_above = param_data.loc[idx_above]
                    
                    sensitivity = (point_above['normalized_cost'] - point_below['normalized_cost']) / \
                                 (point_above['normalized_value'] - point_below['normalized_value'])
                    
                    sensitivity_data.append({
                        'parameter': param,
                        'sensitivity': abs(sensitivity)
                    })
        
        if sensitivity_data:
            # Create bar chart of sensitivities
            sensitivity_df = pd.DataFrame(sensitivity_data)
            sensitivity_df = sensitivity_df.sort_values('sensitivity', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(sensitivity_df['parameter'], sensitivity_df['sensitivity'])
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2.,
                    height,
                    f'{height:.2f}',
                    ha='center',
                    va='bottom'
                )
            
            ax.set_xlabel('Parameter')
            ax.set_ylabel('Absolute Sensitivity')
            ax.set_title('Parameter Sensitivity Ranking')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Save figure
            filename = os.path.join(output_dir, "sensitivity_ranking.png")
            plt.savefig(filename, dpi=300)
            plt.close()
            figure_paths.append(filename)
    
    return figure_paths

def _calculate_elasticity(results_df, baseline_result):
    """
    Calculate elasticity metrics for all parameters.
    
    Args:
        results_df: DataFrame with sensitivity results
        baseline_result: Result from baseline optimization
        
    Returns:
        List of elasticity metrics
    """
    if 'total_cost' not in baseline_result:
        return []
        
    baseline_cost = baseline_result['total_cost']
    parameters = results_df['parameter'].unique()
    
    elasticity_metrics = []
    
    for param in parameters:
        param_data = results_df[results_df['parameter'] == param].copy()
        
        # Need at least two data points to calculate elasticity
        if len(param_data) < 2:
            continue
            
        # Find baseline value for this parameter
        # Use the middle value as baseline
        middle_idx = len(param_data) // 2
        baseline_value = param_data.iloc[middle_idx]['value']
        
        # Calculate elasticity for all other points
        for i, row in param_data.iterrows():
            if row['value'] == baseline_value or row['total_cost'] is None:
                continue
                
            # Calculate elasticity: (ΔCost/Cost) / (ΔParam/Param)
            pct_change_cost = (row['total_cost'] - baseline_cost) / baseline_cost
            pct_change_param = (row['value'] - baseline_value) / baseline_value
            
            if pct_change_param != 0:
                elasticity = pct_change_cost / pct_change_param
            else:
                elasticity = None
                
            elasticity_metrics.append({
                'parameter': param,
                'baseline_value': baseline_value,
                'test_value': row['value'],
                'pct_change_param': pct_change_param * 100,  # Convert to percentage
                'baseline_cost': baseline_cost,
                'test_cost': row['total_cost'],
                'pct_change_cost': pct_change_cost * 100,  # Convert to percentage
                'elasticity': elasticity,
                'abs_elasticity': abs(elasticity) if elasticity is not None else None
            })
    
    return elasticity_metrics

def _generate_report(results_df, elasticity_df, baseline_result, base_params, parameter_ranges, output_dir):
    """
    Generate a text report summarizing sensitivity analysis results.
    
    Args:
        results_df: DataFrame with sensitivity results
        elasticity_df: DataFrame with elasticity metrics
        baseline_result: Result from baseline optimization
        base_params: Base parameters used
        parameter_ranges: Parameter ranges tested
        output_dir: Directory to save the report
        
    Returns:
        Path to the report file
    """
    report_path = os.path.join(output_dir, "sensitivity_report.txt")
    
    # Use utf-8 encoding to support all characters, or open in default encoding and use ASCII-safe characters
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=== SENSITIVITY ANALYSIS REPORT ===\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("== BASELINE PARAMETERS ==\n")
        for key, value in base_params.items():
            f.write(f"{key}: {value}\n")
        
        f.write(f"\nBaseline Cost: {baseline_result.get('total_cost', 'N/A')}\n")
        f.write(f"Baseline Status: {baseline_result.get('status', 'N/A')}\n\n")
        
        f.write("== PARAMETER RANGES TESTED ==\n")
        for param, values in parameter_ranges.items():
            f.write(f"{param}: {values}\n")
        
        f.write("\n== PARAMETER SENSITIVITIES ==\n")
        
        # Group elasticity data by parameter
        if not elasticity_df.empty:
            params = elasticity_df['parameter'].unique()
            
            for param in params:
                param_data = elasticity_df[elasticity_df['parameter'] == param]
                avg_elasticity = param_data['abs_elasticity'].mean()
                
                f.write(f"\n{param}:\n")
                f.write(f"  Average Elasticity: {avg_elasticity:.4f}\n")
                
                # Categorize sensitivity
                if avg_elasticity < 0.5:
                    sensitivity = "LOW"
                elif avg_elasticity < 1.0:
                    sensitivity = "MEDIUM"
                else:
                    sensitivity = "HIGH"
                    
                f.write(f"  Sensitivity Level: {sensitivity}\n")
                
                # Show detailed elasticity for each test point
                # Replace the Unicode arrow with ASCII '->' to avoid encoding issues
                f.write("  Detailed Elasticity Metrics:\n")
                for i, row in param_data.iterrows():
                    f.write(f"    {row['baseline_value']} -> {row['test_value']} ({row['pct_change_param']:.1f}%): "
                           f"Cost change: {row['pct_change_cost']:.1f}%, Elasticity: {row['elasticity']:.4f}\n")
        
        # Add recommendations based on elasticity
        f.write("\n== RECOMMENDATIONS ==\n")
        
        if not elasticity_df.empty:
            # Find most sensitive parameters
            param_sensitivity = {}
            for param in elasticity_df['parameter'].unique():
                param_data = elasticity_df[elasticity_df['parameter'] == param]
                param_sensitivity[param] = param_data['abs_elasticity'].mean()
            
            sorted_params = sorted(param_sensitivity.items(), key=lambda x: x[1], reverse=True)
            
            # Provide recommendations for the most sensitive parameters
            for param, sensitivity in sorted_params:
                if sensitivity > 1.0:
                    f.write(f"\n{param} (High Sensitivity - {sensitivity:.4f}):\n")
                    f.write(f"  - This parameter has a significant impact on total cost\n")
                    f.write(f"  - Consider detailed modeling and accurate data collection for this parameter\n")
                    f.write(f"  - Frequent monitoring of this parameter is recommended\n")
                elif sensitivity > 0.5:
                    f.write(f"\n{param} (Medium Sensitivity - {sensitivity:.4f}):\n")
                    f.write(f"  - This parameter has a moderate impact on total cost\n")
                    f.write(f"  - Periodic review of this parameter is recommended\n")
                else:
                    f.write(f"\n{param} (Low Sensitivity - {sensitivity:.4f}):\n")
                    f.write(f"  - This parameter has minimal impact on total cost\n")
                    f.write(f"  - Less frequent monitoring may be acceptable\n")
        
        f.write("\n=== END OF REPORT ===\n")
    
    return report_path

# Command-line interface
if __name__ == "__main__":
    import argparse
    from meio.io.json_loader import NetworkJsonLoader
    
    parser = argparse.ArgumentParser(description='Run sensitivity analysis on MEIO network')
    parser.add_argument('--json', type=str, required=True, help='Path to network JSON file')
    parser.add_argument('--method', type=str, default='improved_heuristic', 
                        choices=['heuristic', 'improved_heuristic', 'solver'],
                        help='Optimization method to use')
    parser.add_argument('--service-levels', type=str, help='Comma-separated list of service levels to test')
    parser.add_argument('--lead-time-factors', type=str, help='Comma-separated list of lead time factors to test')
    parser.add_argument('--demand-factors', type=str, help='Comma-separated list of demand factors to test')
    parser.add_argument('--inflows', type=str, help='Comma-separated list of inflow values to test')
    parser.add_argument('--output-dir', type=str, help='Output directory')
    parser.add_argument('--no-viz', action='store_true', help='Skip visualizations')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load network
    logger.info(f"Loading network from {args.json}")
    network = NetworkJsonLoader.load(args.json)
    
    # Setup parameter ranges
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
    
    if args.inflows:
        parameter_ranges['inflows'] = [float(x) for x in args.inflows.split(',')]
    
    # Run sensitivity analysis
    result = run_sensitivity_analysis(
        network,
        parameter_ranges,
        method=args.method,
        output_dir=args.output_dir,
        visualize=not args.no_viz
    )
    
    logger.info(f"Sensitivity analysis complete.")
    logger.info(f"Results saved to {result['output_dir']}")
    if 'report_path' in result:
        logger.info(f"Report saved to {result['report_path']}") 