#!/usr/bin/env python
"""
Command-line script to run sensitivity analysis on a MEIO network.
"""
import sys
import argparse
import logging
from datetime import datetime

from meio.io.json_loader import NetworkJsonLoader
from meio.analysis.sensitivity import run_sensitivity_analysis

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run sensitivity analysis on MEIO network')
    parser.add_argument('--json', type=str, required=True, 
                        help='Path to network JSON file')
    
    # Optimization method
    parser.add_argument('--method', type=str, default='improved_heuristic', 
                        choices=['heuristic', 'improved_heuristic', 'solver'],
                        help='Optimization method to use')
    
    # Parameters to vary
    parser.add_argument('--service-levels', type=str, 
                        help='Comma-separated list of service levels to test (e.g., 0.90,0.95,0.98)')
    parser.add_argument('--lead-time-factors', type=str, 
                        help='Comma-separated list of lead time factors to test (e.g., 0.8,1.0,1.2)')
    parser.add_argument('--demand-factors', type=str, 
                        help='Comma-separated list of demand factors to test (e.g., 0.8,1.0,1.2)')
    parser.add_argument('--inflows', type=str, 
                        help='Comma-separated list of inflow values to test (e.g., 0.8,1.0,1.2)')
    
    # Output options
    parser.add_argument('--output-dir', type=str, 
                        help='Output directory (defaults to results/sensitivity_[timestamp])')
    parser.add_argument('--no-viz', action='store_true', 
                        help='Skip generating visualizations')
    
    # Base parameters
    parser.add_argument('--base-service-level', type=float, 
                        help='Base service level for comparison')
    parser.add_argument('--base-inflows', type=float, 
                        help='Base inflow level for comparison')
    
    # Logging
    parser.add_argument('--verbose', '-v', action='store_true', 
                        help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting sensitivity analysis")
    
    # Load network
    try:
        logger.info(f"Loading network from {args.json}")
        network = NetworkJsonLoader.load(args.json)
        logger.info(f"Loaded network with {len(network.nodes)} nodes and {network.num_periods} periods")
    except Exception as e:
        logger.error(f"Failed to load network: {str(e)}")
        return 1
    
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
    
    # Setup base parameters
    base_params = {}
    if args.base_service_level:
        base_params['service_level'] = args.base_service_level
    if args.base_inflows:
        base_params['inflows'] = args.base_inflows
    
    # Run sensitivity analysis
    try:
        logger.info(f"Running sensitivity analysis with parameters: {parameter_ranges}")
        
        result = run_sensitivity_analysis(
            network,
            parameter_ranges,
            base_params=base_params if base_params else None,
            method=args.method,
            output_dir=args.output_dir,
            visualize=not args.no_viz
        )
        
        logger.info(f"Sensitivity analysis complete")
        logger.info(f"Results saved to {result['output_dir']}")
        logger.info(f"Total execution time: {result['execution_time']:.2f} seconds")
        
        # Print key insights
        if 'elasticity_metrics' in result and result['elasticity_metrics']:
            param_sensitivity = {}
            for metric in result['elasticity_metrics']:
                param = metric['parameter']
                elasticity = metric.get('abs_elasticity')
                if elasticity is not None:
                    if param not in param_sensitivity:
                        param_sensitivity[param] = []
                    param_sensitivity[param].append(elasticity)
            
            logger.info("Key insights from sensitivity analysis:")
            for param, values in param_sensitivity.items():
                avg_elasticity = sum(values) / len(values)
                
                if avg_elasticity > 1.0:
                    sensitivity = "HIGH"
                elif avg_elasticity > 0.5:
                    sensitivity = "MEDIUM"
                else:
                    sensitivity = "LOW"
                
                logger.info(f"  {param}: {sensitivity} sensitivity (elasticity = {avg_elasticity:.4f})")
            
            logger.info(f"Detailed report available at: {result['report_path']}")
        
    except Exception as e:
        logger.error(f"Error during sensitivity analysis: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 