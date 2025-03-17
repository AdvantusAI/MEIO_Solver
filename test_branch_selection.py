#!/usr/bin/env python
"""
Test script for branch selection functionality.
"""
import sys
import logging
import os
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test Branch Selection Functionality')
    parser.add_argument('json_path', type=str, help='Path to network JSON file')
    parser.add_argument('--num-branches', type=int, default=3, help='Number of branches to generate')
    parser.add_argument('--method', type=str, default='heuristic', 
                        choices=['heuristic', 'improved_heuristic', 'solver'],
                        help='Optimization method to use')
    parser.add_argument('--selection-criteria', type=str, default='balanced',
                        choices=['balanced', 'cost_focused', 'service_focused', 'robust'],
                        help='Strategy for selecting the best branch')
    parser.add_argument('--use-ai-agent', action='store_true', 
                        help='Use AI Parameter Evolution Agent')
    parser.add_argument('--import-history', type=str, 
                        help='Import historical data for AI agent from file or directory')
    parser.add_argument('--output-dir', type=str, help='Output directory for results')
    
    return parser.parse_args()

def main():
    """Main test function."""
    # Import required modules
    from meio.io.json_loader import NetworkJsonLoader
    from meio.optimization.branch_selection import BranchManager
    from meio.visualization.branch_viz import BranchVisualizer
    
    # Parse command line arguments
    args = parse_args()
    
    json_path = args.json_path
    if not os.path.exists(json_path):
        print(f"Error: JSON file not found: {json_path}")
        return 1
    
    # Load network
    try:
        logging.info(f"Loading network from {json_path}")
        network = NetworkJsonLoader.load(json_path)
        logging.info(f"Loaded network: {network}")
    except Exception as e:
        logging.error(f"Failed to load network: {str(e)}")
        return 1
    
    # Run branch selection
    try:
        # Initialize branch manager with AI agent if requested
        branch_manager = BranchManager(
            output_dir=args.output_dir, 
            use_ai_agent=args.use_ai_agent
        )
        
        # Import historical data if provided
        if args.use_ai_agent and args.import_history:
            import_result = branch_manager.import_historical_data(args.import_history)
            logging.info(f"Import result: {import_result.get('message', 'Unknown')}")
            
            # Show AI performance trends
            trend_data = branch_manager.get_ai_agent_trend()
            if trend_data.get("status") == "success":
                logging.info("\nAI Agent Performance Trends:")
                logging.info(f"Data points: {trend_data.get('data_points', 0)}")
                logging.info(f"Overall trend: {trend_data.get('overall_score_trend', 'unknown')}")
            else:
                logging.info(f"AI trend data: {trend_data.get('message', 'Insufficient data')}")
        
        # Run branch selection with parameters from command line
        results = branch_manager.run_branch_selection(
            network,
            num_branches=args.num_branches,
            criteria=['cost', 'service_level', 'robustness'],
            selection_criteria=args.selection_criteria,
            method=args.method
        )
        
        # Create visualizations
        viz_paths = BranchVisualizer.visualize_branch_selection_summary(results)
        
        # Log selected branch information
        selected_branch = results['selection_results']['selected_branch']
        if selected_branch:
            selected_data = results['branch_results']['branches'][selected_branch]
            logging.info(f"\nSelected Branch: {selected_branch}")
            logging.info(f"Branch Parameters:")
            for k, v in selected_data['params'].items():
                logging.info(f"  {k}: {v}")
            
            logging.info(f"Selection Rationale: {results['selection_results']['rationale']}")
            
            # AI agent information
            if args.use_ai_agent and 'ai_suggestion' in results:
                ai_suggestion = results['ai_suggestion']
                if ai_suggestion:
                    logging.info("\nAI Agent Information:")
                    logging.info(f"Confidence: {ai_suggestion.get('confidence', 'unknown')}")
                    logging.info(f"Data points: {ai_suggestion.get('data_points', 0)}")
                    logging.info(f"Similar networks: {ai_suggestion.get('similar_networks', 0)}")
                    logging.info(f"Rationale: {ai_suggestion.get('rationale', 'Not available')}")
                    
                    # Compare AI suggestion with final selection
                    ai_params = ai_suggestion.get('parameters', {})
                    selected_params = selected_data.get('params', {})
                    
                    logging.info("\nParameter Comparison (AI vs Selected):")
                    common_params = set(ai_params.keys()) & set(selected_params.keys())
                    for param in common_params:
                        if isinstance(ai_params[param], (int, float)) and isinstance(selected_params[param], (int, float)):
                            # Safe to perform numeric comparison and formatting
                            diff = ((selected_params[param] - ai_params[param]) / ai_params[param]) * 100 if ai_params[param] != 0 else 0
                            logging.info(f"  {param}: {ai_params[param]} → {selected_params[param]} ({diff:+.1f}%)")
                        else:
                            # For non-numeric parameters, just show the values without percentage
                            logging.info(f"  {param}: {ai_params[param]} → {selected_params[param]}")
            
            logging.info(f"\nBranch selection results saved to {results['output_dir']}")
            
            # Log visualization paths
            logging.info("Visualization paths:")
            for viz_type, path in viz_paths.items():
                logging.info(f"  {viz_type}: {path}")
        else:
            logging.warning("No valid branch could be selected")
        
        return 0
        
    except Exception as e:
        logging.error(f"Error during branch selection test: {str(e)}")
        import traceback
        logging.debug(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main()) 