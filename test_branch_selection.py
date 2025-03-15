#!/usr/bin/env python
"""
Test script for branch selection functionality.
"""
import sys
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Main test function."""
    # Import required modules
    from meio.io.json_loader import NetworkJsonLoader
    from meio.optimization.branch_selection import BranchManager
    from meio.visualization.branch_viz import BranchVisualizer
    
    # Check if JSON file is provided
    if len(sys.argv) < 2:
        print("Usage: python test_branch_selection.py <path_to_network_json>")
        return 1
    
    json_path = sys.argv[1]
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
        # Initialize branch manager
        branch_manager = BranchManager()
        
        # Run branch selection with default parameters
        results = branch_manager.run_branch_selection(
            network,
            num_branches=3,
            criteria=['cost', 'service_level', 'robustness'],
            selection_criteria='balanced',
            method='heuristic'
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