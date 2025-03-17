#!/usr/bin/env python
"""
Demonstration script for the Parameter Evolution AI Agent.

This script demonstrates how to use the Parameter Evolution AI Agent 
directly to optimize inventory parameters based on historical performance.
"""
import sys
import logging
import os
import argparse
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Parameter Evolution AI Agent Demo')
    parser.add_argument('json_path', type=str, help='Path to network JSON file')
    parser.add_argument('--import-dir', type=str, help='Directory containing historical run data')
    parser.add_argument('--output-dir', type=str, help='Directory to save results')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    
    return parser.parse_args()

def visualize_parameter_trends(historical_data, output_path):
    """
    Visualize parameter trends from historical data.
    
    Args:
        historical_data: List of historical run data
        output_path: Directory to save visualizations
    
    Returns:
        Dict with paths to generated visualizations
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.dates import DateFormatter
    
    # Sort data by timestamp
    sorted_data = sorted(historical_data, key=lambda x: x.get('timestamp', ''))
    
    if not sorted_data:
        return {}
    
    # Extract timestamps and parameters
    timestamps = []
    params_data = {}
    metrics_data = {}
    
    for entry in sorted_data:
        try:
            # Convert timestamp to datetime
            dt = datetime.strptime(entry.get('timestamp', ''), '%Y-%m-%d %H:%M:%S')
            timestamps.append(dt)
            
            # Extract parameters
            for param, value in entry.get('parameters', {}).items():
                if isinstance(value, (int, float)):
                    if param not in params_data:
                        params_data[param] = []
                    params_data[param].append(value)
            
            # Extract metrics
            for metric, value in entry.get('metrics', {}).items():
                if isinstance(value, (int, float)):
                    if metric not in metrics_data:
                        metrics_data[metric] = []
                    metrics_data[metric].append(value)
        except Exception as e:
            logging.warning(f"Error processing entry: {str(e)}")
    
    viz_paths = {}
    
    # Plot parameter trends
    if params_data and len(timestamps) > 1:
        fig, axes = plt.subplots(len(params_data), 1, figsize=(10, 3*len(params_data)))
        
        if len(params_data) == 1:
            axes = [axes]
        
        for i, (param, values) in enumerate(params_data.items()):
            if len(values) == len(timestamps):
                axes[i].plot(timestamps, values, 'o-', label=param)
                axes[i].set_ylabel(param)
                axes[i].grid(True)
                
                # Draw trend line
                if len(timestamps) > 2:
                    z = np.polyfit(np.arange(len(timestamps)), values, 1)
                    p = np.poly1d(z)
                    axes[i].plot(timestamps, p(np.arange(len(timestamps))), "r--", 
                                 label=f"Trend: {z[0]:.4f}x + {z[1]:.2f}")
                
                axes[i].legend()
        
        axes[-1].set_xlabel('Date')
        plt.tight_layout()
        
        # Save figure
        params_path = os.path.join(output_path, 'parameter_trends.png')
        plt.savefig(params_path)
        viz_paths['parameter_trends'] = params_path
        plt.close()
    
    # Plot metrics trends
    if metrics_data and len(timestamps) > 1:
        fig, axes = plt.subplots(len(metrics_data), 1, figsize=(10, 3*len(metrics_data)))
        
        if len(metrics_data) == 1:
            axes = [axes]
        
        for i, (metric, values) in enumerate(metrics_data.items()):
            if len(values) == len(timestamps):
                axes[i].plot(timestamps, values, 'o-', label=metric)
                axes[i].set_ylabel(metric)
                axes[i].grid(True)
                
                # Draw trend line
                if len(timestamps) > 2:
                    z = np.polyfit(np.arange(len(timestamps)), values, 1)
                    p = np.poly1d(z)
                    axes[i].plot(timestamps, p(np.arange(len(timestamps))), "r--", 
                                 label=f"Trend: {z[0]:.4f}x + {z[1]:.2f}")
                
                axes[i].legend()
        
        axes[-1].set_xlabel('Date')
        plt.tight_layout()
        
        # Save figure
        metrics_path = os.path.join(output_path, 'metrics_trends.png')
        plt.savefig(metrics_path)
        viz_paths['metrics_trends'] = metrics_path
        plt.close()
    
    # Plot parameter correlation with overall score
    if params_data and 'overall_score' in metrics_data and len(timestamps) > 1:
        overall_scores = metrics_data['overall_score']
        
        fig, axes = plt.subplots(len(params_data), 1, figsize=(10, 3*len(params_data)))
        
        if len(params_data) == 1:
            axes = [axes]
        
        for i, (param, values) in enumerate(params_data.items()):
            if len(values) == len(overall_scores):
                axes[i].scatter(values, overall_scores)
                axes[i].set_xlabel(param)
                axes[i].set_ylabel('Overall Score')
                axes[i].grid(True)
                
                # Draw trend line
                if len(values) > 2:
                    z = np.polyfit(values, overall_scores, 1)
                    p = np.poly1d(z)
                    x_range = np.linspace(min(values), max(values), 100)
                    axes[i].plot(x_range, p(x_range), "r--", 
                                 label=f"Correlation: {z[0]:.4f}x + {z[1]:.2f}")
                
                axes[i].legend()
        
        plt.tight_layout()
        
        # Save figure
        correlation_path = os.path.join(output_path, 'parameter_correlations.png')
        plt.savefig(correlation_path)
        viz_paths['parameter_correlations'] = correlation_path
        plt.close()
    
    return viz_paths

def main():
    """Main demonstration function."""
    # Import required modules
    from meio.io.json_loader import NetworkJsonLoader
    from meio.optimization.branch_selection import ParameterEvolutionAgent
    
    # Parse command line arguments
    args = parse_args()
    
    json_path = args.json_path
    if not os.path.exists(json_path):
        print(f"Error: JSON file not found: {json_path}")
        return 1
    
    # Set output directory
    output_dir = args.output_dir or os.path.join('results', 'ai_agent_demo')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load network
    try:
        logging.info(f"Loading network from {json_path}")
        network = NetworkJsonLoader.load(json_path)
        logging.info(f"Loaded network with {len(network.nodes)} nodes")
    except Exception as e:
        logging.error(f"Failed to load network: {str(e)}")
        return 1
    
    # Initialize the AI agent
    try:
        logging.info("Initializing Parameter Evolution AI Agent")
        agent = ParameterEvolutionAgent(db_path=os.path.join(output_dir, 'parameter_evolution.json'))
        
        # Import historical data if provided
        if args.import_dir and os.path.exists(args.import_dir):
            logging.info(f"Importing historical data from {args.import_dir}")
            
            # Load historical data
            historical_runs = []
            for filename in os.listdir(args.import_dir):
                if filename.endswith('.json'):
                    try:
                        with open(os.path.join(args.import_dir, filename), 'r') as f:
                            data = json.load(f)
                            if isinstance(data, list):
                                historical_runs.extend(data)
                            else:
                                historical_runs.append(data)
                    except Exception as e:
                        logging.warning(f"Error loading {filename}: {str(e)}")
            
            # Learn from historical data
            learn_result = agent.learn_optimal_parameters(historical_runs)
            logging.info(f"Learning result: {learn_result.get('message', 'Unknown')}")
            
            # Display statistics
            if 'statistics' in learn_result:
                logging.info("\nParameter Statistics:")
                for param, stats in learn_result['statistics'].items():
                    logging.info(f"  {param}:")
                    logging.info(f"    Count: {stats.get('count', 0)}")
                    logging.info(f"    Range: {stats.get('min_value', 'N/A')} - {stats.get('max_value', 'N/A')}")
                    logging.info(f"    Best value: {stats.get('best_value', 'N/A')}")
        
        # Get performance trend
        trend_data = agent.get_performance_trend()
        if trend_data.get("status") == "success":
            logging.info("\nAI Agent Performance Trends:")
            logging.info(f"Data points: {trend_data.get('data_points', 0)}")
            logging.info(f"Overall trend: {trend_data.get('overall_score_trend', 'unknown')}")
            logging.info(f"Cost trend: {trend_data.get('cost_score_trend', 'unknown')}")
            logging.info(f"Service level trend: {trend_data.get('service_score_trend', 'unknown')}")
            logging.info(f"Robustness trend: {trend_data.get('robustness_score_trend', 'unknown')}")
        else:
            logging.info(f"AI trend data: {trend_data.get('message', 'Insufficient data')}")
        
        # Generate parameter suggestions
        logging.info("\nGenerating parameter suggestions for current network")
        suggestion = agent.suggest_parameters(network)
        
        logging.info(f"Suggestion confidence: {suggestion.get('confidence', 'unknown')}")
        logging.info(f"Based on {suggestion.get('similar_networks', 0)} similar networks")
        logging.info(f"Rationale: {suggestion.get('rationale', 'Not available')}")
        
        logging.info("\nSuggested Parameters:")
        for param, value in suggestion.get('parameters', {}).items():
            logging.info(f"  {param}: {value}")
        
        # Generate visualizations if requested
        if args.visualize:
            logging.info("\nGenerating visualizations")
            viz_paths = visualize_parameter_trends(agent.historical_data, output_dir)
            
            logging.info("Visualization paths:")
            for viz_type, path in viz_paths.items():
                logging.info(f"  {viz_type}: {path}")
        
        # Save suggestion to file
        suggestion_path = os.path.join(output_dir, 'latest_suggestion.json')
        with open(suggestion_path, 'w') as f:
            json.dump(suggestion, f, indent=2)
        logging.info(f"\nSuggestion saved to {suggestion_path}")
        
        return 0
        
    except Exception as e:
        logging.error(f"Error in Parameter Evolution AI Agent demo: {str(e)}")
        import traceback
        logging.debug(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main()) 