"""
Branch Selection Visualization Module.

This module provides visualization capabilities for branch selection results.
"""
import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import seaborn as sns
from typing import Dict, List, Any, Optional

from ..utils.path_manager import paths

logger = logging.getLogger(__name__)

class BranchVisualizer:
    """Visualizes branch selection results."""
    
    @staticmethod
    def visualize_branch_comparison(
        results: Dict[str, Any],
        save_path: Optional[str] = None,
        show_plot: bool = False
    ) -> str:
        """
        Create a visualization comparing different branches.
        
        Args:
            results: Results from BranchManager.run_branch_selection()
            save_path: Path to save the visualization (default: auto-generated)
            show_plot: Whether to display the plot
            
        Returns:
            str: Path to the saved visualization
        """
        branches = results['branch_results']['branches']
        scores = results['evaluation_results']['scores']
        selected_branch = results['selection_results']['selected_branch']
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Filter out error branches
        valid_branches = {k: v for k, v in branches.items() if v.get('status') != 'error'}
        
        # Extract data for plotting
        branch_ids = list(valid_branches.keys())
        costs = [branches[b].get('total_cost', 0) for b in branch_ids]
        service_levels = [scores[b]['service_level_score'] for b in branch_ids]
        robustness = [scores[b]['robustness_score'] for b in branch_ids]
        overall_scores = [scores[b]['overall_score'] for b in branch_ids]
        
        # Determine if a branch is selected
        is_selected = [b == selected_branch for b in branch_ids]
        
        # Create color scheme based on selection
        colors = ['#1f77b4' if not sel else '#d62728' for sel in is_selected]
        
        # Plot data
        x = np.arange(len(branch_ids))
        width = 0.2
        
        # Create bar plot
        plt.bar(x - width*1.5, costs, width, label='Cost', color='#1f77b4', alpha=0.7)
        plt.bar(x - width/2, service_levels, width, label='Service Level', color='#ff7f0e', alpha=0.7)
        plt.bar(x + width/2, robustness, width, label='Robustness', color='#2ca02c', alpha=0.7)
        plt.bar(x + width*1.5, overall_scores, width, label='Overall Score', color='#9467bd', alpha=0.7)
        
        # Add indicator for selected branch
        for i in range(len(branch_ids)):
            if is_selected[i]:
                plt.scatter(i, 1.05, marker='v', s=100, color='red')
                plt.text(i, 1.08, 'Selected', ha='center', color='red')
        
        # Customize plot
        plt.xlabel('Branch ID')
        plt.ylabel('Score')
        plt.title('Branch Comparison')
        plt.xticks(x, branch_ids, rotation=45)
        plt.ylim(0, 1.15)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add parameters for each branch
        param_text = ''
        for i, branch_id in enumerate(branch_ids):
            params = branches[branch_id]['params']
            param_str = f"{branch_id}: "
            
            # Get service level with type checking
            service_level = params.get('service_level', 'N/A')
            if isinstance(service_level, (int, float)):
                param_str += f"SL={service_level:.2f}"
            else:
                param_str += f"SL={service_level}"
                
            # Get inflows with type checking
            if 'inflows' in params:
                inflows = params['inflows']
                if isinstance(inflows, (int, float)):
                    param_str += f", Inflows={inflows:.2f}"
                else:
                    param_str += f", Inflows={inflows}"
                    
            # Get lead time factor with type checking
            if 'lead_time_factor' in params:
                lt_factor = params['lead_time_factor']
                if isinstance(lt_factor, (int, float)):
                    param_str += f", LT×{lt_factor:.1f}"
                else:
                    param_str += f", LT×{lt_factor}"
                    
            param_text += param_str + '\n'
        
        plt.figtext(0.02, 0.02, param_text, fontsize=9)
        
        # Add selection rationale
        rationale = results['selection_results']['rationale']
        plt.figtext(0.5, 0.02, rationale, fontsize=9, ha='center')
        
        plt.tight_layout(rect=[0, 0.1, 1, 0.98])
        
        # Save figure
        if save_path is None:
            save_path = paths.get_visualization_path('branch_comparison.png')
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Branch comparison visualization saved to {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
            
        return save_path
    
    @staticmethod
    def visualize_scenario_performance(
        results: Dict[str, Any],
        save_path: Optional[str] = None,
        show_plot: bool = False
    ) -> str:
        """
        Create a visualization showing branch performance across scenarios.
        
        Args:
            results: Results from BranchManager.run_branch_selection()
            save_path: Path to save the visualization (default: auto-generated)
            show_plot: Whether to display the plot
            
        Returns:
            str: Path to the saved visualization
        """
        scenario_results = results['evaluation_results']['scenario_results']
        selected_branch = results['selection_results']['selected_branch']
        
        # Extract data
        data = []
        for scenario_name, branches in scenario_results.items():
            for branch_id, metrics in branches.items():
                data.append({
                    'scenario': scenario_name,
                    'branch_id': branch_id,
                    'service_level': metrics.get('service_level', 0),
                    'cost': metrics.get('total_cost', 0),
                    'robustness': metrics.get('robustness_score', 0),
                    'is_selected': branch_id == selected_branch
                })
        
        df = pd.DataFrame(data)
        
        # Create figure
        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])
        
        # Define color palette
        palette = sns.color_palette("deep", len(df['branch_id'].unique()))
        branch_colors = {b: c for b, c in zip(sorted(df['branch_id'].unique()), palette)}
        
        # Highlight selected branch
        for i, row in df.iterrows():
            if row['is_selected']:
                branch_colors[row['branch_id']] = 'red'
        
        # Plot 1: Service Level by Scenario
        ax1 = plt.subplot(gs[0, 0])
        sns.barplot(x='scenario', y='service_level', hue='branch_id', data=df, ax=ax1, palette=branch_colors)
        ax1.set_title('Service Level by Scenario')
        ax1.set_ylim(0, 1)
        ax1.set_ylabel('Service Level (higher is better)')
        ax1.set_xlabel('Scenario')
        ax1.legend(title='Branch ID')
        
        # Plot 2: Cost by Scenario
        ax2 = plt.subplot(gs[0, 1])
        costs = df.pivot(index='scenario', columns='branch_id', values='cost')
        costs.plot(kind='bar', ax=ax2, color=branch_colors)
        ax2.set_title('Cost by Scenario')
        ax2.set_ylabel('Total Cost (lower is better)')
        ax2.set_xlabel('Scenario')
        ax2.legend(title='Branch ID')
        
        # Plot 3: Robustness by Scenario
        ax3 = plt.subplot(gs[1, 0])
        sns.barplot(x='scenario', y='robustness', hue='branch_id', data=df, ax=ax3, palette=branch_colors)
        ax3.set_title('Robustness by Scenario')
        ax3.set_ylim(0, 1)
        ax3.set_ylabel('Robustness Score (higher is better)')
        ax3.set_xlabel('Scenario')
        ax3.legend(title='Branch ID')
        
        # Plot 4: Radar chart of branch performance
        ax4 = plt.subplot(gs[1, 1], polar=True)
        
        # Aggregate metrics by branch
        branch_metrics = df.groupby('branch_id').agg({
            'service_level': 'mean',
            'robustness': 'mean',
            'cost': 'mean',
            'is_selected': 'first'
        })
        
        # Normalize cost for radar chart (lower is better, so invert)
        max_cost = branch_metrics['cost'].max()
        branch_metrics['cost_normalized'] = 1 - (branch_metrics['cost'] / max_cost) if max_cost > 0 else branch_metrics['cost'].apply(lambda x: 0 if not isinstance(x, (int, float)) or x <= 0 else 1 - (x / max_cost))
        
        # Setup radar chart
        categories = ['Service Level', 'Robustness', 'Cost Efficiency']
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Plot each branch
        for branch_id, metrics in branch_metrics.iterrows():
            values = [metrics['service_level'], metrics['robustness'], metrics['cost_normalized']]
            values += values[:1]  # Close the loop
            
            color = branch_colors[branch_id]
            linewidth = 2 if metrics['is_selected'] else 1
            alpha = 1 if metrics['is_selected'] else 0.7
            
            ax4.plot(angles, values, 'o-', linewidth=linewidth, color=color, alpha=alpha, label=branch_id)
            ax4.fill(angles, values, color=color, alpha=0.1)
        
        # Customize radar chart
        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels(categories)
        ax4.set_yticks([0.25, 0.5, 0.75])
        ax4.set_yticklabels(['0.25', '0.5', '0.75'])
        ax4.set_ylim(0, 1)
        ax4.set_title('Branch Performance Overview')
        ax4.legend(title='Branch ID', loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        
        # Save figure
        if save_path is None:
            save_path = paths.get_visualization_path('scenario_performance.png')
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Scenario performance visualization saved to {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
            
        return save_path
    
    @staticmethod
    def visualize_branch_selection_summary(
        results: Dict[str, Any],
        save_path: Optional[str] = None,
        show_plot: bool = False
    ) -> Dict[str, str]:
        """
        Create a summary visualization of the branch selection process.
        
        Args:
            results: Results from BranchManager.run_branch_selection()
            save_path: Path to save the visualization (default: auto-generated)
            show_plot: Whether to display the plot
            
        Returns:
            Dict[str, str]: Paths to all generated visualizations
        """
        paths = {}
        
        # Create branch comparison visualization
        comparison_path = BranchVisualizer.visualize_branch_comparison(
            results, 
            save_path=None if save_path is None else f"{save_path}_comparison.png",
            show_plot=show_plot
        )
        paths['comparison'] = comparison_path
        
        # Create scenario performance visualization
        scenario_path = BranchVisualizer.visualize_scenario_performance(
            results,
            save_path=None if save_path is None else f"{save_path}_scenarios.png",
            show_plot=show_plot
        )
        paths['scenarios'] = scenario_path
        
        # Create selected branch details visualization
        selected_branch = results['selection_results']['selected_branch']
        if selected_branch:
            branch_data = results['branch_results']['branches'][selected_branch]
            selection_path = BranchVisualizer._visualize_selected_branch(
                results,
                branch_data,
                save_path=None if save_path is None else f"{save_path}_selected.png",
                show_plot=show_plot
            )
            paths['selected'] = selection_path
        
        return paths
    
    @staticmethod
    def _visualize_selected_branch(
        results: Dict[str, Any],
        branch_data: Dict[str, Any],
        save_path: Optional[str] = None,
        show_plot: bool = False
    ) -> str:
        """
        Create a detailed visualization of the selected branch.
        
        Args:
            results: Results from BranchManager.run_branch_selection()
            branch_data: Data for the selected branch
            save_path: Path to save the visualization (default: auto-generated)
            show_plot: Whether to display the plot
            
        Returns:
            str: Path to the saved visualization
        """
        branch_id = branch_data['branch_id']
        params = branch_data['params']
        
        # Create figure
        fig = plt.figure(figsize=(12, 8))
        
        # Create summary text
        summary = f"Branch: {branch_id}\n\nParameters:\n"
        
        for param, value in params.items():
            # Add type checking for parameter values
            if isinstance(value, (int, float)):
                if param == 'service_level':
                    summary += f"- {param}: {value:.3f}\n"
                elif abs(value) < 0.01 or abs(value) >= 1000:
                    summary += f"- {param}: {value:.2e}\n"
                else:
                    summary += f"- {param}: {value:.2f}\n"
            else:
                summary += f"- {param}: {value}\n"
        
        inventory_levels = branch_data.get('inventory_levels', {})
        if inventory_levels:
            # Add some inventory statistics if available
            total_inv = sum(level for (_, _, _), level in inventory_levels.items())
            if isinstance(total_inv, (int, float)):
                summary += f"\nTotal inventory: {total_inv:.2f}\n"
            else:
                summary += f"\nTotal inventory: {total_inv}\n"
                
            total_cost = branch_data.get('total_cost', 0)
            if isinstance(total_cost, (int, float)):
                summary += f"Total cost: {total_cost:.2f}\n"
            else:
                summary += f"Total cost: {total_cost}\n"
        
        # Add selection rationale
        rationale = results['selection_results']['rationale']
        summary += f"\nSelection Rationale:\n{rationale}\n"
        
        # Plot text
        plt.axis('off')
        plt.text(0.1, 0.5, summary, fontsize=12, verticalalignment='center', wrap=True)
        
        plt.title(f"Selected Branch: {branch_id}")
        plt.tight_layout()
        
        # Save figure
        if save_path is None:
            save_path = paths.get_visualization_path(f'selected_branch_{branch_id}.png')
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Selected branch visualization saved to {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
            
        return save_path 