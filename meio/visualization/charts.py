"""
Chart visualizations for the MEIO system.
"""
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from ..config.settings import config

logger = logging.getLogger(__name__)

class ChartVisualizer:
    """Creates visualizations of inventory metrics."""
    
    @staticmethod
    def plot_comparison_chart(network, solver_results, heuristic_results, show=True, save_path=None):
        """
        Create a bar chart comparing solver and heuristic results.
        
        Args:
            network (MultiEchelonNetwork): The network.
            solver_results (dict): Results from mathematical solver.
            heuristic_results (dict): Results from heuristic solver.
            show (bool, optional): Whether to display the plot. Defaults to True.
            save_path (str, optional): Path to save the chart. Defaults to None.
            
        Returns:
            tuple: Figure and axes objects.
        """
        # Get configuration
        figsize = config.get('visualization', 'default_figsize')
        
        # Extract data
        nodes = list(network.nodes.keys())
        solver_totals = []
        heuristic_totals = []
        
        for node_id in nodes:
            solver_total = sum(solver_results['inventory_levels'].get((node_id, prod, t), 0)
                             for prod in network.nodes[node_id].products
                             for t in range(network.num_periods)) if solver_results['status'] == 'optimal' else 0
                             
            heuristic_total = sum(heuristic_results['inventory_levels'].get((node_id, prod, t), 0)
                                for prod in network.nodes[node_id].products
                                for t in range(network.num_periods))
                                
            solver_totals.append(solver_total)
            heuristic_totals.append(heuristic_total)
        
        # Create bar chart
        x = np.arange(len(nodes))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(x - width/2, solver_totals, width, label='Solver', color='skyblue')
        ax.bar(x + width/2, heuristic_totals, width, label='Heuristic', color='lightgreen')
        
        # Add labels and title
        ax.set_ylabel('Total Inventory (units)')
        ax.set_title('Inventory Levels: Solver vs Heuristic')
        ax.set_xticks(x)
        ax.set_xticklabels(nodes, rotation=45)
        ax.legend()
        
        # Add value labels on bars
        for i, v in enumerate(solver_totals):
            ax.text(i - width/2, v + 50, f"{v:.0f}", ha='center')
        for i, v in enumerate(heuristic_totals):
            ax.text(i + width/2, v + 50, f"{v:.0f}", ha='center')
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Comparison chart saved to {save_path}")
        
        # Show if requested
        if show:
            plt.show()
        
        return fig, ax
    
    @staticmethod
    def plot_node_metrics(network, inventory_levels, receptions, node_id=None, show=True, save_path=None):
        """
        Plot metrics for a specific node or all nodes.
        
        Args:
            network (MultiEchelonNetwork): The network.
            inventory_levels (dict): Inventory levels.
            receptions (dict): Inventory receptions.
            node_id (str, optional): Node to visualize. Plots all if None. Defaults to None.
            show (bool, optional): Whether to display the plot. Defaults to True.
            save_path (str, optional): Path to save the chart. Defaults to None.
            
        Returns:
            dict: Mapping of node_id to (fig, ax) tuples.
        """
        # Get configuration
        figsize = config.get('visualization', 'default_figsize')
        
        # Determine which nodes to plot
        nodes_to_plot = [node_id] if node_id else network.nodes.keys()
        result_figs = {}
        
        for node_id in nodes_to_plot:
            node = network.nodes[node_id]
            
            # Create figure
            fig, ax = plt.subplots(figsize=figsize)
            
            # Format x-axis with dates
            date_format = mdates.DateFormatter('%Y-%m-%d')
            ax.xaxis.set_major_formatter(date_format)
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=network.date_interval))
            plt.xticks(rotation=45)
            
            # Plot data for each product
            for prod in node.products:
                # Plot demand
                demand = node.products[prod]['demand_by_date']
                ax.plot(network.dates, demand, label=f"{prod} Demand", 
                       linestyle='-', marker='o', color='blue')
                
                # Plot inventory
                inv_levels = [inventory_levels.get((node_id, prod, t), 0) 
                             for t in range(network.num_periods)]
                ax.plot(network.dates, inv_levels, label=f"{prod} Inventory", 
                       linestyle='--', marker='s', color='green')
                
                # Plot safety stock
                safety_levels = node.products[prod]['safety_stock_by_date']
                ax.plot(network.dates, safety_levels, label=f"{prod} Safety Stock", 
                       linestyle='-.', marker='^', color='red')
                
                # Plot receptions
                recep_levels = receptions[node_id][prod]
                ax.plot(network.dates, recep_levels, label=f"{prod} Receptions", 
                       linestyle=':', marker='d', color='purple')
            
            # Add labels and title
            ax.set_title(f"{node_id} - Supply Chain Metrics")
            ax.set_xlabel("Date")
            ax.set_ylabel("Units")
            ax.legend()
            ax.grid(True)
            
            plt.tight_layout()
            
            # Save if requested
            if save_path:
                node_save_path = f"{save_path}_{node_id}.png"
                plt.savefig(node_save_path, bbox_inches='tight', dpi=300)
                logger.info(f"Node metrics chart for {node_id} saved to {node_save_path}")
            
            # Store figure and axes
            result_figs[node_id] = (fig, ax)
        
        # Show if requested
        if show:
            plt.show()
        
        return result_figs