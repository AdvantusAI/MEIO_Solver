"""
Network visualization for the MEIO system.
"""
import logging
import matplotlib.pyplot as plt
import networkx as nx
from ..config.settings import config

logger = logging.getLogger(__name__)

class NetworkVisualizer:
    """Visualizes the supply chain network structure."""
    
    @staticmethod
    def visualize_network(network, inventory_levels, date_idx=0, show=True, save_path=None):
        """
        Visualize the network structure with inventory levels.
        
        Args:
            network (MultiEchelonNetwork): The network to visualize.
            inventory_levels (dict): Inventory levels by node, product, and period.
            date_idx (int, optional): Index of the date to visualize. Defaults to 0.
            show (bool, optional): Whether to display the plot. Defaults to True.
            save_path (str, optional): Path to save the visualization. Defaults to None.
            
        Returns:
            tuple: Figure and axes objects.
        """
        # Get configuration
        figsize = config.get('visualization', 'default_figsize')
        
        # Create directed graph
        G = nx.DiGraph()
        
        # Add nodes and edges
        for node_id in network.nodes:
            G.add_node(node_id)
            if network.nodes[node_id].parent:
                G.add_edge(network.nodes[node_id].parent.node_id, node_id)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Position nodes using spring layout
        pos = nx.spring_layout(G)
        
        # Color nodes by type
        node_colors = []
        for node_id in G.nodes():
            if network.nodes[node_id].node_type == 'plant':
                node_colors.append('lightblue')
            elif network.nodes[node_id].node_type == 'dc':
                node_colors.append('lightgreen')
            else:  # store
                node_colors.append('lightsalmon')
        
        # Draw nodes
        nx.draw(G, pos, with_labels=False, node_color=node_colors,
               node_size=800, ax=ax)
        
        # Create labels with inventory information
        labels = {}
        date_str = network.dates[date_idx].strftime('%Y-%m-%d')
        
        for node_id in G.nodes():
            total_inv = sum(inventory_levels.get((node_id, p, date_idx), 0)
                          for p in network.nodes[node_id].products)
            labels[node_id] = f"{node_id}\n{total_inv:.0f}"
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, labels, font_size=10)
        
        # Add legend
        plt.plot([0], [0], 'o', color='lightblue', label='Plant')
        plt.plot([0], [0], 'o', color='lightgreen', label='Distribution Center')
        plt.plot([0], [0], 'o', color='lightsalmon', label='Store')
        plt.legend()
        
        # Set title
        plt.title(f"Supply Chain Network - {date_str}")
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Network visualization saved to {save_path}")
        
        # Show if requested
        if show:
            plt.show()
        
        return fig, ax