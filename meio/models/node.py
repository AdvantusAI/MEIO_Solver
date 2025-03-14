"""
Node models for the MEIO system.
"""
import logging
import numpy as np

logger = logging.getLogger(__name__)

class Node:
    """Represents a node in the supply chain network."""
    
    def __init__(self, node_id, node_type, parent=None):
        """
        Initialize a supply chain node.
        
        Args:
            node_id (str): Unique identifier for the node.
            node_type (str): Type of node ('plant', 'dc', or 'store').
            parent (Node, optional): Parent node in the supply chain. Defaults to None.
        """
        self.node_id = node_id
        self.node_type = node_type
        self.parent = parent
        self.children = []
        self.capacity = 0
        self.products = {}
        self.transport_cost = 0
        self.transport_variability = 0
    
    def add_product(self, product_id, lead_time_mean, lead_time_std, holding_cost,
                   shortage_cost, demand_means, demand_stds):
        """
        Add product information to this node.
        
        Args:
            product_id (str): Unique identifier for the product.
            lead_time_mean (float): Mean lead time in periods.
            lead_time_std (float): Standard deviation of lead time.
            holding_cost (float): Cost to hold one unit per period.
            shortage_cost (float): Cost of stockout per unit per period.
            demand_means (list): Mean demand by period.
            demand_stds (list): Standard deviation of demand by period.
            
        Raises:
            ValueError: If demand_means and demand_stds have different lengths.
        """
        if len(demand_means) != len(demand_stds):
            msg = f"Product {product_id}: demand_means ({len(demand_means)}) and demand_stds ({len(demand_stds)}) lengths mismatch"
            logger.error(msg)
            raise ValueError(msg)
            
        self.products[product_id] = {
            'lead_time_mean': lead_time_mean,
            'lead_time_std': lead_time_std,
            'holding_cost': holding_cost,
            'shortage_cost': shortage_cost,
            'demand_by_date': demand_means,
            'demand_std_by_date': demand_stds,
            'safety_stock_by_date': [0] * len(demand_means)
        }
        
        logger.debug(f"Added product {product_id} to node {self.node_id}")
    
    def validate(self):
        """
        Validate the node's data for consistency.
        
        Returns:
            bool: True if validation passes.
            
        Raises:
            ValueError: If validation fails.
        """
        if not self.node_id:
            raise ValueError("Node ID is required")
            
        if self.node_type not in ['plant', 'dc', 'store']:
            raise ValueError(f"Invalid node type: {self.node_type}")
            
        if self.capacity < 0:
            raise ValueError(f"Capacity must be non-negative, got {self.capacity}")
            
        if self.transport_cost < 0:
            raise ValueError(f"Transport cost must be non-negative, got {self.transport_cost}")
            
        if self.transport_variability < 0:
            raise ValueError(f"Transport variability must be non-negative, got {self.transport_variability}")
            
        for prod_id, prod_data in self.products.items():
            if prod_data['lead_time_mean'] < 0:
                raise ValueError(f"Lead time mean must be non-negative for {prod_id}")
                
            if prod_data['lead_time_std'] < 0:
                raise ValueError(f"Lead time std must be non-negative for {prod_id}")
                
            if prod_data['holding_cost'] < 0:
                raise ValueError(f"Holding cost must be non-negative for {prod_id}")
                
            if prod_data['shortage_cost'] < 0:
                raise ValueError(f"Shortage cost must be non-negative for {prod_id}")
                
            if any(d < 0 for d in prod_data['demand_by_date']):
                raise ValueError(f"Demand values must be non-negative for {prod_id}")
                
            if any(d < 0 for d in prod_data['demand_std_by_date']):
                raise ValueError(f"Demand std values must be non-negative for {prod_id}")
        
        return True
    
    def __repr__(self):
        """String representation of the node."""
        return f"Node(id={self.node_id}, type={self.node_type}, products={len(self.products)})"