"""
Network model for the MEIO system.
"""
import logging
from datetime import datetime, timedelta
from ..models.node import Node

logger = logging.getLogger(__name__)

class MultiEchelonNetwork:
    """Represents a multi-echelon supply chain network."""
    
    def __init__(self, start_date=None, end_date=None, date_interval=30):
        """
        Initialize a multi-echelon network.
        
        Args:
            start_date (datetime, optional): Start date for the planning horizon. Defaults to today.
            end_date (datetime, optional): End date for the planning horizon. Defaults to None.
            date_interval (int, optional): Days between planning periods. Defaults to 30.
        """
        self.nodes = {}
        self.start_date = start_date if start_date else datetime.now()
        self.date_interval = date_interval
        self.dates = []
        self.num_periods = 0
        
        if end_date:
            # Calculate periods based on start and end date
            total_days = (end_date - self.start_date).days
            self.num_periods = max(1, total_days // self.date_interval)
            # Generate date list
            self._generate_dates()
    
    def _generate_dates(self):
        """Generate the list of dates for planning periods."""
        self.dates = [self.start_date + timedelta(days=i * self.date_interval) 
                      for i in range(self.num_periods)]
        logger.info(f"Generated {self.num_periods} planning periods from "
                    f"{self.dates[0].strftime('%Y-%m-%d')} to {self.dates[-1].strftime('%Y-%m-%d')}")
    
    def add_node(self, node):
        """
        Add a node to the network.
        
        Args:
            node (Node): Node to add.
            
        Raises:
            ValueError: If a node with this ID already exists.
        """
        if node.node_id in self.nodes:
            raise ValueError(f"Node with ID {node.node_id} already exists")
        
        self.nodes[node.node_id] = node
        logger.debug(f"Added node {node.node_id} to network")
        
    def add_connection(self, parent_id, child_id):
        """
        Connect two nodes in parent-child relationship.
        
        Args:
            parent_id (str): ID of the parent node.
            child_id (str): ID of the child node.
            
        Raises:
            ValueError: If either node doesn't exist.
        """
        if parent_id not in self.nodes:
            raise ValueError(f"Parent node {parent_id} not found")
            
        if child_id not in self.nodes:
            raise ValueError(f"Child node {child_id} not found")
            
        parent = self.nodes[parent_id]
        child = self.nodes[child_id]
        
        child.parent = parent
        parent.children.append(child)
        
        logger.debug(f"Added connection from {parent_id} to {child_id}")
    
    def validate(self):
        """
        Validate the entire network structure.
        
        Returns:
            bool: True if validation passes.
            
        Raises:
            ValueError: If validation fails.
        """
        # Check for empty network
        if not self.nodes:
            raise ValueError("Network has no nodes")
            
        # Check for orphaned nodes (except plants)
        for node_id, node in self.nodes.items():
            if node.node_type != 'plant' and node.parent is None:
                raise ValueError(f"Node {node_id} is not a plant but has no parent")
                
        # Validate individual nodes
        for node in self.nodes.values():
            node.validate()
            
        # Check for cycles
        visited = set()
        temp_visit = set()
        
        def has_cycle(node_id):
            if node_id in temp_visit:
                return True
                
            if node_id in visited:
                return False
                
            temp_visit.add(node_id)
            visited.add(node_id)
            
            for child in self.nodes[node_id].children:
                if has_cycle(child.node_id):
                    return True
                    
            temp_visit.remove(node_id)
            return False
        
        for node_id in self.nodes:
            if node_id not in visited:
                if has_cycle(node_id):
                    raise ValueError("Network contains cycles")
                    
        logger.info("Network validation passed")
        return True
    
    def _map_period_to_date(self, period_values):
        """
        Map period values to date values with appropriate interpolation or trimming.
        
        Args:
            period_values (list): Values by period.
            
        Returns:
            list: Values mapped to dates.
        """
        if len(period_values) == self.num_periods:
            return period_values
        elif len(period_values) > self.num_periods:
            # Trim extra values
            logger.warning(f"Trimming {len(period_values) - self.num_periods} extra period values")
            return period_values[:self.num_periods]
        else:
            # Interpolate to fill missing values
            logger.warning(f"Interpolating {self.num_periods - len(period_values)} missing period values")
            ratio = len(period_values) / self.num_periods
            date_values = []
            for i in range(self.num_periods):
                idx = min(int(i * ratio), len(period_values) - 1)
                date_values.append(period_values[idx])
            return date_values
    
    def __repr__(self):
        """String representation of the network."""
        node_types = {'plant': 0, 'dc': 0, 'store': 0}
        for node in self.nodes.values():
            node_types[node.node_type] += 1
            
        return f"MultiEchelonNetwork(periods={self.num_periods}, plants={node_types['plant']}, " \
               f"dcs={node_types['dc']}, stores={node_types['store']})"