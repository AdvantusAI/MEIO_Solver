"""
DILOP safety stock calculation with parallel processing and caching.
"""
import logging
import numpy as np
from scipy import stats
from ..utils.parallel import ParallelProcessor
from ..utils.caching import cached

logger = logging.getLogger(__name__)


class DilopSafetyStock:
    """Implements the DILOP safety stock calculation with performance optimizations."""
    
    @staticmethod
    def calculate(network, default_service_level=0.95):
        """
        Calculate safety stock levels using the DILOP method with parallel processing.
        
        Args:
            network (MultiEchelonNetwork): The network to optimize.
            default_service_level (float, optional): Default service level. Defaults to 0.95.
            
        Returns:
            dict: Safety stock recommendations by node and product.
        """
        logger.info(f"Calculating safety stock with service level {default_service_level}")
        
        # Process nodes in parallel
        safety_stock_recommendations = ParallelProcessor.process_nodes(
            network,
            DilopSafetyStock._process_node,
            default_service_level=default_service_level
        )
        
        # Update the network nodes with calculated safety stocks
        for node_id, recommendations in safety_stock_recommendations.items():
            for prod, prod_rec in recommendations.items():
                network.nodes[node_id].products[prod]['safety_stock_by_date'] = prod_rec['safety_stock_by_date']
        
        return safety_stock_recommendations
    
    @staticmethod
    def _process_node(node, network, default_service_level):
        """
        Process a single node for safety stock calculation.
        
        Args:
            node (Node): The node to process.
            network (MultiEchelonNetwork): The network.
            default_service_level (float): Default service level.
            
        Returns:
            dict: Safety stock recommendations by product.
        """
        node_results = {}
        
        # Store nodes with IDs like S1-S5 get higher service level
        service_level = 0.98 if (node.node_type == "store" and 'S' in node.node_id and 
                                int(node.node_id[1:]) <= 5) else default_service_level
        
        z_score = stats.norm.ppf(service_level)
        
        for prod, attrs in node.products.items():
            date_safety_stocks = []
            
            for t in range(network.num_periods):
                # Calculate safety stock for this period using cached helper
                safety_stock = DilopSafetyStock._calculate_safety_stock(
                    node=node,
                    prod_id=prod,
                    prod_attrs=attrs,
                    period=t,
                    z_score=z_score
                )
                date_safety_stocks.append(safety_stock)
            
            node_results[prod] = {
                'safety_stock_by_date': date_safety_stocks,
                'avg_safety_stock': np.mean(date_safety_stocks)
            }
        
        return node_results
    
    @staticmethod
    @cached
    def _calculate_safety_stock(node, prod_id, prod_attrs, period, z_score):
        """
        Calculate safety stock for a specific node, product, and period.
        This function is cached to avoid redundant calculations.
        
        Args:
            node (Node): The node.
            prod_id (str): Product ID.
            prod_attrs (dict): Product attributes.
            period (int): Period index.
            z_score (float): Z-score for the service level.
            
        Returns:
            float: Safety stock level.
        """
        # Get demand variability
        demand_std = prod_attrs['demand_std_by_date'][period]
        total_variability = np.sqrt(demand_std**2 + node.transport_variability**2)
        demand_variability = z_score * total_variability
        
        # Calculate lead time factor (incorporating upstream nodes)
        net_lead_time_mean = prod_attrs['lead_time_mean']
        net_lead_time_var = prod_attrs['lead_time_std']**2
        
        # Traverse up the chain to add parent lead times
        parent_node = node.parent
        while parent_node:
            if prod_id in parent_node.products:
                net_lead_time_mean += parent_node.products[prod_id]['lead_time_mean']
                net_lead_time_var += parent_node.products[prod_id]['lead_time_std']**2
            parent_node = parent_node.parent
            
        lead_time_factor = np.sqrt(net_lead_time_mean + net_lead_time_var)
        
        # Calculate position factor based on node type and children
        position_factor = (1.0 if node.node_type == "store" else
                          0.7 / np.sqrt(max(1, len(node.children))) if node.node_type == "dc" else
                          0.5 / np.sqrt(max(1, len(node.children))))
        
        # Calculate cost ratio
        shortage_cost = prod_attrs['shortage_cost']
        holding_cost = prod_attrs['holding_cost']
        cost_ratio = (shortage_cost / (holding_cost + shortage_cost)
                     if shortage_cost > 0 else 1.0)
        
        # Calculate safety stock
        base_safety_stock = demand_variability * lead_time_factor
        adjusted_safety_stock = base_safety_stock * position_factor * cost_ratio
        
        return max(0, adjusted_safety_stock)

