"""
DILOP safety stock calculation for the MEIO system.
"""
import logging
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

class DiloptOpSafetyStock:
    """Implements the DILOP (Distribution Logistics Optimization) safety stock calculation."""
    
    @staticmethod
    def calculate(network, default_service_level=0.95):
        """
        Calculate safety stock levels using the DILOP method.
        
        Args:
            network (MultiEchelonNetwork): The network to optimize.
            default_service_level (float, optional): Default service level. Defaults to 0.95.
            
        Returns:
            dict: Safety stock recommendations by node and product.
        """
        safety_stock_recommendations = {}
        
        for node_id, node in network.nodes.items():
            # Store nodes with IDs like S1-S5 get higher service level
            service_level = 0.98 if (node.node_type == "store" and 'S' in node_id and 
                                     int(node_id[1:]) <= 5) else default_service_level
            
            z_score = stats.norm.ppf(service_level)
            
            safety_stock_recommendations[node_id] = {}
            
            for prod, attrs in node.products.items():
                logger.debug(f"Calculating safety stock for {node_id} - {prod}:")
                date_safety_stocks = []
                
                for t in range(network.num_periods):
                    # Get demand variability
                    demand_std = attrs['demand_std_by_date'][t]
                    total_variability = np.sqrt(demand_std**2 + node.transport_variability**2)
                    demand_variability = z_score * total_variability
                    
                    # Calculate lead time factor (incorporating upstream nodes)
                    net_lead_time_mean = attrs['lead_time_mean']
                    net_lead_time_var = attrs['lead_time_std']**2
                    
                    current = node
                    while current.parent:
                        net_lead_time_mean += current.parent.products[prod]['lead_time_mean']
                        net_lead_time_var += current.parent.products[prod]['lead_time_std']**2
                        current = current.parent
                        
                    lead_time_factor = np.sqrt(net_lead_time_mean + net_lead_time_var)
                    
                    # Calculate position factor based on node type and children
                    position_factor = (1.0 if node.node_type == "store" else
                                      0.7 / np.sqrt(len(node.children)) if node.node_type == "dc" else
                                      0.5 / np.sqrt(len(node.children)))
                    
                    # Calculate cost ratio
                    cost_ratio = (attrs['shortage_cost'] / (attrs['holding_cost'] + attrs['shortage_cost'])
                                 if attrs['shortage_cost'] > 0 else 1.0)
                    
                    # Calculate safety stock
                    base_safety_stock = demand_variability * lead_time_factor
                    adjusted_safety_stock = base_safety_stock * position_factor * cost_ratio
                    date_safety_stocks.append(max(0, adjusted_safety_stock))
                
                # Save results
                attrs['safety_stock_by_date'] = date_safety_stocks
                safety_stock_recommendations[node_id][prod] = {
                    'safety_stock_by_date': date_safety_stocks,
                    'avg_safety_stock': np.mean(date_safety_stocks)
                }
                
                logger.debug(f"Average safety stock for {node_id} - {prod}: {np.mean(date_safety_stocks):.2f}")
        
        return safety_stock_recommendations