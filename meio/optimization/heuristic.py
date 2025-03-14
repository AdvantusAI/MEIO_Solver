"""
Heuristic optimization for the MEIO system.
"""
import logging
import numpy as np
from ..config.settings import config

logger = logging.getLogger(__name__)

class HeuristicSolver:
    """Implements a rule-based heuristic for inventory optimization."""
    
    @staticmethod
    def optimize(network, service_level=None, inflows=None):
        """
        Optimize inventory levels using a heuristic approach.
        
        Args:
            network (MultiEchelonNetwork): The network to optimize.
            service_level (float, optional): Service level. Defaults to config value.
            inflows (float, optional): Base inflow level. Defaults to config value.
            
        Returns:
            dict: Optimization results.
        """
        if service_level is None:
            service_level = config.get('optimization', 'default_service_level')
            
        if inflows is None:
            inflows = config.get('optimization', 'default_inflow')
            
        # First, calculate safety stocks using DILOP
        from .dilop import DiloptOpSafetyStock
        DiloptOpSafetyStock.calculate(network, service_level)
        
        logger.info(f"Starting heuristic optimization with service level {service_level}")
        
        inventory_levels = {}
        total_cost = 0
        
        # Set store inventory levels to safety stock
        logger.debug("Setting store inventory levels...")
        for node_id, node in network.nodes.items():
            if node.node_type == "store":
                for prod in node.products:
                    for t in range(network.num_periods):
                        inv = node.products[prod]['safety_stock_by_date'][t]
                        inventory_levels[(node_id, prod, t)] = inv
                        
                        # Add holding cost
                        total_cost += node.products[prod]['holding_cost'] * inv
                        
                        # Add transport cost
                        total_cost += node.transport_cost * inv
                        
                        # Add shortage cost if safety stock below demand
                        if inv < node.products[prod]['demand_by_date'][t]:
                            shortfall = node.products[prod]['demand_by_date'][t] - inv
                            total_cost += node.products[prod]['shortage_cost'] * shortfall
                            logger.debug(f"Potential stockout at {node_id} for {prod} on period {t}: {shortfall:.2f} units")
        
        # Set DC inventory levels based on store demands
        logger.debug("Setting distribution center inventory levels...")
        for node_id, node in network.nodes.items():
            if node.node_type == "dc":
                for prod in node.products:
                    for t in range(network.num_periods):
                        # Sum children demand
                        children_demand = sum(inventory_levels.get((child.node_id, prod, t), 0)
                                           for child in node.children)
                        
                        # Set inventory to max of children demand or safety stock
                        inv = max(children_demand, node.products[prod]['safety_stock_by_date'][t])
                        
                        # Respect capacity constraint
                        inv = min(inv, node.capacity / len(node.products))
                        
                        inventory_levels[(node_id, prod, t)] = inv
                        
                        # Add costs
                        total_cost += node.products[prod]['holding_cost'] * inv
                        total_cost += node.transport_cost * inv
        
        # Set plant inventory levels based on DC demands and inflows
        logger.debug("Setting plant inventory levels...")
        for node_id, node in network.nodes.items():
            if node.node_type == "plant":
                for prod in node.products:
                    for t in range(network.num_periods):
                        # Sum children demand
                        children_demand = sum(inventory_levels.get((child.node_id, prod, t), 0)
                                           for child in node.children)
                        
                        # Add inflow-based buffer
                        lead_time_buffer = inflows * node.products[prod]['lead_time_mean'] / len(node.products)
                        
                        # Set inventory level
                        inv = max(children_demand, lead_time_buffer)
                        
                        # Respect capacity constraint
                        inv = min(inv, node.capacity / len(node.products))
                        
                        inventory_levels[(node_id, prod, t)] = inv
                        
                        # Add holding cost
                        total_cost += node.products[prod]['holding_cost'] * inv
        
        logger.info(f"Heuristic optimization complete. Total cost: {total_cost:.2f}")
        
        return {
            'inventory_levels': inventory_levels,
            'total_cost': total_cost,
            'status': 'heuristic'
        }
