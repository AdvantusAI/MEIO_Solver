"""
Heuristic optimization methods for the MEIO system.
"""
import logging
import numpy as np
from collections import defaultdict
from ..config.settings import config
from ..utils.caching import cached

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


class ImprovedHeuristicSolver:
    """Implements an improved rule-based heuristic for inventory optimization."""
    
    @staticmethod
    def optimize(network, service_level=None, inflows=None):
        """
        Optimize inventory levels using an improved heuristic approach.
        
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
            
        # First, calculate safety stocks
        from .dilop import DiloptOpSafetyStock
        DiloptOpSafetyStock.calculate(network, service_level)
        
        logger.info(f"Starting improved heuristic optimization with service level {service_level}")
        
        # Initialize results
        inventory_levels = {}
        total_cost = 0
        
        # Process layers of the network from bottom to top
        logger.debug("Processing network layers from stores to plants...")
        
        # 1. Group nodes by type for ordered processing
        nodes_by_type = {
            'store': [],
            'dc': [],
            'plant': []
        }
        
        for node_id, node in network.nodes.items():
            if node.node_type in nodes_by_type:
                nodes_by_type[node.node_type].append(node_id)
        
        # 2. Process stores first (bottom layer)
        logger.debug("Setting store inventory levels...")
        ImprovedHeuristicSolver._process_stores(
            network, nodes_by_type['store'], inventory_levels, total_cost)
        
        # 3. Process distribution centers
        logger.debug("Setting distribution center inventory levels...")
        ImprovedHeuristicSolver._process_dcs(
            network, nodes_by_type['dc'], inventory_levels, total_cost)
        
        # 4. Process plants (top layer)
        logger.debug("Setting plant inventory levels...")
        ImprovedHeuristicSolver._process_plants(
            network, nodes_by_type['plant'], inventory_levels, total_cost, inflows)
        
        # 5. Calculate total cost efficiently
        logger.debug("Calculating total cost...")
        total_cost = ImprovedHeuristicSolver._calculate_total_cost(
            network, inventory_levels)
        
        logger.info(f"Improved heuristic optimization complete. Total cost: {total_cost:.2f}")
        
        return {
            'inventory_levels': inventory_levels,
            'total_cost': total_cost,
            'status': 'heuristic'
        }
    
    @staticmethod
    def _process_stores(network, store_ids, inventory_levels, total_cost):
        """Process store nodes and set inventory levels."""
        for node_id in store_ids:
            node = network.nodes[node_id]
            
            # Use forecast demand for next period to adjust safety stock
            for prod in node.products:
                for t in range(network.num_periods):
                    demand = node.products[prod]['demand_by_date'][t]
                    safety_stock = node.products[prod]['safety_stock_by_date'][t]
                    
                    # Calculate inventory using dynamic safety factor
                    service_factor = 1.0 + (0.05 * (t + 1))  # Increases with time horizon
                    inv = max(demand * 0.2, safety_stock * service_factor)  # At least 20% of demand
                    
                    # Respect capacity constraints
                    capacity_per_product = node.capacity / max(1, len(node.products))
                    inv = min(inv, capacity_per_product)
                    
                    inventory_levels[(node_id, prod, t)] = inv
    
    @staticmethod
    def _process_dcs(network, dc_ids, inventory_levels, total_cost):
        """Process distribution center nodes and set inventory levels."""
        # Pre-compute aggregate demand by DC
        dc_demands = defaultdict(lambda: defaultdict(dict))
        
        for node_id in dc_ids:
            node = network.nodes[node_id]
            
            # Get all children nodes (stores served by this DC)
            for prod in node.products:
                for t in range(network.num_periods):
                    # Calculate demand from all children
                    children_demand = 0
                    children_variability = 0
                    
                    for child in node.children:
                        child_demand = inventory_levels.get((child.node_id, prod, t), 0)
                        children_demand += child_demand
                        
                        # Add demand variability for improved forecasting
                        if t < network.num_periods - 1:
                            next_period = inventory_levels.get((child.node_id, prod, t+1), 0)
                            variability = abs(next_period - child_demand) / max(1, child_demand)
                            children_variability += variability
                    
                    # Add buffer based on children variability
                    buffer_factor = 1.0 + min(0.5, children_variability / max(1, len(node.children)))
                    
                    # Set inventory considering safety stock, demand, and buffer
                    safety_stock = node.products[prod]['safety_stock_by_date'][t]
                    inv = max(children_demand * buffer_factor, safety_stock * 1.1)
                    
                    # Respect capacity constraint
                    capacity_per_product = node.capacity / max(1, len(node.products))
                    inv = min(inv, capacity_per_product)
                    
                    inventory_levels[(node_id, prod, t)] = inv
                    dc_demands[node_id][prod][t] = children_demand
    
    @staticmethod
    def _process_plants(network, plant_ids, inventory_levels, total_cost, inflows):
        """Process plant nodes and set inventory levels."""
        for node_id in plant_ids:
            node = network.nodes[node_id]
            
            # Consider production constraints and economies of scale
            for prod in node.products:
                # Calculate optimal production batch size
                batch_size = ImprovedHeuristicSolver._calculate_batch_size(
                    node, prod, inflows)
                
                for t in range(network.num_periods):
                    # Sum demands from all children (DCs)
                    children_demand = 0
                    for child in node.children:
                        children_demand += inventory_levels.get((child.node_id, prod, t), 0)
                    
                    # Calculate lead time buffer with production consideration
                    lead_time_mean = node.products[prod]['lead_time_mean']
                    production_cycle = max(1, round(batch_size / inflows))
                    
                    # Add buffer for production cycle and lead time
                    lead_time_buffer = (lead_time_mean + production_cycle/2) * inflows / len(node.products)
                    
                    # Determine inventory level
                    inv = max(children_demand * 1.1, lead_time_buffer)
                    
                    # Add cyclical production pattern
                    cycle_position = t % max(1, production_cycle)
                    cycle_factor = 1.0 + 0.2 * (1 - cycle_position/production_cycle)
                    inv *= cycle_factor
                    
                    # Respect capacity constraint
                    capacity_per_product = node.capacity / max(1, len(node.products))
                    inv = min(inv, capacity_per_product)
                    
                    inventory_levels[(node_id, prod, t)] = inv
    
    @staticmethod
    @cached
    def _calculate_batch_size(node, prod_id, inflows):
        """Calculate optimal production batch size based on EOQ principles."""
        holding_cost = node.products[prod_id]['holding_cost']
        setup_cost = 100  # Assumed setup cost
        annual_demand = sum(node.products[prod_id]['demand_by_date'])
        
        # Economic Order Quantity formula
        eoq = np.sqrt((2 * annual_demand * setup_cost) / holding_cost)
        
        # Adjust for production capacity
        return min(eoq, inflows * 7)  # Limit to one week of production
    
    @staticmethod
    def _calculate_total_cost(network, inventory_levels):
        """Calculate total cost efficiently."""
        total_cost = 0
        
        # Group costs by type for more efficient calculation
        holding_costs = 0
        transport_costs = 0
        shortage_costs = 0
        
        for node_id, node in network.nodes.items():
            for prod in node.products:
                for t in range(network.num_periods):
                    inv = inventory_levels.get((node_id, prod, t), 0)
                    
                    # Holding cost
                    holding_costs += node.products[prod]['holding_cost'] * inv
                    
                    # Transport cost
                    if node.parent:
                        transport_costs += node.transport_cost * inv
                    
                    # Shortage cost (only for stores)
                    if node.node_type == "store":
                        demand = node.products[prod]['demand_by_date'][t]
                        if inv < demand:
                            shortage_costs += node.products[prod]['shortage_cost'] * (demand - inv)
        
        total_cost = holding_costs + transport_costs + shortage_costs
        return total_cost