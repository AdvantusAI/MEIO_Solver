"""
Mathematical optimization solver for the MEIO system.
"""
import logging
import numpy as np
from ..config.settings import config

logger = logging.getLogger(__name__)

# Check if PySCIPOpt is available
try:
    from pyscipopt import Model
    HAS_PYSCIPOPT = True
except ImportError:
    HAS_PYSCIPOPT = False
    logger.warning("PySCIPOpt not installed. Only heuristic method will be available.")

class MathematicalSolver:
    """Implements mathematical optimization using PySCIPOpt."""
    
    @staticmethod
    def is_available():
        """Check if solver is available."""
        return HAS_PYSCIPOPT
    
    @staticmethod
    def optimize(network, service_level=None):
        """
        Optimize inventory levels using mathematical programming.
        
        Args:
            network (MultiEchelonNetwork): The network to optimize.
            service_level (float, optional): Service level. Defaults to config value.
            
        Returns:
            dict: Optimization results.
            
        Raises:
            Exception: If PySCIPOpt is not available.
        """
        if not HAS_PYSCIPOPT:
            raise Exception("PySCIPOpt not available. Cannot use mathematical solver.")
        
        if service_level is None:
            service_level = config.get('optimization', 'default_service_level')
            
        # First, calculate safety stocks using DILOP
        from .dilop import DiloptOpSafetyStock
        DiloptOpSafetyStock.calculate(network, service_level)
        
        logger.info(f"Starting mathematical optimization with service level {service_level}")
        
        # Create model
        model = Model("MEIO_Network")
        
        # Set solver parameters
        model.setRealParam('limits/time', config.get('optimization', 'solver_time_limit'))
        model.setRealParam('limits/gap', config.get('optimization', 'solver_gap'))
        
        # Create variables for inventory levels
        inventory = {(n, p, t): model.addVar(f"inv_{n}_{p}_{t}", vtype="C", lb=0)
                   for n in network.nodes for p in network.nodes[n].products
                   for t in range(network.num_periods)}
        
        # Build objective function
        total_cost = 0
        shortage_vars = {}
        
        for node_id, node in network.nodes.items():
            for prod in node.products:
                for t in range(network.num_periods):
                    # Holding cost
                    total_cost += node.products[prod]['holding_cost'] * inventory[(node_id, prod, t)]
                    
                    # Shortage cost (only for stores)
                    if node.node_type == "store":
                        shortage_vars[(node_id, prod, t)] = model.addVar(
                            f"short_{node_id}_{prod}_{t}", lb=0)
                        total_cost += node.products[prod]['shortage_cost'] * shortage_vars[(node_id, prod, t)]
                    
                    # Transport cost
                    if node.parent:
                        total_cost += node.transport_cost * inventory[(node_id, prod, t)]
        
        model.setObjective(total_cost, "minimize")
        
        # Add constraints
        for node_id, node in network.nodes.items():
            # Capacity constraints
            for t in range(network.num_periods):
                total_inv = sum(inventory[(node_id, p, t)] for p in node.products)
                model.addCons(total_inv <= node.capacity, f"capacity_{node_id}_t{t}")
            
            # Safety stock and flow constraints
            for prod in node.products:
                for t in range(network.num_periods):
                    # Safety stock constraint
                    model.addCons(
                        inventory[(node_id, prod, t)] >= node.products[prod]['safety_stock_by_date'][t],
                        f"safety_{node_id}_{prod}_{t}"
                    )
                    
                    # Parent inventory must be at least child inventory
                    if node.parent:
                        parent_id = node.parent.node_id
                        model.addCons(
                            inventory[(parent_id, prod, t)] >= inventory[(node_id, prod, t)],
                            f"flow_{parent_id}_{node_id}_{prod}_{t}"
                        )
        
        # Solve the model
        logger.info("Starting solver...")
        model.optimize()
        status = model.getStatus()
        logger.info(f"Solver finished with status: {status}")
        
        # Collect results
        if status == "optimal":
            results = {
                'inventory_levels': {(n, p, t): model.getVal(inventory[(n, p, t)])
                                   for n in network.nodes for p in network.nodes[n].products
                                   for t in range(network.num_periods)},
                'total_cost': model.getObjVal(),
                'status': 'optimal'
            }
            return results
        else:
            logger.warning(f"Solver failed with status: {status}")
            return {'status': 'infeasible', 'inventory_levels': {}, 'total_cost': 0}
