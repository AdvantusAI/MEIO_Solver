"""
AI agent for analyzing MEIO system results and providing recommendations.
"""
import logging
from typing import Dict, List, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class AIAgent:
    """AI agent for analyzing MEIO system results and providing recommendations."""
    
    def __init__(self):
        """Initialize the AI agent."""
        self.logger = logging.getLogger(__name__)
    
    def analyze_results(self, network_id: str, node_id: str, product_id: str,
                       inventory_levels: Dict[str, float],
                       safety_stock: Dict[str, float],
                       network_stats: Dict[str, float]) -> Dict[str, Any]:
        """
        Analyze results and provide recommendations for a specific node and product.
        
        Args:
            network_id (str): ID of the network
            node_id (str): ID of the node
            product_id (str): ID of the product
            inventory_levels (dict): Dictionary of inventory levels by period
            safety_stock (dict): Dictionary of safety stock values
            network_stats (dict): Dictionary of network statistics
            
        Returns:
            dict: Dictionary containing analysis and recommendations
        """
        try:
            # Calculate key metrics
            avg_inventory = sum(inventory_levels.values()) / len(inventory_levels)
            safety_stock_value = safety_stock.get('avg_safety_stock', 0)
            total_demand = network_stats.get('total_demand', 0)
            avg_demand = network_stats.get('avg_demand', 0)
            demand_std = network_stats.get('demand_std', 0)
            avg_lead_time = network_stats.get('avg_lead_time', 0)
            lead_time_std = network_stats.get('lead_time_std', 0)
            
            # Generate analysis
            analysis = self._generate_analysis(
                node_id, product_id, avg_inventory, safety_stock_value,
                total_demand, avg_demand, demand_std, avg_lead_time, lead_time_std
            )
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                node_id, product_id, avg_inventory, safety_stock_value,
                total_demand, avg_demand, demand_std, avg_lead_time, lead_time_std
            )
            
            return {
                'analysis': analysis,
                'recommendations': recommendations
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing results: {str(e)}")
            raise
    
    def _generate_analysis(self, node_id: str, product_id: str,
                         avg_inventory: float, safety_stock: float,
                         total_demand: float, avg_demand: float,
                         demand_std: float, avg_lead_time: float,
                         lead_time_std: float) -> str:
        """Generate analysis text based on the metrics."""
        analysis = []
        
        # Inventory analysis
        inventory_coverage = avg_inventory / avg_demand if avg_demand > 0 else 0
        analysis.append(f"Current inventory level ({avg_inventory:.2f}) provides {inventory_coverage:.1f} periods of coverage.")
        
        # Safety stock analysis
        if safety_stock > 0:
            analysis.append(f"Safety stock of {safety_stock:.2f} units helps buffer against demand variability.")
        
        # Demand analysis
        cv_demand = demand_std / avg_demand if avg_demand > 0 else 0
        analysis.append(f"Demand variability (CV: {cv_demand:.2f}) indicates {'high' if cv_demand > 0.5 else 'moderate' if cv_demand > 0.2 else 'low'} uncertainty.")
        
        # Lead time analysis
        cv_lead_time = lead_time_std / avg_lead_time if avg_lead_time > 0 else 0
        analysis.append(f"Lead time variability (CV: {cv_lead_time:.2f}) suggests {'high' if cv_lead_time > 0.5 else 'moderate' if cv_lead_time > 0.2 else 'low'} supply uncertainty.")
        
        return " ".join(analysis)
    
    def _generate_recommendations(self, node_id: str, product_id: str,
                                avg_inventory: float, safety_stock: float,
                                total_demand: float, avg_demand: float,
                                demand_std: float, avg_lead_time: float,
                                lead_time_std: float) -> Dict[str, Any]:
        """Generate recommendations based on the metrics."""
        recommendations = {
            'inventory_management': [],
            'safety_stock': [],
            'supply_chain': []
        }
        
        # Inventory management recommendations
        inventory_coverage = avg_inventory / avg_demand if avg_demand > 0 else 0
        if inventory_coverage < 1:
            recommendations['inventory_management'].append({
                'type': 'increase_inventory',
                'reason': 'Current inventory levels are below average demand',
                'suggestion': f'Consider increasing inventory to maintain at least 1 period of coverage'
            })
        elif inventory_coverage > 3:
            recommendations['inventory_management'].append({
                'type': 'reduce_inventory',
                'reason': 'High inventory levels may indicate overstocking',
                'suggestion': f'Review inventory policies to reduce holding costs'
            })
        
        # Safety stock recommendations
        if safety_stock == 0:
            recommendations['safety_stock'].append({
                'type': 'add_safety_stock',
                'reason': 'No safety stock maintained',
                'suggestion': 'Consider implementing safety stock to buffer against demand and supply uncertainty'
            })
        
        # Supply chain recommendations
        cv_demand = demand_std / avg_demand if avg_demand > 0 else 0
        cv_lead_time = lead_time_std / avg_lead_time if avg_lead_time > 0 else 0
        
        if cv_demand > 0.5 or cv_lead_time > 0.5:
            recommendations['supply_chain'].append({
                'type': 'improve_forecasting',
                'reason': 'High demand or lead time variability',
                'suggestion': 'Invest in improved demand forecasting and supplier management'
            })
        
        return recommendations 