"""
Database exporter for saving MEIO system results to Supabase.
"""
import logging
from datetime import datetime
from typing import Dict, List, Any
from supabase import create_client, Client

logger = logging.getLogger(__name__)

class DatabaseExporter:
    """Exports MEIO system results to Supabase database."""
    
    def __init__(self, supabase_url: str, supabase_key: str):
        """
        Initialize the database exporter.
        
        Args:
            supabase_url (str): Supabase project URL
            supabase_key (str): Supabase API key
        """
        self.supabase: Client = create_client(supabase_url, supabase_key)
    
    def save_optimization_run(self, network_id: str, method: str, service_level: float,
                            start_date: datetime, end_date: datetime, total_cost: float,
                            status: str) -> str:
        """
        Save an optimization run to the database.
        
        Args:
            network_id (str): ID of the network
            method (str): Optimization method used ('SCIP Solver' or 'Heuristic')
            service_level (float): Service level used
            start_date (datetime): Start date of the optimization
            end_date (datetime): End date of the optimization
            total_cost (float): Total cost of the solution
            status (str): Status of the optimization ('optimal', 'heuristic', 'failed')
            
        Returns:
            str: ID of the created optimization run
        """
        try:
            data = {
                'network_id': network_id,
                'method': method,
                'service_level': service_level,
                'start_date': start_date.date().isoformat(),
                'end_date': end_date.date().isoformat(),
                'total_cost': total_cost,
                'status': status
            }
            
            response = self.supabase.table('optimization_runs').insert(data).execute()
            if not response.data:
                raise Exception("No data returned from insert operation")
            return response.data[0]['id']
            
        except Exception as e:
            logger.error(f"Error saving optimization run: {str(e)}")
            logger.error(f"Data: {data}")
            raise
    
    def save_inventory_levels(self, optimization_run_id: str, inventory_levels: Dict[tuple, float]):
        """
        Save inventory levels to the database.
        
        Args:
            optimization_run_id (str): ID of the optimization run
            inventory_levels (dict): Dictionary mapping (node_id, product_id, period) to inventory level
        """
        try:
            data = []
            for (node_id, product_id, period), level in inventory_levels.items():
                data.append({
                    'optimization_run_id': optimization_run_id,
                    'node_id': node_id,
                    'product_id': product_id,
                    'period': period,
                    'inventory_level': level
                })
            
            if data:
                response = self.supabase.table('inventory_levels').insert(data).execute()
                if not response.data:
                    raise Exception("No data returned from insert operation")
                
        except Exception as e:
            logger.error(f"Error saving inventory levels: {str(e)}")
            logger.error(f"First few records: {data[:5] if data else 'No data'}")
            raise
    
    def save_stock_alerts(self, optimization_run_id: str, stockouts: Dict[str, Dict[str, List[Dict]]],
                         overstocks: Dict[str, Dict[str, List[Dict]]]):
        """
        Save stock alerts to the database.
        
        Args:
            optimization_run_id (str): ID of the optimization run
            stockouts (dict): Dictionary of stockout alerts
            overstocks (dict): Dictionary of overstock alerts
        """
        try:
            data = []
            
            # Process stockouts
            for node_id, prods in stockouts.items():
                for product_id, alerts in prods.items():
                    for alert in alerts:
                        data.append({
                            'optimization_run_id': optimization_run_id,
                            'node_id': node_id,
                            'product_id': product_id,
                            'alert_type': 'stockout',
                            'period': alert.get('period', 0),
                            'inventory_level': alert['inventory'],
                            'demand': alert['demand'],
                            'shortfall': alert['shortfall']
                        })
            
            # Process overstocks
            for node_id, prods in overstocks.items():
                for product_id, alerts in prods.items():
                    for alert in alerts:
                        data.append({
                            'optimization_run_id': optimization_run_id,
                            'node_id': node_id,
                            'product_id': product_id,
                            'alert_type': 'overstock',
                            'period': alert.get('period', 0),
                            'inventory_level': alert['inventory'],
                            'capacity': alert['capacity'],
                            'excess': alert['excess']
                        })
            
            if data:
                response = self.supabase.table('stock_alerts').insert(data).execute()
                if not response.data:
                    raise Exception("No data returned from insert operation")
                
        except Exception as e:
            logger.error(f"Error saving stock alerts: {str(e)}")
            logger.error(f"First few records: {data[:5] if data else 'No data'}")
            raise
    
    def save_network_statistics(self, network_id: str, statistics: Dict[str, Dict[str, Dict[str, float]]]):
        """
        Save network statistics to the database.
        
        Args:
            network_id (str): ID of the network
            statistics (dict): Dictionary of network statistics
        """
        try:
            data = []
            for node_id, prods in statistics.items():
                for product_id, stats in prods.items():
                    data.append({
                        'network_id': network_id,
                        'node_id': node_id,
                        'product_id': product_id,
                        'total_demand': stats['total_demand'],
                        'avg_demand': stats['avg_demand'],
                        'demand_std': stats['demand_std'],
                        'avg_lead_time': stats['avg_lead_time'],
                        'lead_time_std': stats['lead_time_std'],
                        'holding_cost': stats['holding_cost'],
                        'shortage_cost': stats['shortage_cost']
                    })
            
            if data:
                response = self.supabase.table('network_statistics').insert(data).execute()
                if not response.data:
                    raise Exception("No data returned from insert operation")
                
        except Exception as e:
            logger.error(f"Error saving network statistics: {str(e)}")
            logger.error(f"First few records: {data[:5] if data else 'No data'}")
            raise
    
    def save_safety_stock_recommendations(self, network_id: str, recommendations: Dict[str, Dict[str, Dict[str, float]]],
                                        service_level: float):
        """
        Save safety stock recommendations to the database.
        
        Args:
            network_id (str): ID of the network
            recommendations (dict): Dictionary of safety stock recommendations
            service_level (float): Service level used
        """
        try:
            data = []
            for node_id, prods in recommendations.items():
                for product_id, details in prods.items():
                    data.append({
                        'network_id': network_id,
                        'node_id': node_id,
                        'product_id': product_id,
                        'safety_stock': {
                            'avg_safety_stock': float(details['avg_safety_stock']),
                            'min_safety_stock': float(details.get('min_safety_stock', 0)),
                            'max_safety_stock': float(details.get('max_safety_stock', 0))
                        },
                        'service_level': service_level
                    })
            
            if data:
                response = self.supabase.table('safety_stock_recommendations').insert(data).execute()
                if not response.data:
                    raise Exception("No data returned from insert operation")
                
        except Exception as e:
            logger.error(f"Error saving safety stock recommendations: {str(e)}")
            logger.error(f"First few records: {data[:5] if data else 'No data'}")
            raise
    
    def save_ai_recommendations(self, network_id: str, recommendations: List[Dict[str, Any]]):
        """
        Save AI recommendations to the database.
        
        Args:
            network_id (str): ID of the network
            recommendations (list): List of recommendation dictionaries
        """
        try:
            data = []
            for rec in recommendations:
                data.append({
                    'network_id': network_id,
                    'node_id': rec['node_id'],
                    'product_id': rec['product_id'],
                    'analysis': rec['analysis'],
                    'recommendations': rec['recommendations']
                })
            
            if data:
                response = self.supabase.table('ai_recommendations').insert(data).execute()
                if not response.data:
                    raise Exception("No data returned from insert operation")
                
        except Exception as e:
            logger.error(f"Error saving AI recommendations: {str(e)}")
            logger.error(f"First few records: {data[:5] if data else 'No data'}")
            raise 