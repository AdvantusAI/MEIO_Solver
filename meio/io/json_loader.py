"""
JSON loader for the MEIO system.
"""
import json
import logging
from datetime import datetime, timedelta
from ..models.node import Node
from ..models.network import MultiEchelonNetwork

logger = logging.getLogger(__name__)

class NetworkJsonLoader:
    """Loads network data from JSON files."""
    
    @staticmethod
    def load(json_file, start_date=None, end_date=None, date_interval=30):
        """
        Load a network from a JSON file.
        
        Args:
            json_file (str): Path to JSON file.
            start_date (datetime, optional): Start date. Defaults to None (today).
            end_date (datetime, optional): End date. Defaults to None.
            date_interval (int, optional): Days between periods. Defaults to 30.
            
        Returns:
            MultiEchelonNetwork: The loaded network.
            
        Raises:
            FileNotFoundError: If the file doesn't exist.
            json.JSONDecodeError: If the file isn't valid JSON.
            ValueError: If the data isn't valid.
        """
        # Create network
        network = MultiEchelonNetwork(start_date, end_date, date_interval)
        
        try:
            with open(json_file, 'r') as f:
                content = f.read()
                if not content.strip():
                    raise ValueError("JSON file is empty")
                config = json.loads(content)
                
            logger.info(f"Loaded JSON from {json_file}")
            
            # Validate basic structure
            all_nodes = config.get('plants', []) + config.get('dcs', []) + config.get('stores', [])
            
            if not all_nodes or not all_nodes[0].get('products'):
                raise ValueError("No nodes or products defined in JSON")
                
            # Get number of periods from data
            first_product = all_nodes[0]['products'][0]
            periods_from_json = len(first_product['demand_by_period'])
            
            # Set number of periods if not already set by end_date
            if not end_date:
                network.num_periods = periods_from_json
                network._generate_dates()
                
            logger.info(f"Network has {network.num_periods} periods, with {len(all_nodes)} nodes")
            
            # Create nodes
            NetworkJsonLoader._create_plants(network, config.get('plants', []))
            NetworkJsonLoader._create_dcs(network, config.get('dcs', []))
            NetworkJsonLoader._create_stores(network, config.get('stores', []))
            
            # Create connections
            for connection in config.get('connections', []):
                child_id = connection['to']
                parent_id = connection['from']
                
                try:
                    network.add_connection(parent_id, child_id)
                except ValueError as e:
                    logger.warning(f"Invalid connection {parent_id} -> {child_id}: {e}")
            
            # Validate the network
            network.validate()
            
            return network
            
        except FileNotFoundError:
            logger.error(f"File not found: {json_file}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error loading network: {str(e)}")
            raise
    
    @staticmethod
    def _create_plants(network, plants_data):
        """Create plant nodes from JSON data."""
        for plant in plants_data:
            node = Node(plant['id'], 'plant')
            node.capacity = plant.get('capacity', 0)
            
            for prod in plant.get('products', []):
                # Map periods to dates
                demand_by_date = network._map_period_to_date(prod['demand_by_period'])
                demand_std_by_date = network._map_period_to_date(prod['demand_std_by_period'])
                
                node.add_product(
                    prod['id'], prod['lead_time_mean'], prod['lead_time_std'],
                    prod['holding_cost'], prod['shortage_cost'],
                    demand_by_date, demand_std_by_date
                )
                
            network.add_node(node)
    
    @staticmethod
    def _create_dcs(network, dcs_data):
        """Create distribution center nodes from JSON data."""
        for dc in dcs_data:
            node = Node(dc['id'], 'dc')
            node.capacity = dc.get('capacity', 0)
            node.transport_cost = dc.get('transport_cost', 0)
            node.transport_variability = dc.get('transport_variability', 0)
            
            for prod in dc.get('products', []):
                demand_by_date = network._map_period_to_date(prod['demand_by_period'])
                demand_std_by_date = network._map_period_to_date(prod['demand_std_by_period'])
                
                node.add_product(
                    prod['id'], prod['lead_time_mean'], prod['lead_time_std'],
                    prod['holding_cost'], prod['shortage_cost'],
                    demand_by_date, demand_std_by_date
                )
                
            network.add_node(node)
    
    @staticmethod
    def _create_stores(network, stores_data):
        """Create store nodes from JSON data."""
        for store in stores_data:
            node = Node(store['id'], 'store')
            node.capacity = store.get('capacity', 0)
            node.transport_cost = store.get('transport_cost', 0)
            node.transport_variability = store.get('transport_variability', 0)
            
            for prod in store.get('products', []):
                demand_by_date = network._map_period_to_date(prod['demand_by_period'])
                demand_std_by_date = network._map_period_to_date(prod['demand_std_by_period'])
                
                node.add_product(
                    prod['id'], prod['lead_time_mean'], prod['lead_time_std'],
                    prod['holding_cost'], prod['shortage_cost'],
                    demand_by_date, demand_std_by_date
                )
                
            network.add_node(node)