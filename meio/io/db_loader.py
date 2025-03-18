"""
Database loader for the MEIO system using Supabase.
"""
import logging
from datetime import datetime
from supabase import create_client, Client
from ..models.node import Node
from ..models.network import MultiEchelonNetwork

logger = logging.getLogger(__name__)

class NetworkDBLoader:
    """Loads network data from Supabase database."""
    
    def __init__(self, supabase_url: str, supabase_key: str):
        """
        Initialize the database loader.
        
        Args:
            supabase_url (str): Supabase project URL
            supabase_key (str): Supabase API key
        """
        self.supabase: Client = create_client(supabase_url, supabase_key)
    
    def load(self, network_id: str, start_date=None, end_date=None, date_interval=30):
        """
        Load a network from the database.
        
        Args:
            network_id (str): ID of the network to load
            start_date (datetime, optional): Start date. Defaults to None (today).
            end_date (datetime, optional): End date. Defaults to None.
            date_interval (int, optional): Days between periods. Defaults to 30.
            
        Returns:
            MultiEchelonNetwork: The loaded network.
            
        Raises:
            ValueError: If the data isn't valid.
        """
        # Create network
        network = MultiEchelonNetwork(start_date, end_date, date_interval)
        
        try:
            # Load plants
            plants_data = self._load_plants(network_id)
            self._create_plants(network, plants_data)
            
            # Load distribution centers
            dcs_data = self._load_dcs(network_id)
            self._create_dcs(network, dcs_data)
            
            # Load stores
            stores_data = self._load_stores(network_id)
            self._create_stores(network, stores_data)
            
            # Load connections
            connections = self._load_connections(network_id)
            for connection in connections:
                child_id = connection['to']
                parent_id = connection['from']
                try:
                    network.add_connection(parent_id, child_id)
                except ValueError as e:
                    logger.warning(f"Invalid connection {parent_id} -> {child_id}: {e}")
            
            # Validate the network
            network.validate()
            
            return network
            
        except Exception as e:
            logger.error(f"Error loading network from database: {str(e)}")
            raise
    
    def _load_plants(self, network_id: str):
        """Load plant data from database."""
        response = self.supabase.table('plants').select('*').eq('network_id', network_id).execute()
        return response.data
    
    def _load_dcs(self, network_id: str):
        """Load distribution center data from database."""
        response = self.supabase.table('distribution_centers').select('*').eq('network_id', network_id).execute()
        return response.data
    
    def _load_stores(self, network_id: str):
        """Load store data from database."""
        response = self.supabase.table('stores').select('*').eq('network_id', network_id).execute()
        return response.data
    
    def _load_connections(self, network_id: str):
        """Load connection data from database."""
        response = self.supabase.table('connections').select('*').eq('network_id', network_id).execute()
        return response.data
    
    def _load_products(self, node_type: str, node_id: str):
        """Load product data for a specific node from database."""
        table_name = f"{node_type}_products"
        response = self.supabase.table(table_name).select('*').eq('node_id', node_id).execute()
        return response.data
    
    def _create_plants(self, network, plants_data):
        """Create plant nodes from database data."""
        for plant in plants_data:
            node = Node(plant['id'], 'plant')
            node.capacity = plant.get('capacity', 0)
            
            # Load products for this plant
            products = self._load_products('plant', plant['id'])
            for prod in products:
                # Map periods to dates
                demand_by_date = network._map_period_to_date(prod['demand_by_period'])
                demand_std_by_date = network._map_period_to_date(prod['demand_std_by_period'])
                
                node.add_product(
                    prod['product_id'], prod['lead_time_mean'], prod['lead_time_std'],
                    prod['holding_cost'], prod['shortage_cost'],
                    demand_by_date, demand_std_by_date
                )
                
            network.add_node(node)
    
    def _create_dcs(self, network, dcs_data):
        """Create distribution center nodes from database data."""
        for dc in dcs_data:
            node = Node(dc['id'], 'dc')
            node.capacity = dc.get('capacity', 0)
            node.transport_cost = dc.get('transport_cost', 0)
            node.transport_variability = dc.get('transport_variability', 0)
            
            # Load products for this DC
            products = self._load_products('dc', dc['id'])
            for prod in products:
                demand_by_date = network._map_period_to_date(prod['demand_by_period'])
                demand_std_by_date = network._map_period_to_date(prod['demand_std_by_period'])
                
                node.add_product(
                    prod['product_id'], prod['lead_time_mean'], prod['lead_time_std'],
                    prod['holding_cost'], prod['shortage_cost'],
                    demand_by_date, demand_std_by_date
                )
                
            network.add_node(node)
    
    def _create_stores(self, network, stores_data):
        """Create store nodes from database data."""
        for store in stores_data:
            node = Node(store['id'], 'store')
            node.capacity = store.get('capacity', 0)
            node.transport_cost = store.get('transport_cost', 0)
            node.transport_variability = store.get('transport_variability', 0)
            
            # Load products for this store
            products = self._load_products('store', store['id'])
            for prod in products:
                demand_by_date = network._map_period_to_date(prod['demand_by_period'])
                demand_std_by_date = network._map_period_to_date(prod['demand_std_by_period'])
                
                node.add_product(
                    prod['product_id'], prod['lead_time_mean'], prod['lead_time_std'],
                    prod['holding_cost'], prod['shortage_cost'],
                    demand_by_date, demand_std_by_date
                )
                
            network.add_node(node) 