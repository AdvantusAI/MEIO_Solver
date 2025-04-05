import os
from supabase import create_client, Client
import json
import logging
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('create_tables')

logger.info("Starting table creation script")

# Load database configuration
try:
    with open('meio/config/database_config.json', 'r') as f:
        db_config = json.load(f)
    logger.info(f"Successfully loaded database config from meio/config/database_config.json")
    logger.info(f"Using Supabase URL: {db_config['supabase_url']}")
    # Don't log the full key for security reasons
    logger.info(f"Supabase key present: {'supabase_key' in db_config}")
except Exception as e:
    logger.error(f"Failed to load database config: {str(e)}")
    sys.exit(1)

# Initialize Supabase client
try:
    supabase: Client = create_client(
        db_config['supabase_url'],
        db_config['supabase_key']
    )
    logger.info("Successfully initialized Supabase client")
except Exception as e:
    logger.error(f"Failed to initialize Supabase client: {str(e)}")
    sys.exit(1)

# Helper function to create tables with detailed logging
def try_create_table(table_name, sample_data):
    logger.info(f"Attempting to create/verify {table_name} table")
    try:
        # Test the table existence first
        try:
            # Just fetch a single row to see if the table exists
            test_response = supabase.table(table_name).select('*').limit(1).execute()
            logger.info(f"Table {table_name} exists, returned {len(test_response.data)} rows")
        except Exception as test_e:
            logger.warning(f"Table {table_name} query test failed: {str(test_e)}")
        
        # Try to upsert data to create/verify the table
        response = supabase.table(table_name).upsert(sample_data).execute()
        
        if response.data:
            logger.info(f"Successfully upserted data to {table_name} table")
            logger.info(f"Response data: {response.data}")
        else:
            logger.warning(f"No data returned from {table_name} upsert operation")
            logger.warning(f"Full response: {response}")
        
        return True
    except Exception as e:
        logger.error(f"Error with {table_name} table: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Sample data attempted: {sample_data}")
        return False

# Create locations table
try_create_table('locations', {
    'location_id': 'LOC001',
    'location_name': 'Main Distribution Center',
    'city': 'New York',
    'region': 'Northeast',
    'type': 'distribution_center',
    'working_cal': 'Standard Calendar',
    'borrowing_pct': 0.05,
    'updated_at': '2024-04-04T10:00:00'
})

# Create products table
try_create_table('products', {
    'product_id': 'TEST001',
    'category_id': 'CAT001',
    'category_name': 'Test Category',
    'subcategory_id': 'SUBCAT001',
    'subcategory_name': 'Test Subcategory',
    'product_name': 'Test Product',
    'item_brand': 'Test Brand',
    'item_uom': 'EA',
    'item_weight': 1.0,
    'item_volume': 1.0,
    'item_cost': 10.0,
    'item_price': 20.0,
    'item_status': 'Active'
})

# Create optimization_runs table
try_create_table('optimization_runs', {
    'id': '00000000-0000-0000-0000-000000000000',
    'network_id': 'test',
    'method': 'test',
    'service_level': 0.95,
    'start_date': '2024-03-18',
    'end_date': '2025-03-18',
    'total_cost': 0,
    'status': 'test'
})

# Create inventory_levels table
try_create_table('inventory_levels', {
    'id': '00000000-0000-0000-0000-000000000000',
    'optimization_run_id': '00000000-0000-0000-0000-000000000000',
    'node_id': 'test',
    'product_id': 'test',
    'period': 1,
    'inventory_level': 0
})

# Create stock_alerts table
try_create_table('stock_alerts', {
    'id': '00000000-0000-0000-0000-000000000000',
    'optimization_run_id': '00000000-0000-0000-0000-000000000000',
    'node_id': 'test',
    'product_id': 'test',
    'alert_type': 'test',
    'period': 1,
    'inventory_level': 0,
    'demand': 0,
    'capacity': 0,
    'shortfall': 0,
    'excess': 0
})

# Create network_statistics table
try_create_table('network_statistics', {
    'id': '00000000-0000-0000-0000-000000000000',
    'network_id': 'test',
    'node_id': 'test',
    'product_id': 'test',
    'total_demand': 0,
    'avg_demand': 0,
    'demand_std': 0,
    'avg_lead_time': 0,
    'lead_time_std': 0,
    'holding_cost': 0,
    'shortage_cost': 0
})

# Create safety_stock_recommendations table
try_create_table('safety_stock_recommendations', {
    'id': '00000000-0000-0000-0000-000000000000',
    'network_id': 'test',
    'node_id': 'test',
    'product_id': 'test',
    'safety_stock': 0,
    'service_level': 0.95
})

# Create ai_recommendations table
try_create_table('ai_recommendations', {
    'id': '00000000-0000-0000-0000-000000000000',
    'network_id': 'test',
    'node_id': 'test',
    'product_id': 'test',
    'analysis': 'test analysis',
    'recommendations': json.dumps({'suggestion': 'test recommendation'})
})

# Verify connection is working by selecting from a system table
try:
    schema_response = supabase.table('information_schema.tables').select('table_name').execute()
    if schema_response.data:
        logger.info(f"Database connection verified. Found {len(schema_response.data)} tables in information_schema")
        table_names = [row.get('table_name') for row in schema_response.data]
        logger.info(f"Sample tables: {table_names[:5] if len(table_names) > 5 else table_names}")
    else:
        logger.warning("Database connection verified but no tables found in information_schema")
except Exception as e:
    logger.error(f"Failed to verify database connection: {str(e)}")

logger.info("Finished checking tables") 