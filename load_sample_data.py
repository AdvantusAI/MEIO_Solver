import os
import json
import logging
import sys
import random
from datetime import datetime, timedelta
from supabase import create_client, Client

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('load_sample_data')

# Load database configuration
try:
    with open('meio/config/database_config.json', 'r') as f:
        db_config = json.load(f)
    logger.info(f"Successfully loaded database config from meio/config/database_config.json")
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

# Helper function to insert data with error handling
def insert_data(table_name, data_list):
    logger.info(f"Inserting {len(data_list)} records into {table_name}")
    try:
        response = supabase.table(table_name).upsert(data_list).execute()
        if response.data:
            logger.info(f"Successfully inserted data into {table_name} table")
            logger.info(f"Inserted {len(response.data)} records")
            return True
        else:
            logger.warning(f"No data returned from {table_name} insert operation")
            return False
    except Exception as e:
        logger.error(f"Error inserting data into {table_name}: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        return False

# Sample data for locations
locations_data = [
    {
        'location_id': 'P1',
        'location_name': 'Plant 1',
        'city': 'Chicago',
        'region': 'Midwest',
        'type': 'plant',
        'working_cal': 'Standard-5-2',
        'borrowing_pct': 0.0
    },
    {
        'location_id': 'DC1',
        'location_name': 'Distribution Center 1',
        'city': 'Atlanta',
        'region': 'Southeast',
        'type': 'distribution_center',
        'working_cal': 'Standard-6-1',
        'borrowing_pct': 0.03
    },
    {
        'location_id': 'DC2',
        'location_name': 'Distribution Center 2',
        'city': 'Dallas',
        'region': 'Southwest',
        'type': 'distribution_center',
        'working_cal': 'Standard-6-1',
        'borrowing_pct': 0.02
    },
    {
        'location_id': 'S1',
        'location_name': 'Store 1',
        'city': 'New York',
        'region': 'Northeast',
        'type': 'store',
        'working_cal': 'Retail-7-0',
        'borrowing_pct': 0.05
    },
    {
        'location_id': 'S2',
        'location_name': 'Store 2',
        'city': 'Los Angeles',
        'region': 'West',
        'type': 'store',
        'working_cal': 'Retail-7-0',
        'borrowing_pct': 0.04
    },
    {
        'location_id': 'S3',
        'location_name': 'Store 3',
        'city': 'Miami',
        'region': 'Southeast',
        'type': 'store',
        'working_cal': 'Retail-7-0',
        'borrowing_pct': 0.06
    }
]

# Sample data for products
products_data = [
    {
        'product_id': 'WATER001',
        'product_name': 'Mineral Water 500ml',
        'category_id': 'BEV',
        'category_name': 'Beverages',
        'subcategory_id': 'WATER',
        'subcategory_name': 'Water',
        'item_brand': 'AquaPure',
        'item_uom': 'EA',
        'item_weight': 0.5,
        'item_volume': 0.5,
        'item_cost': 0.25,
        'item_price': 0.99,
        'item_status': 'Active'
    },
    {
        'product_id': 'WATER002',
        'product_name': 'Mineral Water 1L',
        'category_id': 'BEV',
        'category_name': 'Beverages',
        'subcategory_id': 'WATER',
        'subcategory_name': 'Water',
        'item_brand': 'AquaPure',
        'item_uom': 'EA',
        'item_weight': 1.0,
        'item_volume': 1.0,
        'item_cost': 0.40,
        'item_price': 1.49,
        'item_status': 'Active'
    },
    {
        'product_id': 'WATER003',
        'product_name': 'Sparkling Water 750ml',
        'category_id': 'BEV',
        'category_name': 'Beverages',
        'subcategory_id': 'WATER',
        'subcategory_name': 'Water',
        'item_brand': 'BubbleFizz',
        'item_uom': 'EA',
        'item_weight': 0.8,
        'item_volume': 0.75,
        'item_cost': 0.60,
        'item_price': 1.99,
        'item_status': 'Active'
    },
    {
        'product_id': 'SODA001',
        'product_name': 'Cola 330ml',
        'category_id': 'BEV',
        'category_name': 'Beverages',
        'subcategory_id': 'SODA',
        'subcategory_name': 'Soda',
        'item_brand': 'FizzCo',
        'item_uom': 'EA',
        'item_weight': 0.35,
        'item_volume': 0.33,
        'item_cost': 0.30,
        'item_price': 1.29,
        'item_status': 'Active'
    },
    {
        'product_id': 'JUICE001',
        'product_name': 'Orange Juice 1L',
        'category_id': 'BEV',
        'category_name': 'Beverages',
        'subcategory_id': 'JUICE',
        'subcategory_name': 'Juice',
        'item_brand': 'FreshSqueeze',
        'item_uom': 'EA',
        'item_weight': 1.05,
        'item_volume': 1.0,
        'item_cost': 1.20,
        'item_price': 2.99,
        'item_status': 'Active'
    }
]

# Generate a network
network_id = 'network_001'

# Generate optimization run
start_date = datetime.strptime('2024-03-18', '%Y-%m-%d')
end_date = datetime.strptime('2025-03-18', '%Y-%m-%d')
optimization_run = {
    'id': '68f126e0-6c9a-4f62-9f5f-5b846e607499',
    'network_id': network_id,
    'method': 'mathematical',
    'service_level': 0.95,
    'start_date': start_date.strftime('%Y-%m-%d'),
    'end_date': end_date.strftime('%Y-%m-%d'),
    'total_cost': 42500.00,
    'status': 'completed'
}

# Generate some inventory levels
inventory_levels = []
for location in locations_data:
    for product in products_data:
        for period in range(1, 13):  # 12 periods
            base_inventory = random.randint(100, 500)
            if location['type'] == 'plant':
                base_inventory *= 2  # Plants have more inventory
            elif location['type'] == 'distribution_center':
                base_inventory *= 1.5  # DCs have moderate inventory
            
            inventory_levels.append({
                'optimization_run_id': optimization_run['id'],
                'node_id': location['location_id'],
                'product_id': product['product_id'],
                'period': period,
                'inventory_level': base_inventory + random.randint(-50, 50)
            })

# Generate some network statistics
network_statistics = []
for location in locations_data:
    for product in products_data:
        avg_demand = random.randint(50, 300)
        demand_std = avg_demand * random.uniform(0.1, 0.3)  # 10-30% variability
        
        network_statistics.append({
            'network_id': network_id,
            'node_id': location['location_id'],
            'product_id': product['product_id'],
            'total_demand': avg_demand * 12,  # Annual demand
            'avg_demand': avg_demand,
            'demand_std': demand_std,
            'avg_lead_time': random.uniform(1, 5),
            'lead_time_std': random.uniform(0.2, 1.0),
            'holding_cost': product['item_cost'] * 0.25,  # 25% of cost
            'shortage_cost': product['item_price'] * 0.5  # 50% of price
        })

# Generate safety stock recommendations
safety_stock_recommendations = []
for location in locations_data:
    for product in products_data:
        # Base safety stock on location type
        if location['type'] == 'plant':
            ss_value = random.uniform(0, 10)
        elif location['type'] == 'distribution_center':
            ss_value = random.uniform(10, 50)
        else:  # store
            ss_value = random.uniform(50, 200)
            
        safety_stock_recommendations.append({
            'network_id': network_id,
            'node_id': location['location_id'],
            'product_id': product['product_id'],
            'safety_stock': json.dumps({
                'avg_safety_stock': ss_value,
                'min_safety_stock': ss_value * 0.8,
                'max_safety_stock': ss_value * 1.2
            }),
            'service_level': 0.95
        })

# Generate AI recommendations
ai_recommendations = []
for location in locations_data:
    for product in products_data:
        # Create different recommendations based on location type
        if location['type'] == 'plant':
            analysis = f"Plant {location['location_id']} has sufficient capacity for {product['product_name']}. Production planning is optimized."
            recommendations = {
                "inventory_management": [
                    {"type": "maintain_inventory", "reason": "Current inventory levels are optimal", "suggestion": "Continue with current production schedule"}
                ],
                "safety_stock": [
                    {"type": "review_safety_stock", "reason": "Safety stock levels may be higher than necessary", "suggestion": "Consider reducing safety stock by 10%"}
                ],
                "supply_chain": [
                    {"type": "improve_forecasting", "reason": "Production planning can be improved", "suggestion": "Implement more accurate forecasting models"}
                ]
            }
        elif location['type'] == 'distribution_center':
            analysis = f"Distribution Center {location['location_id']} shows moderate demand variability for {product['product_name']}. Lead times from suppliers are stable."
            recommendations = {
                "inventory_management": [
                    {"type": "optimize_replenishment", "reason": "Replenishment frequency can be optimized", "suggestion": "Adjust replenishment frequency to every 5 days"}
                ],
                "safety_stock": [
                    {"type": "maintain_safety_stock", "reason": "Current safety stock levels are appropriate", "suggestion": "Maintain current safety stock levels"}
                ],
                "supply_chain": [
                    {"type": "improve_transportation", "reason": "Transportation costs can be reduced", "suggestion": "Consolidate shipments to reduce transportation costs"}
                ]
            }
        else:  # store
            analysis = f"Store {location['location_id']} shows high demand variability for {product['product_name']}. Seasonal patterns are observed."
            recommendations = {
                "inventory_management": [
                    {"type": "increase_inventory", "reason": "Seasonal demand increase expected", "suggestion": "Increase inventory by 15% for the upcoming peak season"}
                ],
                "safety_stock": [
                    {"type": "adjust_safety_stock", "reason": "High demand variability", "suggestion": "Increase safety stock to cover 7 days of average demand"}
                ],
                "supply_chain": [
                    {"type": "diversify_suppliers", "reason": "Single supplier risk", "suggestion": "Consider adding a secondary supplier for this product"}
                ]
            }
            
        ai_recommendations.append({
            'network_id': network_id,
            'node_id': location['location_id'],
            'product_id': product['product_id'],
            'analysis': analysis,
            'recommendations': json.dumps(recommendations)
        })

# Load data into tables
logger.info("Starting data loading process")

# Load locations
insert_data('locations', locations_data)

# Load products
insert_data('products', products_data)

# Load optimization run
insert_data('optimization_runs', [optimization_run])

# Load inventory levels (in batches to avoid large payloads)
batch_size = 100
for i in range(0, len(inventory_levels), batch_size):
    batch = inventory_levels[i:i+batch_size]
    insert_data('inventory_levels', batch)

# Load network statistics
insert_data('network_statistics', network_statistics)

# Load safety stock recommendations
insert_data('safety_stock_recommendations', safety_stock_recommendations)

# Load AI recommendations
insert_data('ai_recommendations', ai_recommendations)

logger.info("Finished loading sample data") 