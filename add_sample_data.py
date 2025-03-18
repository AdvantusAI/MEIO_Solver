from supabase import create_client
import json
from datetime import datetime, timedelta
import uuid

# Load database configuration
with open('meio/config/database_config.json') as f:
    config = json.load(f)

# Initialize Supabase client
supabase = create_client(config['supabase_url'], config['supabase_key'])

# Delete existing data for network_001
print("Deleting existing data...")
supabase.table('store_products').delete().eq('node_id', 'S1').execute()
supabase.table('store_products').delete().eq('node_id', 'S2').execute()
supabase.table('store_products').delete().eq('node_id', 'S3').execute()
supabase.table('plant_products').delete().eq('node_id', 'P1').execute()
supabase.table('dc_products').delete().eq('node_id', 'DC1').execute()
supabase.table('dc_products').delete().eq('node_id', 'DC2').execute()
supabase.table('connections').delete().eq('network_id', 'network_001').execute()
supabase.table('stores').delete().eq('network_id', 'network_001').execute()
supabase.table('distribution_centers').delete().eq('network_id', 'network_001').execute()
supabase.table('plants').delete().eq('network_id', 'network_001').execute()
print("Existing data deleted.")

# Add plants
plants = [
    {
        'id': 'P1',
        'network_id': 'network_001',
        'capacity': 10000
    }
]

for plant in plants:
    response = supabase.table('plants').insert(plant).execute()
    print(f"Added plant: {plant['id']}")

# Add plant products
plant_products = []
num_periods = 12
for plant in plants:
    plant_products.append({
        'node_id': plant['id'],
        'product_id': 'WATER003',
        'lead_time_mean': 2,
        'lead_time_std': 0.5,
        'holding_cost': 0.05,
        'shortage_cost': 2.0,
        'demand_by_period': [0] * num_periods,  # Plants don't have demand
        'demand_std_by_period': [0] * num_periods
    })

for prod in plant_products:
    response = supabase.table('plant_products').insert(prod).execute()
    print(f"Added product {prod['product_id']} for plant {prod['node_id']}")

# Add distribution centers
dcs = [
    {
        'id': 'DC1',
        'network_id': 'network_001',
        'capacity': 5000,
        'transport_cost': 0.3,
        'transport_variability': 0.1
    },
    {
        'id': 'DC2',
        'network_id': 'network_001',
        'capacity': 5000,
        'transport_cost': 0.4,
        'transport_variability': 0.15
    }
]

for dc in dcs:
    response = supabase.table('distribution_centers').insert(dc).execute()
    print(f"Added distribution center: {dc['id']}")

# Add DC products
dc_products = []
for dc in dcs:
    dc_products.append({
        'node_id': dc['id'],
        'product_id': 'WATER003',
        'lead_time_mean': 3,
        'lead_time_std': 0.8,
        'holding_cost': 0.08,
        'shortage_cost': 1.5,
        'demand_by_period': [0] * num_periods,  # DCs don't have end customer demand
        'demand_std_by_period': [0] * num_periods
    })

for prod in dc_products:
    response = supabase.table('dc_products').insert(prod).execute()
    print(f"Added product {prod['product_id']} for DC {prod['node_id']}")

# Add stores
stores = [
    {
        'id': 'S1',
        'network_id': 'network_001',
        'capacity': 2000,
        'transport_cost': 0.5,
        'transport_variability': 0.2
    },
    {
        'id': 'S2',
        'network_id': 'network_001',
        'capacity': 3000,
        'transport_cost': 0.6,
        'transport_variability': 0.3
    },
    {
        'id': 'S3',
        'network_id': 'network_001',
        'capacity': 4000,
        'transport_cost': 0.7,
        'transport_variability': 0.25
    }
]

for store in stores:
    response = supabase.table('stores').insert(store).execute()
    print(f"Added store: {store['id']}")

# Add connections
connections = [
    # Plant to DC connections
    {
        'id': str(uuid.uuid4()),
        'network_id': 'network_001',
        'from': 'P1',
        'to': 'DC1'
    },
    {
        'id': str(uuid.uuid4()),
        'network_id': 'network_001',
        'from': 'P1',
        'to': 'DC2'
    },
    # DC to Store connections
    {
        'id': str(uuid.uuid4()),
        'network_id': 'network_001',
        'from': 'DC1',
        'to': 'S1'
    },
    {
        'id': str(uuid.uuid4()),
        'network_id': 'network_001',
        'from': 'DC1',
        'to': 'S2'
    },
    {
        'id': str(uuid.uuid4()),
        'network_id': 'network_001',
        'from': 'DC2',
        'to': 'S2'
    },
    {
        'id': str(uuid.uuid4()),
        'network_id': 'network_001',
        'from': 'DC2',
        'to': 'S3'
    }
]

for conn in connections:
    response = supabase.table('connections').insert(conn).execute()
    print(f"Added connection: {conn['id']}")

# Add store products
store_products = []
start_date = datetime.now()
dates = [start_date + timedelta(days=i * 30) for i in range(num_periods)]

for store in stores:
    store_products.append({
        'node_id': store['id'],
        'product_id': 'WATER003',  # Using product_id WATER003 (Agua mineral 600 ml)
        'lead_time_mean': 5,
        'lead_time_std': 1,
        'holding_cost': 0.1,
        'shortage_cost': 1.0,
        'demand_by_period': [100 * (i + 1) for i in range(num_periods)],
        'demand_std_by_period': [20 * (i + 1) for i in range(num_periods)]
    })

for prod in store_products:
    response = supabase.table('store_products').insert(prod).execute()
    print(f"Added product {prod['product_id']} for store {prod['node_id']}") 