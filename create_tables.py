import os
from supabase import create_client, Client
import json

# Load database configuration
with open('meio/config/database_config.json', 'r') as f:
    db_config = json.load(f)

# Initialize Supabase client
supabase: Client = create_client(
    db_config['supabase_url'],
    db_config['supabase_key']
)

# Create optimization_runs table
try:
    response = supabase.table('optimization_runs').upsert({
        'id': '00000000-0000-0000-0000-000000000000',
        'network_id': 'test',
        'method': 'test',
        'service_level': 0.95,
        'start_date': '2024-03-18',
        'end_date': '2025-03-18',
        'total_cost': 0,
        'status': 'test'
    }).execute()
    print("optimization_runs table exists or was created")
except Exception as e:
    print(f"Error with optimization_runs table: {e}")

# Create inventory_levels table
try:
    response = supabase.table('inventory_levels').upsert({
        'id': '00000000-0000-0000-0000-000000000000',
        'optimization_run_id': '00000000-0000-0000-0000-000000000000',
        'node_id': 'test',
        'product_id': 'test',
        'period': 1,
        'inventory_level': 0
    }).execute()
    print("inventory_levels table exists or was created")
except Exception as e:
    print(f"Error with inventory_levels table: {e}")

# Create stock_alerts table
try:
    response = supabase.table('stock_alerts').upsert({
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
    }).execute()
    print("stock_alerts table exists or was created")
except Exception as e:
    print(f"Error with stock_alerts table: {e}")

# Create network_statistics table
try:
    response = supabase.table('network_statistics').upsert({
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
    }).execute()
    print("network_statistics table exists or was created")
except Exception as e:
    print(f"Error with network_statistics table: {e}")

# Create safety_stock_recommendations table
try:
    response = supabase.table('safety_stock_recommendations').upsert({
        'id': '00000000-0000-0000-0000-000000000000',
        'network_id': 'test',
        'node_id': 'test',
        'product_id': 'test',
        'safety_stock': 0,
        'service_level': 0.95
    }).execute()
    print("safety_stock_recommendations table exists or was created")
except Exception as e:
    print(f"Error with safety_stock_recommendations table: {e}")

print("Finished checking tables") 