from supabase import create_client
import json

# Load database configuration
with open('meio/config/database_config.json') as f:
    config = json.load(f)

# Initialize Supabase client
supabase = create_client(config['supabase_url'], config['supabase_key'])

# Query networks table
response = supabase.table('networks').select('*').execute()
print("Networks:", response.data)

# Query plants table
response = supabase.table('plants').select('*').execute()
print("\nPlants:", response.data)

# Query distribution centers table
response = supabase.table('distribution_centers').select('*').execute()
print("\nDistribution Centers:", response.data)

# Query stores table
response = supabase.table('stores').select('*').execute()
print("\nStores:", response.data)

# Query connections table
response = supabase.table('connections').select('*').execute()
print("\nConnections:", response.data) 