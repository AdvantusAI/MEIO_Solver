-- Create extension for UUID generation if it doesn't exist
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create location_type enum if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'location_type') THEN
        CREATE TYPE public.location_type AS ENUM ('plant', 'warehouse', 'store', 'distribution_center');
    END IF;
END
$$;

-- Drop tables if they exist (comment out in production to avoid accidental deletion)
-- DROP TABLE IF EXISTS ai_recommendations;
-- DROP TABLE IF EXISTS safety_stock_recommendations;
-- DROP TABLE IF EXISTS network_statistics;
-- DROP TABLE IF EXISTS stock_alerts;
-- DROP TABLE IF EXISTS inventory_levels;
-- DROP TABLE IF EXISTS optimization_runs;
-- DROP TABLE IF EXISTS products;
-- DROP TABLE IF EXISTS locations;

-- Create locations table
CREATE TABLE IF NOT EXISTS locations (
    location_id TEXT NOT NULL,
    location_name TEXT NULL,
    city TEXT NULL,
    region TEXT NULL,
    type public.location_type NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
    working_cal TEXT NULL,
    borrowing_pct DOUBLE PRECISION NULL,
    updated_at TIMESTAMP WITHOUT TIME ZONE NULL,
    CONSTRAINT locations_pkey PRIMARY KEY (location_id),
    CONSTRAINT locations_location_id_key UNIQUE (location_id)
);

-- Create products table
CREATE TABLE IF NOT EXISTS products (
    category_id TEXT NULL,
    category_name TEXT NULL,
    subcategory_id TEXT NULL,
    subcategory_name TEXT NULL,
    product_id TEXT NOT NULL,
    product_name TEXT NULL,
    item_brand TEXT NULL,
    item_uom TEXT NULL,
    item_weight DOUBLE PRECISION NULL,
    item_volume DOUBLE PRECISION NULL,
    item_cost DOUBLE PRECISION NULL,
    item_price DOUBLE PRECISION NULL,
    item_status TEXT NULL,
    CONSTRAINT products_pkey PRIMARY KEY (product_id)
);

-- Create optimization_runs table
CREATE TABLE IF NOT EXISTS optimization_runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    network_id TEXT NOT NULL,
    method TEXT NOT NULL,
    service_level DECIMAL(5,2) NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    total_cost DECIMAL(15,2) NOT NULL,
    status TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc', NOW())
);

-- Create inventory_levels table
CREATE TABLE IF NOT EXISTS inventory_levels (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    optimization_run_id UUID NOT NULL REFERENCES optimization_runs(id) ON DELETE CASCADE,
    node_id TEXT NOT NULL,
    product_id TEXT NOT NULL,
    period INTEGER NOT NULL,
    inventory_level DECIMAL(15,2) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc', NOW()),
    UNIQUE(optimization_run_id, node_id, product_id, period)
);

-- Create stock_alerts table
CREATE TABLE IF NOT EXISTS stock_alerts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    optimization_run_id UUID NOT NULL REFERENCES optimization_runs(id) ON DELETE CASCADE,
    node_id TEXT NOT NULL,
    product_id TEXT NOT NULL,
    alert_type TEXT NOT NULL,
    period INTEGER NOT NULL,
    inventory_level DECIMAL(15,2) NOT NULL,
    demand DECIMAL(15,2) NOT NULL,
    capacity DECIMAL(15,2) NOT NULL,
    shortfall DECIMAL(15,2) NOT NULL,
    excess DECIMAL(15,2) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc', NOW()),
    UNIQUE(optimization_run_id, node_id, product_id, period, alert_type)
);

-- Create network_statistics table
CREATE TABLE IF NOT EXISTS network_statistics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    network_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    product_id TEXT NOT NULL,
    total_demand DECIMAL(15,2) NOT NULL,
    avg_demand DECIMAL(15,2) NOT NULL,
    demand_std DECIMAL(15,2) NOT NULL,
    avg_lead_time DECIMAL(15,2) NOT NULL,
    lead_time_std DECIMAL(15,2) NOT NULL,
    holding_cost DECIMAL(15,2) NOT NULL,
    shortage_cost DECIMAL(15,2) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc', NOW()),
    UNIQUE(network_id, node_id, product_id)
);

-- Create safety_stock_recommendations table
CREATE TABLE IF NOT EXISTS safety_stock_recommendations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    network_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    product_id TEXT NOT NULL,
    safety_stock JSONB NOT NULL, -- Changed from DECIMAL to JSONB to support complex safety stock data
    service_level DECIMAL(5,2) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc', NOW()),
    UNIQUE(network_id, node_id, product_id, service_level)
);

-- Create ai_recommendations table
CREATE TABLE IF NOT EXISTS ai_recommendations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    network_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    product_id TEXT NOT NULL,
    analysis TEXT NOT NULL,
    recommendations JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc', NOW()),
    UNIQUE(network_id, node_id, product_id)
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_locations_type ON locations(type);
CREATE INDEX IF NOT EXISTS idx_locations_region ON locations(region);
CREATE INDEX IF NOT EXISTS idx_products_category ON products(category_id);
CREATE INDEX IF NOT EXISTS idx_products_subcategory ON products(subcategory_id);
CREATE INDEX IF NOT EXISTS idx_optimization_runs_network_id ON optimization_runs(network_id);
CREATE INDEX IF NOT EXISTS idx_inventory_levels_optimization_run_id ON inventory_levels(optimization_run_id);
CREATE INDEX IF NOT EXISTS idx_inventory_levels_node_product ON inventory_levels(node_id, product_id);
CREATE INDEX IF NOT EXISTS idx_stock_alerts_optimization_run_id ON stock_alerts(optimization_run_id);
CREATE INDEX IF NOT EXISTS idx_stock_alerts_node_product ON stock_alerts(node_id, product_id);
CREATE INDEX IF NOT EXISTS idx_network_statistics_network_id ON network_statistics(network_id);
CREATE INDEX IF NOT EXISTS idx_network_statistics_node_product ON network_statistics(node_id, product_id);
CREATE INDEX IF NOT EXISTS idx_safety_stock_recommendations_network_id ON safety_stock_recommendations(network_id);
CREATE INDEX IF NOT EXISTS idx_safety_stock_recommendations_node_product ON safety_stock_recommendations(node_id, product_id);
CREATE INDEX IF NOT EXISTS idx_ai_recommendations_network_id ON ai_recommendations(network_id);
CREATE INDEX IF NOT EXISTS idx_ai_recommendations_node_product ON ai_recommendations(node_id, product_id);

-- Enable Row-Level Security (RLS) for all tables
ALTER TABLE locations ENABLE ROW LEVEL SECURITY;
ALTER TABLE products ENABLE ROW LEVEL SECURITY;
ALTER TABLE optimization_runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE inventory_levels ENABLE ROW LEVEL SECURITY;
ALTER TABLE stock_alerts ENABLE ROW LEVEL SECURITY;
ALTER TABLE network_statistics ENABLE ROW LEVEL SECURITY;
ALTER TABLE safety_stock_recommendations ENABLE ROW LEVEL SECURITY;
ALTER TABLE ai_recommendations ENABLE ROW LEVEL SECURITY;

-- Create policies to control access
-- For locations
CREATE POLICY "Enable read access for all users" ON locations FOR SELECT USING (true);
CREATE POLICY "Enable insert for authenticated users" ON locations FOR INSERT WITH CHECK (auth.role() = 'authenticated');
CREATE POLICY "Enable update for authenticated users" ON locations FOR UPDATE USING (auth.role() = 'authenticated');

-- For products
CREATE POLICY "Enable read access for all users" ON products FOR SELECT USING (true);
CREATE POLICY "Enable insert for authenticated users" ON products FOR INSERT WITH CHECK (auth.role() = 'authenticated');
CREATE POLICY "Enable update for authenticated users" ON products FOR UPDATE USING (auth.role() = 'authenticated');

-- For optimization_runs
CREATE POLICY "Enable read access for all users" ON optimization_runs FOR SELECT USING (true);
CREATE POLICY "Enable insert for authenticated users" ON optimization_runs FOR INSERT WITH CHECK (auth.role() = 'authenticated');

-- For inventory_levels
CREATE POLICY "Enable read access for all users" ON inventory_levels FOR SELECT USING (true);
CREATE POLICY "Enable insert for authenticated users" ON inventory_levels FOR INSERT WITH CHECK (auth.role() = 'authenticated');

-- For stock_alerts
CREATE POLICY "Enable read access for all users" ON stock_alerts FOR SELECT USING (true);
CREATE POLICY "Enable insert for authenticated users" ON stock_alerts FOR INSERT WITH CHECK (auth.role() = 'authenticated');

-- For network_statistics
CREATE POLICY "Enable read access for all users" ON network_statistics FOR SELECT USING (true);
CREATE POLICY "Enable insert for authenticated users" ON network_statistics FOR INSERT WITH CHECK (auth.role() = 'authenticated');

-- For safety_stock_recommendations
CREATE POLICY "Enable read access for all users" ON safety_stock_recommendations FOR SELECT USING (true);
CREATE POLICY "Enable insert for authenticated users" ON safety_stock_recommendations FOR INSERT WITH CHECK (auth.role() = 'authenticated');

-- For ai_recommendations
CREATE POLICY "Enable read access for all users" ON ai_recommendations FOR SELECT USING (true);
CREATE POLICY "Enable insert for authenticated users" ON ai_recommendations FOR INSERT WITH CHECK (auth.role() = 'authenticated');

-- Grant necessary permissions
GRANT ALL ON locations TO authenticated;
GRANT ALL ON products TO authenticated;
GRANT ALL ON optimization_runs TO authenticated;
GRANT ALL ON inventory_levels TO authenticated;
GRANT ALL ON stock_alerts TO authenticated;
GRANT ALL ON network_statistics TO authenticated;
GRANT ALL ON safety_stock_recommendations TO authenticated;
GRANT ALL ON ai_recommendations TO authenticated; 