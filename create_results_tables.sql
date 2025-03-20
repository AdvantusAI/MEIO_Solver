-- Enable UUID extension if not already enabled
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Drop existing tables if they exist
DROP TABLE IF EXISTS public.safety_stock_recommendations;
DROP TABLE IF EXISTS public.network_statistics;
DROP TABLE IF EXISTS public.stock_alerts;
DROP TABLE IF EXISTS public.inventory_levels;
DROP TABLE IF EXISTS public.optimization_runs;
DROP TABLE IF EXISTS public.ai_recommendations;

-- Table for optimization runs
CREATE TABLE public.optimization_runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    network_id TEXT NOT NULL,
    method TEXT NOT NULL,
    service_level DECIMAL NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    total_cost DECIMAL NOT NULL,
    status TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Table for inventory levels by period
CREATE TABLE public.inventory_levels (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    optimization_run_id UUID NOT NULL,
    node_id TEXT NOT NULL,
    product_id TEXT NOT NULL,
    period INTEGER NOT NULL,
    inventory_level DECIMAL NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Table for stock alerts (stockouts and overstocks)
CREATE TABLE public.stock_alerts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    optimization_run_id UUID NOT NULL,
    node_id TEXT NOT NULL,
    product_id TEXT NOT NULL,
    alert_type TEXT NOT NULL,
    period INTEGER NOT NULL,
    inventory_level DECIMAL NOT NULL,
    demand DECIMAL,
    capacity DECIMAL,
    shortfall DECIMAL,
    excess DECIMAL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Table for network statistics
CREATE TABLE public.network_statistics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    network_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    product_id TEXT NOT NULL,
    total_demand DECIMAL NOT NULL,
    avg_demand DECIMAL NOT NULL,
    demand_std DECIMAL NOT NULL,
    avg_lead_time DECIMAL NOT NULL,
    lead_time_std DECIMAL NOT NULL,
    holding_cost DECIMAL NOT NULL,
    shortage_cost DECIMAL NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Table for safety stock recommendations
CREATE TABLE public.safety_stock_recommendations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    network_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    product_id TEXT NOT NULL,
    safety_stock JSONB NOT NULL,
    service_level DECIMAL NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Table for AI recommendations
CREATE TABLE public.ai_recommendations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    network_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    product_id TEXT NOT NULL,
    analysis TEXT NOT NULL,
    recommendations JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for better query performance
CREATE INDEX idx_optimization_runs_network_id ON public.optimization_runs(network_id);
CREATE INDEX idx_inventory_levels_run_id ON public.inventory_levels(optimization_run_id);
CREATE INDEX idx_stock_alerts_run_id ON public.stock_alerts(optimization_run_id);
CREATE INDEX idx_network_statistics_network_id ON public.network_statistics(network_id);
CREATE INDEX idx_safety_stock_network_id ON public.safety_stock_recommendations(network_id);
CREATE INDEX idx_ai_recommendations_network_id ON public.ai_recommendations(network_id);

-- Enable RLS but allow all operations
ALTER TABLE public.optimization_runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.inventory_levels ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.stock_alerts ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.network_statistics ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.safety_stock_recommendations ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.ai_recommendations ENABLE ROW LEVEL SECURITY;

-- Drop existing policies
DROP POLICY IF EXISTS "Enable read access for all users" ON public.optimization_runs;
DROP POLICY IF EXISTS "Enable insert access for all users" ON public.optimization_runs;
DROP POLICY IF EXISTS "Enable read access for all users" ON public.inventory_levels;
DROP POLICY IF EXISTS "Enable insert access for all users" ON public.inventory_levels;
DROP POLICY IF EXISTS "Enable read access for all users" ON public.stock_alerts;
DROP POLICY IF EXISTS "Enable insert access for all users" ON public.stock_alerts;
DROP POLICY IF EXISTS "Enable read access for all users" ON public.network_statistics;
DROP POLICY IF EXISTS "Enable insert access for all users" ON public.network_statistics;
DROP POLICY IF EXISTS "Enable read access for all users" ON public.safety_stock_recommendations;
DROP POLICY IF EXISTS "Enable insert access for all users" ON public.safety_stock_recommendations;
DROP POLICY IF EXISTS "Enable read access for all users" ON public.ai_recommendations;
DROP POLICY IF EXISTS "Enable insert access for all users" ON public.ai_recommendations;

-- Create policies to allow read and insert operations
CREATE POLICY "Enable read access for all users" ON public.optimization_runs FOR SELECT USING (true);
CREATE POLICY "Enable insert access for all users" ON public.optimization_runs FOR INSERT WITH CHECK (true);

CREATE POLICY "Enable read access for all users" ON public.inventory_levels FOR SELECT USING (true);
CREATE POLICY "Enable insert access for all users" ON public.inventory_levels FOR INSERT WITH CHECK (true);

CREATE POLICY "Enable read access for all users" ON public.stock_alerts FOR SELECT USING (true);
CREATE POLICY "Enable insert access for all users" ON public.stock_alerts FOR INSERT WITH CHECK (true);

CREATE POLICY "Enable read access for all users" ON public.network_statistics FOR SELECT USING (true);
CREATE POLICY "Enable insert access for all users" ON public.network_statistics FOR INSERT WITH CHECK (true);

CREATE POLICY "Enable read access for all users" ON public.safety_stock_recommendations FOR SELECT USING (true);
CREATE POLICY "Enable insert access for all users" ON public.safety_stock_recommendations FOR INSERT WITH CHECK (true);

CREATE POLICY "Enable read access for all users" ON public.ai_recommendations FOR SELECT USING (true);
CREATE POLICY "Enable insert access for all users" ON public.ai_recommendations FOR INSERT WITH CHECK (true); 