-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Table for optimization runs
CREATE TABLE optimization_runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    network_id TEXT NOT NULL,
    method TEXT NOT NULL, -- 'SCIP Solver' or 'Heuristic'
    service_level DECIMAL NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    total_cost DECIMAL NOT NULL,
    status TEXT NOT NULL, -- 'optimal', 'heuristic', 'failed'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT fk_network
        FOREIGN KEY (network_id)
        REFERENCES networks(id)
        ON DELETE CASCADE
);

-- Table for inventory levels by period
CREATE TABLE inventory_levels (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    optimization_run_id UUID NOT NULL,
    node_id TEXT NOT NULL,
    product_id TEXT NOT NULL,
    period INTEGER NOT NULL,
    inventory_level DECIMAL NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT fk_optimization_run
        FOREIGN KEY (optimization_run_id)
        REFERENCES optimization_runs(id)
        ON DELETE CASCADE
);

-- Table for stock alerts (stockouts and overstocks)
CREATE TABLE stock_alerts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    optimization_run_id UUID NOT NULL,
    node_id TEXT NOT NULL,
    product_id TEXT NOT NULL,
    alert_type TEXT NOT NULL, -- 'stockout' or 'overstock'
    period INTEGER NOT NULL,
    inventory_level DECIMAL NOT NULL,
    demand DECIMAL,
    capacity DECIMAL,
    shortfall DECIMAL,
    excess DECIMAL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT fk_optimization_run
        FOREIGN KEY (optimization_run_id)
        REFERENCES optimization_runs(id)
        ON DELETE CASCADE
);

-- Table for network statistics
CREATE TABLE network_statistics (
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
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT fk_network
        FOREIGN KEY (network_id)
        REFERENCES networks(id)
        ON DELETE CASCADE
);

-- Table for safety stock recommendations
CREATE TABLE safety_stock_recommendations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    network_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    product_id TEXT NOT NULL,
    safety_stock DECIMAL NOT NULL,
    service_level DECIMAL NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT fk_network
        FOREIGN KEY (network_id)
        REFERENCES networks(id)
        ON DELETE CASCADE
);

-- Create indexes for better query performance
CREATE INDEX idx_optimization_runs_network_id ON optimization_runs(network_id);
CREATE INDEX idx_inventory_levels_run_id ON inventory_levels(optimization_run_id);
CREATE INDEX idx_stock_alerts_run_id ON stock_alerts(optimization_run_id);
CREATE INDEX idx_network_statistics_network_id ON network_statistics(network_id);
CREATE INDEX idx_safety_stock_network_id ON safety_stock_recommendations(network_id);

-- Add RLS (Row Level Security) policies
ALTER TABLE optimization_runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE inventory_levels ENABLE ROW LEVEL SECURITY;
ALTER TABLE stock_alerts ENABLE ROW LEVEL SECURITY;
ALTER TABLE network_statistics ENABLE ROW LEVEL SECURITY;
ALTER TABLE safety_stock_recommendations ENABLE ROW LEVEL SECURITY;

-- Create policies to allow all operations (you may want to restrict these based on your security requirements)
CREATE POLICY "Allow all operations on optimization_runs" ON optimization_runs
    FOR ALL USING (true);

CREATE POLICY "Allow all operations on inventory_levels" ON inventory_levels
    FOR ALL USING (true);

CREATE POLICY "Allow all operations on stock_alerts" ON stock_alerts
    FOR ALL USING (true);

CREATE POLICY "Allow all operations on network_statistics" ON network_statistics
    FOR ALL USING (true);

CREATE POLICY "Allow all operations on safety_stock_recommendations" ON safety_stock_recommendations
    FOR ALL USING (true); 