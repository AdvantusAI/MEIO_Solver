-- Create AI recommendations table
CREATE TABLE IF NOT EXISTS ai_recommendations (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    network_id TEXT NOT NULL REFERENCES networks(id),
    node_id TEXT NOT NULL,
    product_id TEXT NOT NULL,
    analysis TEXT NOT NULL,
    recommendations JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc', NOW()),
    FOREIGN KEY (network_id, node_id) REFERENCES nodes(network_id, id),
    FOREIGN KEY (network_id, product_id) REFERENCES products(network_id, id)
);

-- Create index for faster lookups
CREATE INDEX IF NOT EXISTS ai_recommendations_network_id_idx ON ai_recommendations(network_id);
CREATE INDEX IF NOT EXISTS ai_recommendations_node_id_idx ON ai_recommendations(node_id);
CREATE INDEX IF NOT EXISTS ai_recommendations_product_id_idx ON ai_recommendations(product_id);

-- Enable row level security
ALTER TABLE ai_recommendations ENABLE ROW LEVEL SECURITY;

-- Create policies
CREATE POLICY "Enable read access for all users" ON ai_recommendations
    FOR SELECT USING (true);

CREATE POLICY "Enable insert for authenticated users" ON ai_recommendations
    FOR INSERT WITH CHECK (auth.role() = 'authenticated');

-- Grant permissions
GRANT SELECT, INSERT ON ai_recommendations TO authenticated;
GRANT USAGE ON SEQUENCE ai_recommendations_id_seq TO authenticated; 