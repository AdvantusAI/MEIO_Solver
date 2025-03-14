# Example of parallel processing for node calculations
safety_stock_recommendations = ParallelProcessor.process_nodes(
    network, 
    DilopSafetyStock._process_node,
    network=network,
    default_service_level=default_service_level
)
