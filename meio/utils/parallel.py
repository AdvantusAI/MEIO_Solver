"""
Parallel processing utilities for the MEIO system.
"""
import logging
import concurrent.futures
from functools import partial

logger = logging.getLogger(__name__)

class ParallelProcessor:
    """Provides utilities for parallel processing of independent tasks."""
    
    @staticmethod
    def map(func, items, max_workers=None, **kwargs):
        """
        Process items in parallel using the provided function.
        
        Args:
            func (callable): Function to apply to each item.
            items (iterable): Items to process.
            max_workers (int, optional): Maximum number of worker processes. Defaults to None (auto).
            **kwargs: Additional keyword arguments to pass to the function.
            
        Returns:
            list: Results in the same order as items.
        """
        # If single item or very few items, process sequentially
        if not hasattr(items, '__len__') or len(items) <= 1:
            return [func(item, **kwargs) for item in items]
            
        # Wrap function to include kwargs
        if kwargs:
            wrapped_func = partial(func, **kwargs)
        else:
            wrapped_func = func
        
        results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(wrapped_func, item): i for i, item in enumerate(items)}
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    idx = futures[future]
                    results.append((idx, result))
                except Exception as e:
                    logger.error(f"Error in parallel processing: {str(e)}")
                    raise
        
        # Sort by original order
        results.sort()
        return [r[1] for r in results]
    
    @staticmethod
    def process_nodes(network, node_func, **kwargs):
        """
        Process network nodes in parallel using the provided function.
        
        Args:
            network (MultiEchelonNetwork): The network.
            node_func (callable): Function to apply to each node.
            **kwargs: Additional keyword arguments to pass to the function.
            
        Returns:
            dict: Results by node_id.
        """
        # Create a list of (node_id, node) tuples
        node_items = list(network.nodes.items())  # FIX: Use network.nodes.items() instead of network.items()
        # Process in parallel
        def process_node_item(item, **kwargs):
            node_id, node = item
            return node_id, node_func(node, **kwargs)
        
        results = ParallelProcessor.map(process_node_item, node_items, **kwargs)
        
        # Convert to dictionary
        return dict(results)

