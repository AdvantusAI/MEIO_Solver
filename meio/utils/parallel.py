# File: meio/utils/parallel.py
"""
Parallel processing utilities for the MEIO system.
"""
import logging
import concurrent.futures
import threading
from functools import partial

logger = logging.getLogger(__name__)

# Define this at module level so it can be pickled
def _process_node_item(item, node_func=None, **kwargs):
    """Process a single (node_id, node) pair with the given function."""
    node_id, node = item
    return node_id, node_func(node, **kwargs)

class ParallelProcessor:
    """Provides utilities for parallel processing of independent tasks."""
    
    @staticmethod
    def map(func, items, max_workers=None, use_threads=False, **kwargs):
        """
        Process items in parallel using the provided function.
        
        Args:
            func (callable): Function to apply to each item.
            items (iterable): Items to process.
            max_workers (int, optional): Maximum number of worker processes. Defaults to None (auto).
            use_threads (bool, optional): Use threads instead of processes. Defaults to False.
            **kwargs: Additional keyword arguments to pass to the function.
            
        Returns:
            list: Results in the same order as items.
        """
        # If single item or very few items, process sequentially
        items_list = list(items)  # Convert to list for length check and indexing
        if len(items_list) <= 1:
            return [func(item, **kwargs) for item in items_list]
            
        # Wrap function to include kwargs
        if kwargs:
            wrapped_func = partial(func, **kwargs)
        else:
            wrapped_func = func
        
        results = []
        
        # Choose executor based on use_threads flag
        executor_class = concurrent.futures.ThreadPoolExecutor if use_threads else concurrent.futures.ProcessPoolExecutor
        
        with executor_class(max_workers=max_workers) as executor:
            futures = {executor.submit(wrapped_func, item): i for i, item in enumerate(items_list)}
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    idx = futures[future]
                    results.append((idx, result))
                except Exception as e:
                    logger.error(f"Error in parallel execution: {str(e)}")
                    raise
        
        # Sort by original order
        results.sort()
        return [r[1] for r in results]
    
    @staticmethod
    def process_nodes(network, node_func, use_threads=True, **kwargs):
        """
        Process network nodes in parallel using the provided function.
        
        Args:
            network (MultiEchelonNetwork): The network.
            node_func (callable): Function to apply to each node.
            use_threads (bool, optional): Use threads instead of processes. Defaults to True.
            **kwargs: Additional keyword arguments to pass to the function.
            
        Returns:
            dict: Results by node_id.
        """
        # Create a list of (node_id, node) tuples
        node_items = list(network.nodes.items())
        
        # Process in parallel with the module-level function
        process_kwargs = {
            'node_func': node_func,
            **kwargs
        }
        
        results = ParallelProcessor.map(
            _process_node_item, 
            node_items, 
            use_threads=use_threads, 
            **process_kwargs
        )
        
        # Convert to dictionary
        return dict(results)
    
    @staticmethod
    def sequential_map(func, items, **kwargs):
        """
        Process items sequentially - a fallback for when parallel processing fails.
        
        Args:
            func (callable): Function to apply to each item.
            items (iterable): Items to process.
            **kwargs: Additional keyword arguments to pass to the function.
            
        Returns:
            list: Results in the same order as items.
        """
        results = []
        for i, item in enumerate(items):
            try:
                result = func(item, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing item {i}: {str(e)}")
                raise
        return results