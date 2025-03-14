"""
Caching utilities for the MEIO system.
"""
import logging
import functools
import hashlib
import pickle
import os
import time
from ..config.settings import config

logger = logging.getLogger(__name__)

class Cache:
    """Provides caching functionality to avoid redundant calculations."""
    
    def __init__(self, cache_dir=None, max_age=3600):
        """
        Initialize the cache.
        
        Args:
            cache_dir (str, optional): Directory for cached files. Defaults to config value.
            max_age (int, optional): Maximum age of cache entries in seconds. Defaults to 1 hour.
        """
        self.cache_dir = cache_dir or os.path.join(config.get('paths', 'data_dir'), 'cache')
        self.max_age = max_age
        os.makedirs(self.cache_dir, exist_ok=True)
        logger.debug(f"Cache initialized with directory: {self.cache_dir}")
    
    def _get_cache_path(self, key):
        """Get the file path for a cache key."""
        # Convert key to a hash
        key_hash = hashlib.md5(str(key).encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{key_hash}.cache")
    
    def get(self, key):
        """
        Get a cached value.
        
        Args:
            key: Cache key.
            
        Returns:
            The cached value or None if not found or expired.
        """
        cache_path = self._get_cache_path(key)
        
        if not os.path.exists(cache_path):
            return None
            
        # Check if expired
        if self.max_age > 0:
            mod_time = os.path.getmtime(cache_path)
            if time.time() - mod_time > self.max_age:
                logger.debug(f"Cache entry expired: {key}")
                os.remove(cache_path)
                return None
        
        # Load from cache
        try:
            with open(cache_path, 'rb') as f:
                value = pickle.load(f)
            logger.debug(f"Cache hit: {key}")
            return value
        except Exception as e:
            logger.warning(f"Failed to load cache entry: {str(e)}")
            return None
    
    def set(self, key, value):
        """
        Set a cached value.
        
        Args:
            key: Cache key.
            value: Value to cache.
            
        Returns:
            bool: True if successful.
        """
        cache_path = self._get_cache_path(key)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f)
            logger.debug(f"Cache set: {key}")
            return True
        except Exception as e:
            logger.warning(f"Failed to save cache entry: {str(e)}")
            return False

def cached(func=None, *, key_func=None, cache_instance=None):
    """
    Decorator for caching function results.
    
    Args:
        func (callable, optional): Function to decorate.
        key_func (callable, optional): Function to generate cache key. Defaults to None.
        cache_instance (Cache, optional): Cache instance to use. Defaults to None (creates new).
        
    Returns:
        callable: Decorated function.
    """
    if func is None:
        return lambda f: cached(f, key_func=key_func, cache_instance=cache_instance)
    
    cache = cache_instance or Cache()
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Generate cache key
        if key_func:
            cache_key = key_func(*args, **kwargs)
        else:
            # Default key generation based on function name, args, and kwargs
            key_parts = [func.__module__, func.__name__]
            key_parts.extend([str(arg) for arg in args])
            key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
            cache_key = ":".join(key_parts)
        
        # Try to get from cache
        cached_result = cache.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        # Compute and cache result
        result = func(*args, **kwargs)
        cache.set(cache_key, result)
        return result
    
    return wrapper