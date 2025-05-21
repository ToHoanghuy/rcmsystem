"""
Optimized recommendation cache implementation.
Provides in-memory caching for recommendations with TTL expiration
and LRU (Least Recently Used) eviction policy.
"""

import time
import logging
import threading
from collections import OrderedDict
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OptimizedRecommendationCache:
    """
    An optimized cache for recommendations with TTL expiration
    and LRU (Least Recently Used) eviction policy.
    Thread-safe implementation for concurrent access.
    """
    def __init__(self, max_size=1000, expiry_seconds=3600):
        """
        Initialize the cache
        
        Parameters:
        -----------
        max_size : int
            Maximum number of items in the cache
        expiry_seconds : int
            Seconds after which a cached item expires
        """
        self.max_size = max_size
        self.expiry_seconds = expiry_seconds
        # Using OrderedDict to maintain insertion order for LRU functionality
        self.cache = OrderedDict()
        # Maps a key to its expiration time
        self.expiry_times = {}
        # Lock for thread safety
        self.lock = threading.RLock()
        # Stats
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
        # Start background thread to clean expired items
        self.cleanup_thread = threading.Thread(target=self._cleanup_expired, daemon=True)
        self.cleanup_thread.start()
    
    def set(self, key, value):
        """
        Set a value in the cache
        
        Parameters:
        -----------
        key : str
            Cache key
        value : any
            Value to cache
            
        Returns:
        --------
        bool
            True if set successfully
        """
        with self.lock:
            # Remove key if it already exists
            if key in self.cache:
                self.cache.pop(key)
                
            # If cache is full, remove the least recently used item (first item)
            if len(self.cache) >= self.max_size:
                try:
                    oldest_key, _ = self.cache.popitem(last=False)
                    self.expiry_times.pop(oldest_key, None)
                    self.evictions += 1
                except KeyError:
                    # Cache might be empty due to concurrent access
                    pass
            
            # Add new item to cache with expiry time
            self.cache[key] = value
            self.expiry_times[key] = time.time() + self.expiry_seconds
            
            return True
    
    def get(self, key):
        """
        Get a value from the cache
        
        Parameters:
        -----------
        key : str
            Cache key
            
        Returns:
        --------
        any
            Cached value or None if not found or expired
        """
        with self.lock:
            if key in self.cache:
                # Check if expired
                if time.time() > self.expiry_times.get(key, 0):
                    # Remove expired item
                    self.cache.pop(key)
                    self.expiry_times.pop(key)
                    self.misses += 1
                    return None
                
                # Move the item to the end (most recently used)
                value = self.cache.pop(key)
                self.cache[key] = value
                
                # Update expiry time (sliding expiration)
                self.expiry_times[key] = time.time() + self.expiry_seconds
                
                self.hits += 1
                return value
            
            self.misses += 1
            return None
    
    def delete(self, key):
        """
        Delete an item from the cache
        
        Parameters:
        -----------
        key : str
            Cache key
            
        Returns:
        --------
        bool
            True if item was deleted, False if not found
        """
        with self.lock:
            if key in self.cache:
                self.cache.pop(key)
                self.expiry_times.pop(key, None)
                return True
            return False
    
    def clear(self):
        """
        Clear the entire cache
        """
        with self.lock:
            self.cache.clear()
            self.expiry_times.clear()
    
    def _cleanup_expired(self):
        """
        Background thread to clean expired items from the cache
        """
        while True:
            time.sleep(60)  # Check every minute
            try:
                with self.lock:
                    # Find all expired keys
                    now = time.time()
                    expired_keys = [k for k, exp_time in self.expiry_times.items() if now > exp_time]
                    
                    # Remove expired items
                    for key in expired_keys:
                        self.cache.pop(key, None)
                        self.expiry_times.pop(key, None)
                    
                    if expired_keys:
                        logger.debug(f"Cleaned up {len(expired_keys)} expired cache items")
            except Exception as e:
                logger.error(f"Error in cache cleanup: {str(e)}")
    
    def get_stats(self):
        """
        Get cache statistics
        
        Returns:
        --------
        dict
            Cache statistics
        """
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0
            
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': hit_rate,
                'evictions': self.evictions
            }
    
    def get_keys(self):
        """
        Get all keys in the cache
        
        Returns:
        --------
        list
            List of keys
        """
        with self.lock:
            return list(self.cache.keys())
    
    def __len__(self):
        """
        Get the number of items in the cache
        
        Returns:
        --------
        int
            Number of items
        """
        with self.lock:
            return len(self.cache)
    
    def __contains__(self, key):
        """
        Check if a key is in the cache
        
        Parameters:
        -----------
        key : str
            Cache key
            
        Returns:
        --------
        bool
            True if key is in the cache and not expired
        """
        with self.lock:
            if key in self.cache:
                # Check if expired
                if time.time() > self.expiry_times.get(key, 0):
                    return False
                return True
            return False
