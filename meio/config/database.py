"""
Database configuration for the MEIO system.
"""
import os
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class DatabaseConfig:
    """Handles database configuration and credentials."""
    
    def __init__(self, config_file=None):
        """
        Initialize database configuration.
        
        Args:
            config_file (str, optional): Path to configuration file. Defaults to None.
        """
        self.supabase_url = None
        self.supabase_key = None
        
        # Try to load from environment variables first
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_key = os.getenv('SUPABASE_KEY')
        
        # If not in environment, try config file
        if not self.supabase_url or not self.supabase_key:
            if config_file:
                self._load_from_file(config_file)
            else:
                # Try default config file location
                default_config = Path(__file__).parent / 'database_config.json'
                if default_config.exists():
                    self._load_from_file(default_config)
    
    def _load_from_file(self, config_file):
        """
        Load configuration from a file.
        
        Args:
            config_file (str): Path to configuration file.
        """
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                self.supabase_url = config.get('supabase_url')
                self.supabase_key = config.get('supabase_key')
        except Exception as e:
            logger.error(f"Error loading database config from {config_file}: {str(e)}")
    
    def get_credentials(self):
        """
        Get database credentials.
        
        Returns:
            tuple: (supabase_url, supabase_key)
            
        Raises:
            ValueError: If credentials are not configured.
        """
        if not self.supabase_url or not self.supabase_key:
            raise ValueError(
                "Database credentials not configured. "
                "Set SUPABASE_URL and SUPABASE_KEY environment variables "
                "or provide a database_config.json file."
            )
        return self.supabase_url, self.supabase_key 