"""
Configuration settings for the MEIO system.
"""
import os
import logging
import json
from pathlib import Path
from meio.utils.path_manager import paths

# Legacy constants for backward compatibility
RESULTS_DIR = paths.RESULTS_DIR
VISUALIZATIONS_DIR = paths.VISUALIZATION_DIR
DEFAULT_SERVICE_LEVEL = 0.95
DEFAULT_INFLOW = 1000

class Config:
    """Configuration class for MEIO system."""
    
    def __init__(self, config_file=None):
        """
        Initialize configuration with default settings or from a file.
        
        Args:
            config_file (str, optional): Path to configuration file. Defaults to None.
        """
        # Default settings
        self.DEFAULT_CONFIG = {
            'paths': {
                'data_dir': paths.DATA_DIR,
                'output_dir': paths.RESULTS_DIR,
                'log_dir': paths.LOG_DIR,
                'optimization_dir': paths.OPTIMIZATION_RESULTS_DIR,
                'benchmark_dir': paths.BENCHMARK_RESULTS_DIR,
                'visualization_dir': paths.VISUALIZATION_DIR,
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'file': paths.get_log_path('meio.log'),
            },
            'optimization': {
                'default_service_level': DEFAULT_SERVICE_LEVEL,
                'default_inflow': DEFAULT_INFLOW,
                'solver_time_limit': 600,  # seconds
                'solver_gap': 0.01,        # 1% gap
            },
            'visualization': {
                'default_figsize': (12, 8),
                'color_palette': 'viridis',
            }
        }
        
        self.config = self.DEFAULT_CONFIG.copy()
        
        # Load from file if specified
        if config_file:
            self.load_config(config_file)
            
        # Ensure directories exist
        self._ensure_directories()
        
        # Setup logging
        self._setup_logging()
    
    def load_config(self, config_file):
        """
        Load configuration from a file.
        
        Args:
            config_file (str): Path to configuration file.
            
        Raises:
            FileNotFoundError: If the config file doesn't exist.
            json.JSONDecodeError: If the config file isn't valid JSON.
        """
        try:
            with open(config_file, 'r') as f:
                user_config = json.load(f)
                
            # Update default config with user-provided values
            self._update_nested_dict(self.config, user_config)
            
        except FileNotFoundError:
            logging.error(f"Configuration file not found: {config_file}")
            raise
        except json.JSONDecodeError:
            logging.error(f"Invalid JSON in configuration file: {config_file}")
            raise
    
    def _update_nested_dict(self, d, u):
        """Recursively update nested dictionary."""
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = self._update_nested_dict(d.get(k, {}), v)
            else:
                d[k] = v
        return d
    
    def _ensure_directories(self):
        """Create necessary directories if they don't exist."""
        # No need to create directories here as they are handled by PathManager
        pass
    
    def _setup_logging(self):
        """Configure logging based on settings."""
        log_config = self.config['logging']
        log_dir = os.path.dirname(log_config['file'])
        os.makedirs(log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, log_config['level']),
            format=log_config['format'],
            handlers=[
                logging.FileHandler(log_config['file']),
                logging.StreamHandler()
            ]
        )
    
    def get(self, section, key=None):
        """
        Get a configuration value.
        
        Args:
            section (str): Configuration section name.
            key (str, optional): Specific key within section. Defaults to None.
            
        Returns:
            The configuration value or entire section dict if key is None.
            
        Raises:
            KeyError: If the section or key doesn't exist.
        """
        if section not in self.config:
            raise KeyError(f"Configuration section not found: {section}")
        
        if key is None:
            return self.config[section]
        
        if key not in self.config[section]:
            raise KeyError(f"Configuration key not found: {section}.{key}")
        
        return self.config[section][key]
    
    def set(self, section, key, value):
        """
        Set a configuration value.
        
        Args:
            section (str): Configuration section name.
            key (str): Key within section.
            value: New value to set.
        """
        if section not in self.config:
            self.config[section] = {}
        
        self.config[section][key] = value

# Global config instance
config = Config()