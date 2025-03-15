"""
Path management for the MEIO system.
Centralizes all file and directory paths used throughout the application.
"""
import os
import logging

logger = logging.getLogger(__name__)

class PathManager:
    """Manages all paths used in the MEIO system."""
    
    # Base paths
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Main directories
    RESULTS_DIR = os.path.join(_PROJECT_ROOT, 'results')
    DATA_DIR = os.path.join(_PROJECT_ROOT, 'data')
    CONFIG_DIR = os.path.join(_PROJECT_ROOT, 'meio', 'config')
    LOG_DIR = os.path.join(_PROJECT_ROOT, 'logs')
    
    # Output subdirectories
    OPTIMIZATION_RESULTS_DIR = os.path.join(RESULTS_DIR, 'optimization')
    BENCHMARK_RESULTS_DIR = os.path.join(RESULTS_DIR, 'benchmarks')
    VISUALIZATION_DIR = os.path.join(RESULTS_DIR, 'visualizations')
    SENSITIVITY_RESULTS_DIR = os.path.join(RESULTS_DIR, 'sensitivity')
    
    # Ensure all directories exist
    @classmethod
    def initialize(cls):
        """Create all necessary directories if they don't exist."""
        directories = [
            cls.RESULTS_DIR,
            cls.DATA_DIR, 
            cls.LOG_DIR,
            cls.OPTIMIZATION_RESULTS_DIR,
            cls.BENCHMARK_RESULTS_DIR,
            cls.VISUALIZATION_DIR,
            cls.SENSITIVITY_RESULTS_DIR
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logger.debug(f"Ensured directory exists: {directory}")
            
        return cls
    
    @classmethod
    def get_file_path(cls, directory, filename):
        """
        Get complete path for a file in the specified directory.
        
        Args:
            directory (str): Base directory (use class constants).
            filename (str): Name of the file.
            
        Returns:
            str: Complete file path.
        """
        return os.path.join(directory, filename)
    
    @classmethod
    def get_optimization_result_path(cls, filename):
        """Get path for an optimization result file."""
        return cls.get_file_path(cls.OPTIMIZATION_RESULTS_DIR, filename)
    
    @classmethod
    def get_benchmark_result_path(cls, filename):
        """Get path for a benchmark result file."""
        return cls.get_file_path(cls.BENCHMARK_RESULTS_DIR, filename)
    
    @classmethod
    def get_visualization_path(cls, filename):
        """Get path for a visualization file."""
        return cls.get_file_path(cls.VISUALIZATION_DIR, filename)
    
    @classmethod
    def get_log_path(cls, filename="meio.log"):
        """Get path for a log file."""
        return cls.get_file_path(cls.LOG_DIR, filename)
    
    @classmethod
    def get_config_path(cls, filename):
        """Get path for a config file."""
        return cls.get_file_path(cls.CONFIG_DIR, filename)

# Initialize paths at module import time
paths = PathManager.initialize() 