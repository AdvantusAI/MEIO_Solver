"""
Configuration settings for the MEIO system.
"""
import os
import logging

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Results directory
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Paths for specific result files
OPTIMIZATION_RESULTS_FILE = os.path.join(RESULTS_DIR, 'optimization_results.csv')
INVENTORY_LEVELS_FILE = os.path.join(RESULTS_DIR, 'inventory_levels.csv')
STOCK_ALERTS_FILE = os.path.join(RESULTS_DIR, 'stock_alerts.csv')

# Visualization directory (subfolder of results)
VISUALIZATIONS_DIR = os.path.join(RESULTS_DIR, 'visualizations')
os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)

# Logging configuration
LOG_DIR = os.path.join(BASE_DIR, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, 'meio.log')

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

# Optimization defaults
DEFAULT_SERVICE_LEVEL = 0.95
DEFAULT_INFLOW = 1000

# Get logger
logger = logging.getLogger(__name__)
logger.info(f"Results will be saved to: {RESULTS_DIR}")