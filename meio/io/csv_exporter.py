"""
CSV exporter for the MEIO system.
"""
import os
import csv
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class CSVExporter:
    """Exports results to CSV files."""
    
    def __init__(self, output_dir=None):
        """
        Initialize the CSV exporter.
        
        Args:
            output_dir (str, optional): Output directory. Uses config if None.
        """
        from ..config.settings import config
        self.output_dir = output_dir or config.get('paths', 'output_dir')
        os.makedirs(self.output_dir, exist_ok=True)
        logger.debug(f"CSV Exporter initialized with output directory: {self.output_dir}")
    
    def save_to_csv(self, filename, data, headers, mode='a'):
        """
        Write data to a CSV file.
        
        Args:
            filename (str): CSV filename (without path).
            data (dict or list): Data to write.
            headers (list): Column headers.
            mode (str, optional): File mode ('w' or 'a'). Defaults to 'a'.
        """
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            with open(filepath, mode, newline='') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                
                # Write header if file is new or in write mode
                if mode == 'w' or f.tell() == 0:
                    writer.writeheader()
                    
                if isinstance(data, list):
                    writer.writerows(data)
                else:
                    writer.writerow(data)
                    
            logger.info(f"CSV data written to {filepath}")
            
        except Exception as e:
            logger.error(f"Error writing CSV to {filepath}: {str(e)}")
            raise
    
    def save_optimization_results(self, network, method, results):
        """
        Save optimization results to CSV.
        
        Args:
            network (MultiEchelonNetwork): The network.
            method (str): Optimization method name.
            results (dict): Optimization results.
            
        Returns:
            str: Optimization ID.
        """
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        opt_id = f"{method}_{timestamp}"
        
        # Save basic optimization info
        opt_data = {
            'optimization_id': opt_id,
            'method': method,
            'status': results['status'],
            'total_cost': results.get('total_cost', None),
            'run_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        self.save_to_csv(
            'optimization_results.csv', 
            opt_data,
            ['optimization_id', 'method', 'status', 'total_cost', 'run_timestamp']
        )
        
        # Save inventory levels
        if results['status'] in ['optimal', 'heuristic']:
            inventory_data = []
            
            for (node_id, prod, t), inv in results['inventory_levels'].items():
                inventory_data.append({
                    'optimization_id': opt_id,
                    'node_id': node_id,
                    'product_id': prod,
                    'date': network.dates[t].strftime('%Y-%m-%d'),
                    'inventory': inv
                })
                
            self.save_to_csv(
                'inventory_levels.csv', 
                inventory_data, 
                ['optimization_id', 'node_id', 'product_id', 'date', 'inventory']
            )
            
        return opt_id
    
    def save_stock_alerts(self, optimization_id, stockouts, overstocks, method):
        """
        Save stock alerts to CSV.
        
        Args:
            optimization_id (str): Optimization ID.
            stockouts (dict): Stockout alerts.
            overstocks (dict): Overstock alerts.
            method (str): Optimization method name.
        """
        alert_data = []
        
        # Process stockouts
        for node_id, prods in stockouts.items():
            for prod, alerts in prods.items():
                for alert in alerts:
                    alert_data.append({
                        'optimization_id': optimization_id,
                        'method': method,
                        'node_id': node_id,
                        'product_id': prod,
                        'date': alert['date'],
                        'alert_type': 'Stockout',
                        'inventory': alert['inventory'],
                        'demand': alert['demand'],
                        'shortfall': alert['shortfall'],
                        'total_inventory': None,
                        'capacity': None,
                        'excess': None
                    })
        
        # Process overstocks
        for node_id, prods in overstocks.items():
            for prod, alerts in prods.items():
                for alert in alerts:
                    alert_data.append({
                        'optimization_id': optimization_id,
                        'method': method,
                        'node_id': node_id,
                        'product_id': prod,
                        'date': alert['date'],
                        'alert_type': 'Overstock',
                        'inventory': alert['inventory'],
                        'demand': None,
                        'shortfall': None,
                        'total_inventory': alert['total_inventory'],
                        'capacity': alert['capacity'],
                        'excess': alert['excess']
                    })
        
        if alert_data:
            self.save_to_csv(
                'stock_alerts.csv', 
                alert_data, 
                ['optimization_id', 'method', 'node_id', 'product_id', 'date', 
                 'alert_type', 'inventory', 'demand', 'shortfall', 
                 'total_inventory', 'capacity', 'excess']
            )
            logger.info(f"Saved {len(alert_data)} stock alerts to CSV for optimization {optimization_id}")
        else:
            logger.info(f"No stock alerts to save for optimization {optimization_id}")
