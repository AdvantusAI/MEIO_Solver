"""
Validation utilities for the MEIO system.
"""
import logging
import json
import re
from datetime import datetime

logger = logging.getLogger(__name__)

class Validators:
    """Provides validation functions for MEIO data."""
    
    @staticmethod
    def validate_json_structure(json_data):
        """
        Validate the structure of a JSON config.
        
        Args:
            json_data (dict): JSON data to validate.
            
        Returns:
            tuple: (is_valid, errors) where is_valid is a boolean and errors is a list.
        """
        errors = []
        
        # Check required sections
        required_sections = ['plants', 'connections']
        missing_sections = [s for s in required_sections if s not in json_data]
        
        if missing_sections:
            errors.append(f"Missing required sections: {', '.join(missing_sections)}")
        
        # Check that we have at least one node
        all_nodes = json_data.get('plants', []) + json_data.get('dcs', []) + json_data.get('stores', [])
        if not all_nodes:
            errors.append("No nodes defined in JSON")
        
        # Check node IDs are unique
        node_ids = [node.get('id') for node in all_nodes if 'id' in node]
        duplicates = [nid for nid in set(node_ids) if node_ids.count(nid) > 1]
        
        if duplicates:
            errors.append(f"Duplicate node IDs: {', '.join(duplicates)}")
        
        # Check connections refer to valid nodes
        for conn in json_data.get('connections', []):
            if 'from' not in conn or 'to' not in conn:
                errors.append(f"Invalid connection: {conn}")
                continue
                
            if conn['from'] not in node_ids:
                errors.append(f"Connection refers to non-existent 'from' node: {conn['from']}")
                
            if conn['to'] not in node_ids:
                errors.append(f"Connection refers to non-existent 'to' node: {conn['to']}")
        
        # Check for product data consistency
        if all_nodes:
            first_node = all_nodes[0]
            first_product = first_node.get('products', [])
            
            if not first_product:
                errors.append("No products defined")
            else:
                first_product = first_product[0]
                period_length = len(first_product.get('demand_by_period', []))
                
                for node in all_nodes:
                    for prod in node.get('products', []):
                        demand_len = len(prod.get('demand_by_period', []))
                        std_len = len(prod.get('demand_std_by_period', []))
                        
                        if demand_len != period_length:
                            errors.append(f"Inconsistent demand_by_period length in {node.get('id')} - {prod.get('id')}: "
                                         f"got {demand_len}, expected {period_length}")
                            
                        if std_len != period_length:
                            errors.append(f"Inconsistent demand_std_by_period length in {node.get('id')} - {prod.get('id')}: "
                                         f"got {std_len}, expected {period_length}")
        
        return (len(errors) == 0, errors)
    
    @staticmethod
    def validate_date_string(date_str):
        """
        Validate a date string is in YYYY-MM-DD format.
        
        Args:
            date_str (str): Date string to validate.
            
        Returns:
            tuple: (is_valid, datetime_obj or None, error_msg or None)
        """
        if not date_str:
            return (False, None, "Date string is empty")
            
        # Check format
        if not re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
            return (False, None, f"Date {date_str} is not in YYYY-MM-DD format")
            
        try:
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            return (True, date_obj, None)
        except ValueError as e:
            return (False, None, f"Invalid date: {str(e)}")
    
    @staticmethod
    def validate_file_path(file_path, must_exist=True):
        """
        Validate a file path.
        
        Args:
            file_path (str): Path to validate.
            must_exist (bool, optional): Whether the file must exist. Defaults to True.
            
        Returns:
            tuple: (is_valid, error_msg or None)
        """
        import os
        
        if not file_path:
            return (False, "File path is empty")
            
        if must_exist and not os.path.exists(file_path):
            return (False, f"File does not exist: {file_path}")
            
        if must_exist and not os.path.isfile(file_path):
            return (False, f"Path is not a file: {file_path}")
            
        return (True, None)
