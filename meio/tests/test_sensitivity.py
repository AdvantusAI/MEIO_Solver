"""
Tests for the sensitivity analysis module.
"""
import unittest
import os
import shutil
import tempfile
from datetime import datetime, timedelta

from meio.models.network import MultiEchelonNetwork
from meio.analysis.sensitivity import run_sensitivity_analysis, _modify_network_lead_times, _modify_network_demand

class TestSensitivityAnalysis(unittest.TestCase):
    """Test cases for sensitivity analysis."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test outputs
        self.test_dir = tempfile.mkdtemp()
        
        # Create a simple test network
        self.network = self._create_test_network()
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Clean up the temporary directory
        shutil.rmtree(self.test_dir)
    
    def _create_test_network(self):
        """Create a simple test network for testing."""
        # Create dates
        start_date = datetime.now()
        dates = [start_date + timedelta(days=i*30) for i in range(3)]
        
        # Create network
        network = MultiEchelonNetwork(dates)
        
        # Add nodes
        network.add_node("DC", {
            "type": "distribution_center",
            "capacity": 1000,
            "holding_cost": 0.1,
            "products": {
                "product1": {
                    "demand_by_date": [0, 0, 0]
                }
            }
        })
        
        network.add_node("Store1", {
            "type": "store",
            "capacity": 100,
            "holding_cost": 0.2,
            "products": {
                "product1": {
                    "demand_by_date": [10, 15, 20]
                }
            }
        })
        
        # Set lead times
        network.nodes["DC"].lead_time = {"product1": 7}
        network.nodes["Store1"].lead_time = {"product1": 3}
        
        # Add connections
        network.add_connection("DC", "Store1", {
            "transit_time": 2,
            "products": {
                "product1": {"cost": 5.0}
            }
        })
        
        return network
    
    def test_modify_network_lead_times(self):
        """Test modifying lead times in the network."""
        # Create a copy of the network
        network_copy = self._create_test_network()
        
        # Get original lead times
        original_dc_lead_time = network_copy.nodes["DC"].lead_time["product1"]
        original_store_lead_time = network_copy.nodes["Store1"].lead_time["product1"]
        
        # Modify lead times
        factor = 1.5
        _modify_network_lead_times(network_copy, factor)
        
        # Check that lead times were modified correctly
        self.assertEqual(network_copy.nodes["DC"].lead_time["product1"], int(original_dc_lead_time * factor))
        self.assertEqual(network_copy.nodes["Store1"].lead_time["product1"], int(original_store_lead_time * factor))
    
    def test_modify_network_demand(self):
        """Test modifying demand in the network."""
        # Create a copy of the network
        network_copy = self._create_test_network()
        
        # Get original demand
        original_demand = network_copy.nodes["Store1"].products["product1"]["demand_by_date"].copy()
        
        # Modify demand
        factor = 1.2
        _modify_network_demand(network_copy, factor)
        
        # Check that demand was modified correctly
        for i in range(len(original_demand)):
            self.assertAlmostEqual(
                network_copy.nodes["Store1"].products["product1"]["demand_by_date"][i],
                original_demand[i] * factor
            )
    
    def test_run_sensitivity_analysis(self):
        """Test running sensitivity analysis."""
        # Define parameters to test
        parameter_ranges = {
            'service_level': [0.90, 0.95],
            'lead_time_factor': [0.8, 1.0]
        }
        
        # Run sensitivity analysis
        result = run_sensitivity_analysis(
            self.network,
            parameter_ranges,
            output_dir=self.test_dir,
            visualize=True
        )
        
        # Check that results were generated
        self.assertIn('analysis_id', result)
        self.assertIn('output_dir', result)
        self.assertIn('results_csv', result)
        self.assertIn('elasticity_csv', result)
        self.assertIn('report_path', result)
        
        # Check that output files exist
        self.assertTrue(os.path.exists(result['results_csv']))
        self.assertTrue(os.path.exists(result['elasticity_csv']))
        self.assertTrue(os.path.exists(result['report_path']))
        
        # Check visualization files
        if result['figure_paths']:
            for fig_path in result['figure_paths']:
                self.assertTrue(os.path.exists(fig_path))
    
    def test_service_level_sensitivity(self):
        """Test sensitivity to service level only."""
        # Define parameters to test
        parameter_ranges = {
            'service_level': [0.90, 0.95, 0.98]
        }
        
        # Run sensitivity analysis
        result = run_sensitivity_analysis(
            self.network,
            parameter_ranges,
            output_dir=self.test_dir,
            visualize=False  # Skip visualization for faster testing
        )
        
        # Check that results were generated
        self.assertIn('all_results', result)
        self.assertEqual(len(result['all_results']), len(parameter_ranges['service_level']))

if __name__ == '__main__':
    unittest.main() 