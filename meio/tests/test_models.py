"""
Tests for the MEIO models.
"""
import unittest
from ..models.node import Node
from ..models.network import MultiEchelonNetwork

class TestNode(unittest.TestCase):
    """Tests for the Node class."""
    
    def test_node_initialization(self):
        """Test node initialization."""
        node = Node("N1", "store")
        self.assertEqual(node.node_id, "N1")
        self.assertEqual(node.node_type, "store")
        self.assertEqual(node.parent, None)
        self.assertEqual(node.children, [])
        self.assertEqual(node.capacity, 0)
        self.assertEqual(node.products, {})
        self.assertEqual(node.transport_cost, 0)
        self.assertEqual(node.transport_variability, 0)
    
    def test_add_product(self):
        """Test adding a product to a node."""
        node = Node("N1", "store")
        node.add_product("P1", 1.0, 0.5, 2.0, 20.0, [100, 110], [10, 11])
        
        self.assertIn("P1", node.products)
        self.assertEqual(node.products["P1"]["lead_time_mean"], 1.0)
        self.assertEqual(node.products["P1"]["lead_time_std"], 0.5)
        self.assertEqual(node.products["P1"]["holding_cost"], 2.0)
        self.assertEqual(node.products["P1"]["shortage_cost"], 20.0)
        self.assertEqual(node.products["P1"]["demand_by_date"], [100, 110])
        self.assertEqual(node.products["P1"]["demand_std_by_date"], [10, 11])
        self.assertEqual(node.products["P1"]["safety_stock_by_date"], [0, 0])
    
    def test_add_product_mismatched_lengths(self):
        """Test adding a product with mismatched demand and std lengths."""
        node = Node("N1", "store")
        with self.assertRaises(ValueError):
            node.add_product("P1", 1.0, 0.5, 2.0, 20.0, [100, 110], [10])
    
    def test_validate(self):
        """Test node validation."""
        node = Node("N1", "store")
        node.add_product("P1", 1.0, 0.5, 2.0, 20.0, [100, 110], [10, 11])
        self.assertTrue(node.validate())
        
        # Test invalid node type
        node = Node("N1", "invalid")
        with self.assertRaises(ValueError):
            node.validate()
            
        # Test negative values
        node = Node("N1", "store")
        node.capacity = -1
        with self.assertRaises(ValueError):
            node.validate()

class TestNetwork(unittest.TestCase):
    """Tests for the MultiEchelonNetwork class."""
    
    def test_network_initialization(self):
        """Test network initialization."""
        from datetime import datetime, timedelta
        
        start_date = datetime.now()
        end_date = start_date + timedelta(days=365)
        network = MultiEchelonNetwork(start_date, end_date, 30)
        
        self.assertEqual(network.start_date, start_date)
        self.assertEqual(network.date_interval, 30)
        self.assertEqual(len(network.dates), 12)  # 365/30 = 12.16 -> 12
        self.assertEqual(network.num_periods, 12)
    
    def test_add_node(self):
        """Test adding a node to the network."""
        network = MultiEchelonNetwork()
        node = Node("N1", "store")
        network.add_node(node)
        
        self.assertIn("N1", network.nodes)
        self.assertEqual(network.nodes["N1"], node)
        
        # Test adding duplicate node
        with self.assertRaises(ValueError):
            network.add_node(node)
    
    def test_add_connection(self):
        """Test adding a connection between nodes."""
        network = MultiEchelonNetwork()
        parent = Node("P1", "plant")
        child = Node("S1", "store")
        
        network.add_node(parent)
        network.add_node(child)
        network.add_connection("P1", "S1")
        
        self.assertEqual(child.parent, parent)
        self.assertIn(child, parent.children)
        
        # Test invalid connection
        with self.assertRaises(ValueError):
            network.add_connection("P1", "S2")
    
    def test_validate(self):
        """Test network validation."""
        network = MultiEchelonNetwork()
        parent = Node("P1", "plant")
        child = Node("S1", "store")
        
        network.add_node(parent)
        network.add_node(child)
        network.add_connection("P1", "S1")
        
        self.assertTrue(network.validate())
        
        # Test empty network
        network = MultiEchelonNetwork()
        with self.assertRaises(ValueError):
            network.validate()
            
        # Test orphaned node
        network = MultiEchelonNetwork()
        network.add_node(Node("S1", "store"))
        with self.assertRaises(ValueError):
            network.validate()