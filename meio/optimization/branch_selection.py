"""
Branch Selection Strategy Module for MEIO System.

This module provides functionality to generate multiple inventory policy options (branches),
evaluate them under different scenarios, and select the most appropriate strategy using a
formalized decision process.
"""
import logging
import copy
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional, Union

from ..config.settings import config
from ..utils.path_manager import paths
from ..io.csv_exporter import CSVExporter
from .heuristic import HeuristicSolver, ImprovedHeuristicSolver
from .solver import MathematicalSolver

logger = logging.getLogger(__name__)

class BranchGenerator:
    """Generates multiple inventory policy options (branches) based on different parameter sets."""
    
    @staticmethod
    def generate_branches(
        network, 
        num_branches: int = 5,
        criteria: List[str] = None,
        base_params: Dict[str, Any] = None,
        method: str = 'improved_heuristic'
    ) -> Dict[str, Any]:
        """
        Generate multiple inventory policy branches with varying parameters.
        
        Args:
            network: The network to optimize
            num_branches: Number of branches to generate
            criteria: List of criteria to consider (e.g., ['cost', 'service_level', 'robustness'])
            base_params: Base parameters to use for optimization
            method: Optimization method to use
            
        Returns:
            Dict with generated branches and their metadata
        """
        if criteria is None:
            criteria = ['cost', 'service_level', 'robustness']
            
        if base_params is None:
            base_params = {
                'service_level': config.get('optimization', 'default_service_level'),
                'inflows': config.get('optimization', 'default_inflow')
            }
            
        # Select optimizer based on method
        optimizer = BranchGenerator._get_optimizer(method)
        
        # Generate parameter variations for branches
        branch_params = BranchGenerator._generate_branch_parameters(
            base_params, num_branches, criteria)
        
        logger.info(f"Generating {num_branches} branches using {method} method")
        
        branches = {}
        for branch_id, params in branch_params.items():
            logger.info(f"Generating branch {branch_id} with params: {params}")
            
            # Create a copy of the network to avoid modifying the original
            branch_network = copy.deepcopy(network)
            
            try:
                # Apply any network adjustments based on params
                if 'lead_time_factor' in params:
                    BranchGenerator._adjust_lead_times(branch_network, params['lead_time_factor'])
                
                if 'demand_factor' in params:
                    BranchGenerator._adjust_demand(branch_network, params['demand_factor'])
                
                # Filter optimizer parameters (remove network adjustment parameters)
                optimizer_params = {k: v for k, v in params.items() 
                                  if k not in ['lead_time_factor', 'demand_factor']}
                
                # Run optimization with filtered parameters
                result = optimizer(branch_network, **optimizer_params)
                
                # Store branch results
                branches[branch_id] = {
                    'branch_id': branch_id,
                    'params': params,
                    'inventory_levels': result.get('inventory_levels', {}),
                    'total_cost': result.get('total_cost', 0),
                    'status': result.get('status', 'unknown'),
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
            except Exception as e:
                logger.error(f"Error generating branch {branch_id}: {str(e)}")
                branches[branch_id] = {
                    'branch_id': branch_id,
                    'params': params,
                    'status': 'error',
                    'error': str(e),
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
        
        return {
            'branches': branches,
            'base_params': base_params,
            'criteria': criteria,
            'method': method
        }
    
    @staticmethod
    def _get_optimizer(method: str):
        """Get the appropriate optimizer function based on method name."""
        if method == 'solver' and MathematicalSolver.is_available():
            return MathematicalSolver.optimize
        elif method == 'improved_heuristic':
            return ImprovedHeuristicSolver.optimize
        else:
            return HeuristicSolver.optimize
    
    @staticmethod
    def _generate_branch_parameters(
        base_params: Dict[str, Any], 
        num_branches: int,
        criteria: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Generate parameter sets for different branches."""
        branch_params = {}
        
        # Branch 0 uses base parameters
        branch_params['branch_0'] = copy.deepcopy(base_params)
        
        # Generate variations based on criteria
        for i in range(1, num_branches):
            params = copy.deepcopy(base_params)
            
            # Vary service level
            if 'service_level' in criteria:
                # Create more aggressive and conservative service levels
                if i % 3 == 1:
                    # More aggressive (lower service level)
                    params['service_level'] = max(0.85, base_params.get('service_level', 0.95) - 0.05)
                elif i % 3 == 2:
                    # More conservative (higher service level)
                    params['service_level'] = min(0.995, base_params.get('service_level', 0.95) + 0.03)
            
            # Vary lead time estimates
            if 'robustness' in criteria:
                # Add lead time buffer for more robust policies
                if i % 2 == 0:
                    params['lead_time_factor'] = 1.2  # 20% buffer on lead times
            
            # Vary inflow levels
            if 'cost' in criteria and 'inflows' in base_params:
                # Adjust inflow based on branch number
                adjustment = 0.8 + (i % 5) * 0.1  # Between 0.8 and 1.2
                params['inflows'] = base_params['inflows'] * adjustment
            
            branch_params[f'branch_{i}'] = params
        
        return branch_params
    
    @staticmethod
    def _adjust_lead_times(network, factor: float):
        """Adjust lead times in the network by a factor."""
        for node_id, node in network.nodes.items():
            for prod in node.products:
                node.products[prod]['lead_time_mean'] *= factor
                if 'lead_time_std' in node.products[prod]:
                    node.products[prod]['lead_time_std'] *= factor
    
    @staticmethod
    def _adjust_demand(network, factor: float):
        """Adjust demand in the network by a factor."""
        for node_id, node in network.nodes.items():
            for prod in node.products:
                for t in range(len(node.products[prod]['demand_by_date'])):
                    node.products[prod]['demand_by_date'][t] *= factor


class BranchEvaluator:
    """Evaluates branches across multiple criteria and scenarios."""
    
    @staticmethod
    def evaluate_branches(
        network, 
        branches: Dict[str, Any],
        scenarios: List[Dict[str, Any]] = None,
        weights: Dict[str, float] = None
    ) -> Dict[str, Any]:
        """
        Evaluate branches across multiple criteria and scenarios.
        
        Args:
            network: The original network
            branches: Dictionary of branches generated by BranchGenerator
            scenarios: List of scenario definitions to test against
            weights: Dictionary mapping criteria to their weights
            
        Returns:
            Dict with evaluation results
        """
        if scenarios is None:
            scenarios = BranchEvaluator._generate_default_scenarios()
            
        if weights is None:
            weights = {
                'cost': 0.4,
                'service_level': 0.4,
                'robustness': 0.2
            }
            
        logger.info(f"Evaluating {len(branches)} branches against {len(scenarios)} scenarios")
        
        results = {}
        scenario_results = {}
        
        # Evaluate each branch against each scenario
        for scenario_id, scenario in enumerate(scenarios):
            scenario_name = scenario.get('name', f'scenario_{scenario_id}')
            logger.info(f"Evaluating against scenario: {scenario_name}")
            
            # Create modified network for this scenario
            scenario_network = BranchEvaluator._create_scenario_network(network, scenario)
            
            # Evaluate each branch on this scenario
            branch_evaluations = {}
            for branch_id, branch in branches.items():
                if branch.get('status') != 'error':
                    evaluation = BranchEvaluator._evaluate_branch_on_scenario(
                        scenario_network, branch, scenario)
                    branch_evaluations[branch_id] = evaluation
            
            scenario_results[scenario_name] = branch_evaluations
        
        # Calculate overall scores
        scores = BranchEvaluator._calculate_overall_scores(scenario_results, weights)
        
        # Determine best branch
        best_branch = max(scores.items(), key=lambda x: x[1]['overall_score'])[0]
        
        return {
            'scenario_results': scenario_results,
            'scores': scores,
            'best_branch': best_branch,
            'weights': weights
        }
    
    @staticmethod
    def _generate_default_scenarios() -> List[Dict[str, Any]]:
        """Generate default scenarios for evaluation."""
        return [
            {
                'name': 'baseline',
                'description': 'Baseline scenario with no modifications',
                'lead_time_factor': 1.0,
                'demand_factor': 1.0
            },
            {
                'name': 'high_demand',
                'description': 'High demand scenario',
                'lead_time_factor': 1.0,
                'demand_factor': 1.3
            },
            {
                'name': 'supply_disruption',
                'description': 'Supply chain disruption with longer lead times',
                'lead_time_factor': 1.5,
                'demand_factor': 1.0
            },
            {
                'name': 'combined_stress',
                'description': 'Combined stress with high demand and supply disruption',
                'lead_time_factor': 1.3,
                'demand_factor': 1.2
            }
        ]
    
    @staticmethod
    def _create_scenario_network(network, scenario: Dict[str, Any]):
        """Create a modified network based on scenario parameters."""
        scenario_network = copy.deepcopy(network)
        
        # Apply scenario modifications
        if 'lead_time_factor' in scenario:
            BranchGenerator._adjust_lead_times(scenario_network, scenario['lead_time_factor'])
        
        if 'demand_factor' in scenario:
            BranchGenerator._adjust_demand(scenario_network, scenario['demand_factor'])
        
        return scenario_network
    
    @staticmethod
    def _evaluate_branch_on_scenario(network, branch: Dict[str, Any], scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a branch on a specific scenario."""
        inventory_levels = branch.get('inventory_levels', {})
        
        # Calculate stockout metrics
        stockout_count, total_demand, stockout_rate = BranchEvaluator._calculate_stockout_metrics(
            network, inventory_levels)
        
        # Calculate cost metrics under this scenario
        total_cost = BranchEvaluator._calculate_scenario_cost(network, inventory_levels)
        
        # Calculate robustness metrics
        robustness_score = BranchEvaluator._calculate_robustness(network, inventory_levels)
        
        return {
            'total_cost': total_cost,
            'stockout_count': stockout_count,
            'total_demand': total_demand,
            'stockout_rate': stockout_rate,
            'service_level': 1.0 - stockout_rate,
            'robustness_score': robustness_score
        }
    
    @staticmethod
    def _calculate_stockout_metrics(network, inventory_levels: Dict) -> Tuple[int, float, float]:
        """Calculate stockout metrics for a given inventory policy."""
        stockout_count = 0
        total_demand = 0
        
        for node_id, node in network.nodes.items():
            if node.node_type == "store":  # Only count stockouts at store level
                for prod in node.products:
                    for t in range(network.num_periods):
                        inv = inventory_levels.get((node_id, prod, t), 0)
                        demand = node.products[prod]['demand_by_date'][t]
                        total_demand += demand
                        
                        if inv < demand:
                            stockout_count += 1
        
        stockout_rate = stockout_count / max(1, total_demand)
        return stockout_count, total_demand, stockout_rate
    
    @staticmethod
    def _calculate_scenario_cost(network, inventory_levels: Dict) -> float:
        """Calculate total cost for a given inventory policy under a scenario."""
        total_cost = 0
        
        for node_id, node in network.nodes.items():
            for prod in node.products:
                for t in range(network.num_periods):
                    inv = inventory_levels.get((node_id, prod, t), 0)
                    
                    # Holding cost
                    total_cost += node.products[prod]['holding_cost'] * inv
                    
                    # Shortage cost (for stores)
                    if node.node_type == "store":
                        demand = node.products[prod]['demand_by_date'][t]
                        if inv < demand:
                            shortage = demand - inv
                            total_cost += node.products[prod]['shortage_cost'] * shortage
                    
                    # Transport cost
                    if node.parent:
                        total_cost += node.transport_cost * inv
        
        return total_cost
    
    @staticmethod
    def _calculate_robustness(network, inventory_levels: Dict) -> float:
        """
        Calculate a robustness score for an inventory policy.
        Higher is better.
        """
        # Calculate average safety margin across all nodes/products/periods
        safety_margins = []
        
        for node_id, node in network.nodes.items():
            for prod in node.products:
                for t in range(network.num_periods):
                    inv = inventory_levels.get((node_id, prod, t), 0)
                    demand = node.products[prod]['demand_by_date'][t]
                    
                    if demand > 0:
                        safety_margin = (inv - demand) / demand
                        safety_margins.append(max(-1.0, safety_margin))  # Cap negative margins at -100%
        
        # Normalize robustness score to 0-1 range
        if safety_margins:
            avg_margin = np.mean(safety_margins)
            # Transform to 0-1 scale where 0 = -1 margin, 1 = 1 margin
            robustness_score = (avg_margin + 1) / 2
            return max(0, min(1, robustness_score))
        else:
            return 0.5  # Default if no data
    
    @staticmethod
    def _calculate_overall_scores(
        scenario_results: Dict[str, Dict[str, Dict[str, Any]]],
        weights: Dict[str, float]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate overall scores for each branch across all scenarios."""
        scores = {}
        
        # Get list of all branches
        all_branches = set()
        for scenario_data in scenario_results.values():
            all_branches.update(scenario_data.keys())
        
        # Calculate scores for each branch
        for branch_id in all_branches:
            branch_scores = {
                'cost_score': 0,
                'service_level_score': 0,
                'robustness_score': 0,
                'overall_score': 0,
                'scenario_scores': {}
            }
            
            scenarios_count = 0
            
            # Calculate scores across scenarios
            for scenario_name, scenario_data in scenario_results.items():
                if branch_id in scenario_data:
                    evaluation = scenario_data[branch_id]
                    scenarios_count += 1
                    
                    # Normalize cost (lower is better, so invert)
                    cost = evaluation.get('total_cost', float('inf'))
                    cost_score = 1.0 / (1.0 + cost / 10000)  # Normalize to 0-1 scale
                    
                    # Service level (higher is better)
                    service_level = evaluation.get('service_level', 0)
                    
                    # Robustness (higher is better)
                    robustness = evaluation.get('robustness_score', 0)
                    
                    # Store scenario scores
                    branch_scores['scenario_scores'][scenario_name] = {
                        'cost_score': cost_score,
                        'service_level_score': service_level,
                        'robustness_score': robustness
                    }
                    
                    # Add to total scores
                    branch_scores['cost_score'] += cost_score
                    branch_scores['service_level_score'] += service_level
                    branch_scores['robustness_score'] += robustness
            
            # Average scores across scenarios
            if scenarios_count > 0:
                branch_scores['cost_score'] /= scenarios_count
                branch_scores['service_level_score'] /= scenarios_count
                branch_scores['robustness_score'] /= scenarios_count
                
                # Calculate weighted average
                branch_scores['overall_score'] = (
                    weights.get('cost', 0.33) * branch_scores['cost_score'] +
                    weights.get('service_level', 0.33) * branch_scores['service_level_score'] +
                    weights.get('robustness', 0.34) * branch_scores['robustness_score']
                )
            
            scores[branch_id] = branch_scores
        
        return scores


class BranchSelector:
    """Selects the most appropriate branch based on evaluation results."""
    
    @staticmethod
    def select_branch(
        evaluation_results: Dict[str, Any],
        selection_criteria: str = 'balanced'
    ) -> Dict[str, Any]:
        """
        Select the most appropriate branch based on evaluation results.
        
        Args:
            evaluation_results: Results from BranchEvaluator
            selection_criteria: Strategy for selection ('balanced', 'cost_focused', 
                               'service_focused', or 'robust')
            
        Returns:
            Dict with selected branch and rationale
        """
        scores = evaluation_results.get('scores', {})
        
        if not scores:
            return {'selected_branch': None, 'rationale': 'No valid branches to select from'}
        
        # Define criteria weights based on selection strategy
        if selection_criteria == 'cost_focused':
            weights = {'cost_score': 0.6, 'service_level_score': 0.3, 'robustness_score': 0.1}
        elif selection_criteria == 'service_focused':
            weights = {'cost_score': 0.3, 'service_level_score': 0.6, 'robustness_score': 0.1}
        elif selection_criteria == 'robust':
            weights = {'cost_score': 0.2, 'service_level_score': 0.3, 'robustness_score': 0.5}
        else:  # balanced
            weights = {'cost_score': 0.33, 'service_level_score': 0.33, 'robustness_score': 0.34}
        
        # Calculate weighted scores for each branch
        weighted_scores = {}
        for branch_id, branch_scores in scores.items():
            weighted_score = (
                weights['cost_score'] * branch_scores['cost_score'] +
                weights['service_level_score'] * branch_scores['service_level_score'] +
                weights['robustness_score'] * branch_scores['robustness_score']
            )
            weighted_scores[branch_id] = weighted_score
        
        # Select branch with highest weighted score
        selected_branch = max(weighted_scores.items(), key=lambda x: x[1])[0]
        
        # Generate rationale
        selected_branch_scores = scores[selected_branch]
        rationale = (
            f"Branch {selected_branch} was selected using '{selection_criteria}' criteria. "
            f"It scored {selected_branch_scores['cost_score']:.2f} for cost, "
            f"{selected_branch_scores['service_level_score']:.2f} for service level, and "
            f"{selected_branch_scores['robustness_score']:.2f} for robustness, "
            f"with an overall score of {weighted_scores[selected_branch]:.2f}."
        )
        
        return {
            'selected_branch': selected_branch,
            'selection_criteria': selection_criteria,
            'rationale': rationale,
            'weighted_scores': weighted_scores
        }


class BranchManager:
    """Main class for managing the branch selection process."""
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize the branch manager.
        
        Args:
            output_dir: Directory to save results (uses default if None)
        """
        self.output_dir = output_dir or paths.BRANCH_SELECTION_DIR
        self.csv_exporter = CSVExporter(output_dir=self.output_dir)
        
        # Create output directory if it doesn't exist
        import os
        os.makedirs(self.output_dir, exist_ok=True)
    
    def run_branch_selection(
        self,
        network,
        num_branches: int = 5,
        criteria: List[str] = None,
        weights: Dict[str, float] = None,
        scenarios: List[Dict[str, Any]] = None,
        selection_criteria: str = 'balanced',
        method: str = 'improved_heuristic'
    ) -> Dict[str, Any]:
        """
        Run the full branch selection process.
        
        Args:
            network: The network to optimize
            num_branches: Number of branches to generate
            criteria: List of criteria to consider
            weights: Weights for each criterion
            scenarios: List of scenarios to evaluate against
            selection_criteria: Strategy for branch selection
            method: Optimization method to use
            
        Returns:
            Dict with complete results
        """
        logger.info(f"Starting branch selection process with {num_branches} branches")
        
        # Generate branches
        branch_results = BranchGenerator.generate_branches(
            network, num_branches, criteria, method=method)
        
        # Evaluate branches
        evaluation_results = BranchEvaluator.evaluate_branches(
            network, branch_results['branches'], scenarios, weights)
        
        # Select best branch
        selection_results = BranchSelector.select_branch(
            evaluation_results, selection_criteria)
        
        # Combine all results
        results = {
            'branch_results': branch_results,
            'evaluation_results': evaluation_results,
            'selection_results': selection_results,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'output_dir': self.output_dir
        }
        
        # Save results to CSV
        self._save_results_to_csv(results)
        
        return results
    
    def _save_results_to_csv(self, results: Dict[str, Any]):
        """Save all results to CSV files."""
        # Save branch parameters
        branch_params = []
        # Collect all possible parameter keys across all branches
        all_param_keys = set(['branch_id', 'status', 'total_cost'])
        
        for branch_id, branch in results['branch_results']['branches'].items():
            params = copy.deepcopy(branch['params'])
            params['branch_id'] = branch_id
            params['status'] = branch['status']
            params['total_cost'] = branch.get('total_cost', None)
            branch_params.append(params)
            # Collect all parameter keys
            all_param_keys.update(params.keys())
        
        # Convert to list and ensure essential fields come first
        fieldnames = ['branch_id', 'status', 'total_cost']
        for key in sorted(all_param_keys):
            if key not in fieldnames:
                fieldnames.append(key)
        
        self.csv_exporter.save_to_csv(
            'branch_parameters.csv', branch_params,
            fieldnames
        )
        
        # Save scenario results
        scenario_results = []
        for scenario_name, branches in results['evaluation_results']['scenario_results'].items():
            for branch_id, evaluation in branches.items():
                scenario_results.append({
                    'scenario': scenario_name,
                    'branch_id': branch_id,
                    'total_cost': evaluation.get('total_cost', None),
                    'service_level': evaluation.get('service_level', None),
                    'robustness_score': evaluation.get('robustness_score', None),
                    'stockout_count': evaluation.get('stockout_count', None)
                })
        
        self.csv_exporter.save_to_csv(
            'scenario_results.csv', scenario_results,
            ['scenario', 'branch_id', 'total_cost', 'service_level', 'robustness_score', 'stockout_count']
        )
        
        # Save branch scores
        branch_scores = []
        for branch_id, scores in results['evaluation_results']['scores'].items():
            branch_scores.append({
                'branch_id': branch_id,
                'cost_score': scores['cost_score'],
                'service_level_score': scores['service_level_score'],
                'robustness_score': scores['robustness_score'],
                'overall_score': scores['overall_score'],
                'is_selected': branch_id == results['selection_results']['selected_branch']
            })
        
        self.csv_exporter.save_to_csv(
            'branch_scores.csv', branch_scores,
            ['branch_id', 'cost_score', 'service_level_score', 'robustness_score', 
             'overall_score', 'is_selected']
        )
