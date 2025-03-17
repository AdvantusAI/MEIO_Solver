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
        
        # Ensure scores are formatted as floats
        cost_score = selected_branch_scores['cost_score']
        service_level_score = selected_branch_scores['service_level_score']
        robustness_score = selected_branch_scores['robustness_score']
        overall_score = weighted_scores[selected_branch]
        
        # Ensure all values are numeric
        cost_score = float(cost_score) if isinstance(cost_score, (int, float)) else 0
        service_level_score = float(service_level_score) if isinstance(service_level_score, (int, float)) else 0
        robustness_score = float(robustness_score) if isinstance(robustness_score, (int, float)) else 0
        overall_score = float(overall_score) if isinstance(overall_score, (int, float)) else 0
        
        rationale = (
            f"Branch {selected_branch} was selected using '{selection_criteria}' criteria. "
            f"It scored {cost_score:.2f} for cost, "
            f"{service_level_score:.2f} for service level, and "
            f"{robustness_score:.2f} for robustness, "
            f"with an overall score of {overall_score:.2f}."
        )
        
        return {
            'selected_branch': selected_branch,
            'selection_criteria': selection_criteria,
            'rationale': rationale,
            'weighted_scores': weighted_scores
        }


class BranchManager:
    """Main class for managing the branch selection process."""
    
    def __init__(self, output_dir: Optional[str] = None, use_ai_agent: bool = False):
        """
        Initialize the branch manager.
        
        Args:
            output_dir: Directory to save results (uses default if None)
            use_ai_agent: Whether to use the Parameter Evolution Agent for parameter suggestions
        """
        self.output_dir = output_dir or paths.BRANCH_SELECTION_DIR
        self.csv_exporter = CSVExporter(output_dir=self.output_dir)
        self.use_ai_agent = use_ai_agent
        self.parameter_agent = ParameterEvolutionAgent() if use_ai_agent else None
        
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
        method: str = 'improved_heuristic',
        base_params: Dict[str, Any] = None
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
            base_params: Base parameters to use for optimization (uses defaults if None)
            
        Returns:
            Dict with complete results
        """
        logger.info(f"Starting branch selection process with {num_branches} branches")
        
        # Use AI agent to suggest parameters if enabled
        ai_suggestion = None
        if self.use_ai_agent and self.parameter_agent:
            ai_suggestion = self.parameter_agent.suggest_parameters(network, base_params)
            if ai_suggestion and ai_suggestion.get('confidence') != 'low':
                logger.info(f"Using AI-suggested parameters with {ai_suggestion.get('confidence')} confidence")
                base_params = ai_suggestion.get('parameters', base_params)
            else:
                logger.info("AI agent suggestion has low confidence, using default parameters")
        
        # Generate branches
        branch_results = BranchGenerator.generate_branches(
            network, num_branches, criteria, base_params=base_params, method=method)
        
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
            'output_dir': self.output_dir,
            'ai_suggestion': ai_suggestion
        }
        
        # Save results to CSV
        self._save_results_to_csv(results)
        
        # Record results in AI agent if enabled
        if self.use_ai_agent and self.parameter_agent:
            self.parameter_agent.record_run_results(network, results)
            
            # Include AI agent performance trend in results
            results['ai_performance_trend'] = self.parameter_agent.get_performance_trend()
        
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
        
        # Save AI agent parameters if used
        if 'ai_suggestion' in results and results['ai_suggestion']:
            self.csv_exporter.save_to_csv(
                'ai_suggested_parameters.csv', 
                [{'parameter': k, 'value': v} for k, v in results['ai_suggestion']['parameters'].items() 
                 if isinstance(v, (int, float, str))],
                ['parameter', 'value']
            )
    
    def get_ai_agent_trend(self) -> Dict[str, Any]:
        """
        Get performance trend data from the AI agent.
        
        Returns:
            Dict with trend data or error message if AI agent is not enabled
        """
        if not self.use_ai_agent or not self.parameter_agent:
            return {"status": "error", "message": "AI agent is not enabled"}
        
        return self.parameter_agent.get_performance_trend()
    
    def import_historical_data(self, data_source: str) -> Dict[str, Any]:
        """
        Import historical data for AI agent learning.
        
        Args:
            data_source: Path to JSON file or directory containing historical run results
            
        Returns:
            Dict with import results
        """
        if not self.use_ai_agent or not self.parameter_agent:
            return {"status": "error", "message": "AI agent is not enabled"}
        
        import os
        import json
        import glob
        
        historical_runs = []
        
        try:
            if os.path.isfile(data_source):
                # Single file import
                with open(data_source, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        historical_runs.extend(data)
                    else:
                        historical_runs.append(data)
            elif os.path.isdir(data_source):
                # Directory import
                for filepath in glob.glob(os.path.join(data_source, "*.json")):
                    try:
                        with open(filepath, 'r') as f:
                            data = json.load(f)
                            if isinstance(data, list):
                                historical_runs.extend(data)
                            else:
                                historical_runs.append(data)
                    except Exception as e:
                        logger.warning(f"Error importing {filepath}: {str(e)}")
            else:
                return {"status": "error", "message": f"Data source not found: {data_source}"}
            
            # Learn from imported data
            result = self.parameter_agent.learn_optimal_parameters(historical_runs)
            
            return {
                "status": "success",
                "message": f"Imported and learned from {len(historical_runs)} historical runs",
                "details": result
            }
            
        except Exception as e:
            logger.error(f"Error importing historical data: {str(e)}")
            return {"status": "error", "message": str(e)}


class ParameterEvolutionAgent:
    """AI agent that evolves optimization parameters based on historical performance."""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the Parameter Evolution Agent.
        
        Args:
            db_path: Path to store historical parameter data (uses default if None)
        """
        import os
        import json
        from datetime import datetime
        
        self.db_path = db_path or os.path.join(paths.RESULTS_DIR, "parameter_evolution.json")
        self.historical_data = []
        self.network_features_cache = {}
        
        # Load historical data if available
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'r') as f:
                    self.historical_data = json.load(f)
                logger.info(f"Loaded {len(self.historical_data)} historical parameter records")
            except Exception as e:
                logger.error(f"Error loading historical parameter data: {str(e)}")
    
    def learn_optimal_parameters(self, historical_runs: List[Dict[str, Any]]):
        """
        Learn optimal parameters from past branch selections and their outcomes.
        
        Args:
            historical_runs: List of dictionaries containing past branch selection results
        
        Returns:
            Dict with learning results and statistics
        """
        if not historical_runs:
            logger.warning("No historical runs provided for learning")
            return {"status": "error", "message": "No historical runs provided"}
        
        try:
            logger.info(f"Learning from {len(historical_runs)} historical runs")
            
            # Extract features and outcomes from historical runs
            for run in historical_runs:
                self._process_historical_run(run)
            
            # Save updated historical data
            self._save_historical_data()
            
            # Calculate some basic statistics
            param_counts = self._calculate_parameter_statistics()
            
            return {
                "status": "success",
                "message": f"Successfully learned from {len(historical_runs)} runs",
                "statistics": param_counts
            }
        
        except Exception as e:
            logger.error(f"Error learning from historical runs: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _process_historical_run(self, run: Dict[str, Any]):
        """Process a single historical run and extract learning data."""
        if 'selection_results' not in run or 'evaluation_results' not in run or 'branch_results' not in run:
            logger.warning("Invalid historical run format - missing required sections")
            return
        
        # Get network features if available
        network_features = run.get('network_features', {})
        if not network_features and 'network_hash' in run:
            network_features = self.network_features_cache.get(run['network_hash'], {})
        
        # Get selected branch
        selected_branch_id = run['selection_results'].get('selected_branch')
        if not selected_branch_id:
            return
        
        # Get branch parameters and performance metrics
        branches = run['branch_results'].get('branches', {})
        if selected_branch_id not in branches:
            return
        
        branch = branches[selected_branch_id]
        params = branch.get('params', {})
        
        # Get performance metrics
        metrics = {}
        scores = run['evaluation_results'].get('scores', {}).get(selected_branch_id, {})
        metrics.update({
            'cost_score': scores.get('cost_score', 0),
            'service_level_score': scores.get('service_level_score', 0),
            'robustness_score': scores.get('robustness_score', 0),
            'overall_score': scores.get('overall_score', 0),
        })
        
        # Add to historical data
        entry = {
            'timestamp': branch.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
            'network_features': network_features,
            'parameters': params,
            'metrics': metrics,
            'branch_id': selected_branch_id,
            'scenario_performance': self._extract_scenario_performance(run, selected_branch_id)
        }
        
        self.historical_data.append(entry)
    
    def _extract_scenario_performance(self, run: Dict[str, Any], branch_id: str) -> Dict[str, Any]:
        """Extract performance across different scenarios for a branch."""
        scenario_results = {}
        
        if 'evaluation_results' in run and 'scenario_results' in run['evaluation_results']:
            for scenario_name, branches in run['evaluation_results']['scenario_results'].items():
                if branch_id in branches:
                    scenario_results[scenario_name] = branches[branch_id]
        
        return scenario_results
    
    def _save_historical_data(self):
        """Save historical data to disk."""
        import json
        import os
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        try:
            with open(self.db_path, 'w') as f:
                json.dump(self.historical_data, f, indent=2)
            logger.info(f"Saved {len(self.historical_data)} parameter records to {self.db_path}")
        except Exception as e:
            logger.error(f"Error saving historical parameter data: {str(e)}")
    
    def _calculate_parameter_statistics(self) -> Dict[str, Any]:
        """Calculate statistics about parameter values and their performance."""
        if not self.historical_data:
            return {}
        
        param_values = {}
        
        # Collect all parameter names and their values
        for entry in self.historical_data:
            for param_name, param_value in entry.get('parameters', {}).items():
                if param_name not in param_values:
                    param_values[param_name] = []
                param_values[param_name].append({
                    'value': param_value,
                    'overall_score': entry.get('metrics', {}).get('overall_score', 0)
                })
        
        # Calculate statistics for each parameter
        param_stats = {}
        for param_name, values in param_values.items():
            param_stats[param_name] = {
                'count': len(values),
                'min_value': min([v['value'] for v in values if isinstance(v['value'], (int, float))], default=None),
                'max_value': max([v['value'] for v in values if isinstance(v['value'], (int, float))], default=None),
                'best_value': max(values, key=lambda x: x['overall_score'])['value'] if values else None
            }
        
        return param_stats
    
    def suggest_parameters(self, network, base_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Suggest parameters for current optimization run based on historical learning.
        
        Args:
            network: The current network to optimize
            base_params: Base parameters to start from (uses defaults if None)
            
        Returns:
            Dict with suggested parameters and confidence levels
        """
        if base_params is None:
            base_params = {
                'service_level': config.get('optimization', 'default_service_level'),
                'inflows': config.get('optimization', 'default_inflow')
            }
        
        # If we don't have enough historical data, return base params with low confidence
        if len(self.historical_data) < 5:
            logger.info("Not enough historical data for confident parameter suggestions")
            return {
                'parameters': base_params,
                'confidence': 'low',
                'rationale': 'Insufficient historical data for learning',
                'data_points': len(self.historical_data)
            }
        
        # Extract features from the current network
        network_features = self._extract_network_features(network)
        
        # Find similar historical networks and their performance
        similar_runs = self._find_similar_networks(network_features)
        
        if not similar_runs:
            logger.info("No similar networks found in historical data")
            return {
                'parameters': base_params,
                'confidence': 'low',
                'rationale': 'No similar networks found in historical data',
                'data_points': len(self.historical_data)
            }
        
        # Generate suggested parameters based on similar networks
        suggested_params = self._generate_suggested_parameters(similar_runs, base_params)
        
        confidence = 'medium' if len(similar_runs) >= 3 and len(self.historical_data) >= 10 else 'low'
        if len(similar_runs) >= 5 and len(self.historical_data) >= 20:
            confidence = 'high'
        
        return {
            'parameters': suggested_params,
            'confidence': confidence,
            'rationale': f"Based on {len(similar_runs)} similar historical networks",
            'data_points': len(self.historical_data),
            'similar_networks': len(similar_runs)
        }
    
    def _extract_network_features(self, network) -> Dict[str, Any]:
        """Extract key features from a network for similarity comparison."""
        features = {
            'node_count': len(network.nodes),
            'product_count': sum(len(node.products) for node_id, node in network.nodes.items()),
            'avg_lead_time': 0,
            'avg_demand': 0,
            'network_depth': self._calculate_network_depth(network),
            'network_type': self._determine_network_type(network)
        }
        
        # Calculate average lead time and demand
        lead_times = []
        demands = []
        
        for node_id, node in network.nodes.items():
            for prod in node.products:
                # Ensure lead_time_mean is numeric
                if 'lead_time_mean' in node.products[prod]:
                    lead_time = node.products[prod]['lead_time_mean']
                    if isinstance(lead_time, (int, float)):
                        lead_times.append(lead_time)
                
                # Ensure demand values are numeric
                if 'demand_by_date' in node.products[prod]:
                    for demand in node.products[prod]['demand_by_date']:
                        if isinstance(demand, (int, float)):
                            demands.append(demand)
        
        if lead_times:
            features['avg_lead_time'] = sum(lead_times) / len(lead_times)
        
        if demands:
            features['avg_demand'] = sum(demands) / len(demands)
        
        # Calculate demand variability
        if demands:
            import numpy as np
            mean_demand = np.mean(demands)
            if mean_demand > 0.001:  # Avoid division by zero
                features['demand_variability'] = np.std(demands) / mean_demand
            else:
                features['demand_variability'] = 0
        
        # Calculate hash for this network
        import hashlib
        import json
        
        # Create a serializable version of features
        hash_features = {k: (str(v) if not isinstance(v, (int, float, str, bool)) else v) 
                         for k, v in features.items()}
        
        network_hash = hashlib.md5(json.dumps(hash_features, sort_keys=True).encode()).hexdigest()
        features['hash'] = network_hash
        
        # Cache these features
        self.network_features_cache[network_hash] = features
        
        return features
    
    def _calculate_network_depth(self, network) -> int:
        """Calculate the maximum depth of the network."""
        max_depth = 0
        
        # Start with nodes that have no children
        leaf_nodes = [node_id for node_id, node in network.nodes.items() 
                    if not any(n.parent == node_id for n_id, n in network.nodes.items())]
        
        # Calculate depth for each leaf node
        for leaf_id in leaf_nodes:
            depth = 0
            current_id = leaf_id
            
            # Traverse up to root
            while current_id is not None:
                current_node = network.nodes.get(current_id)
                if current_node is None:
                    break
                    
                depth += 1
                current_id = current_node.parent
            
            max_depth = max(max_depth, depth)
        
        return max_depth
    
    def _determine_network_type(self, network) -> str:
        """Determine the type of network structure."""
        # Count nodes by type
        store_count = sum(1 for n_id, n in network.nodes.items() if n.node_type == "store")
        dc_count = sum(1 for n_id, n in network.nodes.items() if n.node_type == "dc")
        supplier_count = sum(1 for n_id, n in network.nodes.items() if n.node_type == "supplier")
        
        # Determine network complexity
        if len(network.nodes) <= 5:
            size = "small"
        elif len(network.nodes) <= 20:
            size = "medium"
        else:
            size = "large"
        
        # Determine structure
        if dc_count == 0:
            structure = "direct"
        elif dc_count == 1:
            structure = "simple_hub"
        else:
            structure = "multi_tier"
        
        return f"{size}_{structure}"
    
    def _find_similar_networks(self, current_features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find historically optimized networks similar to the current one."""
        similar_runs = []
        
        for entry in self.historical_data:
            hist_features = entry.get('network_features', {})
            if not hist_features:
                continue
            
            # Calculate similarity score
            similarity = self._calculate_similarity(current_features, hist_features)
            
            if similarity >= 0.7:  # At least 70% similar
                similar_runs.append({
                    'entry': entry,
                    'similarity': similarity
                })
        
        # Sort by similarity (most similar first)
        similar_runs.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Return top 5 most similar
        return similar_runs[:5]
    
    def _calculate_similarity(self, features1: Dict[str, Any], features2: Dict[str, Any]) -> float:
        """Calculate similarity score between two network feature sets."""
        if not features1 or not features2:
            return 0.0
        
        # Fast path for identical networks
        if features1.get('hash') == features2.get('hash'):
            return 1.0
        
        # Define feature weights
        weights = {
            'node_count': 0.2,
            'product_count': 0.2,
            'avg_lead_time': 0.15,
            'avg_demand': 0.15,
            'network_depth': 0.1,
            'demand_variability': 0.1,
            'network_type': 0.1
        }
        
        # Calculate similarity for numeric features
        num_sim_score = 0.0
        num_weight_sum = 0.0
        
        for feature, weight in weights.items():
            if feature == 'network_type':
                continue
                
            if feature in features1 and feature in features2:
                val1 = features1[feature]
                val2 = features2[feature]
                
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    # Normalize to 0-1 range
                    max_val = max(val1, val2, 0.001)  # Avoid division by zero
                    sim = 1.0 - min(1.0, abs(val1 - val2) / max_val)
                    num_sim_score += sim * weight
                    num_weight_sum += weight
        
        # Normalize numeric similarity
        if num_weight_sum > 0:
            num_sim_score /= num_weight_sum
        
        # Calculate categorical similarity
        cat_sim_score = 0.0
        if features1.get('network_type') == features2.get('network_type'):
            cat_sim_score = 1.0
        
        # Combine scores
        total_sim = num_sim_score * 0.9 + cat_sim_score * 0.1
        
        return total_sim
    
    def _generate_suggested_parameters(
        self, 
        similar_runs: List[Dict[str, Any]], 
        base_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate suggested parameters based on similar historical runs.
        
        Uses a weighted average of parameters from similar networks, with
        weights based on similarity and performance.
        """
        suggested_params = copy.deepcopy(base_params)
        
        # Collect all parameter names across similar runs
        all_params = set()
        for run in similar_runs:
            all_params.update(run['entry'].get('parameters', {}).keys())
        
        # Calculate weighted average for each parameter
        for param_name in all_params:
            if param_name not in base_params:
                continue
                
            # Skip if base parameter is not numeric
            base_value = base_params[param_name]
            if not isinstance(base_value, (int, float)):
                continue
            
            # Collect values and weights
            values = []
            weights = []
            
            for run in similar_runs:
                entry = run['entry']
                similarity = run['similarity']
                
                if param_name in entry.get('parameters', {}):
                    param_value = entry['parameters'][param_name]
                    if isinstance(param_value, (int, float)):
                        # Try to get the performance score, defaulting to 0.5 if not available or not numeric
                        performance = entry.get('metrics', {}).get('overall_score', 0.5)
                        if not isinstance(performance, (int, float)):
                            try:
                                performance = float(performance)
                            except (ValueError, TypeError):
                                performance = 0.5
                        
                        # Weight is product of similarity and performance
                        weight = similarity * performance
                        values.append(param_value)
                        weights.append(weight)
            
            # Calculate weighted average
            if values and sum(weights) > 0:
                weighted_avg = sum(v * w for v, w in zip(values, weights)) / sum(weights)
                
                # Blend with base parameter (80% historical, 20% base)
                suggested_params[param_name] = 0.8 * weighted_avg + 0.2 * base_value
                
                # Round service level parameters to 3 decimal places
                if param_name == 'service_level':
                    suggested_params[param_name] = round(suggested_params[param_name], 3)
                    # Keep within valid range
                    suggested_params[param_name] = max(0.8, min(0.999, suggested_params[param_name]))
        
        return suggested_params
    
    def record_run_results(self, network, run_results: Dict[str, Any]) -> bool:
        """
        Record results from a completed branch selection run.
        
        Args:
            network: The network that was optimized
            run_results: Results from BranchManager.run_branch_selection()
            
        Returns:
            Boolean indicating success
        """
        try:
            # Extract network features
            network_features = self._extract_network_features(network)
            
            # Add network features to results
            run_results['network_features'] = network_features
            run_results['network_hash'] = network_features.get('hash')
            
            # Process this run
            self._process_historical_run(run_results)
            
            # Save updated historical data
            self._save_historical_data()
            
            return True
        
        except Exception as e:
            logger.error(f"Error recording run results: {str(e)}")
            return False
            
    def get_performance_trend(self) -> Dict[str, Any]:
        """
        Calculate performance trends over time from historical data.
        
        Returns:
            Dict with trend analysis and statistics
        """
        if len(self.historical_data) < 2:
            return {"status": "insufficient_data", "message": "Need at least 2 data points for trend analysis"}
        
        # Sort data by timestamp
        sorted_data = sorted(self.historical_data, key=lambda x: x.get('timestamp', ''))
        
        # Extract overall scores over time
        timestamps = []
        overall_scores = []
        cost_scores = []
        service_scores = []
        robustness_scores = []
        
        for entry in sorted_data:
            timestamps.append(entry.get('timestamp', ''))
            metrics = entry.get('metrics', {})
            
            # Ensure all values are numeric
            overall_score = metrics.get('overall_score', 0)
            cost_score = metrics.get('cost_score', 0)
            service_score = metrics.get('service_level_score', 0)
            robustness_score = metrics.get('robustness_score', 0)
            
            # Convert to float if they're strings but represent numbers
            try:
                if not isinstance(overall_score, (int, float)):
                    overall_score = float(overall_score) if overall_score.replace('.', '', 1).isdigit() else 0
                if not isinstance(cost_score, (int, float)):
                    cost_score = float(cost_score) if cost_score.replace('.', '', 1).isdigit() else 0
                if not isinstance(service_score, (int, float)):
                    service_score = float(service_score) if service_score.replace('.', '', 1).isdigit() else 0
                if not isinstance(robustness_score, (int, float)):
                    robustness_score = float(robustness_score) if robustness_score.replace('.', '', 1).isdigit() else 0
            except (ValueError, AttributeError):
                # If conversion fails, use default value 0
                if not isinstance(overall_score, (int, float)): overall_score = 0
                if not isinstance(cost_score, (int, float)): cost_score = 0
                if not isinstance(service_score, (int, float)): service_score = 0
                if not isinstance(robustness_score, (int, float)): robustness_score = 0
            
            overall_scores.append(overall_score)
            cost_scores.append(cost_score)
            service_scores.append(service_score)
            robustness_scores.append(robustness_score)
        
        # Calculate trend (simple linear regression)
        import numpy as np
        x = np.arange(len(timestamps))
        
        # Overall score trend
        if len(x) >= 2:
            overall_slope = np.polyfit(x, overall_scores, 1)[0]
            cost_slope = np.polyfit(x, cost_scores, 1)[0]
            service_slope = np.polyfit(x, service_scores, 1)[0]
            robustness_slope = np.polyfit(x, robustness_scores, 1)[0]
        else:
            overall_slope = cost_slope = service_slope = robustness_slope = 0
        
        return {
            "status": "success",
            "data_points": len(sorted_data),
            "latest_timestamp": timestamps[-1],
            "overall_score_trend": "improving" if overall_slope > 0.01 else "stable" if abs(overall_slope) <= 0.01 else "declining",
            "cost_score_trend": "improving" if cost_slope > 0.01 else "stable" if abs(cost_slope) <= 0.01 else "declining",
            "service_score_trend": "improving" if service_slope > 0.01 else "stable" if abs(service_slope) <= 0.01 else "declining",
            "robustness_score_trend": "improving" if robustness_slope > 0.01 else "stable" if abs(robustness_slope) <= 0.01 else "declining",
            "latest_scores": {
                "overall": overall_scores[-1],
                "cost": cost_scores[-1],
                "service_level": service_scores[-1],
                "robustness": robustness_scores[-1]
            }
        }
