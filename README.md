# MEIO Branch Selection Module

## Overview
The Branch Selection Strategy Module is a component of the MEIO (Multi-Echelon Inventory Optimization) system that enables the generation and evaluation of multiple inventory policy options (branches) to determine the optimal inventory management strategy.

This module provides a systematic approach to inventory optimization by:
1. Generating multiple policy branches with varying parameters
2. Evaluating these branches under different scenarios
3. Selecting the most appropriate branch based on defined criteria
4. (Optional) Using AI to evolve optimization parameters based on historical data

## Key Components

### BranchGenerator
Generates multiple inventory policy options (branches) based on different parameter sets.

### BranchEvaluator
Evaluates branches across multiple criteria and scenarios.

### BranchSelector
Selects the most appropriate branch based on evaluation results.

### BranchManager
Main class for managing the branch selection process.

### ParameterEvolutionAgent
AI agent that evolves optimization parameters based on historical performance.

## Execution Options

### Basic Branch Selection Process

The most straightforward way to use this module is through the `BranchManager` class:

```python
from meio.optimization.branch_selection import BranchManager

# Initialize the branch manager
branch_manager = BranchManager()

# Run the full branch selection process
results = branch_manager.run_branch_selection(
    network,                     # Your supply chain network
    num_branches=5,              # Number of branches to generate
    criteria=['cost', 'service_level', 'robustness'],  # Criteria to consider
    selection_criteria='balanced'  # Strategy for branch selection
)

# The selected branch and details can be accessed from results
selected_branch = results['selection_results']['selected_branch']
rationale = results['selection_results']['rationale']
```

### Customizing Branch Generation

For more control over branch generation:

```python
from meio.optimization.branch_selection import BranchGenerator

# Generate branches with custom parameters
base_params = {
    'service_level': 0.95,
    'inflows': 100
}

branch_results = BranchGenerator.generate_branches(
    network,
    num_branches=10,
    criteria=['cost', 'service_level', 'robustness'],
    base_params=base_params,
    method='improved_heuristic'  # Options: 'solver', 'improved_heuristic', 'heuristic'
)
```

### Manual Branch Evaluation

To manually evaluate branches against specific scenarios:

```python
from meio.optimization.branch_selection import BranchEvaluator

# Define custom scenarios
custom_scenarios = [
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
    }
]

# Custom weights for different criteria
weights = {
    'cost': 0.4,
    'service_level': 0.4,
    'robustness': 0.2
}

# Evaluate branches against scenarios
evaluation_results = BranchEvaluator.evaluate_branches(
    network,
    branch_results['branches'],
    scenarios=custom_scenarios,
    weights=weights
)
```

### Custom Branch Selection

To customize the branch selection process:

```python
from meio.optimization.branch_selection import BranchSelector

# Select branch using custom criteria
selection_results = BranchSelector.select_branch(
    evaluation_results,
    selection_criteria='cost_focused'  # Options: 'balanced', 'cost_focused', 'service_focused', 'robust'
)
```

### Using the AI Parameter Agent

To leverage the AI agent for parameter optimization:

```python
from meio.optimization.branch_selection import BranchManager

# Initialize with AI agent enabled
branch_manager = BranchManager(use_ai_agent=True)

# Run branch selection with AI parameter suggestions
results = branch_manager.run_branch_selection(
    network,
    num_branches=5
)

# Import historical data for AI learning
import_result = branch_manager.import_historical_data('path/to/historical_data.json')

# Get AI agent trend information
trend_data = branch_manager.get_ai_agent_trend()
```

### Executing Step by Step

For maximum control, you can execute the process step by step:

```python
from meio.optimization.branch_selection import BranchGenerator, BranchEvaluator, BranchSelector

# Step 1: Generate branches
branch_results = BranchGenerator.generate_branches(network, num_branches=5)

# Step 2: Evaluate branches
evaluation_results = BranchEvaluator.evaluate_branches(
    network, 
    branch_results['branches']
)

# Step 3: Select best branch
selection_results = BranchSelector.select_branch(evaluation_results)

# Step 4: Extract and use the selected branch
selected_branch_id = selection_results['selected_branch']
selected_inventory_levels = branch_results['branches'][selected_branch_id]['inventory_levels']
```

## Parameter Options

### Branch Generation Parameters

- `num_branches`: Number of branches to generate (default: 5)
- `criteria`: List of criteria to consider (default: ['cost', 'service_level', 'robustness'])
- `base_params`: Base parameters for optimization
  - `service_level`: Target service level (default from config)
  - `inflows`: Default inflow value (default from config)
- `method`: Optimization method ('solver', 'improved_heuristic', 'heuristic')

### Scenario Parameters

- `lead_time_factor`: Adjustment factor for lead times (1.0 = no change)
- `demand_factor`: Adjustment factor for demand (1.0 = no change)

### Branch Selection Criteria

- `balanced`: Equal weight to cost, service level, and robustness
- `cost_focused`: Higher weight on cost reduction
- `service_focused`: Higher weight on service level
- `robust`: Higher weight on robustness to disruptions

## Example End-to-End Process

```python
from meio.optimization.branch_selection import BranchManager

# Initialize branch manager with AI assistance
branch_manager = BranchManager(use_ai_agent=True)

# Define custom weights
weights = {
    'cost': 0.5,
    'service_level': 0.3,
    'robustness': 0.2
}

# Define custom scenarios
scenarios = [
    {
        'name': 'baseline',
        'lead_time_factor': 1.0,
        'demand_factor': 1.0
    },
    {
        'name': 'peak_season',
        'lead_time_factor': 1.1,
        'demand_factor': 1.5
    },
    {
        'name': 'supply_disruption',
        'lead_time_factor': 2.0,
        'demand_factor': 0.9
    }
]

# Run the full process
results = branch_manager.run_branch_selection(
    network,
    num_branches=8,
    criteria=['cost', 'service_level', 'robustness'],
    weights=weights,
    scenarios=scenarios,
    selection_criteria='robust',
    method='improved_heuristic'
)

# Save results to CSV
# (automatically done by BranchManager)

# Extract selected branch details
selected_branch_id = results['selection_results']['selected_branch']
selected_branch = results['branch_results']['branches'][selected_branch_id]
inventory_levels = selected_branch['inventory_levels']
```
