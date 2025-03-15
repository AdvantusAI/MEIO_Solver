# Branch Selection Strategy Guide

The Branch Selection Strategy is a powerful feature of the MEIO Solver that helps supply chain planners deal with the complexity of multi-objective decision making. This guide explains how to use the feature and how it works under the hood.

## Overview

In real-world multi-echelon inventory management, optimizing for a single objective (like cost) often doesn't capture the full complexity of decision making. The Branch Selection Strategy allows you to:

1. Generate multiple inventory policy options (branches) with different parameters
2. Evaluate these options under various scenarios (e.g., demand spikes, supply disruptions)
3. Score each option based on multiple criteria (cost, service level, robustness)
4. Select the most appropriate policy using a formalized decision process

## Quick Start

```bash
# Basic usage
python -m meio.main --json network.json --branch-selection

# Advanced usage with customized parameters
python -m meio.main --json network.json --branch-selection \
    --num-branches 8 \
    --branch-criteria cost,service_level,robustness \
    --branch-weights 0.3,0.5,0.2 \
    --selection-criteria service_focused \
    --branch-method improved_heuristic
```

## Command-line Options

| Option | Description | Default Value |
|--------|-------------|---------------|
| `--branch-selection` | Enable branch selection | N/A |
| `--num-branches` | Number of branches to generate | 5 |
| `--branch-criteria` | Criteria for evaluation | cost,service_level,robustness |
| `--branch-weights` | Weights for each criterion | 0.4,0.4,0.2 |
| `--selection-criteria` | Strategy for selecting the best branch | balanced |
| `--branch-method` | Optimization method to use | improved_heuristic |

## Selection Criteria

The system supports four pre-defined selection criteria:

| Criteria | Description | Weight Distribution |
|----------|-------------|---------------------|
| `balanced` | Equal weight to all criteria | Cost: 33%, Service: 33%, Robustness: 34% |
| `cost_focused` | Prioritizes cost-efficiency | Cost: 60%, Service: 30%, Robustness: 10% |
| `service_focused` | Prioritizes service level | Cost: 30%, Service: 60%, Robustness: 10% |
| `robust` | Prioritizes policy robustness | Cost: 20%, Service: 30%, Robustness: 50% |

## How It Works

### 1. Branch Generation

The system generates inventory policy "branches" by varying key parameters:

- **Service Level**: Determines safety stock levels
- **Lead Time Factor**: Adds buffers to expected lead times
- **Inflow Levels**: Adjusts base inflow buffer levels

Each branch represents a different inventory policy with its own trade-offs.

### 2. Scenario Testing

Each branch is evaluated against multiple scenarios:

- **Baseline**: Normal operations
- **High Demand**: Unexpected increase in demand
- **Supply Disruption**: Longer lead times
- **Combined Stress**: Both high demand and supply disruption

### 3. Branch Evaluation

Branches are scored on three key metrics:

- **Cost Efficiency**: Total operating cost (holding, shortage, transport)
- **Service Level**: Ability to meet demand without stockouts
- **Robustness**: Performance stability across different scenarios

### 4. Branch Selection

The final selection applies weights to each criterion according to the chosen selection strategy and selects the branch with the highest overall score.

## Output and Visualizations

The branch selection process produces:

1. **CSV Files**:
   - `branch_parameters.csv`: Parameters for each branch
   - `scenario_results.csv`: Performance metrics by scenario
   - `branch_scores.csv`: Final scores and selection status

2. **Visualizations**:
   - Branch comparison chart
   - Scenario performance analysis
   - Selected branch details with implementation guidance

## Advanced Usage

### Custom Scenarios

You can define custom scenarios by modifying the default scenarios in the `BranchEvaluator._generate_default_scenarios` method:

```python
custom_scenarios = [
    {
        'name': 'seasonal_peak',
        'description': 'Holiday season peak demand',
        'lead_time_factor': 1.0,
        'demand_factor': 1.5
    },
    # Add more custom scenarios
]

# Use when calling run_branch_selection
results = branch_manager.run_branch_selection(
    network,
    scenarios=custom_scenarios,
    # other parameters
)
```

### Programmatic API

You can use the branch selection system programmatically:

```python
from meio.optimization.branch_selection import BranchManager
from meio.visualization.branch_viz import BranchVisualizer

# Initialize branch manager
branch_manager = BranchManager(output_dir="my_results")

# Run branch selection
results = branch_manager.run_branch_selection(
    network,
    num_branches=5,
    criteria=['cost', 'service_level', 'robustness'],
    weights={'cost': 0.4, 'service_level': 0.4, 'robustness': 0.2},
    selection_criteria='balanced',
    method='improved_heuristic'
)

# Create visualizations
viz_paths = BranchVisualizer.visualize_branch_selection_summary(results)

# Access selected branch
selected_branch = results['selection_results']['selected_branch']
selected_data = results['branch_results']['branches'][selected_branch]
```

## Implementation in Real-Life

When implementing the selected strategy in a real supply chain:

1. Apply the service level settings from the selected branch
2. If the branch includes a lead time factor, add the specified buffer to lead time estimates
3. If the branch includes custom inflow levels, adjust reorder policies accordingly
4. Monitor performance and be prepared to adjust as needed

## Example Use Case

A consumer goods company facing seasonal demand volatility used the Branch Selection Strategy to evaluate different inventory policies:

- **Branch 0**: Conservative baseline (SL=0.95)
- **Branch 1**: Cost-focused option (SL=0.90)
- **Branch 2**: Service-focused option (SL=0.98)
- **Branch 3**: Robust option (SL=0.95, LT×1.2)

After evaluation across multiple scenarios, the system selected Branch 3 (the robust option) for implementation because it provided the best balance between service level and cost while being resilient to supply chain disruptions.

## Troubleshooting

If you encounter any issues with the Branch Selection Strategy:

1. **Error during branch generation**: Check if the optimizer method specified is available
2. **Path-related errors**: Ensure the MEIO system paths are correctly initialized
3. **Visualization errors**: Check if the required Python packages (matplotlib, seaborn) are installed

For more assistance, please consult the main MEIO Solver documentation or submit an issue to the project repository. 