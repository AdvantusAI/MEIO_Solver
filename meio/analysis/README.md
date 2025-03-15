# Sensitivity Analysis Module for MEIO System

This module provides functionality to analyze how sensitive inventory strategies are to changes in various parameters such as service levels, lead times, and demand patterns.

## Overview

The sensitivity analysis module helps you understand how different parameters impact your inventory optimization results. By systematically varying one parameter at a time, it reveals:

- Which parameters have the greatest impact on total cost
- How sensitive your inventory strategy is to changes in service level, lead times, or demand
- Where you should focus your attention for data accuracy and monitoring

## Key Features

- **Multiple Parameter Testing**: Systematically analyze the impact of various parameters
- **Elasticity Calculation**: Measure the percentage change in cost relative to percentage change in input parameters
- **Visualization**: Generate charts showing the relationship between parameters and costs
- **Detailed Reporting**: Comprehensive reports with insights and recommendations
- **Performance Metrics**: Track service levels, stockouts, and costs across parameter variations

## How to Use

### Command Line Interface

You can run sensitivity analysis from the command line using the dedicated script:

```bash
python -m meio.analyze_sensitivity --json path/to/network.json
```

#### Options:

- `--json`: (Required) Path to network JSON file
- `--method`: Optimization method to use (choices: 'heuristic', 'improved_heuristic', 'solver')
- `--service-levels`: Comma-separated list of service levels to test (e.g., 0.90,0.95,0.98)
- `--lead-time-factors`: Comma-separated list of lead time factors to test (e.g., 0.8,1.0,1.2)
- `--demand-factors`: Comma-separated list of demand factors to test (e.g., 0.8,1.0,1.2)
- `--inflows`: Comma-separated list of inflow values to test (e.g., 0.8,1.0,1.2)
- `--output-dir`: Output directory (defaults to results/sensitivity_[timestamp])
- `--no-viz`: Skip generating visualizations
- `--verbose`: Enable verbose logging

### Via main.py

You can also run sensitivity analysis through the main script by adding the `--sensitivity` flag:

```bash
python -m meio.main --json path/to/network.json --sensitivity
```

#### Additional sensitivity options:

- `--sensitivity`: Run sensitivity analysis
- `--service-levels`: Comma-separated list of service levels to test
- `--lead-time-factors`: Comma-separated list of lead time factors to test
- `--demand-factors`: Comma-separated list of demand factors for sensitivity analysis
- `--inflow-levels`: Comma-separated list of inflow levels for sensitivity analysis
- `--sensitivity-method`: Optimization method to use for sensitivity analysis

### From Python Code

You can also use the sensitivity analysis module directly in your Python code:

```python
from meio.io.json_loader import NetworkJsonLoader
from meio.analysis.sensitivity import run_sensitivity_analysis

# Load network
network = NetworkJsonLoader.load("path/to/network.json")

# Define parameters to test
parameter_ranges = {
    'service_level': [0.90, 0.95, 0.98],
    'lead_time_factor': [0.8, 1.0, 1.2],
    'demand_factor': [0.8, 1.0, 1.2]
}

# Run sensitivity analysis
result = run_sensitivity_analysis(
    network,
    parameter_ranges,
    method='improved_heuristic',
    visualize=True
)

# Access results
print(f"Analysis completed. Results saved to: {result['output_dir']}")
print(f"Report: {result['report_path']}")
```

## Output

Sensitivity analysis generates the following outputs:

1. **CSV Files**:
   - `sensitivity_results.csv`: Overall results
   - `sensitivity_[parameter].csv`: Parameter-specific results
   - `elasticity_metrics.csv`: Detailed elasticity calculations

2. **Visualizations**:
   - Parameter sensitivity charts
   - Comparative sensitivity analysis
   - Sensitivity ranking chart

3. **Report**:
   - `sensitivity_report.txt`: Detailed analysis with recommendations

## Interpreting Results

The key metric to focus on is the **elasticity** value, which measures how much the total cost changes relative to changes in the parameter.

- **High Elasticity (>1.0)**: Parameters with high elasticity have a significant impact on cost. Small changes in these parameters lead to large changes in cost.
- **Medium Elasticity (0.5-1.0)**: Parameters with medium elasticity have a moderate impact on cost.
- **Low Elasticity (<0.5)**: Parameters with low elasticity have minimal impact on cost.

Focus your data collection and monitoring efforts on parameters with high elasticity, as these have the greatest impact on your inventory strategy.

## Examples

### Example 1: Testing Service Levels

```bash
python -m meio.analyze_sensitivity --json network.json --service-levels 0.90,0.95,0.98,0.99
```

### Example 2: Full Sensitivity Analysis

```bash
python -m meio.analyze_sensitivity --json network.json \
    --service-levels 0.90,0.95,0.98 \
    --lead-time-factors 0.8,1.0,1.2 \
    --demand-factors 0.8,1.0,1.2 \
    --inflows 0.8,1.0,1.2
``` 