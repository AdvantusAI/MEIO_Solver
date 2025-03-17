# Parameter Evolution AI Agent

The MEIO Solver now includes a powerful AI agent that can learn from historical optimization runs and suggest improved parameters for future optimizations. This document explains how to use this feature effectively.

## Overview

The Parameter Evolution Agent is an artificial intelligence module that:

1. **Learns from historical runs**: Analyzes past branch selection results to understand what parameter combinations led to the best outcomes.

2. **Identifies patterns**: Recognizes which parameters work best for specific types of networks and scenarios.

3. **Suggests optimized parameters**: Provides parameter suggestions tailored to your specific network structure.

4. **Continuously improves**: Gets smarter over time as more optimization data is collected.

## Key Benefits

- **Better starting points**: Start optimizations with parameters that have historically worked well for similar networks
- **Faster convergence**: Reduce the number of branches needed to find optimal solutions
- **Knowledge retention**: Preserve optimization knowledge that would otherwise be lost
- **Trend tracking**: Monitor how optimization performance changes over time

## Using the AI Agent

### Command-Line Integration

The Parameter Evolution Agent is integrated into the main MEIO system and can be enabled with command-line arguments:

```bash
python -m meio.main --json your_network.json --branch-selection --use-ai-agent
```

#### Additional AI Agent Options

```bash
# Import historical data for agent learning
python -m meio.main --json your_network.json --branch-selection --use-ai-agent --import-history path/to/history_directory

# Show AI agent performance trends
python -m meio.main --json your_network.json --branch-selection --use-ai-agent --show-ai-trends
```

### Test Script

For testing the branch selection with AI agent:

```bash
python test_branch_selection.py your_network.json --use-ai-agent
```

#### Test Script Options

```bash
python test_branch_selection.py your_network.json --use-ai-agent --import-history path/to/history_directory --num-branches 5 --method improved_heuristic
```

### Direct Agent Usage

For more advanced users, a dedicated demo script showcases direct interaction with the Parameter Evolution Agent:

```bash
python demo_ai_agent.py your_network.json --import-dir path/to/history_directory --visualize
```

## How it Works

### Learning Process

1. The agent stores results from successful branch selection runs
2. It extracts features from each network to identify similar network structures
3. It analyzes which parameters produced the best results for each network type
4. It maintains a database of this knowledge for future reference

### Parameter Suggestion

When suggesting parameters, the agent:

1. Analyzes the current network structure
2. Finds similar networks from its historical database
3. Identifies which parameters worked best for those similar networks
4. Calculates a weighted average of those parameters, favoring more similar networks
5. Returns these suggestions with a confidence level

### Confidence Levels

The agent provides confidence levels for its suggestions:

- **High**: Based on 5+ similar networks and 20+ total historical runs
- **Medium**: Based on 3+ similar networks and 10+ total historical runs
- **Low**: Limited historical data or no similar networks found

## Data Storage

The agent stores historical data in:

- `results/parameter_evolution.json`: Main historical database
- `results/branch_selection/ai_suggested_parameters.csv`: Suggested parameters for current run

## Visualizations

When the `--visualize` option is used with the demo script, the agent generates:

- **Parameter trends**: How parameter values have evolved over time
- **Metrics trends**: How performance metrics have changed over time
- **Parameter correlations**: Relationship between parameter values and overall performance

## Advanced Integration

For programmatic usage in custom code:

```python
from meio.optimization.branch_selection import ParameterEvolutionAgent

# Initialize the agent
agent = ParameterEvolutionAgent()

# Suggest parameters for a network
suggestions = agent.suggest_parameters(network)

# Track performance trends
trends = agent.get_performance_trend()

# Import historical data
with open('path/to/results.json', 'r') as f:
    historical_data = json.load(f)
    agent.learn_optimal_parameters(historical_data)
```

## Best Practices

1. **Start with a baseline**: Run a few optimizations without the AI agent to build initial historical data

2. **Import relevant history**: Use the `--import-history` option to give the agent relevant historical data

3. **Regularly export results**: Save optimization results to import into future runs

4. **Check confidence levels**: Be more cautious with "low" confidence suggestions

5. **Monitor trends**: Use the trend reports to ensure optimization is improving over time

## Troubleshooting

If you encounter issues with the Parameter Evolution Agent:

- **Insufficient data warning**: Agent needs more historical data to make confident suggestions. Run more optimizations or import additional historical data.

- **No similar networks found**: Your current network structure differs significantly from historical data. Consider running more diverse optimizations.

- **Low confidence suggestions**: Agent has limited relevant data for your network type. Consider starting with default parameters and gradually incorporating agent suggestions.

## Technical Details

The Parameter Evolution Agent uses a similarity-based learning approach:

1. **Network feature extraction**: Extracts key features (node count, product count, lead times, etc.)
2. **Similarity calculation**: Computes similarity scores between networks
3. **Weighted parameter averaging**: Weights historical parameters by similarity and performance
4. **Confidence estimation**: Estimates confidence based on data quantity and similarity

The agent is designed to be lightweight and doesn't require external machine learning libraries, making it easy to use in any environment where the MEIO system runs. 