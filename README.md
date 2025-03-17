# Example of parallel processing for node calculations
safety_stock_recommendations = ParallelProcessor.process_nodes(
    network, 
    DilopSafetyStock._process_node,
    network=network,
    default_service_level=default_service_level
)

## Branch Selection Strategy

The MEIO Solver now includes a Branch Selection Strategy system that allows you to generate multiple inventory policy options, evaluate them under different scenarios, and select the most appropriate strategy using a formalized decision process.

### Key Features

- Generate multiple inventory policy branches with varying parameters
- Evaluate branches across multiple criteria (cost, service level, robustness)
- Test branch performance under various scenarios (baseline, high demand, supply disruption)
- Select the optimal branch based on configurable selection criteria
- Visualize branch comparison and performance metrics

### Usage

```bash
python -m meio.main --json network.json --branch-selection --num-branches 5 --selection-criteria balanced
```

### Command-line Options

- `--branch-selection` - Enable branch selection analysis
- `--num-branches N` - Number of branches to generate (default: 5)
- `--branch-criteria` - Comma-separated list of criteria for evaluation (default: cost,service_level,robustness)
- `--branch-weights` - Comma-separated list of weights for each criterion (default: 0.4,0.4,0.2)
- `--selection-criteria` - Strategy for selecting the best branch (balanced, cost_focused, service_focused, robust)
- `--branch-method` - Optimization method to use for branch generation (heuristic, improved_heuristic, solver)

### Output

The branch selection process produces detailed CSV files with results and visualizations including:
- Branch comparison charts
- Scenario performance analysis
- Selected branch details with implementation guidance

This system helps decision-makers understand trade-offs between competing objectives and select the most appropriate inventory policy for their specific business context.

## Parameter Evolution AI Agent

The MEIO Solver now includes an intelligent AI agent that learns from historical optimization runs and suggests improved parameters for future optimizations.

### Key Features

- **Continuous Learning**: The agent learns from past optimization runs to improve future recommendations
- **Network Feature Analysis**: Automatically extracts network structure features to find similar historical cases
- **Parameter Suggestion**: Recommends optimized parameters based on historical results from similar networks
- **Performance Tracking**: Monitors optimization performance trends over time

### Usage

```bash
# Use the AI agent with branch selection
python -m meio.main --json network.json --branch-selection --use-ai-agent

# Import historical data for agent learning
python -m meio.main --json network.json --branch-selection --use-ai-agent --import-history path/to/history_directory

# Show AI performance trends
python -m meio.main --json network.json --branch-selection --use-ai-agent --show-ai-trends
```

### Direct Agent Interaction

For advanced users, a dedicated demo script provides direct interaction with the Parameter Evolution Agent:

```bash
python demo_ai_agent.py your_network.json --import-dir path/to/history_directory --visualize
```

### Benefits

- **Better Starting Points**: Begin optimizations with parameters that have historically worked well for similar networks
- **Knowledge Retention**: Preserve optimization knowledge that would otherwise be lost
- **Continuous Improvement**: Optimization performance improves over time as the agent learns from more runs
- **Reduced Iterations**: Find optimal solutions faster by starting with better parameter estimates

See the [AI Agent Documentation](docs/ai_agent.md) for detailed usage instructions and implementation details.
