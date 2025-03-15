#!/bin/bash
# Example commands for running sensitivity analysis in the MEIO system

# Set path to your network JSON file
NETWORK_JSON="meio/config/supply_chain_network.json"

# 1. Basic sensitivity analysis with default parameters
echo "Running basic sensitivity analysis..."
python -m meio.analyze_sensitivity --json ${NETWORK_JSON}

# 2. Testing service level sensitivity
echo "Running service level sensitivity analysis..."
python -m meio.analyze_sensitivity --json ${NETWORK_JSON} \
    --service-levels 0.90,0.92,0.95,0.97,0.99

# 3. Testing lead time sensitivity
echo "Running lead time sensitivity analysis..."
python -m meio.analyze_sensitivity --json ${NETWORK_JSON} \
    --lead-time-factors 0.6,0.8,1.0,1.2,1.4

# 4. Testing demand sensitivity
echo "Running demand sensitivity analysis..."
python -m meio.analyze_sensitivity --json ${NETWORK_JSON} \
    --demand-factors 0.6,0.8,1.0,1.2,1.4

# 5. Testing inflow sensitivity
echo "Running inflow sensitivity analysis..."
python -m meio.analyze_sensitivity --json ${NETWORK_JSON} \
    --inflows 0.6,0.8,1.0,1.2,1.4

# 6. Testing multiple parameters (comprehensive analysis)
echo "Running comprehensive sensitivity analysis..."
python -m meio.analyze_sensitivity --json ${NETWORK_JSON} \
    --service-levels 0.90,0.95,0.99 \
    --lead-time-factors 0.8,1.0,1.2 \
    --demand-factors 0.8,1.0,1.2 \
    --inflows 0.8,1.0,1.2

# 7. Using a different optimization method
echo "Running sensitivity analysis with mathematical solver..."
python -m meio.analyze_sensitivity --json ${NETWORK_JSON} \
    --service-levels 0.90,0.95,0.99 \
    --method solver

# 8. Saving results to a custom directory
echo "Running sensitivity analysis with custom output directory..."
python -m meio.analyze_sensitivity --json ${NETWORK_JSON} \
    --service-levels 0.90,0.95,0.99 \
    --output-dir ./results/my_sensitivity_analysis

# 9. Running via main.py
echo "Running sensitivity analysis via main.py..."
python -m meio.main --json ${NETWORK_JSON} \
    --sensitivity \
    --service-levels 0.90,0.95,0.99

echo "All examples completed!" 