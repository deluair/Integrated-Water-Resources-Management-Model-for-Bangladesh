# Integrated Water Resources Management Model for Bangladesh

A comprehensive Python-based simulation platform designed to model Bangladesh's complex water resource challenges across coastal salinity intrusion, groundwater depletion, and freshwater scarcity while optimizing water allocation, pricing, and infrastructure systems.

## Overview

This simulator addresses critical water management challenges in Bangladesh:
- **Coastal Salinity**: Salinity levels reaching 40.0 ppt (equivalent to seawater) in coastal regions
- **Groundwater Depletion**: Approximately 1 meter annual depletion since 2000 in northwestern Bangladesh
- **Urban Water Crisis**: Water levels dropped 60-75 meters below ground surface in Dhaka
- **Agricultural Impact**: 80% of agricultural land depends on groundwater irrigation

## Key Features

### Core Simulation Modules
1. **Regional Groundwater Management Engine**
2. **Coastal Salinity Intrusion Module**
3. **Surface Water Allocation & Management System**
4. **Agricultural Water Demand & Productivity Module**
5. **Urban Water Supply & Infrastructure Module**
6. **Water Economics & Pricing Module**
7. **Integrated Policy Simulation Engine**
8. **Synthetic Data Generator**: For creating realistic datasets for various water parameters when real data is scarce.

### Bangladesh-Specific Scenarios
- Extreme salinity events from cyclone-induced saltwater intrusion
- Barind Tract groundwater crisis simulation
- Farakka impact on southwestern Bangladesh water flows
- Climate change adaptation modeling

## Installation

```bash
# Clone the repository
git clone https://github.com/deluair/Integrated-Water-Resources-Management-Model-for-Bangladesh.git
cd Integrated-Water-Resources-Management-Model-for-Bangladesh

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package (if setup.py is configured for editable install)
# Or ensure the package is in your PYTHONPATH
pip install -e .
```

## Quick Start (Using the Python API)

```python
from bangladesh_water_management import WaterResourcesSimulator
from bangladesh_water_management.config import load_config

# Load configuration (or use default if path not specified)
# config = load_config('path/to/your/bangladesh_config.yaml') 
config = load_config() # Uses default config if no argument is passed

# Initialize simulator
simulator = WaterResourcesSimulator(config)

# Example: Run groundwater depletion scenario
results_gw = simulator.run_groundwater_scenario(
    region='barind_tract',
    years=10,
    extraction_rate=1.2  # 20% increase
)
print(f"Groundwater Scenario Results: {results_gw}")

# Example: Analyze salinity intrusion
results_salinity = simulator.run_salinity_scenario(
    region='coastal_southwest',
    years=15,
    sea_level_rise=0.3,  # 30cm rise
    cyclone_frequency=1.5  # 50% increase
)
print(f"Salinity Scenario Results: {results_salinity}")

# Example: Run an integrated scenario
integrated_regions = ['dhaka_metro', 'coastal_southwest']
results_integrated = simulator.run_integrated_scenario(
    regions=integrated_regions,
    years=5
)
print(f"Integrated Scenario Results for {integrated_regions}: {results_integrated}")

# Generate policy recommendations (example based on one of the results)
# recommendations = simulator.generate_policy_recommendations(results_gw)
# print(f"Policy Recommendations: {recommendations}")
```

## Command Line Interface (CLI) Usage

The model can also be run using the Command Line Interface (`cli.py`).

```bash
python -m bangladesh_water_management.cli --help
```

This will display all available options. Key arguments include:

- `--config <path_to_yaml>`: Specify a custom configuration file. If not provided, the default configuration (`bangladesh_water_management/config/bangladesh_config.yaml`) is attempted, and if not found, an internal default is used.
- `--scenario <scenario_type>`: Choose the scenario to run.
  - `groundwater`: Runs the groundwater depletion scenario.
  - `salinity`: Runs the coastal salinity intrusion scenario.
  - `integrated`: Runs a comprehensive integrated scenario.
  - `none` (default): Initializes the simulator but does not run a specific scenario. Useful for checking setup.
- `--region <region_name>`: Specify the target region for the scenario (e.g., `dhaka_metro`, `barind_tract`). Default is `dhaka_metro`.
- `--years <num_years>`: Set the simulation period in years. Default is `5`.

**Examples:**

Run the default groundwater scenario for Dhaka Metro for 10 years:
```bash
python -m bangladesh_water_management.cli --scenario groundwater --region dhaka_metro --years 10
```

Run an integrated scenario for specific regions using a custom config:
```bash
python -m bangladesh_water_management.cli --scenario integrated --config path/to/my_config.yaml --region coastal_southwest --years 20 
# Note: For integrated scenario, multiple regions are typically defined within the config or via more complex CLI arguments if implemented. 
# The CLI currently uses --region for a primary region, and the integrated scenario might use a default set or this primary region.
```

Initialize the simulator without running a scenario (useful for testing configuration):
```bash
python -m bangladesh_water_management.cli --scenario none
```


## Project Structure

The project is organized as follows:

```
Integrated-Water-Resources-Management-Model-for-Bangladesh/
├── bangladesh_water_management/    # Main package
│   ├── __init__.py
│   ├── cli.py                      # Command Line Interface entry point
│   ├── config.py                   # Configuration loading and defaults
│   ├── simulator.py                # Main WaterResourcesSimulator class
│   │
│   ├── config/                     # Default configuration files
│   │   └── bangladesh_config.yaml
│   │
│   ├── data/                       # Data handling modules
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   ├── synthetic_data.py
│   │   └── data_validator.py       # (Assumed, based on typical structure)
│   │
│   ├── modules/                    # Core simulation and analysis modules
│   │   ├── __init__.py
│   │   ├── agriculture.py
│   │   ├── economic.py
│   │   ├── groundwater.py
│   │   ├── policy.py
│   │   ├── salinity.py
│   │   ├── surface_water.py
│   │   └── urban.py
│   │
│   ├── utils/                      # Utility functions and helpers
│   │   ├── __init__.py
│   │   ├── logging_config.py
│   │   ├── validation.py
│   │   └── export.py               # (Assumed, based on simulator methods)
│   │
│   └── visualization/              # Visualization tools (e.g., dashboards)
│       ├── __init__.py
│       └── dashboard.py
│
├── tests/                          # Test suite for the package
│   └── ...
│
├── data/                           # Placeholder for actual input datasets (not in package)
│   └── ... 
│
├── outputs/                        # Default directory for simulation results
│   └── ...
│
├── venv/                           # Virtual environment (if created as per instructions)
├── .gitignore
├── LICENSE                         # Project License (e.g., MIT)
├── README.md                       # This file
└── requirements.txt                # Python dependencies
```

## Data Sources

The model can integrate data from various sources, including:
- **Bangladesh Water Development Board (BWDB)**: Hydrological monitoring data
- **Department of Public Health Engineering (DPHE)**: Water quality and treatment data
- **Department of Agricultural Extension (DAE)**: Irrigation and crop data
- **NASA Satellite Data**: For parameters like groundwater, precipitation, etc.
- **World Bank Datasets**: Economic and infrastructure data
- **Synthetic Data**: Generated by the model for testing and baseline scenarios.

## Key Outputs

### Real-Time Monitoring & Simulation Outputs
- Groundwater status (levels, depletion rates) across regions
- Surface water availability (flow, storage)
- Salinity tracking and intrusion forecasts
- Water quality assessments (various parameters)
- Agricultural water demand and yield projections
- Urban water supply-demand balance

### Economic Analysis
- Agricultural productivity impacts under different water scenarios
- Cost-benefit analysis of infrastructure investments
- Economic impacts of water-related health issues
- Regional development implications of water policies

### Policy Support
- Optimization of water allocation strategies
- Analysis of water pricing policies
- Planning for climate-resilient infrastructure
- Development of emergency response protocols for floods/droughts

## Contributing

We welcome contributions from water resource professionals, researchers, and developers. Please see our `CONTRIBUTING.md` (to be created) for details on how to contribute, coding standards, and the development process.

## License

This project is licensed under the MIT License - see the `LICENSE` file (to be created/updated) for details.

## Acknowledgments

- Bangladesh Water Development Board
- Ministry of Water Resources, Bangladesh
- International Water Management Institute
- University of Tennessee Research Team (and other collaborators)

## Contact

For questions and support regarding this model:
- Primary Contact: [User/Organization Email or GitHub Profile]
- Documentation: [Link to ReadTheDocs or Wiki if available]
- Issues: [https://github.com/deluair/Integrated-Water-Resources-Management-Model-for-Bangladesh/issues](https://github.com/deluair/Integrated-Water-Resources-Management-Model-for-Bangladesh/issues)