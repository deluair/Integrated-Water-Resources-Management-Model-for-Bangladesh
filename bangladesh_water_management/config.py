"""Configuration management for Bangladesh Water Resources Management Model.

This module handles loading and validation of configuration parameters
for all simulation modules.
"""

import yaml
import os
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from loguru import logger

# Placeholder for JSON schema for configuration validation
CONFIG_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "simulation_settings": {"type": "object"},
        "regional_settings": {"type": "object"},
        "sectoral_settings": {"type": "object"},
        # Add other top-level sections from get_default_config as needed by tests
        "simulation": {"type": "object", "properties": {"start_year": {"type": "integer"}}}, # Example based on test_get_default_config
        "regions": {"type": "object"},
        "groundwater": {"type": "object"},
        "salinity": {"type": "object"},
        "surface_water": {"type": "object"},
        "agriculture": {"type": "object"},
        "urban": {"type": "object"},
        "economics": {"type": "object"}, # Renamed from 'economics' to 'economic_config' in some places, ensure consistency
        "economic": {"type": "object"}, # Added to match EconomicAnalyzer init
        "climate": {"type": "object"},
        "data_settings": {"type": "object"}, # Added based on DataLoader usage
        "data": {"type": "object"},
        "output_settings": {"type": "object"}, # Added based on test_config.py
        "output": {"type": "object"}
    },
    "required": ["simulation", "regions", "groundwater", "salinity", "surface_water", "agriculture", "urban", "economics", "data", "output"]
    # "required": ["simulation_settings", "regional_settings"] # Example, adjust based on actual requirements
}

import jsonschema # Add this import

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file. If None, uses default config.
        
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        config_path = Path(__file__).parent / 'config' / 'bangladesh_config.yaml'
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Validate configuration
        validate_config(config)
        
        logger.info(f"Configuration loaded from {config_path}")
        return config
        
    except FileNotFoundError:
        logger.warning(f"Config file not found at {config_path}, using default configuration")
        return get_default_config()
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML config: {e}")
        raise


def get_default_config() -> Dict[str, Any]:
    """Get default configuration for Bangladesh water management model.
    
    Returns:
        Default configuration dictionary
    """
    return {
        'simulation': {
            'default_years': 10,
            'time_step': 'monthly',
            'start_year': 2024,
            'random_seed': 42
        },
        
        'regions': {
            'all': [
                'dhaka_metro', 'chittagong', 'sylhet', 'rajshahi', 'khulna',
                'barisal', 'rangpur', 'mymensingh', 'coastal_southwest',
                'coastal_southeast', 'barind_tract', 'haor_region'
            ],
            'coastal': [
                'coastal_southwest', 'coastal_southeast', 'chittagong',
                'barisal', 'khulna'
            ],
            'groundwater_dependent': [
                'dhaka_metro', 'rajshahi', 'barind_tract', 'rangpur'
            ],
            'urban': [
                'dhaka_metro', 'chittagong', 'sylhet', 'rajshahi', 'khulna'
            ],
            'agricultural': [
                'barind_tract', 'rangpur', 'rajshahi', 'mymensingh', 'haor_region'
            ]
        },
        
        'groundwater': {
            'aquifer_layers': 3,
            'depletion_rates': {
                'dhaka_metro': 1.2,  # meters per year
                'barind_tract': 0.8,
                'rajshahi': 0.6,
                'rangpur': 0.7
            },
            'recharge_rates': {
                'monsoon_multiplier': 3.5,
                'dry_season_multiplier': 0.2
            },
            'critical_depth': 60,  # meters below surface
            'sustainability_threshold': 0.8
        },
        
        'salinity': {
            'intrusion_rates': {
                'coastal_southwest': 2.5,  # km per year
                'coastal_southeast': 1.8,
                'chittagong': 1.2
            },
            'salinity_thresholds': {
                'freshwater': 0.5,  # ppt
                'brackish': 5.0,
                'saline': 15.0,
                'seawater': 35.0
            },
            'cyclone_impact_multiplier': 2.5,
            'sea_level_rise_factor': 1.5
        },
        
        'surface_water': {
            'major_rivers': [
                'ganges', 'brahmaputra', 'meghna', 'padma', 'jamuna'
            ],
            'seasonal_variation': {
                'monsoon_flow_multiplier': 4.0,
                'dry_season_flow_multiplier': 0.3
            },
            'farakka_impact': {
                'dry_season_reduction': 0.4,  # 40% reduction
                'affected_regions': ['coastal_southwest', 'khulna']
            }
        },
        
        'agriculture': {
            'irrigation_efficiency': {
                'flood': 0.4,
                'sprinkler': 0.7,
                'drip': 0.9,
                'awd': 0.6  # Alternate Wetting and Drying
            },
            'crop_water_requirements': {
                'rice_boro': 1200,  # mm per season
                'rice_aman': 800,
                'wheat': 400,
                'maize': 500,
                'jute': 600
            },
            'salinity_tolerance': {
                'rice': 3.0,  # dS/m
                'wheat': 6.0,
                'barley': 8.0,
                'cotton': 7.7
            }
        },
        
        'urban': {
            'per_capita_demand': {
                'dhaka_metro': 150,  # liters per day
                'chittagong': 120,
                'other_cities': 100
            },
            'population_growth_rates': {
                'dhaka_metro': 0.035,  # 3.5% per year
                'chittagong': 0.025,
                'other_cities': 0.015
            },
            'infrastructure_capacity': {
                'treatment_plants': 0.8,  # utilization factor
                'distribution_efficiency': 0.7,
                'storage_days': 3
            }
        },
        
        'economics': {
            'water_prices': {
                'irrigation': 0.02,  # USD per cubic meter
                'domestic': 0.15,
                'industrial': 0.25
            },
            'health_costs': {
                'arsenic_poisoning': 500,  # USD per case per year
                'waterborne_disease': 50,
                'salinity_health_impact': 200
            },
            'agricultural_productivity': {
                'water_stress_factor': 0.3,  # yield reduction per unit stress
                'salinity_impact_factor': 0.4
            },
            'infrastructure_costs': {
                'desalination_plant': 1000,  # USD per cubic meter capacity
                'treatment_plant': 500,
                'distribution_network': 200
            }
        },
        
        'climate': {
            'scenarios': {
                'conservative': {
                    'temperature_increase': 1.5,  # Celsius by 2050
                    'precipitation_change': 0.05,  # 5% increase
                    'sea_level_rise': 0.2  # meters by 2050
                },
                'moderate': {
                    'temperature_increase': 2.5,
                    'precipitation_change': 0.10,
                    'sea_level_rise': 0.4
                },
                'severe': {
                    'temperature_increase': 4.0,
                    'precipitation_change': 0.15,
                    'sea_level_rise': 0.8
                }
            }
        },
        
        'data': {
            'synthetic_data_size': 10000,
            'validation_split': 0.2,
            'noise_level': 0.05,
            'missing_data_rate': 0.02
        },
        
        'output': {
            'default_format': 'csv',
            'precision': 3,
            'include_metadata': True,
            'dashboard_port': 8050
        }
    }


def validate_config(config: Dict[str, Any]) -> Tuple[bool, Optional[List[jsonschema.ValidationError]]]:
    """Validate configuration parameters using JSONSchema.

    Args:
        config: Configuration dictionary to validate

    Returns:
        A tuple: (is_valid, list_of_errors_or_None)
    """
    try:
        jsonschema.validate(instance=config, schema=CONFIG_SCHEMA)
        logger.info("Configuration validation passed.")
        return True, None
    except jsonschema.ValidationError as e:
        # Log the primary error, but return all errors if multiple occur (though validate stops on first by default)
        logger.warning(f"Configuration validation failed: {e.message}")
        # To get all errors, one would use a validatingDraft7Validator and iterate over iter_errors
        # For simplicity with the current test structure, we'll return a list containing the single error.
        # If tests expect more detailed error reporting, this part needs refinement.
        # validator = jsonschema.Draft7Validator(CONFIG_SCHEMA)
        # errors = sorted(validator.iter_errors(config), key=lambda e: e.path)
        # return False, errors
        return False, [e] # Return the single error in a list to match test expectations
    except Exception as e:
        logger.error(f"An unexpected error occurred during configuration validation: {e}")
        # Create a generic ValidationError-like object or re-raise if appropriate
        # For now, treat as a validation failure with a custom message
        class GenericValidationError:
            def __init__(self, message):
                self.message = message
                self.path = [] # Placeholder
        return False, [GenericValidationError(str(e))]


def get_nested_value(config: Dict[str, Any], path: str) -> Any:
    """Get nested value from configuration using dot notation.
    
    Args:
        config: Configuration dictionary
        path: Dot-separated path (e.g., 'simulation.default_years')
        
    Returns:
        Value at the specified path, or None if not found
    """
    keys = path.split('.')
    value = config
    
    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return None


def update_config(config: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Update configuration with new values.
    
    Args:
        config: Original configuration
        updates: Dictionary of updates to apply
        
    Returns:
        Updated configuration
    """
    import copy
    updated_config = copy.deepcopy(config)
    
    def deep_update(base_dict, update_dict):
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    deep_update(updated_config, updates)
    validate_config(updated_config)
    
    return updated_config


def save_config(config: Dict[str, Any], output_path: str) -> bool:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary to save
        output_path: Path where to save the configuration
    Returns:
        True if successful, False otherwise.
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        logger.info(f"Configuration saved to {output_path}")
        return True
    except IOError as e:
        logger.error(f"Failed to save config to {output_path}: {e}")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred while saving config to {output_path}: {e}")
        return False
    """Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary to save
        output_path: Path where to save the configuration
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    logger.info(f"Configuration saved to {output_path}")