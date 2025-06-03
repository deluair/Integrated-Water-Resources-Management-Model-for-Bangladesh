"""Data Management Package for Bangladesh Water Management.

This package handles data collection, processing, validation, storage,
and integration for the water resources management model.

Modules:
    - data_loader: Data loading and preprocessing utilities
    - data_validator: Data quality validation and cleaning
    - synthetic_data: Synthetic data generation for modeling
    - database: Database management and operations
    - api_client: External data API integration
"""

from .data_loader import DataLoader
from .synthetic_data import SyntheticDataGenerator
from .data_validator import DataValidator

__version__ = "1.0.0"
__author__ = "Bangladesh Water Management Team"

# Package-level configuration
DATA_CONFIG = {
    'default_data_path': 'data/',
    'supported_formats': ['csv', 'json', 'xlsx', 'shp', 'geojson'],
    'validation_rules': {
        'required_fields': ['timestamp', 'location', 'value'],
        'data_types': {'timestamp': 'datetime', 'value': 'numeric'},
        'quality_thresholds': {'completeness': 0.8, 'accuracy': 0.9}
    },
    'synthetic_data_config': {
        'random_seed': 42,
        'default_years': 10,
        'temporal_resolution': 'daily'
    }
}

# Export main classes and functions
__all__ = [
    'DataLoader',
    'SyntheticDataGenerator', 
    'DataValidator',
    'DATA_CONFIG'
]