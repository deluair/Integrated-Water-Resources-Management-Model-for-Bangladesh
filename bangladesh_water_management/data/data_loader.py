"""Data Loading and Preprocessing Module for Bangladesh Water Management.

This module handles loading data from various sources including files, databases,
and APIs, with preprocessing and standardization capabilities.
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import json
import yaml
from datetime import datetime, timedelta
from loguru import logger
import warnings
warnings.filterwarnings('ignore')


class DataLoader:
    """Handles loading and preprocessing of water management data.
    
    This class provides methods to load data from various sources,
    standardize formats, and prepare data for analysis and modeling.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize data loader.
        
        Args:
            config: Configuration dictionary containing data parameters
        """
        self.config = config
        self.data_config = config.get('data', {})
        self.base_path = Path(self.data_config.get('base_path', 'data/'))
        
        # Initialize data schemas
        self.data_schemas = self._initialize_data_schemas()
        
        # Initialize data sources
        self.data_sources = self._initialize_data_sources()
        
        # Cache for loaded data
        self._data_cache = {}
        
        logger.info("Data Loader initialized")
    
    def _initialize_data_schemas(self) -> Dict[str, Dict]:
        """Initialize data schemas for different data types."""
        return {
            'meteorological': {
                'required_columns': ['date', 'location', 'temperature', 'precipitation', 'humidity'],
                'optional_columns': ['wind_speed', 'solar_radiation', 'evapotranspiration'],
                'data_types': {
                    'date': 'datetime64[ns]',
                    'location': 'string',
                    'temperature': 'float64',
                    'precipitation': 'float64',
                    'humidity': 'float64'
                },
                'units': {
                    'temperature': 'celsius',
                    'precipitation': 'mm',
                    'humidity': 'percent'
                }
            },
            'hydrological': {
                'required_columns': ['date', 'station_id', 'water_level', 'discharge'],
                'optional_columns': ['velocity', 'sediment_load', 'water_quality'],
                'data_types': {
                    'date': 'datetime64[ns]',
                    'station_id': 'string',
                    'water_level': 'float64',
                    'discharge': 'float64'
                },
                'units': {
                    'water_level': 'meters',
                    'discharge': 'cubic_meters_per_second'
                }
            },
            'groundwater': {
                'required_columns': ['date', 'well_id', 'water_table_depth', 'extraction_rate'],
                'optional_columns': ['water_quality', 'aquifer_type', 'well_depth'],
                'data_types': {
                    'date': 'datetime64[ns]',
                    'well_id': 'string',
                    'water_table_depth': 'float64',
                    'extraction_rate': 'float64'
                },
                'units': {
                    'water_table_depth': 'meters_below_ground',
                    'extraction_rate': 'cubic_meters_per_day'
                }
            },
            'water_quality': {
                'required_columns': ['date', 'location', 'ph', 'dissolved_oxygen', 'turbidity'],
                'optional_columns': ['salinity', 'nitrates', 'phosphates', 'heavy_metals'],
                'data_types': {
                    'date': 'datetime64[ns]',
                    'location': 'string',
                    'ph': 'float64',
                    'dissolved_oxygen': 'float64',
                    'turbidity': 'float64'
                },
                'units': {
                    'ph': 'ph_scale',
                    'dissolved_oxygen': 'mg_per_liter',
                    'turbidity': 'ntu'
                }
            },
            'agricultural': {
                'required_columns': ['date', 'district', 'crop_type', 'irrigation_demand', 'yield'],
                'optional_columns': ['cropped_area', 'water_stress_index', 'soil_moisture'],
                'data_types': {
                    'date': 'datetime64[ns]',
                    'district': 'string',
                    'crop_type': 'string',
                    'irrigation_demand': 'float64',
                    'yield': 'float64'
                },
                'units': {
                    'irrigation_demand': 'mm',
                    'yield': 'tons_per_hectare'
                }
            },
            'urban_water': {
                'required_columns': ['date', 'city', 'water_demand', 'water_supply', 'population'],
                'optional_columns': ['water_losses', 'treatment_capacity', 'service_coverage'],
                'data_types': {
                    'date': 'datetime64[ns]',
                    'city': 'string',
                    'water_demand': 'float64',
                    'water_supply': 'float64',
                    'population': 'int64'
                },
                'units': {
                    'water_demand': 'million_liters_per_day',
                    'water_supply': 'million_liters_per_day'
                }
            },
            'economic': {
                'required_columns': ['date', 'sector', 'water_cost', 'economic_value'],
                'optional_columns': ['investment', 'employment', 'gdp_contribution'],
                'data_types': {
                    'date': 'datetime64[ns]',
                    'sector': 'string',
                    'water_cost': 'float64',
                    'economic_value': 'float64'
                },
                'units': {
                    'water_cost': 'usd_per_cubic_meter',
                    'economic_value': 'million_usd'
                }
            },
            'spatial': {
                'required_columns': ['geometry', 'name', 'type'],
                'optional_columns': ['area', 'population', 'elevation'],
                'geometry_types': ['Point', 'LineString', 'Polygon'],
                'crs': 'EPSG:4326'  # WGS84
            }
        }
    
    def _initialize_data_sources(self) -> Dict[str, Dict]:
        """Initialize available data sources and their configurations."""
        return {
            'bangladesh_meteorological_department': {
                'description': 'Weather and climate data',
                'data_types': ['meteorological'],
                'update_frequency': 'daily',
                'spatial_coverage': 'national',
                'temporal_coverage': '1950-present',
                'quality_rating': 'good',
                'access_method': 'api',
                'api_endpoint': 'http://bmd.gov.bd/api/data',
                'authentication_required': True
            },
            'bangladesh_water_development_board': {
                'description': 'Hydrological and water resources data',
                'data_types': ['hydrological', 'groundwater'],
                'update_frequency': 'daily',
                'spatial_coverage': 'national',
                'temporal_coverage': '1960-present',
                'quality_rating': 'excellent',
                'access_method': 'database',
                'database_config': {
                    'host': 'bwdb.gov.bd',
                    'database': 'water_resources',
                    'authentication_required': True
                }
            },
            'department_of_environment': {
                'description': 'Water quality and environmental data',
                'data_types': ['water_quality'],
                'update_frequency': 'monthly',
                'spatial_coverage': 'major_rivers_cities',
                'temporal_coverage': '1990-present',
                'quality_rating': 'good',
                'access_method': 'file',
                'file_formats': ['csv', 'xlsx']
            },
            'department_of_agricultural_extension': {
                'description': 'Agricultural water use and crop data',
                'data_types': ['agricultural'],
                'update_frequency': 'seasonal',
                'spatial_coverage': 'district_level',
                'temporal_coverage': '1980-present',
                'quality_rating': 'moderate',
                'access_method': 'file',
                'file_formats': ['csv', 'xlsx']
            },
            'water_and_sewerage_authorities': {
                'description': 'Urban water supply and demand data',
                'data_types': ['urban_water'],
                'update_frequency': 'monthly',
                'spatial_coverage': 'urban_areas',
                'temporal_coverage': '2000-present',
                'quality_rating': 'good',
                'access_method': 'multiple',
                'data_availability': 'varies_by_city'
            },
            'bangladesh_bureau_of_statistics': {
                'description': 'Socioeconomic and demographic data',
                'data_types': ['economic', 'demographic'],
                'update_frequency': 'annual',
                'spatial_coverage': 'national',
                'temporal_coverage': '1970-present',
                'quality_rating': 'excellent',
                'access_method': 'file',
                'file_formats': ['csv', 'xlsx']
            },
            'satellite_data': {
                'description': 'Remote sensing data for water resources',
                'data_types': ['spatial', 'meteorological'],
                'update_frequency': 'daily',
                'spatial_coverage': 'national',
                'temporal_coverage': '2000-present',
                'quality_rating': 'good',
                'sources': ['MODIS', 'Landsat', 'Sentinel'],
                'access_method': 'api'
            }
        }
    
    def load_data(self,
                  data_type: str,
                  source: Optional[str] = None,
                  file_path: Optional[Union[str, Path]] = None,
                  date_range: Optional[Tuple[str, str]] = None,
                  spatial_filter: Optional[Dict] = None,
                  **kwargs) -> pd.DataFrame:
        """Load data from specified source.
        
        Args:
            data_type: Type of data to load (e.g., 'meteorological', 'hydrological')
            source: Data source name (optional)
            file_path: Path to data file (for file-based sources)
            date_range: Tuple of start and end dates (YYYY-MM-DD format)
            spatial_filter: Spatial filtering criteria
            **kwargs: Additional parameters for data loading
            
        Returns:
            Loaded and preprocessed DataFrame
        """
        # Check cache first
        cache_key = self._generate_cache_key(data_type, source, file_path, date_range)
        if cache_key in self._data_cache:
            logger.info(f"Loading {data_type} data from cache")
            return self._data_cache[cache_key].copy()
        
        # Validate data type
        if data_type not in self.data_schemas:
            raise ValueError(f"Unsupported data type: {data_type}")
        
        # Load data based on source type
        if file_path:
            data = self._load_from_file(file_path, data_type, **kwargs)
        elif source:
            data = self._load_from_source(source, data_type, date_range, spatial_filter, **kwargs)
        else:
            # Try to find default source for data type
            default_source = self._find_default_source(data_type)
            if default_source:
                data = self._load_from_source(default_source, data_type, date_range, spatial_filter, **kwargs)
            else:
                raise ValueError(f"No source specified and no default source found for {data_type}")
        
        # Validate and standardize data
        data = self._validate_and_standardize(data, data_type)
        
        # Apply filters
        if date_range:
            data = self._apply_date_filter(data, date_range)
        
        if spatial_filter:
            data = self._apply_spatial_filter(data, spatial_filter)
        
        # Cache the result
        self._data_cache[cache_key] = data.copy()
        
        logger.info(f"Loaded {len(data)} records of {data_type} data")
        return data
    
    def _generate_cache_key(self,
                           data_type: str,
                           source: Optional[str],
                           file_path: Optional[Union[str, Path]],
                           date_range: Optional[Tuple[str, str]]) -> str:
        """Generate cache key for data loading."""
        key_parts = [data_type]
        
        if source:
            key_parts.append(source)
        if file_path:
            key_parts.append(str(file_path))
        if date_range:
            key_parts.extend(date_range)
        
        return "_".join(key_parts)
    
    def _load_from_file(self,
                       file_path: Union[str, Path],
                       data_type: str,
                       **kwargs) -> pd.DataFrame:
        """Load data from file."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        # Determine file format and load accordingly
        file_extension = file_path.suffix.lower()
        
        if file_extension == '.csv':
            data = pd.read_csv(file_path, **kwargs)
        elif file_extension in ['.xlsx', '.xls']:
            data = pd.read_excel(file_path, **kwargs)
        elif file_extension == '.json':
            data = pd.read_json(file_path, **kwargs)
        elif file_extension == '.parquet':
            data = pd.read_parquet(file_path, **kwargs)
        elif file_extension in ['.shp', '.geojson']:
            # Spatial data
            data = gpd.read_file(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        logger.info(f"Loaded data from file: {file_path}")
        return data
    
    def _load_from_source(self,
                         source: str,
                         data_type: str,
                         date_range: Optional[Tuple[str, str]],
                         spatial_filter: Optional[Dict],
                         **kwargs) -> pd.DataFrame:
        """Load data from external source."""
        if source not in self.data_sources:
            raise ValueError(f"Unknown data source: {source}")
        
        source_config = self.data_sources[source]
        access_method = source_config['access_method']
        
        if access_method == 'api':
            return self._load_from_api(source, data_type, date_range, spatial_filter, **kwargs)
        elif access_method == 'database':
            return self._load_from_database(source, data_type, date_range, spatial_filter, **kwargs)
        elif access_method == 'file':
            return self._load_from_source_files(source, data_type, date_range, **kwargs)
        else:
            raise ValueError(f"Unsupported access method: {access_method}")
    
    def _load_from_api(self,
                      source: str,
                      data_type: str,
                      date_range: Optional[Tuple[str, str]],
                      spatial_filter: Optional[Dict],
                      **kwargs) -> pd.DataFrame:
        """Load data from API (placeholder implementation)."""
        logger.warning(f"API loading for {source} not implemented. Generating synthetic data.")
        
        # For demonstration, generate synthetic data
        from .synthetic_data import SyntheticDataGenerator
        generator = SyntheticDataGenerator(self.config)
        
        if data_type == 'meteorological':
            return generator.generate_meteorological_data()
        elif data_type == 'hydrological':
            return generator.generate_hydrological_data()
        elif data_type == 'groundwater':
            return generator.generate_groundwater_data()
        else:
            # Generate basic synthetic data
            return generator.generate_basic_timeseries(data_type)
    
    def _load_from_database(self,
                           source: str,
                           data_type: str,
                           date_range: Optional[Tuple[str, str]],
                           spatial_filter: Optional[Dict],
                           **kwargs) -> pd.DataFrame:
        """Load data from database (placeholder implementation)."""
        logger.warning(f"Database loading for {source} not implemented. Generating synthetic data.")
        
        # For demonstration, generate synthetic data
        from .synthetic_data import SyntheticDataGenerator
        generator = SyntheticDataGenerator(self.config)
        
        if data_type == 'hydrological':
            return generator.generate_hydrological_data()
        elif data_type == 'groundwater':
            return generator.generate_groundwater_data()
        else:
            return generator.generate_basic_timeseries(data_type)
    
    def _load_from_source_files(self,
                               source: str,
                               data_type: str,
                               date_range: Optional[Tuple[str, str]],
                               **kwargs) -> pd.DataFrame:
        """Load data from source-specific files."""
        # Look for files in source-specific directory
        source_dir = self.base_path / source.lower().replace(' ', '_')
        
        if not source_dir.exists():
            logger.warning(f"Source directory not found: {source_dir}. Generating synthetic data.")
            from .synthetic_data import SyntheticDataGenerator
            generator = SyntheticDataGenerator(self.config)
            return generator.generate_basic_timeseries(data_type)
        
        # Find relevant files
        pattern = f"*{data_type}*"
        files = list(source_dir.glob(pattern + ".csv")) + list(source_dir.glob(pattern + ".xlsx"))
        
        if not files:
            logger.warning(f"No {data_type} files found in {source_dir}. Generating synthetic data.")
            from .synthetic_data import SyntheticDataGenerator
            generator = SyntheticDataGenerator(self.config)
            return generator.generate_basic_timeseries(data_type)
        
        # Load and combine files
        dataframes = []
        for file_path in files:
            df = self._load_from_file(file_path, data_type, **kwargs)
            dataframes.append(df)
        
        # Combine all dataframes
        combined_data = pd.concat(dataframes, ignore_index=True)
        return combined_data
    
    def _find_default_source(self, data_type: str) -> Optional[str]:
        """Find default source for a data type."""
        for source_name, source_config in self.data_sources.items():
            if data_type in source_config.get('data_types', []):
                return source_name
        return None
    
    def _validate_and_standardize(self, data: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """Validate and standardize data format."""
        schema = self.data_schemas[data_type]
        
        # Check for required columns
        required_cols = schema['required_columns']
        missing_cols = [col for col in required_cols if col not in data.columns]
        
        if missing_cols:
            logger.warning(f"Missing required columns for {data_type}: {missing_cols}")
            # Try to map common column name variations
            data = self._map_column_names(data, data_type)
            
            # Check again after mapping
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Convert data types
        if 'data_types' in schema:
            for col, dtype in schema['data_types'].items():
                if col in data.columns:
                    try:
                        if dtype == 'datetime64[ns]':
                            data[col] = pd.to_datetime(data[col])
                        else:
                            data[col] = data[col].astype(dtype)
                    except Exception as e:
                        logger.warning(f"Could not convert {col} to {dtype}: {e}")
        
        # Remove duplicates
        initial_len = len(data)
        data = data.drop_duplicates()
        if len(data) < initial_len:
            logger.info(f"Removed {initial_len - len(data)} duplicate records")
        
        # Sort by date if date column exists
        date_cols = ['date', 'timestamp', 'datetime']
        for date_col in date_cols:
            if date_col in data.columns:
                data = data.sort_values(date_col)
                break
        
        return data
    
    def _map_column_names(self, data: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """Map common column name variations to standard names."""
        column_mappings = {
            'meteorological': {
                'temp': 'temperature',
                'precip': 'precipitation',
                'rainfall': 'precipitation',
                'humid': 'humidity',
                'rh': 'humidity',
                'wind': 'wind_speed'
            },
            'hydrological': {
                'level': 'water_level',
                'stage': 'water_level',
                'flow': 'discharge',
                'q': 'discharge',
                'station': 'station_id'
            },
            'groundwater': {
                'depth': 'water_table_depth',
                'gwl': 'water_table_depth',
                'extraction': 'extraction_rate',
                'pumping': 'extraction_rate',
                'well': 'well_id'
            },
            'water_quality': {
                'do': 'dissolved_oxygen',
                'oxygen': 'dissolved_oxygen',
                'turb': 'turbidity'
            }
        }
        
        if data_type in column_mappings:
            mappings = column_mappings[data_type]
            # Create reverse mapping for partial matches
            for old_name in data.columns:
                for partial, standard in mappings.items():
                    if partial.lower() in old_name.lower() and standard not in data.columns:
                        data = data.rename(columns={old_name: standard})
                        logger.info(f"Mapped column '{old_name}' to '{standard}'")
                        break
        
        return data
    
    def _apply_date_filter(self, data: pd.DataFrame, date_range: Tuple[str, str]) -> pd.DataFrame:
        """Apply date range filter to data."""
        start_date, end_date = date_range
        
        # Find date column
        date_cols = ['date', 'timestamp', 'datetime']
        date_col = None
        for col in date_cols:
            if col in data.columns:
                date_col = col
                break
        
        if not date_col:
            logger.warning("No date column found for date filtering")
            return data
        
        # Convert date strings to datetime
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # Apply filter
        mask = (data[date_col] >= start_dt) & (data[date_col] <= end_dt)
        filtered_data = data[mask]
        
        logger.info(f"Date filter applied: {len(filtered_data)}/{len(data)} records retained")
        return filtered_data
    
    def _apply_spatial_filter(self, data: pd.DataFrame, spatial_filter: Dict) -> pd.DataFrame:
        """Apply spatial filter to data."""
        # Simplified spatial filtering implementation
        if 'location' in spatial_filter and 'location' in data.columns:
            locations = spatial_filter['location']
            if isinstance(locations, str):
                locations = [locations]
            
            filtered_data = data[data['location'].isin(locations)]
            logger.info(f"Spatial filter applied: {len(filtered_data)}/{len(data)} records retained")
            return filtered_data
        
        if 'district' in spatial_filter and 'district' in data.columns:
            districts = spatial_filter['district']
            if isinstance(districts, str):
                districts = [districts]
            
            filtered_data = data[data['district'].isin(districts)]
            logger.info(f"District filter applied: {len(filtered_data)}/{len(data)} records retained")
            return filtered_data
        
        logger.warning("Spatial filter could not be applied - no matching columns found")
        return data
    
    def load_multiple_datasets(self,
                              dataset_configs: List[Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
        """Load multiple datasets with different configurations.
        
        Args:
            dataset_configs: List of dataset configuration dictionaries
            
        Returns:
            Dictionary of loaded datasets
        """
        datasets = {}
        
        for config in dataset_configs:
            dataset_name = config.get('name', f"dataset_{len(datasets)}")
            
            try:
                data = self.load_data(**config)
                datasets[dataset_name] = data
                logger.info(f"Successfully loaded dataset: {dataset_name}")
            except Exception as e:
                logger.error(f"Failed to load dataset {dataset_name}: {e}")
        
        return datasets
    
    def get_data_summary(self, data: pd.DataFrame, data_type: str) -> Dict[str, Any]:
        """Generate summary statistics for loaded data.
        
        Args:
            data: DataFrame to summarize
            data_type: Type of data
            
        Returns:
            Summary statistics dictionary
        """
        summary = {
            'data_type': data_type,
            'total_records': len(data),
            'columns': list(data.columns),
            'date_range': None,
            'spatial_coverage': None,
            'data_quality': None,
            'missing_data': None
        }
        
        # Date range analysis
        date_cols = ['date', 'timestamp', 'datetime']
        for date_col in date_cols:
            if date_col in data.columns:
                summary['date_range'] = {
                    'start': data[date_col].min(),
                    'end': data[date_col].max(),
                    'duration_days': (data[date_col].max() - data[date_col].min()).days
                }
                break
        
        # Spatial coverage analysis
        spatial_cols = ['location', 'station_id', 'district', 'city']
        for spatial_col in spatial_cols:
            if spatial_col in data.columns:
                summary['spatial_coverage'] = {
                    'column': spatial_col,
                    'unique_locations': data[spatial_col].nunique(),
                    'locations': list(data[spatial_col].unique())
                }
                break
        
        # Data quality assessment
        total_cells = data.size
        missing_cells = data.isnull().sum().sum()
        completeness = 1 - (missing_cells / total_cells) if total_cells > 0 else 0
        
        summary['data_quality'] = {
            'completeness': completeness,
            'completeness_percentage': completeness * 100
        }
        
        # Missing data analysis
        missing_by_column = data.isnull().sum()
        summary['missing_data'] = {
            'total_missing_cells': missing_cells,
            'missing_by_column': missing_by_column.to_dict(),
            'columns_with_missing': list(missing_by_column[missing_by_column > 0].index)
        }
        
        return summary
    
    def export_data(self,
                   data: pd.DataFrame,
                   file_path: Union[str, Path],
                   format: str = 'csv',
                   **kwargs) -> None:
        """Export data to file.
        
        Args:
            data: DataFrame to export
            file_path: Output file path
            format: Export format ('csv', 'xlsx', 'json', 'parquet')
            **kwargs: Additional parameters for export
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'csv':
            data.to_csv(file_path, index=False, **kwargs)
        elif format == 'xlsx':
            data.to_excel(file_path, index=False, **kwargs)
        elif format == 'json':
            data.to_json(file_path, **kwargs)
        elif format == 'parquet':
            data.to_parquet(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Data exported to: {file_path}")
    
    def clear_cache(self) -> None:
        """Clear the data cache."""
        self._data_cache.clear()
        logger.info("Data cache cleared")
    
    def get_available_sources(self, data_type: Optional[str] = None) -> Dict[str, Dict]:
        """Get information about available data sources.
        
        Args:
            data_type: Filter sources by data type (optional)
            
        Returns:
            Dictionary of available sources
        """
        if data_type:
            return {
                name: config for name, config in self.data_sources.items()
                if data_type in config.get('data_types', [])
            }
        else:
            return self.data_sources.copy()
    
    def get_data_schema(self, data_type: str) -> Dict[str, Any]:
        """Get data schema for a specific data type.
        
        Args:
            data_type: Type of data
            
        Returns:
            Data schema dictionary
        """
        if data_type not in self.data_schemas:
            raise ValueError(f"Unknown data type: {data_type}")
        
        return self.data_schemas[data_type].copy()