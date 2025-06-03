"""Data Validation and Cleaning Module for Bangladesh Water Management.

This module provides tools for validating data quality, applying rules,
and cleaning datasets for the water management model.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from pandera import DataFrameSchema, Column, Check, Index
from pandera.errors import SchemaError
from loguru import logger
import warnings
warnings.filterwarnings('ignore')


class DataValidator:
    """Validates and cleans water management data.
    
    This class uses Pandera for schema validation and provides custom
    cleaning and imputation methods.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize data validator.
        
        Args:
            config: Configuration dictionary containing validation parameters
        """
        self.config = config
        self.validation_config = config.get('validation', {})
        
        # Initialize validation schemas
        self.validation_schemas = self._initialize_validation_schemas()
        
        logger.info("Data Validator initialized")
    
    def _initialize_validation_schemas(self) -> Dict[str, DataFrameSchema]:
        """Initialize Pandera validation schemas for different data types."""
        schemas = {}
        
        # Meteorological Data Schema
        schemas['meteorological'] = DataFrameSchema(
            columns={
                "date": Column(pd.Timestamp, nullable=False, unique=True),
                "location": Column(str, nullable=False),
                "temperature": Column(float, checks=Check.in_range(-50, 60), nullable=True),
                "precipitation": Column(float, checks=Check.ge(0), nullable=True),
                "humidity": Column(float, checks=Check.in_range(0, 100), nullable=True),
                "wind_speed": Column(float, checks=Check.ge(0), nullable=True),
                "solar_radiation": Column(float, checks=Check.ge(0), nullable=True),
                "evapotranspiration": Column(float, checks=Check.ge(0), nullable=True)
            },
            index=Index(int),
            strict=False,  # Allow extra columns
            ordered=False
        )
        
        # Hydrological Data Schema
        schemas['hydrological'] = DataFrameSchema(
            columns={
                "date": Column(pd.Timestamp, nullable=False),
                "station_id": Column(str, nullable=False),
                "water_level": Column(float, nullable=True),
                "discharge": Column(float, checks=Check.ge(0), nullable=True),
                "velocity": Column(float, checks=Check.ge(0), nullable=True),
                "sediment_load": Column(float, checks=Check.ge(0), nullable=True)
            },
            index=Index(int),
            strict=False,
            ordered=False
        )
        
        # Groundwater Data Schema
        schemas['groundwater'] = DataFrameSchema(
            columns={
                "date": Column(pd.Timestamp, nullable=False),
                "well_id": Column(str, nullable=False),
                "water_table_depth": Column(float, nullable=True),
                "extraction_rate": Column(float, checks=Check.ge(0), nullable=True),
                "aquifer_type": Column(str, nullable=True),
                "well_depth": Column(float, checks=Check.ge(0), nullable=True)
            },
            index=Index(int),
            strict=False,
            ordered=False
        )
        
        # Water Quality Data Schema
        schemas['water_quality'] = DataFrameSchema(
            columns={
                "date": Column(pd.Timestamp, nullable=False),
                "location": Column(str, nullable=False),
                "ph": Column(float, checks=Check.in_range(0, 14), nullable=True),
                "dissolved_oxygen": Column(float, checks=Check.ge(0), nullable=True),
                "turbidity": Column(float, checks=Check.ge(0), nullable=True),
                "salinity": Column(float, checks=Check.ge(0), nullable=True),
                "nitrates": Column(float, checks=Check.ge(0), nullable=True),
                "phosphates": Column(float, checks=Check.ge(0), nullable=True)
            },
            index=Index(int),
            strict=False,
            ordered=False
        )
        
        # Agricultural Data Schema
        schemas['agricultural'] = DataFrameSchema(
            columns={
                "date": Column(pd.Timestamp, nullable=False),
                "district": Column(str, nullable=False),
                "crop_type": Column(str, nullable=False),
                "irrigation_demand": Column(float, checks=Check.ge(0), nullable=True),
                "yield": Column(float, checks=Check.ge(0), nullable=True),
                "cropped_area": Column(float, checks=Check.ge(0), nullable=True),
                "soil_moisture": Column(float, checks=Check.in_range(0, 1), nullable=True)
            },
            index=Index(int),
            strict=False,
            ordered=False
        )
        
        # Urban Water Data Schema
        schemas['urban_water'] = DataFrameSchema(
            columns={
                "date": Column(pd.Timestamp, nullable=False),
                "city": Column(str, nullable=False),
                "water_demand": Column(float, checks=Check.ge(0), nullable=True),
                "water_supply": Column(float, checks=Check.ge(0), nullable=True),
                "population": Column(int, checks=Check.ge(0), nullable=True),
                "water_losses": Column(float, checks=Check.in_range(0, 1), nullable=True),
                "treatment_capacity": Column(float, checks=Check.ge(0), nullable=True),
                "service_coverage": Column(float, checks=Check.in_range(0, 1), nullable=True)
            },
            index=Index(int),
            strict=False,
            ordered=False
        )
        
        return schemas
    
    def validate_data(self,
                      data: pd.DataFrame,
                      data_type: str,
                      raise_exception: bool = False) -> Tuple[bool, Optional[pd.DataFrame]]:
        """Validate data against a Pandera schema.
        
        Args:
            data: DataFrame to validate
            data_type: Type of data (e.g., 'meteorological', 'hydrological')
            raise_exception: If True, raise SchemaError on validation failure
            
        Returns:
            Tuple (is_valid, error_report_df or None)
        """
        if data_type not in self.validation_schemas:
            logger.error(f"No validation schema found for data type: {data_type}")
            return False, None
        
        schema = self.validation_schemas[data_type]
        
        try:
            schema.validate(data, lazy=True)
            logger.info(f"Data validation successful for {data_type}")
            return True, None
        except SchemaError as e:
            logger.warning(f"Data validation failed for {data_type}: {e.failure_cases}")
            if raise_exception:
                raise e
            return False, e.failure_cases
    
    def clean_data(self,
                   data: pd.DataFrame,
                   data_type: str,
                   imputation_strategy: str = 'mean',
                   outlier_method: str = 'iqr',
                   outlier_threshold: float = 1.5) -> pd.DataFrame:
        """Clean data by handling missing values and outliers.
        
        Args:
            data: DataFrame to clean
            data_type: Type of data
            imputation_strategy: Method for imputing missing values ('mean', 'median', 'mode', 'ffill', 'bfill')
            outlier_method: Method for outlier detection ('iqr', 'zscore')
            outlier_threshold: Threshold for outlier detection
            
        Returns:
            Cleaned DataFrame
        """
        cleaned_data = data.copy()
        
        # Handle missing values
        cleaned_data = self._impute_missing_values(cleaned_data, data_type, imputation_strategy)
        
        # Handle outliers
        cleaned_data = self._handle_outliers(cleaned_data, data_type, outlier_method, outlier_threshold)
        
        # Ensure data types are correct after cleaning
        schema = self.validation_schemas.get(data_type)
        if schema:
            for col_name, column_schema in schema.columns.items():
                if col_name in cleaned_data.columns:
                    try:
                        if column_schema.dtype == pd.Timestamp:
                            cleaned_data[col_name] = pd.to_datetime(cleaned_data[col_name])
                        else:
                            cleaned_data[col_name] = cleaned_data[col_name].astype(column_schema.dtype)
                    except Exception as e:
                        logger.warning(f"Could not convert column {col_name} to {column_schema.dtype} after cleaning: {e}")
        
        logger.info(f"Data cleaning completed for {data_type}")
        return cleaned_data
    
    def _impute_missing_values(self,
                               data: pd.DataFrame,
                               data_type: str,
                               strategy: str) -> pd.DataFrame:
        """Impute missing values in the DataFrame."""
        for col in data.columns:
            if data[col].isnull().any():
                if pd.api.types.is_numeric_dtype(data[col]):
                    if strategy == 'mean':
                        fill_value = data[col].mean()
                    elif strategy == 'median':
                        fill_value = data[col].median()
                    elif strategy == 'mode':
                        fill_value = data[col].mode()[0] if not data[col].mode().empty else data[col].mean()
                    elif strategy == 'ffill':
                        data[col] = data[col].ffill()
                        continue
                    elif strategy == 'bfill':
                        data[col] = data[col].bfill()
                        continue
                    else:
                        fill_value = data[col].mean()  # Default to mean
                    
                    data[col] = data[col].fillna(fill_value)
                    logger.debug(f"Imputed missing values in column '{col}' using {strategy} (value: {fill_value})")
                
                elif pd.api.types.is_datetime64_any_dtype(data[col]):
                    # For datetime, use ffill or bfill
                    if strategy in ['ffill', 'bfill']:
                        data[col] = data[col].fillna(method=strategy)
                    else: # Default to ffill for datetime
                        data[col] = data[col].ffill()
                    logger.debug(f"Imputed missing datetime values in column '{col}' using {strategy or 'ffill'}")
                
                else:
                    # For categorical or string data, use mode or a placeholder
                    if strategy == 'mode':
                        fill_value = data[col].mode()[0] if not data[col].mode().empty else 'Unknown'
                    else:
                        fill_value = 'Unknown' # Default placeholder
                    
                    data[col] = data[col].fillna(fill_value)
                    logger.debug(f"Imputed missing categorical values in column '{col}' using mode (value: {fill_value})")
        
        return data
    
    def _handle_outliers(self,
                         data: pd.DataFrame,
                         data_type: str,
                         method: str,
                         threshold: float) -> pd.DataFrame:
        """Handle outliers in numerical columns."""
        for col in data.select_dtypes(include=np.number).columns:
            if method == 'iqr':
                q1 = data[col].quantile(0.25)
                q3 = data[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - threshold * iqr
                upper_bound = q3 + threshold * iqr
            elif method == 'zscore':
                mean = data[col].mean()
                std = data[col].std()
                lower_bound = mean - threshold * std
                upper_bound = mean + threshold * std
            else:
                continue # No outlier handling for this method
            
            # Cap outliers
            outliers_low = data[col] < lower_bound
            outliers_high = data[col] > upper_bound
            
            if outliers_low.any() or outliers_high.any():
                data[col] = np.clip(data[col], lower_bound, upper_bound)
                logger.debug(f"Capped outliers in column '{col}' using {method} method")
        
        return data
    
    def generate_validation_report(self,
                                   data: pd.DataFrame,
                                   data_type: str) -> Dict[str, Any]:
        """Generate a comprehensive validation report.
        
        Args:
            data: DataFrame to validate
            data_type: Type of data
            
        Returns:
            Validation report dictionary
        """
        report = {
            'data_type': data_type,
            'num_records': len(data),
            'num_columns': len(data.columns),
            'validation_status': None,
            'schema_errors': None,
            'data_quality_metrics': {}
        }
        
        # Perform schema validation
        is_valid, errors_df = self.validate_data(data, data_type)
        report['validation_status'] = 'Valid' if is_valid else 'Invalid'
        
        if errors_df is not None and not errors_df.empty:
            report['schema_errors'] = errors_df.to_dict(orient='records')
        
        # Calculate data quality metrics
        report['data_quality_metrics']['missing_values'] = self._calculate_missing_values(data)
        report['data_quality_metrics']['duplicate_records'] = self._calculate_duplicate_records(data)
        report['data_quality_metrics']['outlier_counts'] = self._count_outliers(data)
        report['data_quality_metrics']['column_summaries'] = self._summarize_columns(data)
        
        logger.info(f"Generated validation report for {data_type}")
        return report
    
    def _calculate_missing_values(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate missing value statistics."""
        missing_info = {}
        total_missing = data.isnull().sum().sum()
        total_cells = np.prod(data.shape)
        
        missing_info['total_missing_cells'] = int(total_missing)
        missing_info['percentage_missing'] = (total_missing / total_cells) * 100 if total_cells > 0 else 0
        missing_info['missing_by_column'] = data.isnull().sum().to_dict()
        
        return missing_info
    
    def _calculate_duplicate_records(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate duplicate record statistics."""
        num_duplicates = data.duplicated().sum()
        return {
            'num_duplicate_records': int(num_duplicates),
            'percentage_duplicates': (num_duplicates / len(data)) * 100 if len(data) > 0 else 0
        }
    
    def _count_outliers(self, data: pd.DataFrame, method: str = 'iqr', threshold: float = 1.5) -> Dict[str, int]:
        """Count outliers in numerical columns."""
        outlier_counts = {}
        for col in data.select_dtypes(include=np.number).columns:
            if method == 'iqr':
                q1 = data[col].quantile(0.25)
                q3 = data[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - threshold * iqr
                upper_bound = q3 + threshold * iqr
            elif method == 'zscore':
                mean = data[col].mean()
                std = data[col].std()
                if std == 0: # Avoid division by zero for constant columns
                    outlier_counts[col] = 0
                    continue
                lower_bound = mean - threshold * std
                upper_bound = mean + threshold * std
            else:
                outlier_counts[col] = 0
                continue
            
            num_outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)][col].count()
            outlier_counts[col] = int(num_outliers)
            
        return outlier_counts
    
    def _summarize_columns(self, data: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Generate summary statistics for each column."""
        summaries = {}
        for col in data.columns:
            summary = {
                'dtype': str(data[col].dtype),
                'num_unique': data[col].nunique(),
                'num_missing': int(data[col].isnull().sum())
            }
            
            if pd.api.types.is_numeric_dtype(data[col]):
                summary.update(data[col].describe().to_dict())
            elif pd.api.types.is_datetime64_any_dtype(data[col]):
                summary['min_date'] = str(data[col].min())
                summary['max_date'] = str(data[col].max())
            else: # Categorical/Object
                summary['top_values'] = data[col].value_counts().nlargest(5).to_dict()
            
            summaries[col] = summary
        return summaries

    def get_validation_schema(self, data_type: str) -> Optional[DataFrameSchema]:
        """Retrieve a validation schema.

        Args:
            data_type: The type of data for which to get the schema.

        Returns:
            The Pandera DataFrameSchema if found, else None.
        """
        return self.validation_schemas.get(data_type)

    def add_custom_validation_rule(self, 
                                   data_type: str, 
                                   column_name: str, 
                                   check: Check, 
                                   rule_name: Optional[str] = None) -> bool:
        """Add a custom validation rule to an existing schema.

        Args:
            data_type: The data type schema to modify.
            column_name: The column to add the rule to.
            check: The Pandera Check object representing the rule.
            rule_name: Optional name for the check.

        Returns:
            True if rule was added successfully, False otherwise.
        """
        if data_type not in self.validation_schemas:
            logger.error(f"Schema for data type '{data_type}' not found.")
            return False

        schema = self.validation_schemas[data_type]
        if column_name not in schema.columns:
            logger.error(f"Column '{column_name}' not found in schema for '{data_type}'.")
            return False

        try:
            # Pandera schemas are immutable by default after creation in some contexts.
            # Recreating the column with the new check is a safer way.
            existing_column = schema.columns[column_name]
            new_checks = existing_column.checks + [check] if existing_column.checks else [check]
            
            updated_column = Column(
                dtype=existing_column.dtype,
                checks=new_checks,
                nullable=existing_column.nullable,
                unique=existing_column.unique,
                coerce=existing_column.coerce,
                required=existing_column.required,
                regex=existing_column.regex,
                name=existing_column.name
            )
            
            new_columns = schema.columns.copy()
            new_columns[column_name] = updated_column
            
            self.validation_schemas[data_type] = DataFrameSchema(
                columns=new_columns,
                index=schema.index,
                dtype=schema.dtype,
                coerce=schema.coerce,
                strict=schema.strict,
                name=schema.name,
                ordered=schema.ordered,
                unique=schema.unique,
                report_grouping=schema.report_grouping
            )
            logger.info(f"Added custom validation rule '{rule_name or check.name}' to column '{column_name}' in '{data_type}' schema.")
            return True
        except Exception as e:
            logger.error(f"Failed to add custom validation rule: {e}")
            return False

# Example usage (for testing purposes)
if __name__ == '__main__':
    # Sample configuration (replace with actual config loading)
    config = {
        'validation': {
            'default_imputation': 'median',
            'default_outlier_method': 'iqr'
        }
    }
    
    validator = DataValidator(config)
    
    # Sample meteorological data with errors
    sample_met_data = pd.DataFrame({
        'date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-01']), # Duplicate date
        'location': ['Dhaka', 'Chittagong', 'Sylhet', 'Dhaka'],
        'temperature': [25.0, 26.5, 70.0, 25.5],  # Outlier
        'precipitation': [0.0, 5.2, -2.0, 1.0], # Negative value
        'humidity': [75.0, 80.0, 110.0, 78.0], # Out of range
        'extra_col': [1,2,3,4] # Extra column
    })
    
    logger.info("--- Validating Sample Meteorological Data ---")
    is_valid, errors = validator.validate_data(sample_met_data, 'meteorological')
    logger.info(f"Validation status: {'Valid' if is_valid else 'Invalid'}")
    if errors is not None:
        logger.info(f"Validation errors:\n{errors}")
    
    logger.info("--- Cleaning Sample Meteorological Data ---")
    # Make a copy for cleaning to avoid SettingWithCopyWarning
    data_to_clean = sample_met_data.copy()
    data_to_clean.loc[0, 'temperature'] = np.nan # Introduce NaN for imputation test
    cleaned_data = validator.clean_data(data_to_clean, 'meteorological')
    logger.info(f"Cleaned data:\n{cleaned_data}")
    
    logger.info("--- Re-validating Cleaned Data ---")
    is_valid_cleaned, errors_cleaned = validator.validate_data(cleaned_data, 'meteorological')
    logger.info(f"Cleaned data validation status: {'Valid' if is_valid_cleaned else 'Invalid'}")
    if errors_cleaned is not None:
        logger.info(f"Cleaned data validation errors:\n{errors_cleaned}")

    logger.info("--- Generating Validation Report for Cleaned Data ---")
    report = validator.generate_validation_report(cleaned_data, 'meteorological')
    import json
    logger.info(f"Validation report:\n{json.dumps(report, indent=2, default=str)}")

    # Test adding custom rule
    new_check = Check(lambda s: s > 0, name="positive_check", error="Value must be positive")
    added = validator.add_custom_validation_rule('meteorological', 'temperature', new_check, 'temp_positive')
    if added:
        logger.info("Custom rule added. Schema for temperature:")
        logger.info(validator.get_validation_schema('meteorological').columns['temperature'])