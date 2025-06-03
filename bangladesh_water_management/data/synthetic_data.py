"""Synthetic Data Generation Module for Bangladesh Water Management.

This module provides tools for generating synthetic datasets for various
water management parameters, useful for testing, calibration, and scenarios
where real data is scarce.
"""

import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from loguru import logger
import warnings
warnings.filterwarnings('ignore')


class SyntheticDataGenerator:
    """Generates synthetic data for water management modeling."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize synthetic data generator.
        
        Args:
            config: Configuration dictionary with parameters for data generation.
        """
        self.config = config.get('synthetic_data_generation', {})
        self.faker = Faker('en_BD') # Use Bangladesh locale for relevant fake data
        logger.info("Synthetic Data Generator initialized.")

    def _generate_date_range(self, start_date_str: str, end_date_str: str, freq: str = 'D') -> pd.DatetimeIndex:
        """Helper to generate a date range."""
        try:
            start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
            return pd.date_range(start=start_date, end=end_date, freq=freq)
        except ValueError as e:
            logger.error(f"Invalid date format: {e}. Using default dates.")
            return pd.date_range(start='2020-01-01', end='2023-12-31', freq=freq)

    def _generate_seasonal_pattern(self, dates: pd.DatetimeIndex, base_value: float, amplitude: float, phase_shift: float = 0) -> np.ndarray:
        """Generates a seasonal pattern for a given date range."""
        day_of_year = dates.dayofyear
        return base_value + amplitude * np.sin(2 * np.pi * (day_of_year - phase_shift) / 365.25)

    def _add_noise(self, data: np.ndarray, noise_level: float) -> np.ndarray:
        """Adds random noise to the data."""
        data_array = np.asarray(data) # Ensure data is an array
        noise_scale = noise_level * np.abs(data_array).mean() if np.abs(data_array).mean() > 0 else noise_level
        return data_array + np.random.normal(0, noise_scale, len(data_array))

    def generate_meteorological_data(self, 
                                     locations: List[str], 
                                     start_date: str = '2020-01-01', 
                                     end_date: str = '2023-12-31',
                                     params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Generates synthetic meteorological data."""
        params = params or self.config.get('meteorological', {})
        dates = self._generate_date_range(start_date, end_date)
        all_data = []

        for location in locations:
            loc_params = params.get(location, params.get('default', {}))
            
            temp_base = loc_params.get('temperature_base', 25)
            temp_amp = loc_params.get('temperature_amplitude', 5)
            precip_base = loc_params.get('precipitation_base', 2)
            precip_amp = loc_params.get('precipitation_amplitude', 5)
            precip_skew = loc_params.get('precipitation_skew', 0.8) # Probability of no rain

            temperature = self._generate_seasonal_pattern(dates, temp_base, temp_amp, phase_shift=120) # Peak in summer
            temperature = self._add_noise(temperature, loc_params.get('temperature_noise', 0.1))
            temperature = np.clip(temperature, -10, 50)

            precipitation = self._generate_seasonal_pattern(dates, precip_base, precip_amp, phase_shift=180) # Peak in monsoon
            precipitation = self._add_noise(precipitation, loc_params.get('precipitation_noise', 0.5))
            precipitation[precipitation < 0] = 0
            # Introduce dry days
            precipitation[np.random.rand(len(precipitation)) < precip_skew] = 0
            
            humidity = 100 - (temperature - 10) * 2 # Simplistic inverse relation to temp
            humidity = self._add_noise(humidity, 0.05)
            humidity = np.clip(humidity, 20, 100)

            wind_speed = np.random.gamma(2, 2, len(dates))
            wind_speed = np.clip(wind_speed, 0, 30)

            solar_radiation = self._generate_seasonal_pattern(dates, 200, 100, phase_shift=150)
            solar_radiation = self._add_noise(solar_radiation, 0.15)
            solar_radiation = np.clip(solar_radiation, 0, 1000)

            # Simplified Penman-Monteith for ET0 (conceptual)
            evapotranspiration = 0.05 * temperature + 0.01 * solar_radiation 
            evapotranspiration = self._add_noise(evapotranspiration, 0.2)
            evapotranspiration = np.clip(evapotranspiration, 0, 15)

            df = pd.DataFrame({
                'date': dates,
                'location': location,
                'temperature': temperature,
                'precipitation': precipitation,
                'humidity': humidity,
                'wind_speed': wind_speed,
                'solar_radiation': solar_radiation,
                'evapotranspiration': evapotranspiration
            })
            all_data.append(df)
        
        logger.info(f"Generated synthetic meteorological data for {len(locations)} locations.")
        return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

    def generate_hydrological_data(self, 
                                   station_ids: List[str], 
                                   start_date: str = '2020-01-01', 
                                   end_date: str = '2023-12-31',
                                   params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Generates synthetic hydrological data (river flow, water level)."""
        params = params or self.config.get('hydrological', {})
        dates = self._generate_date_range(start_date, end_date)
        all_data = []

        for station_id in station_ids:
            st_params = params.get(station_id, params.get('default', {}))
            
            level_base = st_params.get('level_base', 5)
            level_amp = st_params.get('level_amplitude', 3)
            flow_base = st_params.get('flow_base', 1000)
            flow_amp = st_params.get('flow_amplitude', 2000)

            water_level = self._generate_seasonal_pattern(dates, level_base, level_amp, phase_shift=180) # Monsoon peak
            water_level = self._add_noise(water_level, st_params.get('level_noise', 0.15))
            water_level = np.clip(water_level, 0.1, 20)

            # Flow often related to level (e.g., Q = a * H^b)
            discharge = flow_base * (water_level / level_base)**1.5 
            discharge = self._add_noise(discharge, st_params.get('flow_noise', 0.25))
            discharge = np.clip(discharge, 10, 50000)
            
            velocity = discharge / (water_level * st_params.get('channel_width_factor', 50)) # Simplified
            velocity = self._add_noise(velocity, 0.1)
            velocity = np.clip(velocity, 0.01, 5)

            sediment_load = discharge * st_params.get('sediment_factor', 0.001) * (1 + 0.5 * np.random.rand(len(dates)))
            sediment_load = self._add_noise(sediment_load, 0.3)
            sediment_load = np.clip(sediment_load, 0, 1000)

            df = pd.DataFrame({
                'date': dates,
                'station_id': station_id,
                'water_level': water_level,
                'discharge': discharge,
                'velocity': velocity,
                'sediment_load': sediment_load
            })
            all_data.append(df)
        
        logger.info(f"Generated synthetic hydrological data for {len(station_ids)} stations.")
        return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

    def generate_groundwater_data(self, 
                                  well_ids: List[str], 
                                  start_date: str = '2020-01-01', 
                                  end_date: str = '2023-12-31',
                                  params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Generates synthetic groundwater data."""
        params = params or self.config.get('groundwater', {})
        dates = self._generate_date_range(start_date, end_date, freq='M') # Monthly data often for GW
        all_data = []

        aquifer_types = ['Shallow Tube Well', 'Deep Tube Well', 'Hand Pump']

        for well_id in well_ids:
            well_params = params.get(well_id, params.get('default', {}))
            
            depth_base = well_params.get('depth_base', 10) # Depth below ground
            depth_amp = well_params.get('depth_amplitude', 2)
            extraction_mean = well_params.get('extraction_mean', 50)

            # GW levels often lag surface water, lower in dry season
            water_table_depth = self._generate_seasonal_pattern(dates, depth_base, depth_amp, phase_shift=270) # Lowest before monsoon
            water_table_depth = self._add_noise(water_table_depth, well_params.get('depth_noise', 0.1))
            # Simulate depletion trend
            depletion_trend = np.linspace(0, well_params.get('annual_depletion_rate', 0.1) * (len(dates)/12), len(dates))
            water_table_depth += depletion_trend
            water_table_depth = np.clip(water_table_depth, 1, 100)

            extraction_rate = np.random.gamma(extraction_mean / 5, 5, len(dates))
            extraction_rate = self._add_noise(extraction_rate, well_params.get('extraction_noise', 0.2))
            extraction_rate = np.clip(extraction_rate, 0, 500)
            
            aquifer_type = np.random.choice(aquifer_types)
            well_depth = depth_base + np.random.uniform(5, 50) if aquifer_type == 'Deep Tube Well' else depth_base + np.random.uniform(1,10)
            well_depth = np.clip(well_depth, 5, 150)

            df = pd.DataFrame({
                'date': dates,
                'well_id': well_id,
                'water_table_depth': water_table_depth,
                'extraction_rate': extraction_rate,
                'aquifer_type': aquifer_type,
                'well_depth': well_depth
            })
            all_data.append(df)
        
        logger.info(f"Generated synthetic groundwater data for {len(well_ids)} wells.")
        return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

    def generate_water_quality_data(self, 
                                    locations: List[str], 
                                    start_date: str = '2020-01-01', 
                                    end_date: str = '2023-12-31',
                                    params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Generates synthetic water quality data."""
        params = params or self.config.get('water_quality', {})
        dates = self._generate_date_range(start_date, end_date, freq='M') # WQ often monthly/quarterly
        all_data = []

        for location in locations:
            loc_params = params.get(location, params.get('default', {}))
            
            ph = np.random.normal(loc_params.get('ph_mean', 7.5), loc_params.get('ph_std', 0.5), len(dates))
            ph = np.clip(ph, 5, 9)
            
            dissolved_oxygen = self._generate_seasonal_pattern(dates, loc_params.get('do_base', 6), loc_params.get('do_amplitude', 1.5), phase_shift=30) # Higher in cooler months
            dissolved_oxygen = self._add_noise(dissolved_oxygen, 0.1)
            dissolved_oxygen = np.clip(dissolved_oxygen, 2, 12)

            turbidity = np.random.gamma(loc_params.get('turbidity_shape', 2), loc_params.get('turbidity_scale', 10), len(dates))
            turbidity = np.clip(turbidity, 1, 500) # Higher during monsoon
            if 'monsoon_factor' in loc_params:
                 turbidity[dates.month.isin([6,7,8,9])] *= loc_params['monsoon_factor']

            salinity_base = loc_params.get('salinity_base', 0.5) # PSU or mg/L depending on context
            salinity_amp = loc_params.get('salinity_amplitude', 0.2)
            salinity = self._generate_seasonal_pattern(dates, salinity_base, salinity_amp, phase_shift=90) # Higher in dry season for coastal areas
            salinity = self._add_noise(salinity, 0.2)
            salinity = np.clip(salinity, 0, 35) # Max for seawater

            nitrates = np.random.gamma(loc_params.get('nitrates_shape', 1), loc_params.get('nitrates_scale', 2), len(dates))
            nitrates = np.clip(nitrates, 0, 50) # mg/L

            phosphates = np.random.gamma(loc_params.get('phosphates_shape', 0.5), loc_params.get('phosphates_scale', 1), len(dates))
            phosphates = np.clip(phosphates, 0, 10) # mg/L

            df = pd.DataFrame({
                'date': dates,
                'location': location,
                'ph': ph,
                'dissolved_oxygen': dissolved_oxygen,
                'turbidity': turbidity,
                'salinity': salinity,
                'nitrates': nitrates,
                'phosphates': phosphates
            })
            all_data.append(df)
        
        logger.info(f"Generated synthetic water quality data for {len(locations)} locations.")
        return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

    def generate_agricultural_data(self, 
                                   districts: List[str], 
                                   crop_types: List[str], 
                                   start_date: str = '2020-01-01', 
                                   end_date: str = '2023-12-31', # Yearly data often
                                   params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Generates synthetic agricultural data."""
        params = params or self.config.get('agricultural', {})
        # Agricultural data is often annual or seasonal, using annual for simplicity here
        years = self._generate_date_range(start_date, end_date, freq='AS') # Annual Start
        all_data = []

        for district in districts:
            dist_params = params.get(district, params.get('default', {}))
            for year_date in years:
                for crop in crop_types:
                    crop_params = dist_params.get(crop, dist_params.get('default_crop', {}))

                    irrigation_demand = np.random.normal(crop_params.get('irrigation_mean', 500), crop_params.get('irrigation_std', 100)) # mm/season
                    irrigation_demand = np.clip(irrigation_demand, 100, 1500)

                    crop_yield = np.random.normal(crop_params.get('yield_mean', 3.5), crop_params.get('yield_std', 0.5)) # tons/ha
                    crop_yield = np.clip(crop_yield, 0.5, 10)

                    cropped_area = np.random.normal(crop_params.get('area_mean', 10000), crop_params.get('area_std', 2000)) # ha
                    cropped_area = np.clip(cropped_area, 100, 50000)
                    
                    soil_moisture = np.random.uniform(0.2, 0.8) # Fraction

                    df = pd.DataFrame({
                        'date': [year_date],
                        'district': [district],
                        'crop_type': [crop],
                        'irrigation_demand': [irrigation_demand],
                        'yield': [crop_yield],
                        'cropped_area': [cropped_area],
                        'soil_moisture': [soil_moisture]
                    })
                    all_data.append(df)
        
        logger.info(f"Generated synthetic agricultural data for {len(districts)} districts and {len(crop_types)} crops.")
        return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

    def generate_urban_water_data(self, 
                                  cities: List[str], 
                                  start_date: str = '2020-01-01', 
                                  end_date: str = '2023-12-31', # Yearly data
                                  params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Generates synthetic urban water supply and demand data."""
        params = params or self.config.get('urban_water', {})
        years = self._generate_date_range(start_date, end_date, freq='AS')
        all_data = []

        for city in cities:
            city_params = params.get(city, params.get('default', {}))
            base_pop = city_params.get('population_base', 1000000)
            pop_growth_rate = city_params.get('population_growth_rate', 0.02)
            
            for i, year_date in enumerate(years):
                population = base_pop * (1 + pop_growth_rate)**i
                population = np.round(population)

                per_capita_demand = np.random.normal(city_params.get('per_capita_demand_mean', 150), city_params.get('per_capita_demand_std', 20)) # LPCD
                water_demand = population * per_capita_demand / 1000 # MLD (Million Liters per Day)
                water_demand = np.clip(water_demand, population * 50 / 1000, population * 300 / 1000)

                supply_coverage = np.random.uniform(city_params.get('supply_coverage_min', 0.7), city_params.get('supply_coverage_max', 0.95))
                water_supply = water_demand * supply_coverage * np.random.uniform(0.9, 1.1) # Supply close to demand with some variability
                water_supply = np.clip(water_supply, 0, water_demand * 1.2)

                water_losses = np.random.uniform(city_params.get('losses_min', 0.15), city_params.get('losses_max', 0.40))
                
                treatment_capacity = water_supply * np.random.uniform(city_params.get('treatment_capacity_factor_min', 0.5), city_params.get('treatment_capacity_factor_max', 0.8))
                service_coverage = np.random.uniform(city_params.get('service_coverage_min', 0.6), city_params.get('service_coverage_max', 0.9))

                df = pd.DataFrame({
                    'date': [year_date],
                    'city': [city],
                    'population': [population],
                    'water_demand': [water_demand],
                    'water_supply': [water_supply],
                    'water_losses': [water_losses],
                    'treatment_capacity': [treatment_capacity],
                    'service_coverage': [service_coverage]
                })
                all_data.append(df)
        
        logger.info(f"Generated synthetic urban water data for {len(cities)} cities.")
        return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

# Example Usage (for testing purposes)
if __name__ == '__main__':
    # Sample configuration (replace with actual config loading)
    config_data = {
        'synthetic_data_generation': {
            'meteorological': {
                'default': {'temperature_base': 26, 'temperature_amplitude': 6, 'precipitation_base': 3, 'precipitation_amplitude': 7},
                'Dhaka': {'temperature_base': 27, 'precipitation_base': 4}
            },
            'hydrological': {
                'default': {'level_base': 4, 'flow_base': 800},
                'StationA': {'level_base': 6, 'flow_base': 1200}
            },
            'groundwater': {
                'default': {'depth_base': 8, 'annual_depletion_rate': 0.05}
            },
            'water_quality': {
                'default': {'ph_mean': 7.2, 'do_base': 5.5},
                'CoastalSite': {'salinity_base': 5, 'salinity_amplitude': 3}
            },
            'agricultural': {
                'default': {'default_crop': {'irrigation_mean': 400, 'yield_mean': 3.0, 'area_mean': 5000}}
            },
            'urban_water': {
                'default': {'population_base': 500000, 'per_capita_demand_mean': 130},
                'MegaCity': {'population_base': 5000000, 'per_capita_demand_mean': 180}
            }
        }
    }

    generator = SyntheticDataGenerator(config_data)

    logger.info("--- Generating Meteorological Data ---")
    met_data = generator.generate_meteorological_data(['Dhaka', 'Sylhet'], '2022-01-01', '2022-03-31')
    logger.info(f"Meteorological Data Sample:\n{met_data.head()}")

    logger.info("--- Generating Hydrological Data ---")
    hydro_data = generator.generate_hydrological_data(['StationA', 'StationB'], '2022-01-01', '2022-02-28')
    logger.info(f"Hydrological Data Sample:\n{hydro_data.head()}")

    logger.info("--- Generating Groundwater Data ---")
    gw_data = generator.generate_groundwater_data(['Well1', 'Well2', 'Well3'], '2022-01-01', '2022-12-31')
    logger.info(f"Groundwater Data Sample:\n{gw_data.head()}")

    logger.info("--- Generating Water Quality Data ---")
    wq_data = generator.generate_water_quality_data(['RiverPoint1', 'CoastalSite'], '2022-01-01', '2022-06-30')
    logger.info(f"Water Quality Data Sample:\n{wq_data.head()}")

    logger.info("--- Generating Agricultural Data ---")
    agri_data = generator.generate_agricultural_data(['Rajshahi', 'Comilla'], ['Rice', 'Wheat'], '2021-01-01', '2022-12-31')
    logger.info(f"Agricultural Data Sample:\n{agri_data.head()}")

    logger.info("--- Generating Urban Water Data ---")
    urban_data = generator.generate_urban_water_data(['SmallTown', 'MegaCity'], '2021-01-01', '2022-12-31')
    logger.info(f"Urban Water Data Sample:\n{urban_data.head()}")