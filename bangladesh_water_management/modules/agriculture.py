"""Agricultural Water Management Module for Bangladesh.

This module handles irrigation systems, crop water requirements, agricultural sustainability,
and climate adaptation strategies for Bangladesh's agricultural sector.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from loguru import logger
from scipy.optimize import minimize
from scipy.interpolate import interp1d


class AgriculturalWaterManager:
    """Manages agricultural water resources and irrigation systems.
    
    This class implements crop water requirement calculations, irrigation optimization,
    and agricultural sustainability assessment for Bangladesh.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize agricultural water manager.
        
        Args:
            config: Configuration dictionary containing agricultural parameters
        """
        self.config = config
        self.agriculture_config = config['agriculture']
        self.regions_config = config['regions']
        
        # Initialize crop parameters
        self.crop_parameters = self._initialize_crop_parameters()
        
        # Initialize irrigation systems
        self.irrigation_systems = self._initialize_irrigation_systems()
        
        # Initialize regional agricultural data
        self.regional_agriculture = self._initialize_regional_agriculture()
        
        logger.info("Agricultural Water Manager initialized")
    
    def _initialize_crop_parameters(self) -> Dict[str, Dict]:
        """Initialize crop-specific parameters for Bangladesh."""
        crops = {}
        
        # Rice (Boro, Aman, Aus seasons)
        crops['rice'] = {
            'seasons': {
                'boro': {  # Dry season rice (Dec-May)
                    'planting_month': 12,
                    'harvesting_month': 5,
                    'duration_days': 150,
                    'kc_stages': [0.6, 1.15, 1.20, 0.6],  # Crop coefficients
                    'stage_lengths': [30, 40, 50, 30],     # Days per stage
                    'yield_potential_tha': 6.5,
                    'water_requirement_mm': 1200
                },
                'aman': {  # Monsoon rice (Jun-Dec)
                    'planting_month': 6,
                    'harvesting_month': 12,
                    'duration_days': 140,
                    'kc_stages': [0.6, 1.10, 1.15, 0.6],
                    'stage_lengths': [25, 35, 50, 30],
                    'yield_potential_tha': 5.8,
                    'water_requirement_mm': 800  # Lower due to monsoon
                },
                'aus': {   # Pre-monsoon rice (Mar-Jul)
                    'planting_month': 3,
                    'harvesting_month': 7,
                    'duration_days': 120,
                    'kc_stages': [0.6, 1.05, 1.10, 0.6],
                    'stage_lengths': [20, 30, 45, 25],
                    'yield_potential_tha': 4.2,
                    'water_requirement_mm': 950
                }
            },
            'salinity_tolerance': 2.0,  # dS/m
            'drought_tolerance': 'Low',
            'economic_value_usd_per_ton': 400,
            'area_coverage_percent': 75,  # 75% of agricultural land
            'water_use_efficiency': 0.45  # 45% efficiency
        }
        
        # Wheat
        crops['wheat'] = {
            'seasons': {
                'rabi': {  # Winter wheat (Nov-Apr)
                    'planting_month': 11,
                    'harvesting_month': 4,
                    'duration_days': 120,
                    'kc_stages': [0.4, 0.7, 1.15, 0.4],
                    'stage_lengths': [20, 25, 50, 25],
                    'yield_potential_tha': 3.2,
                    'water_requirement_mm': 450
                }
            },
            'salinity_tolerance': 6.0,
            'drought_tolerance': 'Medium',
            'economic_value_usd_per_ton': 350,
            'area_coverage_percent': 8,
            'water_use_efficiency': 0.65
        }
        
        # Jute
        crops['jute'] = {
            'seasons': {
                'kharif': {  # Monsoon season (Apr-Sep)
                    'planting_month': 4,
                    'harvesting_month': 9,
                    'duration_days': 120,
                    'kc_stages': [0.5, 0.8, 1.0, 0.7],
                    'stage_lengths': [25, 30, 40, 25],
                    'yield_potential_tha': 2.5,
                    'water_requirement_mm': 600
                }
            },
            'salinity_tolerance': 4.0,
            'drought_tolerance': 'Medium',
            'economic_value_usd_per_ton': 800,
            'area_coverage_percent': 3,
            'water_use_efficiency': 0.55
        }
        
        # Sugarcane
        crops['sugarcane'] = {
            'seasons': {
                'annual': {  # Year-round crop
                    'planting_month': 2,
                    'harvesting_month': 1,  # Next year
                    'duration_days': 365,
                    'kc_stages': [0.5, 0.8, 1.25, 0.8],
                    'stage_lengths': [60, 90, 150, 65],
                    'yield_potential_tha': 55,
                    'water_requirement_mm': 1800
                }
            },
            'salinity_tolerance': 1.7,
            'drought_tolerance': 'Medium',
            'economic_value_usd_per_ton': 45,
            'area_coverage_percent': 2,
            'water_use_efficiency': 0.50
        }
        
        # Vegetables (mixed)
        crops['vegetables'] = {
            'seasons': {
                'winter': {  # Oct-Mar
                    'planting_month': 10,
                    'harvesting_month': 3,
                    'duration_days': 90,
                    'kc_stages': [0.6, 0.8, 1.0, 0.8],
                    'stage_lengths': [15, 25, 35, 15],
                    'yield_potential_tha': 25,
                    'water_requirement_mm': 400
                },
                'summer': {  # Mar-Jun
                    'planting_month': 3,
                    'harvesting_month': 6,
                    'duration_days': 75,
                    'kc_stages': [0.6, 0.8, 1.0, 0.8],
                    'stage_lengths': [12, 20, 30, 13],
                    'yield_potential_tha': 20,
                    'water_requirement_mm': 350
                }
            },
            'salinity_tolerance': 2.5,
            'drought_tolerance': 'Low',
            'economic_value_usd_per_ton': 300,
            'area_coverage_percent': 8,
            'water_use_efficiency': 0.60
        }
        
        # Pulses (lentils, chickpeas)
        crops['pulses'] = {
            'seasons': {
                'rabi': {  # Nov-Mar
                    'planting_month': 11,
                    'harvesting_month': 3,
                    'duration_days': 100,
                    'kc_stages': [0.4, 0.7, 1.0, 0.4],
                    'stage_lengths': [20, 25, 35, 20],
                    'yield_potential_tha': 1.8,
                    'water_requirement_mm': 300
                }
            },
            'salinity_tolerance': 3.0,
            'drought_tolerance': 'High',
            'economic_value_usd_per_ton': 900,
            'area_coverage_percent': 2,
            'water_use_efficiency': 0.70
        }
        
        return crops
    
    def _initialize_irrigation_systems(self) -> Dict[str, Dict]:
        """Initialize irrigation system parameters for Bangladesh."""
        systems = {}
        
        # Surface water irrigation
        systems['surface_water'] = {
            'coverage_area_ha': 1200000,  # 1.2 million hectares
            'efficiency': 0.40,  # 40% efficiency
            'infrastructure': {
                'canals_km': 8500,
                'regulators': 450,
                'pumping_stations': 180
            },
            'operational_cost_usd_per_ha': 85,
            'maintenance_cost_usd_per_ha': 25,
            'water_source_reliability': 0.75,
            'seasonal_availability': {
                'monsoon': 0.95,
                'post_monsoon': 0.80,
                'winter': 0.60,
                'pre_monsoon': 0.35
            }
        }
        
        # Groundwater irrigation (STW - Shallow Tube Wells)
        systems['shallow_tubewells'] = {
            'coverage_area_ha': 3500000,  # 3.5 million hectares
            'efficiency': 0.65,
            'infrastructure': {
                'number_of_wells': 1500000,
                'average_depth_m': 25,
                'pump_capacity_lps': 15
            },
            'operational_cost_usd_per_ha': 120,
            'maintenance_cost_usd_per_ha': 35,
            'water_source_reliability': 0.85,
            'energy_requirement_kwh_per_ha': 450
        }
        
        # Deep Tube Wells (DTW)
        systems['deep_tubewells'] = {
            'coverage_area_ha': 800000,
            'efficiency': 0.70,
            'infrastructure': {
                'number_of_wells': 35000,
                'average_depth_m': 80,
                'pump_capacity_lps': 50
            },
            'operational_cost_usd_per_ha': 180,
            'maintenance_cost_usd_per_ha': 45,
            'water_source_reliability': 0.90,
            'energy_requirement_kwh_per_ha': 650
        }
        
        # Low Lift Pumps (LLP)
        systems['low_lift_pumps'] = {
            'coverage_area_ha': 600000,
            'efficiency': 0.50,
            'infrastructure': {
                'number_of_pumps': 180000,
                'pump_capacity_lps': 25
            },
            'operational_cost_usd_per_ha': 95,
            'maintenance_cost_usd_per_ha': 20,
            'water_source_reliability': 0.70,
            'energy_requirement_kwh_per_ha': 300
        }
        
        # Drip irrigation (modern systems)
        systems['drip_irrigation'] = {
            'coverage_area_ha': 50000,  # Limited but growing
            'efficiency': 0.90,
            'infrastructure': {
                'installation_cost_usd_per_ha': 1200,
                'drip_lines_km_per_ha': 8
            },
            'operational_cost_usd_per_ha': 150,
            'maintenance_cost_usd_per_ha': 80,
            'water_source_reliability': 0.95,
            'water_savings_percent': 40
        }
        
        return systems
    
    def _initialize_regional_agriculture(self) -> Dict[str, Dict]:
        """Initialize regional agricultural characteristics."""
        regions = {}
        
        # Northern Bangladesh
        regions['northern'] = {
            'total_agricultural_area_ha': 2800000,
            'irrigated_area_ha': 1680000,
            'dominant_crops': ['rice', 'wheat', 'jute'],
            'cropping_intensity': 1.85,  # 185%
            'groundwater_dependency': 0.75,
            'soil_characteristics': {
                'type': 'Alluvial',
                'fertility': 'High',
                'drainage': 'Good',
                'salinity_risk': 'Low'
            },
            'climate_risks': {
                'drought_frequency': 0.25,
                'flood_frequency': 0.35,
                'temperature_stress': 'Medium'
            }
        }
        
        # Central Bangladesh
        regions['central'] = {
            'total_agricultural_area_ha': 3200000,
            'irrigated_area_ha': 1920000,
            'dominant_crops': ['rice', 'vegetables', 'pulses'],
            'cropping_intensity': 1.95,
            'groundwater_dependency': 0.80,
            'soil_characteristics': {
                'type': 'Alluvial',
                'fertility': 'High',
                'drainage': 'Moderate',
                'salinity_risk': 'Low'
            },
            'climate_risks': {
                'drought_frequency': 0.20,
                'flood_frequency': 0.28,
                'temperature_stress': 'Medium'
            }
        }
        
        # Southern Bangladesh (Coastal)
        regions['southern'] = {
            'total_agricultural_area_ha': 2100000,
            'irrigated_area_ha': 840000,  # Lower due to salinity
            'dominant_crops': ['rice', 'shrimp_aquaculture'],
            'cropping_intensity': 1.45,  # Lower due to salinity constraints
            'groundwater_dependency': 0.45,  # Limited due to salinity
            'soil_characteristics': {
                'type': 'Coastal alluvium',
                'fertility': 'Medium',
                'drainage': 'Poor',
                'salinity_risk': 'High'
            },
            'climate_risks': {
                'drought_frequency': 0.15,
                'flood_frequency': 0.22,
                'salinity_intrusion': 'High',
                'cyclone_risk': 'High'
            }
        }
        
        # Eastern Bangladesh (Hill areas)
        regions['eastern'] = {
            'total_agricultural_area_ha': 800000,
            'irrigated_area_ha': 240000,
            'dominant_crops': ['rice', 'vegetables', 'fruits'],
            'cropping_intensity': 1.60,
            'groundwater_dependency': 0.30,  # More surface water
            'soil_characteristics': {
                'type': 'Hill soil',
                'fertility': 'Medium',
                'drainage': 'Excellent',
                'salinity_risk': 'Very Low'
            },
            'climate_risks': {
                'drought_frequency': 0.30,
                'flood_frequency': 0.31,
                'landslide_risk': 'Medium'
            }
        }
        
        return regions
    
    def calculate_crop_water_requirements(self,
                                        crop: str,
                                        season: str,
                                        region: str,
                                        area_ha: float,
                                        climate_data: Optional[Dict] = None) -> Dict[str, Any]:
        """Calculate crop water requirements using FAO Penman-Monteith method.
        
        Args:
            crop: Crop type
            season: Growing season
            region: Geographic region
            area_ha: Cultivated area in hectares
            climate_data: Optional climate data
            
        Returns:
            Dictionary containing water requirement calculations
        """
        if crop not in self.crop_parameters:
            raise ValueError(f"Crop {crop} not supported")
        
        if season not in self.crop_parameters[crop]['seasons']:
            raise ValueError(f"Season {season} not supported for crop {crop}")
        
        crop_params = self.crop_parameters[crop]['seasons'][season]
        
        # Reference evapotranspiration (ET0) - simplified calculation
        # In practice, this would use detailed meteorological data
        et0_monthly = self._get_reference_evapotranspiration(region, climate_data)
        
        # Calculate crop evapotranspiration (ETc) for each growth stage
        etc_stages = []
        water_requirements = []
        
        for i, (kc, stage_length) in enumerate(zip(crop_params['kc_stages'], crop_params['stage_lengths'])):
            # Get appropriate ET0 for this stage
            stage_month = (crop_params['planting_month'] + sum(crop_params['stage_lengths'][:i]) // 30) % 12
            stage_et0 = et0_monthly[stage_month]
            
            # Calculate crop evapotranspiration
            etc_daily = kc * stage_et0
            etc_stage = etc_daily * stage_length
            etc_stages.append(etc_stage)
            
            # Water requirement considering efficiency
            irrigation_efficiency = self._get_irrigation_efficiency(region)
            water_req_stage = etc_stage / irrigation_efficiency
            water_requirements.append(water_req_stage)
        
        # Total seasonal water requirement
        total_etc = sum(etc_stages)
        total_water_req = sum(water_requirements)
        
        # Calculate for total area
        total_volume_m3 = total_water_req * area_ha * 10  # Convert mm to m³/ha
        
        # Effective rainfall consideration
        effective_rainfall = self._calculate_effective_rainfall(region, season)
        net_irrigation_req = max(0, total_water_req - effective_rainfall)
        net_volume_m3 = net_irrigation_req * area_ha * 10
        
        # Monthly distribution
        monthly_distribution = self._distribute_monthly_requirements(
            crop_params, water_requirements, effective_rainfall
        )
        
        return {
            'crop': crop,
            'season': season,
            'region': region,
            'area_ha': area_ha,
            'crop_evapotranspiration': {
                'stages_mm': etc_stages,
                'total_mm': total_etc,
                'daily_peak_mm': max([kc * et0_monthly[6] for kc in crop_params['kc_stages']])
            },
            'water_requirements': {
                'gross_requirement_mm': total_water_req,
                'net_requirement_mm': net_irrigation_req,
                'total_volume_m3': total_volume_m3,
                'net_volume_m3': net_volume_m3,
                'monthly_distribution': monthly_distribution
            },
            'efficiency_factors': {
                'irrigation_efficiency': irrigation_efficiency,
                'effective_rainfall_mm': effective_rainfall
            },
            'economic_analysis': {
                'potential_yield_tons': crop_params['yield_potential_tha'] * area_ha,
                'gross_value_usd': crop_params['yield_potential_tha'] * area_ha * 
                                 self.crop_parameters[crop]['economic_value_usd_per_ton'],
                'water_cost_usd': self._calculate_water_cost(region, net_volume_m3)
            }
        }
    
    def _get_reference_evapotranspiration(self, region: str, climate_data: Optional[Dict]) -> List[float]:
        """Get reference evapotranspiration values by month."""
        # Simplified ET0 values for Bangladesh regions (mm/day)
        # In practice, these would be calculated from detailed climate data
        regional_et0 = {
            'northern': [2.1, 2.8, 4.2, 5.1, 5.8, 5.2, 4.8, 4.6, 4.2, 3.5, 2.8, 2.2],
            'central': [2.0, 2.6, 4.0, 4.9, 5.6, 5.0, 4.6, 4.4, 4.0, 3.3, 2.6, 2.1],
            'southern': [2.2, 2.9, 4.3, 5.2, 5.9, 5.3, 4.9, 4.7, 4.3, 3.6, 2.9, 2.3],
            'eastern': [1.9, 2.5, 3.8, 4.7, 5.4, 4.8, 4.4, 4.2, 3.8, 3.1, 2.5, 2.0]
        }
        
        if climate_data and 'temperature' in climate_data:
            # Adjust ET0 based on temperature changes
            temp_adjustment = (climate_data['temperature'] - 25) * 0.1  # 10% per degree
            base_et0 = regional_et0.get(region, regional_et0['central'])
            return [et0 * (1 + temp_adjustment) for et0 in base_et0]
        
        return regional_et0.get(region, regional_et0['central'])
    
    def _get_irrigation_efficiency(self, region: str) -> float:
        """Get irrigation efficiency for the region."""
        # Weighted average efficiency based on irrigation system mix
        regional_systems = {
            'northern': {'shallow_tubewells': 0.6, 'surface_water': 0.3, 'deep_tubewells': 0.1},
            'central': {'shallow_tubewells': 0.7, 'surface_water': 0.2, 'deep_tubewells': 0.1},
            'southern': {'surface_water': 0.4, 'low_lift_pumps': 0.4, 'shallow_tubewells': 0.2},
            'eastern': {'surface_water': 0.5, 'shallow_tubewells': 0.3, 'low_lift_pumps': 0.2}
        }
        
        region_mix = regional_systems.get(region, regional_systems['central'])
        weighted_efficiency = 0
        
        for system, proportion in region_mix.items():
            system_efficiency = self.irrigation_systems[system]['efficiency']
            weighted_efficiency += system_efficiency * proportion
        
        return weighted_efficiency
    
    def _calculate_effective_rainfall(self, region: str, season: str) -> float:
        """Calculate effective rainfall for irrigation planning."""
        # Monthly rainfall data for Bangladesh regions (mm)
        regional_rainfall = {
            'northern': [8, 18, 32, 68, 140, 280, 320, 290, 220, 85, 12, 5],
            'central': [10, 22, 38, 75, 155, 310, 350, 320, 240, 95, 15, 7],
            'southern': [12, 25, 42, 82, 170, 340, 380, 350, 260, 105, 18, 9],
            'eastern': [15, 30, 50, 95, 200, 380, 420, 390, 290, 120, 22, 12]
        }
        
        monthly_rainfall = regional_rainfall.get(region, regional_rainfall['central'])
        
        # Season-specific effective rainfall calculation
        if season in ['boro', 'rabi', 'winter']:  # Dry season crops
            # Months: Nov-May
            relevant_months = monthly_rainfall[10:12] + monthly_rainfall[0:5]
            effective_rainfall = sum(relevant_months) * 0.7  # 70% effectiveness
        elif season in ['aman', 'kharif', 'monsoon']:  # Monsoon crops
            # Months: Jun-Nov
            relevant_months = monthly_rainfall[5:11]
            effective_rainfall = sum(relevant_months) * 0.8  # 80% effectiveness
        else:  # Other seasons
            effective_rainfall = sum(monthly_rainfall) * 0.75 / 2  # Half year average
        
        return effective_rainfall
    
    def _distribute_monthly_requirements(self,
                                       crop_params: Dict,
                                       water_requirements: List[float],
                                       effective_rainfall: float) -> Dict[int, float]:
        """Distribute water requirements by month."""
        planting_month = crop_params['planting_month']
        stage_lengths = crop_params['stage_lengths']
        
        monthly_req = {}
        current_month = planting_month
        days_in_month = 0
        stage_idx = 0
        stage_days_used = 0
        
        for month_offset in range(12):  # Maximum 12 months
            month = (planting_month + month_offset) % 12
            if month == 0:
                month = 12
            
            monthly_req[month] = 0
            days_remaining = 30  # Simplified 30 days per month
            
            while days_remaining > 0 and stage_idx < len(stage_lengths):
                stage_days_left = stage_lengths[stage_idx] - stage_days_used
                days_to_use = min(days_remaining, stage_days_left)
                
                # Proportional water requirement for this month
                stage_daily_req = water_requirements[stage_idx] / stage_lengths[stage_idx]
                monthly_req[month] += stage_daily_req * days_to_use
                
                days_remaining -= days_to_use
                stage_days_used += days_to_use
                
                if stage_days_used >= stage_lengths[stage_idx]:
                    stage_idx += 1
                    stage_days_used = 0
            
            if stage_idx >= len(stage_lengths):
                break
        
        # Adjust for effective rainfall (distributed proportionally)
        total_gross_req = sum(monthly_req.values())
        if total_gross_req > 0:
            rainfall_factor = max(0, 1 - effective_rainfall / total_gross_req)
            for month in monthly_req:
                monthly_req[month] *= rainfall_factor
        
        return monthly_req
    
    def _calculate_water_cost(self, region: str, volume_m3: float) -> float:
        """Calculate cost of water for irrigation."""
        # Simplified cost calculation based on regional irrigation mix
        regional_costs = {
            'northern': 0.08,  # USD per m³
            'central': 0.09,
            'southern': 0.12,  # Higher due to salinity treatment
            'eastern': 0.07
        }
        
        cost_per_m3 = regional_costs.get(region, 0.09)
        return volume_m3 * cost_per_m3
    
    def optimize_cropping_pattern(self,
                                region: str,
                                available_water_m3: float,
                                total_area_ha: float,
                                economic_objective: str = 'maximize_profit') -> Dict[str, Any]:
        """Optimize cropping pattern based on water availability and objectives.
        
        Args:
            region: Geographic region
            available_water_m3: Available water for irrigation
            total_area_ha: Total available agricultural area
            economic_objective: Optimization objective
            
        Returns:
            Optimized cropping pattern and analysis
        """
        regional_data = self.regional_agriculture[region]
        dominant_crops = regional_data['dominant_crops']
        
        # Define decision variables: area allocated to each crop-season combination
        crop_seasons = []
        for crop in dominant_crops:
            if crop in self.crop_parameters:
                for season in self.crop_parameters[crop]['seasons']:
                    crop_seasons.append((crop, season))
        
        # Calculate water requirements and profits for each crop-season
        crop_data = []
        for crop, season in crop_seasons:
            # Calculate for 1 hectare to get per-hectare values
            water_calc = self.calculate_crop_water_requirements(crop, season, region, 1.0)
            
            crop_info = {
                'crop': crop,
                'season': season,
                'water_req_m3_per_ha': water_calc['water_requirements']['net_volume_m3'],
                'gross_value_usd_per_ha': water_calc['economic_analysis']['gross_value_usd'],
                'water_cost_usd_per_ha': water_calc['economic_analysis']['water_cost_usd'],
                'net_profit_usd_per_ha': (water_calc['economic_analysis']['gross_value_usd'] - 
                                        water_calc['economic_analysis']['water_cost_usd']),
                'yield_tons_per_ha': self.crop_parameters[crop]['seasons'][season]['yield_potential_tha']
            }
            crop_data.append(crop_info)
        
        # Optimization problem setup
        n_crops = len(crop_data)
        
        def objective(x):
            """Objective function based on selected criteria."""
            if economic_objective == 'maximize_profit':
                return -sum(x[i] * crop_data[i]['net_profit_usd_per_ha'] for i in range(n_crops))
            elif economic_objective == 'maximize_production':
                return -sum(x[i] * crop_data[i]['yield_tons_per_ha'] for i in range(n_crops))
            elif economic_objective == 'minimize_water_use':
                return sum(x[i] * crop_data[i]['water_req_m3_per_ha'] for i in range(n_crops))
            else:
                return -sum(x[i] * crop_data[i]['net_profit_usd_per_ha'] for i in range(n_crops))
        
        # Constraints
        constraints = []
        
        # Land constraint
        def land_constraint(x):
            return total_area_ha - sum(x)
        
        constraints.append({'type': 'ineq', 'fun': land_constraint})
        
        # Water constraint
        def water_constraint(x):
            return available_water_m3 - sum(x[i] * crop_data[i]['water_req_m3_per_ha'] for i in range(n_crops))
        
        constraints.append({'type': 'ineq', 'fun': water_constraint})
        
        # Minimum food security constraint (ensure minimum rice production)
        rice_indices = [i for i, crop_info in enumerate(crop_data) if crop_info['crop'] == 'rice']
        if rice_indices:
            def rice_constraint(x):
                rice_area = sum(x[i] for i in rice_indices)
                min_rice_area = total_area_ha * 0.6  # Minimum 60% rice
                return rice_area - min_rice_area
            
            constraints.append({'type': 'ineq', 'fun': rice_constraint})
        
        # Bounds (non-negative areas)
        bounds = [(0, total_area_ha) for _ in range(n_crops)]
        
        # Initial guess (equal distribution)
        x0 = [total_area_ha / n_crops] * n_crops
        
        # Solve optimization
        from scipy.optimize import minimize
        
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            optimal_areas = result.x
            
            # Calculate results
            total_water_use = sum(optimal_areas[i] * crop_data[i]['water_req_m3_per_ha'] 
                                for i in range(n_crops))
            total_profit = sum(optimal_areas[i] * crop_data[i]['net_profit_usd_per_ha'] 
                             for i in range(n_crops))
            total_production = sum(optimal_areas[i] * crop_data[i]['yield_tons_per_ha'] 
                                 for i in range(n_crops))
            
            # Detailed crop allocation
            crop_allocation = {}
            for i, (crop, season) in enumerate(crop_seasons):
                if optimal_areas[i] > 1:  # Only include significant allocations
                    key = f"{crop}_{season}"
                    crop_allocation[key] = {
                        'area_ha': optimal_areas[i],
                        'water_use_m3': optimal_areas[i] * crop_data[i]['water_req_m3_per_ha'],
                        'profit_usd': optimal_areas[i] * crop_data[i]['net_profit_usd_per_ha'],
                        'production_tons': optimal_areas[i] * crop_data[i]['yield_tons_per_ha']
                    }
            
            # Performance metrics
            water_productivity = total_profit / total_water_use if total_water_use > 0 else 0
            land_use_efficiency = sum(optimal_areas) / total_area_ha
            water_use_efficiency = total_water_use / available_water_m3 if available_water_m3 > 0 else 0
            
            return {
                'optimization_success': True,
                'crop_allocation': crop_allocation,
                'summary': {
                    'total_area_used_ha': sum(optimal_areas),
                    'total_water_use_m3': total_water_use,
                    'total_profit_usd': total_profit,
                    'total_production_tons': total_production
                },
                'efficiency_metrics': {
                    'water_productivity_usd_per_m3': water_productivity,
                    'land_use_efficiency': land_use_efficiency,
                    'water_use_efficiency': water_use_efficiency,
                    'profit_per_hectare': total_profit / sum(optimal_areas) if sum(optimal_areas) > 0 else 0
                },
                'resource_utilization': {
                    'water_utilization_percent': water_use_efficiency * 100,
                    'land_utilization_percent': land_use_efficiency * 100,
                    'remaining_water_m3': available_water_m3 - total_water_use,
                    'remaining_land_ha': total_area_ha - sum(optimal_areas)
                }
            }
        else:
            return {
                'optimization_success': False,
                'error_message': result.message,
                'fallback_strategy': self._generate_fallback_strategy(region, total_area_ha)
            }
    
    def _generate_fallback_strategy(self, region: str, total_area_ha: float) -> Dict[str, Any]:
        """Generate a fallback cropping strategy when optimization fails."""
        regional_data = self.regional_agriculture[region]
        
        # Simple proportional allocation based on current practices
        rice_area = total_area_ha * 0.75  # 75% rice (typical for Bangladesh)
        other_area = total_area_ha * 0.25
        
        return {
            'strategy': 'Traditional cropping pattern',
            'rice_area_ha': rice_area,
            'other_crops_area_ha': other_area,
            'cropping_intensity': regional_data['cropping_intensity']
        }
    
    def assess_climate_adaptation(self,
                                region: str,
                                climate_scenarios: Dict[str, Dict]) -> Dict[str, Any]:
        """Assess agricultural adaptation strategies for climate change.
        
        Args:
            region: Geographic region
            climate_scenarios: Climate change scenarios
            
        Returns:
            Adaptation assessment and recommendations
        """
        regional_data = self.regional_agriculture[region]
        adaptation_strategies = {
            'immediate_actions': [],
            'medium_term_investments': [],
            'long_term_transformations': [],
            'cost_estimates': {},
            'effectiveness_scores': {}
        }
        
        # Analyze climate risks
        climate_risks = self._analyze_climate_risks(region, climate_scenarios)
        
        # Water-related adaptations
        if climate_risks['water_stress_risk'] > 0.6:
            adaptation_strategies['immediate_actions'].extend([
                'Promote water-efficient crops',
                'Improve irrigation scheduling',
                'Rainwater harvesting systems'
            ])
            
            adaptation_strategies['medium_term_investments'].extend([
                'Drip irrigation installation',
                'Groundwater recharge systems',
                'Water storage infrastructure'
            ])
        
        # Temperature-related adaptations
        if climate_risks['temperature_stress_risk'] > 0.5:
            adaptation_strategies['immediate_actions'].extend([
                'Heat-tolerant crop varieties',
                'Adjusted planting dates',
                'Shade management systems'
            ])
        
        # Salinity-related adaptations (for coastal regions)
        if region == 'southern' and climate_risks.get('salinity_risk', 0) > 0.4:
            adaptation_strategies['immediate_actions'].extend([
                'Salt-tolerant crop varieties',
                'Soil desalinization techniques',
                'Alternative livelihoods (aquaculture)'
            ])
            
            adaptation_strategies['long_term_transformations'].extend([
                'Managed retreat from highly saline areas',
                'Integrated aquaculture-agriculture systems'
            ])
        
        # Flood-related adaptations
        if climate_risks['flood_risk'] > 0.3:
            adaptation_strategies['medium_term_investments'].extend([
                'Improved drainage systems',
                'Flood-resistant crop varieties',
                'Early warning systems'
            ])
        
        # Cost estimates (simplified)
        total_agricultural_area = regional_data['total_agricultural_area_ha']
        
        adaptation_strategies['cost_estimates'] = {
            'water_efficiency_improvements': total_agricultural_area * 150,  # $150/ha
            'climate_resilient_varieties': total_agricultural_area * 50,     # $50/ha
            'infrastructure_upgrades': total_agricultural_area * 300,        # $300/ha
            'capacity_building': total_agricultural_area * 25               # $25/ha
        }
        
        # Effectiveness scores (0-1 scale)
        adaptation_strategies['effectiveness_scores'] = {
            'water_management': 0.8,
            'crop_diversification': 0.7,
            'infrastructure_improvement': 0.9,
            'technology_adoption': 0.6
        }
        
        # Priority ranking
        adaptation_strategies['priority_ranking'] = self._rank_adaptation_priorities(
            climate_risks, regional_data
        )
        
        return {
            'climate_risks': climate_risks,
            'adaptation_strategies': adaptation_strategies,
            'implementation_timeline': {
                'immediate_0_2_years': adaptation_strategies['immediate_actions'],
                'medium_term_2_10_years': adaptation_strategies['medium_term_investments'],
                'long_term_10_plus_years': adaptation_strategies['long_term_transformations']
            },
            'monitoring_indicators': {
                'crop_yield_trends': 'Track annual yield changes',
                'water_use_efficiency': 'Monitor irrigation water productivity',
                'farmer_adoption_rates': 'Track technology and practice adoption',
                'economic_resilience': 'Monitor farm income stability'
            }
        }
    
    def _analyze_climate_risks(self, region: str, climate_scenarios: Dict) -> Dict[str, float]:
        """Analyze climate-related risks for agriculture."""
        regional_data = self.regional_agriculture[region]
        
        # Base climate risks from regional data
        base_risks = regional_data['climate_risks']
        
        # Adjust risks based on climate scenarios
        risks = {
            'drought_risk': base_risks.get('drought_frequency', 0.2),
            'flood_risk': base_risks.get('flood_frequency', 0.25),
            'temperature_stress_risk': 0.3,  # Base temperature stress
            'water_stress_risk': regional_data['groundwater_dependency'] * 0.4
        }
        
        # Apply climate scenario adjustments
        for scenario, changes in climate_scenarios.items():
            if 'temperature_increase' in changes:
                temp_increase = changes['temperature_increase']
                risks['temperature_stress_risk'] += temp_increase * 0.15
                risks['water_stress_risk'] += temp_increase * 0.1
            
            if 'precipitation_change' in changes:
                precip_change = changes['precipitation_change']
                if precip_change < 0:  # Decreased precipitation
                    risks['drought_risk'] += abs(precip_change) * 0.5
                    risks['water_stress_risk'] += abs(precip_change) * 0.3
                else:  # Increased precipitation
                    risks['flood_risk'] += precip_change * 0.4
        
        # Regional-specific adjustments
        if region == 'southern':
            risks['salinity_risk'] = 0.6  # High baseline salinity risk
            if 'sea_level_rise' in climate_scenarios.get('rcp85', {}):
                slr = climate_scenarios['rcp85']['sea_level_rise']
                risks['salinity_risk'] += slr * 0.2
        
        # Ensure risks are bounded between 0 and 1
        for risk_type in risks:
            risks[risk_type] = max(0, min(1, risks[risk_type]))
        
        return risks
    
    def _rank_adaptation_priorities(self, climate_risks: Dict, regional_data: Dict) -> List[str]:
        """Rank adaptation priorities based on risk levels and regional characteristics."""
        priorities = []
        
        # High priority adaptations based on dominant risks
        if climate_risks['water_stress_risk'] > 0.6:
            priorities.append('Water use efficiency improvements')
        
        if climate_risks['temperature_stress_risk'] > 0.5:
            priorities.append('Heat-tolerant crop varieties')
        
        if climate_risks['flood_risk'] > 0.4:
            priorities.append('Drainage and flood management')
        
        if climate_risks.get('salinity_risk', 0) > 0.5:
            priorities.append('Salinity management systems')
        
        if climate_risks['drought_risk'] > 0.4:
            priorities.append('Drought-resistant crops and water storage')
        
        # Add general priorities
        priorities.extend([
            'Crop diversification',
            'Improved irrigation systems',
            'Climate information systems',
            'Farmer capacity building'
        ])
        
        return priorities[:8]  # Return top 8 priorities