"""Salinity Intrusion Management Module for Bangladesh.

This module handles coastal salinity intrusion modeling, saltwater penetration analysis,
and agricultural/health impact assessment for Bangladesh's coastal regions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from loguru import logger
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit


class SalinityManager:
    """Manages coastal salinity intrusion modeling and impact assessment.
    
    This class implements saltwater intrusion models, tidal effects simulation,
    and agricultural/health impact analysis for Bangladesh's coastal regions.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize salinity manager.
        
        Args:
            config: Configuration dictionary containing salinity parameters
        """
        self.config = config
        self.salinity_config = config['salinity']
        self.regions_config = config['regions']
        
        # Initialize salinity parameters for coastal regions
        self.salinity_params = self._initialize_salinity_parameters()
        
        logger.info("Salinity Manager initialized")
    
    def _initialize_salinity_parameters(self) -> Dict[str, Dict]:
        """Initialize salinity parameters for different coastal regions."""
        params = {}
        
        # Coastal Southwest (Satkhira, Khulna) - Most severely affected
        params['coastal_southwest'] = {
            'baseline_salinity': {
                'dry_season': 15.0,  # ppt
                'monsoon': 3.0,
                'peak_intrusion': 40.0  # Equivalent to seawater in Ashashuni
            },
            'intrusion_distance': {
                'current': 150,  # km inland
                'historical': 80,   # km in 1970s
                'projected_2050': 200
            },
            'affected_area': 12000,  # km²
            'population_affected': 3500000,
            'agricultural_area': 800000,  # hectares
            'soil_salinity': {
                'assasuni': 8.24,  # dS/m
                'dacope': 8.08,
                'morrelganj': 7.96
            },
            'tidal_amplitude': 3.5,  # meters
            'river_flow_dependency': 0.8  # Dependency on upstream flow
        }
        
        # Coastal Southeast (Chittagong, Cox's Bazar)
        params['coastal_southeast'] = {
            'baseline_salinity': {
                'dry_season': 8.0,
                'monsoon': 1.5,
                'peak_intrusion': 25.0
            },
            'intrusion_distance': {
                'current': 80,
                'historical': 50,
                'projected_2050': 120
            },
            'affected_area': 8000,
            'population_affected': 2800000,
            'agricultural_area': 400000,
            'soil_salinity': {
                'average': 5.5
            },
            'tidal_amplitude': 4.2,
            'river_flow_dependency': 0.6
        }
        
        # Barisal coastal areas
        params['barisal'] = {
            'baseline_salinity': {
                'dry_season': 12.0,
                'monsoon': 2.5,
                'peak_intrusion': 30.0
            },
            'intrusion_distance': {
                'current': 100,
                'historical': 60,
                'projected_2050': 140
            },
            'affected_area': 6000,
            'population_affected': 1800000,
            'agricultural_area': 350000,
            'soil_salinity': {
                'average': 6.8
            },
            'tidal_amplitude': 3.0,
            'river_flow_dependency': 0.7
        }
        
        return params
    
    def simulate_intrusion(self,
                          region: str,
                          sea_level_rise: float = 0.0,
                          cyclone_frequency: float = 1.0,
                          years: int = 10,
                          hydrological_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Simulate salinity intrusion over time.
        
        Args:
            region: Coastal region to simulate
            sea_level_rise: Sea level rise in meters
            cyclone_frequency: Multiplier for cyclone frequency
            years: Number of years to simulate
            hydrological_data: Historical hydrological data
            
        Returns:
            Dictionary containing simulation results
        """
        if region not in self.salinity_params:
            raise ValueError(f"Region {region} not supported for salinity simulation")
        
        params = self.salinity_params[region]
        
        # Time array (monthly time steps)
        t = np.linspace(0, years, years * 12)
        
        # Simulate salinity levels over time
        salinity_results = self._simulate_salinity_levels(
            t, params, sea_level_rise, cyclone_frequency
        )
        
        # Simulate intrusion distance
        intrusion_results = self._simulate_intrusion_distance(
            t, params, sea_level_rise
        )
        
        # Calculate affected areas and populations
        impact_results = self._calculate_salinity_impacts(
            salinity_results, intrusion_results, params
        )
        
        # Agricultural impact assessment
        agricultural_impacts = self._assess_agricultural_impacts(
            salinity_results, params
        )
        
        # Water quality assessment
        water_quality = self._assess_water_quality_impacts(
            salinity_results, params
        )
        
        return {
            'salinity_levels': salinity_results,
            'intrusion_distance': intrusion_results,
            'impact_assessment': impact_results,
            'agricultural_impacts': agricultural_impacts,
            'water_quality_impacts': water_quality,
            'simulation_params': {
                'region': region,
                'sea_level_rise': sea_level_rise,
                'cyclone_frequency': cyclone_frequency,
                'years': years
            }
        }
    
    def _simulate_salinity_levels(self,
                                 t: np.ndarray,
                                 params: Dict,
                                 sea_level_rise: float,
                                 cyclone_frequency: float) -> Dict[str, np.ndarray]:
        """Simulate salinity levels over time."""
        # Base seasonal pattern
        seasonal_pattern = np.sin(2 * np.pi * t) * 0.5 + 0.5  # 0 to 1
        
        # Dry season salinity (high) to monsoon salinity (low)
        base_salinity = (
            params['baseline_salinity']['dry_season'] * (1 - seasonal_pattern) +
            params['baseline_salinity']['monsoon'] * seasonal_pattern
        )
        
        # Sea level rise effect (increases salinity)
        slr_effect = sea_level_rise * 2.5  # 2.5 ppt per meter of SLR
        
        # Long-term trend (increasing salinity)
        trend_effect = t * 0.1  # 0.1 ppt per year increase
        
        # Cyclone events (temporary spikes)
        cyclone_events = self._generate_cyclone_events(t, cyclone_frequency)
        cyclone_effect = cyclone_events * params['baseline_salinity']['peak_intrusion'] * 0.3
        
        # Combined salinity
        total_salinity = base_salinity + slr_effect + trend_effect + cyclone_effect
        
        # Ensure realistic bounds
        total_salinity = np.clip(total_salinity, 0, params['baseline_salinity']['peak_intrusion'])
        
        # Different measurement points
        results = {
            'time': t,
            'surface_water': total_salinity,
            'shallow_groundwater': total_salinity * 0.7,  # Slightly lower
            'deep_groundwater': total_salinity * 0.3,     # Much lower
            'soil_salinity': total_salinity * 0.4,        # Accumulated in soil
            'seasonal_pattern': seasonal_pattern,
            'cyclone_events': cyclone_events
        }
        
        return results
    
    def _generate_cyclone_events(self, t: np.ndarray, frequency_multiplier: float) -> np.ndarray:
        """Generate cyclone events that cause salinity spikes."""
        # Base cyclone frequency: ~3 per year during cyclone season
        base_frequency = 3.0
        adjusted_frequency = base_frequency * frequency_multiplier
        
        # Cyclone season (May-November)
        cyclone_season = np.sin(2 * np.pi * (t - 0.33)) > 0.3
        
        # Random cyclone events
        np.random.seed(42)  # For reproducibility
        random_events = np.random.poisson(adjusted_frequency / 12, len(t))
        
        # Apply seasonal mask
        cyclone_events = random_events * cyclone_season
        
        # Smooth the spikes (cyclones have lasting effects)
        from scipy.ndimage import gaussian_filter1d
        smoothed_events = gaussian_filter1d(cyclone_events.astype(float), sigma=2)
        
        return smoothed_events
    
    def _simulate_intrusion_distance(self,
                                   t: np.ndarray,
                                   params: Dict,
                                   sea_level_rise: float) -> Dict[str, np.ndarray]:
        """Simulate saltwater intrusion distance inland."""
        current_distance = params['intrusion_distance']['current']
        historical_distance = params['intrusion_distance']['historical']
        projected_distance = params['intrusion_distance']['projected_2050']
        
        # Base trend (linear increase)
        years_since_baseline = 2024 - 1970  # 54 years
        annual_increase = (current_distance - historical_distance) / years_since_baseline
        
        # Future projection
        base_intrusion = current_distance + annual_increase * t
        
        # Sea level rise effect (accelerates intrusion)
        slr_acceleration = sea_level_rise * 15  # 15 km per meter of SLR
        
        # Seasonal variation (further during dry season)
        seasonal_variation = np.sin(2 * np.pi * t) * 10  # ±10 km seasonal variation
        
        total_intrusion = base_intrusion + slr_acceleration - seasonal_variation
        
        # Ensure realistic bounds
        max_possible = projected_distance * 1.5
        total_intrusion = np.clip(total_intrusion, historical_distance, max_possible)
        
        return {
            'time': t,
            'intrusion_distance_km': total_intrusion,
            'base_trend': base_intrusion,
            'slr_effect': slr_acceleration,
            'seasonal_variation': seasonal_variation
        }
    
    def _calculate_salinity_impacts(self,
                                  salinity_results: Dict,
                                  intrusion_results: Dict,
                                  params: Dict) -> Dict[str, Any]:
        """Calculate impacts of salinity on population and area."""
        intrusion_distance = intrusion_results['intrusion_distance_km']
        surface_salinity = salinity_results['surface_water']
        
        # Calculate affected area (assumes linear relationship with intrusion distance)
        base_affected_area = params['affected_area']
        current_intrusion = params['intrusion_distance']['current']
        
        affected_area_ratio = intrusion_distance / current_intrusion
        affected_area = base_affected_area * affected_area_ratio
        
        # Calculate affected population
        base_population = params['population_affected']
        affected_population = base_population * affected_area_ratio
        
        # Categorize salinity levels
        freshwater_threshold = self.salinity_config['salinity_thresholds']['freshwater']
        brackish_threshold = self.salinity_config['salinity_thresholds']['brackish']
        saline_threshold = self.salinity_config['salinity_thresholds']['saline']
        
        # Calculate area under different salinity categories
        freshwater_area = affected_area * (surface_salinity < freshwater_threshold).mean()
        brackish_area = affected_area * ((surface_salinity >= freshwater_threshold) & 
                                       (surface_salinity < brackish_threshold)).mean()
        saline_area = affected_area * ((surface_salinity >= brackish_threshold) & 
                                     (surface_salinity < saline_threshold)).mean()
        hypersaline_area = affected_area * (surface_salinity >= saline_threshold).mean()
        
        return {
            'total_affected_area_km2': affected_area,
            'total_affected_population': affected_population,
            'area_by_salinity': {
                'freshwater_km2': freshwater_area,
                'brackish_km2': brackish_area,
                'saline_km2': saline_area,
                'hypersaline_km2': hypersaline_area
            },
            'population_by_risk': {
                'low_risk': affected_population * 0.3,
                'medium_risk': affected_population * 0.4,
                'high_risk': affected_population * 0.3
            },
            'max_intrusion_distance': np.max(intrusion_distance),
            'average_salinity': np.mean(surface_salinity)
        }
    
    def _assess_agricultural_impacts(self,
                                   salinity_results: Dict,
                                   params: Dict) -> Dict[str, Any]:
        """Assess agricultural impacts of salinity intrusion."""
        soil_salinity = salinity_results['soil_salinity']
        agricultural_area = params['agricultural_area']
        
        # Crop tolerance levels (from config)
        crop_tolerances = self.config['agriculture']['salinity_tolerance']
        
        # Calculate yield impacts for different crops
        crop_impacts = {}
        for crop, tolerance in crop_tolerances.items():
            # Yield reduction based on salinity stress
            salinity_stress = np.maximum(0, soil_salinity - tolerance)
            yield_reduction = np.minimum(0.8, salinity_stress * 0.1)  # Max 80% reduction
            
            crop_impacts[crop] = {
                'average_yield_reduction': np.mean(yield_reduction),
                'max_yield_reduction': np.max(yield_reduction),
                'affected_area_percent': (yield_reduction > 0.1).mean() * 100
            }
        
        # Calculate economic losses
        # Simplified calculation - in reality would use detailed crop economics
        crop_values = {
            'rice': 500,  # USD per hectare per season
            'wheat': 400,
            'barley': 300,
            'cotton': 800
        }
        
        total_economic_loss = 0
        for crop, impact in crop_impacts.items():
            if crop in crop_values:
                crop_area = agricultural_area * 0.25  # Assume equal distribution
                loss_per_hectare = crop_values[crop] * impact['average_yield_reduction']
                total_loss = crop_area * loss_per_hectare
                total_economic_loss += total_loss
                crop_impacts[crop]['economic_loss_usd'] = total_loss
        
        # Land abandonment risk
        severe_salinity_area = agricultural_area * (soil_salinity > 8.0).mean()
        abandonment_risk = severe_salinity_area / agricultural_area
        
        return {
            'crop_impacts': crop_impacts,
            'total_economic_loss_usd': total_economic_loss,
            'affected_agricultural_area_ha': agricultural_area,
            'severe_salinity_area_ha': severe_salinity_area,
            'land_abandonment_risk': abandonment_risk,
            'adaptation_urgency': 'Critical' if abandonment_risk > 0.3 else
                                 'High' if abandonment_risk > 0.15 else
                                 'Medium' if abandonment_risk > 0.05 else 'Low'
        }
    
    def _assess_water_quality_impacts(self,
                                    salinity_results: Dict,
                                    params: Dict) -> Dict[str, Any]:
        """Assess water quality impacts of salinity intrusion."""
        surface_salinity = salinity_results['surface_water']
        shallow_gw_salinity = salinity_results['shallow_groundwater']
        
        # WHO drinking water guidelines
        who_guideline = 0.5  # ppt for drinking water
        
        # Calculate population with access to safe drinking water
        safe_surface_water = (surface_salinity <= who_guideline).mean()
        safe_shallow_gw = (shallow_gw_salinity <= who_guideline).mean()
        
        affected_population = params['population_affected']
        
        # Health risk categories
        low_risk_pop = affected_population * safe_shallow_gw
        medium_risk_pop = affected_population * ((shallow_gw_salinity > who_guideline) & 
                                                (shallow_gw_salinity <= 5.0)).mean()
        high_risk_pop = affected_population * (shallow_gw_salinity > 5.0).mean()
        
        # Health impacts
        health_impacts = self._calculate_health_impacts(
            low_risk_pop, medium_risk_pop, high_risk_pop
        )
        
        # Water treatment requirements
        treatment_needs = self._assess_treatment_needs(
            surface_salinity, shallow_gw_salinity, affected_population
        )
        
        return {
            'safe_water_access': {
                'surface_water_percent': safe_surface_water * 100,
                'shallow_groundwater_percent': safe_shallow_gw * 100
            },
            'population_by_risk': {
                'low_risk': low_risk_pop,
                'medium_risk': medium_risk_pop,
                'high_risk': high_risk_pop
            },
            'health_impacts': health_impacts,
            'treatment_needs': treatment_needs,
            'average_salinity': {
                'surface_water': np.mean(surface_salinity),
                'shallow_groundwater': np.mean(shallow_gw_salinity)
            }
        }
    
    def _calculate_health_impacts(self,
                                low_risk_pop: float,
                                medium_risk_pop: float,
                                high_risk_pop: float) -> Dict[str, Any]:
        """Calculate health impacts from salinity exposure."""
        # Health impact rates (cases per 1000 people per year)
        impact_rates = {
            'hypertension': {'low': 5, 'medium': 15, 'high': 30},
            'kidney_disease': {'low': 1, 'medium': 5, 'high': 12},
            'skin_problems': {'low': 10, 'medium': 25, 'high': 45},
            'pregnancy_complications': {'low': 2, 'medium': 8, 'high': 18}
        }
        
        health_cases = {}
        total_health_cost = 0
        
        # Cost per case per year (USD)
        case_costs = {
            'hypertension': 200,
            'kidney_disease': 1500,
            'skin_problems': 50,
            'pregnancy_complications': 800
        }
        
        for condition, rates in impact_rates.items():
            cases_low = (low_risk_pop / 1000) * rates['low']
            cases_medium = (medium_risk_pop / 1000) * rates['medium']
            cases_high = (high_risk_pop / 1000) * rates['high']
            
            total_cases = cases_low + cases_medium + cases_high
            condition_cost = total_cases * case_costs[condition]
            
            health_cases[condition] = {
                'total_cases': total_cases,
                'annual_cost_usd': condition_cost
            }
            
            total_health_cost += condition_cost
        
        return {
            'health_cases_by_condition': health_cases,
            'total_annual_health_cost_usd': total_health_cost,
            'health_cost_per_capita': total_health_cost / (low_risk_pop + medium_risk_pop + high_risk_pop)
        }
    
    def _assess_treatment_needs(self,
                              surface_salinity: np.ndarray,
                              gw_salinity: np.ndarray,
                              population: float) -> Dict[str, Any]:
        """Assess water treatment needs for salinity removal."""
        # Treatment technology requirements based on salinity levels
        avg_surface_salinity = np.mean(surface_salinity)
        avg_gw_salinity = np.mean(gw_salinity)
        
        treatment_options = {
            'reverse_osmosis': {
                'salinity_range': (5.0, 40.0),
                'efficiency': 0.95,
                'cost_per_m3': 0.8,
                'energy_kwh_per_m3': 4.5
            },
            'electrodialysis': {
                'salinity_range': (1.0, 10.0),
                'efficiency': 0.85,
                'cost_per_m3': 0.5,
                'energy_kwh_per_m3': 2.5
            },
            'ion_exchange': {
                'salinity_range': (0.5, 5.0),
                'efficiency': 0.80,
                'cost_per_m3': 0.3,
                'energy_kwh_per_m3': 1.0
            }
        }
        
        # Determine appropriate treatment for each source
        recommended_treatments = []
        
        for source, salinity in [('surface', avg_surface_salinity), ('groundwater', avg_gw_salinity)]:
            for treatment, specs in treatment_options.items():
                if specs['salinity_range'][0] <= salinity <= specs['salinity_range'][1]:
                    recommended_treatments.append({
                        'source': source,
                        'treatment': treatment,
                        'salinity_level': salinity,
                        'efficiency': specs['efficiency'],
                        'cost_per_m3': specs['cost_per_m3']
                    })
                    break
        
        # Calculate treatment capacity needs
        per_capita_demand = 150  # liters per day
        total_demand_m3_per_day = population * per_capita_demand / 1000
        
        # Estimate costs
        if recommended_treatments:
            avg_cost = np.mean([t['cost_per_m3'] for t in recommended_treatments])
            annual_treatment_cost = total_demand_m3_per_day * 365 * avg_cost
        else:
            annual_treatment_cost = 0
        
        return {
            'recommended_treatments': recommended_treatments,
            'required_capacity_m3_per_day': total_demand_m3_per_day,
            'annual_treatment_cost_usd': annual_treatment_cost,
            'treatment_urgency': 'Critical' if avg_surface_salinity > 10 else
                               'High' if avg_surface_salinity > 5 else
                               'Medium' if avg_surface_salinity > 1 else 'Low'
        }
    
    def generate_adaptation_strategies(self, region: str, simulation_results: Dict) -> Dict[str, Any]:
        """Generate adaptation strategies for salinity management.
        
        Args:
            region: Target region
            simulation_results: Results from salinity simulation
            
        Returns:
            Comprehensive adaptation strategy recommendations
        """
        strategies = {
            'immediate_actions': [],
            'medium_term_investments': [],
            'long_term_planning': [],
            'cost_estimates': {},
            'priority_ranking': []
        }
        
        # Analyze severity
        avg_salinity = simulation_results['impact_assessment']['average_salinity']
        affected_area = simulation_results['impact_assessment']['total_affected_area_km2']
        agricultural_loss = simulation_results['agricultural_impacts']['total_economic_loss_usd']
        
        # Immediate actions (0-2 years)
        if avg_salinity > 10:
            strategies['immediate_actions'].extend([
                'Emergency freshwater supply through tankers',
                'Rainwater harvesting system installation',
                'Community-level water treatment units',
                'Salt-tolerant crop variety distribution'
            ])
        
        # Medium-term investments (2-10 years)
        strategies['medium_term_investments'].extend([
            'Desalination plant construction',
            'Improved drainage systems',
            'Freshwater pond excavation',
            'Mangrove restoration programs',
            'Alternative livelihood programs'
        ])
        
        # Long-term planning (10+ years)
        strategies['long_term_planning'].extend([
            'Managed retreat from highly saline areas',
            'Regional water transfer systems',
            'Climate-resilient infrastructure',
            'Integrated coastal zone management'
        ])
        
        # Cost estimates (simplified)
        strategies['cost_estimates'] = {
            'emergency_response': affected_area * 1000,  # $1000 per km²
            'water_treatment': simulation_results['water_quality_impacts']['treatment_needs']['annual_treatment_cost_usd'],
            'agricultural_adaptation': agricultural_loss * 0.3,  # 30% of losses for adaptation
            'infrastructure': affected_area * 50000  # $50,000 per km² for infrastructure
        }
        
        # Priority ranking based on cost-effectiveness
        strategies['priority_ranking'] = [
            'Rainwater harvesting',
            'Salt-tolerant crops',
            'Community water treatment',
            'Improved drainage',
            'Desalination plants'
        ]
        
        return strategies