"""Surface Water Management Module for Bangladesh.

This module handles river flow modeling, flood simulation, water allocation,
and surface water quality management for Bangladesh's river systems.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from loguru import logger
from scipy.integrate import odeint
from scipy.optimize import minimize
from scipy.stats import gamma, norm


class SurfaceWaterManager:
    """Manages surface water resources including rivers, floods, and allocation.
    
    This class implements hydrological modeling, flood simulation,
    and water allocation optimization for Bangladesh's river systems.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize surface water manager.
        
        Args:
            config: Configuration dictionary containing surface water parameters
        """
        self.config = config
        self.surface_water_config = config['surface_water']
        self.regions_config = config['regions']
        
        # Initialize river system parameters
        self.river_systems = self._initialize_river_systems()
        
        # Initialize flood parameters
        self.flood_params = self._initialize_flood_parameters()
        
        logger.info("Surface Water Manager initialized")
    
    def _initialize_river_systems(self) -> Dict[str, Dict]:
        """Initialize parameters for major river systems in Bangladesh."""
        systems = {}
        
        # Ganges-Padma River System
        systems['ganges_padma'] = {
            'length_km': 366,
            'catchment_area_km2': 46300,
            'average_discharge_m3s': 11000,
            'peak_discharge_m3s': 75000,
            'low_flow_m3s': 1200,
            'major_tributaries': ['Mahananda', 'Atrai', 'Chalan Beel'],
            'flow_variability': {
                'monsoon_multiplier': 6.8,
                'dry_season_multiplier': 0.11,
                'cv': 0.85  # Coefficient of variation
            },
            'water_quality': {
                'baseline_tds': 180,  # mg/L
                'pollution_load': 'High',
                'arsenic_risk': 'Medium'
            },
            'infrastructure': {
                'major_barrages': ['Farakka', 'Hardinge Bridge'],
                'irrigation_offtakes': 45,
                'flood_control_structures': 12
            },
            'transboundary': {
                'upstream_country': 'India',
                'dependency_ratio': 0.91,  # 91% flow from upstream
                'treaty_allocation': 'Ganges Water Treaty 1996'
            }
        }
        
        # Brahmaputra-Jamuna River System
        systems['brahmaputra_jamuna'] = {
            'length_km': 337,
            'catchment_area_km2': 47000,
            'average_discharge_m3s': 19800,
            'peak_discharge_m3s': 100000,
            'low_flow_m3s': 4800,
            'major_tributaries': ['Teesta', 'Dharla', 'Dudhkumar'],
            'flow_variability': {
                'monsoon_multiplier': 5.1,
                'dry_season_multiplier': 0.24,
                'cv': 0.72
            },
            'water_quality': {
                'baseline_tds': 120,
                'pollution_load': 'Medium',
                'arsenic_risk': 'Low'
            },
            'infrastructure': {
                'major_barrages': ['Bangabandhu Bridge', 'Jamuna Bridge'],
                'irrigation_offtakes': 38,
                'flood_control_structures': 18
            },
            'transboundary': {
                'upstream_country': 'India',
                'dependency_ratio': 0.83,
                'treaty_allocation': 'No formal treaty'
            }
        }
        
        # Meghna River System
        systems['meghna'] = {
            'length_km': 164,
            'catchment_area_km2': 82000,
            'average_discharge_m3s': 4000,
            'peak_discharge_m3s': 25000,
            'low_flow_m3s': 900,
            'major_tributaries': ['Surma', 'Kushiyara', 'Kalni'],
            'flow_variability': {
                'monsoon_multiplier': 6.3,
                'dry_season_multiplier': 0.23,
                'cv': 0.78
            },
            'water_quality': {
                'baseline_tds': 95,
                'pollution_load': 'Medium',
                'arsenic_risk': 'Low'
            },
            'infrastructure': {
                'major_barrages': ['Meghna Bridge'],
                'irrigation_offtakes': 25,
                'flood_control_structures': 8
            },
            'transboundary': {
                'upstream_country': 'India',
                'dependency_ratio': 0.75,
                'treaty_allocation': 'No formal treaty'
            }
        }
        
        # Chittagong Hill Tracts Rivers
        systems['chittagong_rivers'] = {
            'length_km': 200,  # Combined major rivers
            'catchment_area_km2': 13000,
            'average_discharge_m3s': 800,
            'peak_discharge_m3s': 5000,
            'low_flow_m3s': 150,
            'major_tributaries': ['Karnaphuli', 'Sangu', 'Matamuhuri'],
            'flow_variability': {
                'monsoon_multiplier': 6.7,
                'dry_season_multiplier': 0.19,
                'cv': 0.88
            },
            'water_quality': {
                'baseline_tds': 65,
                'pollution_load': 'Low',
                'arsenic_risk': 'Very Low'
            },
            'infrastructure': {
                'major_barrages': ['Kaptai Dam'],
                'irrigation_offtakes': 15,
                'flood_control_structures': 5
            },
            'transboundary': {
                'upstream_country': 'Myanmar/India',
                'dependency_ratio': 0.45,
                'treaty_allocation': 'Limited agreements'
            }
        }
        
        return systems
    
    def _initialize_flood_parameters(self) -> Dict[str, Dict]:
        """Initialize flood modeling parameters for different regions."""
        params = {}
        
        # National flood statistics
        params['national'] = {
            'annual_flood_probability': 0.26,  # 26% of country floods annually
            'severe_flood_probability': 0.06,  # 6% severe flooding
            'flood_duration_days': {
                'normal': 45,
                'severe': 68,
                'extreme': 95
            },
            'economic_damage_per_km2': {
                'normal': 50000,  # USD
                'severe': 150000,
                'extreme': 400000
            }
        }
        
        # Regional flood characteristics
        params['regions'] = {
            'northern': {
                'flood_frequency': 0.35,
                'flash_flood_risk': 'High',
                'main_causes': ['Brahmaputra overflow', 'Hill torrents'],
                'vulnerable_area_km2': 15000
            },
            'central': {
                'flood_frequency': 0.28,
                'flash_flood_risk': 'Medium',
                'main_causes': ['River confluence', 'Drainage congestion'],
                'vulnerable_area_km2': 25000
            },
            'southern': {
                'flood_frequency': 0.22,
                'flash_flood_risk': 'Low',
                'main_causes': ['Tidal surge', 'Cyclone-induced'],
                'vulnerable_area_km2': 18000
            },
            'eastern': {
                'flood_frequency': 0.31,
                'flash_flood_risk': 'Very High',
                'main_causes': ['Hill runoff', 'Sudden rainfall'],
                'vulnerable_area_km2': 8000
            }
        }
        
        return params
    
    def simulate_river_flow(self,
                          river_system: str,
                          years: int = 10,
                          climate_scenario: str = 'baseline',
                          upstream_flow_change: float = 0.0) -> Dict[str, Any]:
        """Simulate river flow over time.
        
        Args:
            river_system: River system to simulate
            years: Number of years to simulate
            climate_scenario: Climate change scenario
            upstream_flow_change: Change in upstream flow (fraction)
            
        Returns:
            Dictionary containing flow simulation results
        """
        if river_system not in self.river_systems:
            raise ValueError(f"River system {river_system} not supported")
        
        system_params = self.river_systems[river_system]
        
        # Time array (daily time steps)
        t = np.arange(0, years * 365)
        
        # Generate base flow pattern
        base_flow = self._generate_base_flow_pattern(t, system_params)
        
        # Apply climate change effects
        climate_modified_flow = self._apply_climate_effects(
            base_flow, climate_scenario, years
        )
        
        # Apply upstream flow changes
        if upstream_flow_change != 0.0:
            dependency = system_params['transboundary']['dependency_ratio']
            climate_modified_flow *= (1 + upstream_flow_change * dependency)
        
        # Calculate flow statistics
        flow_stats = self._calculate_flow_statistics(climate_modified_flow, system_params)
        
        # Assess water availability
        water_availability = self._assess_water_availability(
            climate_modified_flow, system_params
        )
        
        # Calculate environmental flows
        environmental_flows = self._calculate_environmental_flows(
            climate_modified_flow, system_params
        )
        
        return {
            'time_days': t,
            'daily_flow_m3s': climate_modified_flow,
            'flow_statistics': flow_stats,
            'water_availability': water_availability,
            'environmental_flows': environmental_flows,
            'simulation_params': {
                'river_system': river_system,
                'years': years,
                'climate_scenario': climate_scenario,
                'upstream_flow_change': upstream_flow_change
            }
        }
    
    def _generate_base_flow_pattern(self, t: np.ndarray, params: Dict) -> np.ndarray:
        """Generate realistic base flow pattern with seasonal variation."""
        # Seasonal pattern (monsoon peak around day 200, dry season minimum around day 50)
        seasonal_pattern = (
            0.5 * np.sin(2 * np.pi * (t % 365) / 365 - np.pi/2) + 0.5
        )
        
        # Apply flow variability
        monsoon_mult = params['flow_variability']['monsoon_multiplier']
        dry_mult = params['flow_variability']['dry_season_multiplier']
        
        # Scale seasonal pattern
        flow_multiplier = (
            dry_mult + (monsoon_mult - dry_mult) * seasonal_pattern
        )
        
        # Base flow with seasonal variation
        base_flow = params['average_discharge_m3s'] * flow_multiplier
        
        # Add stochastic variability
        np.random.seed(42)  # For reproducibility
        cv = params['flow_variability']['cv']
        noise = np.random.normal(1.0, cv * 0.3, len(t))
        
        # Apply noise with smoothing to avoid unrealistic spikes
        from scipy.ndimage import gaussian_filter1d
        smoothed_noise = gaussian_filter1d(noise, sigma=5)
        
        stochastic_flow = base_flow * smoothed_noise
        
        # Ensure realistic bounds
        min_flow = params['low_flow_m3s']
        max_flow = params['peak_discharge_m3s']
        
        return np.clip(stochastic_flow, min_flow, max_flow)
    
    def _apply_climate_effects(self,
                             base_flow: np.ndarray,
                             scenario: str,
                             years: int) -> np.ndarray:
        """Apply climate change effects to flow patterns."""
        # Climate change factors by scenario
        climate_factors = {
            'baseline': {'annual_change': 0.0, 'variability_change': 1.0},
            'rcp45': {'annual_change': -0.005, 'variability_change': 1.15},  # -0.5% per year, +15% variability
            'rcp85': {'annual_change': -0.012, 'variability_change': 1.35}   # -1.2% per year, +35% variability
        }
        
        if scenario not in climate_factors:
            scenario = 'baseline'
        
        factors = climate_factors[scenario]
        
        # Apply gradual annual change
        t_years = np.arange(len(base_flow)) / 365.0
        annual_trend = (1 + factors['annual_change']) ** t_years
        
        # Apply increased variability
        if factors['variability_change'] != 1.0:
            mean_flow = np.mean(base_flow)
            deviation = base_flow - mean_flow
            enhanced_deviation = deviation * factors['variability_change']
            modified_flow = mean_flow + enhanced_deviation
        else:
            modified_flow = base_flow.copy()
        
        return modified_flow * annual_trend
    
    def _calculate_flow_statistics(self, flow: np.ndarray, params: Dict) -> Dict[str, Any]:
        """Calculate comprehensive flow statistics."""
        # Basic statistics
        stats = {
            'mean_flow_m3s': np.mean(flow),
            'median_flow_m3s': np.median(flow),
            'std_flow_m3s': np.std(flow),
            'cv': np.std(flow) / np.mean(flow),
            'min_flow_m3s': np.min(flow),
            'max_flow_m3s': np.max(flow)
        }
        
        # Flow duration curve (percentiles)
        percentiles = [5, 10, 25, 50, 75, 90, 95]
        flow_duration = {}
        for p in percentiles:
            flow_duration[f'Q{p}'] = np.percentile(flow, 100 - p)
        
        stats['flow_duration_curve'] = flow_duration
        
        # Seasonal statistics
        daily_flows = flow.reshape(-1, 365)  # Reshape to years x days
        
        # Monsoon (June-September, days 152-273)
        monsoon_flows = daily_flows[:, 152:274].flatten()
        dry_season_flows = np.concatenate([
            daily_flows[:, :152].flatten(),  # Jan-May
            daily_flows[:, 274:].flatten()   # Oct-Dec
        ])
        
        stats['seasonal_statistics'] = {
            'monsoon': {
                'mean_flow_m3s': np.mean(monsoon_flows),
                'peak_flow_m3s': np.max(monsoon_flows),
                'total_volume_bcm': np.sum(monsoon_flows) * 86400 / 1e9  # Convert to BCM
            },
            'dry_season': {
                'mean_flow_m3s': np.mean(dry_season_flows),
                'min_flow_m3s': np.min(dry_season_flows),
                'total_volume_bcm': np.sum(dry_season_flows) * 86400 / 1e9
            }
        }
        
        # Annual volumes
        annual_volumes = []
        for year_flows in daily_flows:
            annual_volume_bcm = np.sum(year_flows) * 86400 / 1e9
            annual_volumes.append(annual_volume_bcm)
        
        stats['annual_volumes'] = {
            'mean_bcm': np.mean(annual_volumes),
            'std_bcm': np.std(annual_volumes),
            'min_bcm': np.min(annual_volumes),
            'max_bcm': np.max(annual_volumes)
        }
        
        return stats
    
    def _assess_water_availability(self, flow: np.ndarray, params: Dict) -> Dict[str, Any]:
        """Assess water availability for different uses."""
        # Water demand estimates (simplified)
        demands = {
            'irrigation': np.mean(flow) * 0.65,  # 65% for irrigation
            'domestic': np.mean(flow) * 0.08,    # 8% for domestic use
            'industrial': np.mean(flow) * 0.12,  # 12% for industry
            'environmental': np.mean(flow) * 0.15 # 15% environmental flow
        }
        
        total_demand = sum(demands.values())
        
        # Calculate availability metrics
        availability = {
            'total_demand_m3s': total_demand,
            'demand_by_sector': demands,
            'supply_reliability': (flow >= total_demand).mean(),
            'deficit_frequency': (flow < total_demand).mean(),
            'average_deficit_m3s': np.mean(np.maximum(0, total_demand - flow)),
            'maximum_deficit_m3s': np.max(np.maximum(0, total_demand - flow))
        }
        
        # Seasonal availability
        daily_flows = flow.reshape(-1, 365)
        monsoon_flows = daily_flows[:, 152:274].flatten()
        dry_season_flows = np.concatenate([
            daily_flows[:, :152].flatten(),
            daily_flows[:, 274:].flatten()
        ])
        
        availability['seasonal_availability'] = {
            'monsoon_reliability': (monsoon_flows >= total_demand).mean(),
            'dry_season_reliability': (dry_season_flows >= total_demand).mean(),
            'critical_period_deficit': np.mean(np.maximum(0, total_demand - dry_season_flows))
        }
        
        # Water stress indicators
        availability['stress_indicators'] = {
            'water_stress_index': total_demand / np.mean(flow),
            'scarcity_risk': 'High' if total_demand / np.mean(flow) > 0.4 else
                           'Medium' if total_demand / np.mean(flow) > 0.2 else 'Low',
            'adaptation_priority': 'Critical' if availability['supply_reliability'] < 0.8 else
                                 'High' if availability['supply_reliability'] < 0.9 else 'Medium'
        }
        
        return availability
    
    def _calculate_environmental_flows(self, flow: np.ndarray, params: Dict) -> Dict[str, Any]:
        """Calculate environmental flow requirements."""
        # Environmental flow methods
        methods = {}
        
        # Tennant Method (percentage of mean annual flow)
        mean_flow = np.mean(flow)
        methods['tennant'] = {
            'excellent': mean_flow * 0.60,
            'good': mean_flow * 0.40,
            'fair': mean_flow * 0.30,
            'poor': mean_flow * 0.20,
            'minimum': mean_flow * 0.10
        }
        
        # Flow Duration Curve Method (Q90 - flow exceeded 90% of time)
        q90 = np.percentile(flow, 10)  # 90% exceedance
        methods['q90_method'] = q90
        
        # Seasonal environmental flows
        daily_flows = flow.reshape(-1, 365)
        monsoon_flows = daily_flows[:, 152:274].flatten()
        dry_season_flows = np.concatenate([
            daily_flows[:, :152].flatten(),
            daily_flows[:, 274:].flatten()
        ])
        
        methods['seasonal_requirements'] = {
            'monsoon_minimum': np.percentile(monsoon_flows, 20),  # 80% exceedance
            'dry_season_minimum': np.percentile(dry_season_flows, 10),  # 90% exceedance
            'spawning_season': mean_flow * 0.50  # Higher flows for fish spawning
        }
        
        # Recommended environmental flow
        recommended_eflow = max(
            methods['tennant']['fair'],
            methods['q90_method'],
            methods['seasonal_requirements']['dry_season_minimum']
        )
        
        # Assessment of current environmental flow status
        eflow_deficit = np.mean(np.maximum(0, recommended_eflow - flow))
        eflow_compliance = (flow >= recommended_eflow).mean()
        
        return {
            'methods': methods,
            'recommended_eflow_m3s': recommended_eflow,
            'current_compliance_rate': eflow_compliance,
            'average_deficit_m3s': eflow_deficit,
            'ecological_risk': 'High' if eflow_compliance < 0.7 else
                             'Medium' if eflow_compliance < 0.85 else 'Low'
        }
    
    def simulate_flood_events(self,
                            region: str,
                            years: int = 10,
                            climate_scenario: str = 'baseline') -> Dict[str, Any]:
        """Simulate flood events for a specific region.
        
        Args:
            region: Region to simulate floods for
            years: Number of years to simulate
            climate_scenario: Climate change scenario
            
        Returns:
            Dictionary containing flood simulation results
        """
        if region not in self.flood_params['regions']:
            raise ValueError(f"Region {region} not supported for flood simulation")
        
        region_params = self.flood_params['regions'][region]
        national_params = self.flood_params['national']
        
        # Generate flood events
        flood_events = self._generate_flood_events(
            years, region_params, climate_scenario
        )
        
        # Calculate flood impacts
        flood_impacts = self._calculate_flood_impacts(
            flood_events, region_params, national_params
        )
        
        # Assess flood risk
        risk_assessment = self._assess_flood_risk(
            flood_events, region_params
        )
        
        return {
            'flood_events': flood_events,
            'flood_impacts': flood_impacts,
            'risk_assessment': risk_assessment,
            'simulation_params': {
                'region': region,
                'years': years,
                'climate_scenario': climate_scenario
            }
        }
    
    def _generate_flood_events(self,
                             years: int,
                             region_params: Dict,
                             climate_scenario: str) -> List[Dict]:
        """Generate realistic flood events."""
        # Climate change effects on flood frequency
        climate_multipliers = {
            'baseline': 1.0,
            'rcp45': 1.25,  # 25% increase in flood frequency
            'rcp85': 1.55   # 55% increase in flood frequency
        }
        
        multiplier = climate_multipliers.get(climate_scenario, 1.0)
        adjusted_frequency = region_params['flood_frequency'] * multiplier
        
        events = []
        np.random.seed(42)  # For reproducibility
        
        for year in range(years):
            # Determine if flood occurs this year
            if np.random.random() < adjusted_frequency:
                # Determine flood severity
                severity_prob = np.random.random()
                if severity_prob < 0.15:  # 15% chance of severe flood
                    severity = 'severe'
                    duration = np.random.normal(68, 15)
                    affected_area_ratio = np.random.uniform(0.6, 0.9)
                elif severity_prob < 0.05:  # 5% chance of extreme flood
                    severity = 'extreme'
                    duration = np.random.normal(95, 20)
                    affected_area_ratio = np.random.uniform(0.8, 1.0)
                else:
                    severity = 'normal'
                    duration = np.random.normal(45, 10)
                    affected_area_ratio = np.random.uniform(0.3, 0.6)
                
                # Flood timing (mostly during monsoon)
                if np.random.random() < 0.85:  # 85% during monsoon
                    start_day = np.random.randint(152, 274)  # June-September
                else:
                    start_day = np.random.randint(1, 365)
                
                duration = max(7, int(duration))  # Minimum 7 days
                affected_area = region_params['vulnerable_area_km2'] * affected_area_ratio
                
                events.append({
                    'year': year,
                    'start_day': start_day,
                    'duration_days': duration,
                    'severity': severity,
                    'affected_area_km2': affected_area,
                    'peak_water_level_m': np.random.uniform(2, 8) if severity == 'normal' else
                                        np.random.uniform(6, 12) if severity == 'severe' else
                                        np.random.uniform(10, 18)
                })
        
        return events
    
    def _calculate_flood_impacts(self,
                               events: List[Dict],
                               region_params: Dict,
                               national_params: Dict) -> Dict[str, Any]:
        """Calculate economic and social impacts of flood events."""
        total_damage = 0
        total_affected_area = 0
        total_duration = 0
        
        damage_by_severity = {'normal': 0, 'severe': 0, 'extreme': 0}
        
        for event in events:
            severity = event['severity']
            affected_area = event['affected_area_km2']
            duration = event['duration_days']
            
            # Calculate economic damage
            damage_per_km2 = national_params['economic_damage_per_km2'][severity]
            event_damage = affected_area * damage_per_km2
            
            # Duration multiplier (longer floods cause more damage)
            duration_multiplier = 1 + (duration - 30) * 0.02  # 2% increase per day over 30
            event_damage *= max(0.5, duration_multiplier)
            
            total_damage += event_damage
            total_affected_area += affected_area
            total_duration += duration
            damage_by_severity[severity] += event_damage
        
        # Calculate average impacts
        num_events = len(events)
        if num_events > 0:
            avg_damage_per_event = total_damage / num_events
            avg_affected_area = total_affected_area / num_events
            avg_duration = total_duration / num_events
        else:
            avg_damage_per_event = avg_affected_area = avg_duration = 0
        
        # Estimate population impacts
        population_density = 1200  # people per km²
        affected_population = total_affected_area * population_density
        
        # Agricultural impacts
        agricultural_ratio = 0.65  # 65% of affected area is agricultural
        affected_agricultural_area = total_affected_area * agricultural_ratio
        crop_loss_per_ha = 800  # USD per hectare
        agricultural_damage = affected_agricultural_area * 100 * crop_loss_per_ha  # Convert km² to ha
        
        return {
            'total_economic_damage_usd': total_damage,
            'agricultural_damage_usd': agricultural_damage,
            'damage_by_severity': damage_by_severity,
            'total_affected_area_km2': total_affected_area,
            'affected_population': affected_population,
            'number_of_events': num_events,
            'average_impacts': {
                'damage_per_event_usd': avg_damage_per_event,
                'affected_area_per_event_km2': avg_affected_area,
                'duration_per_event_days': avg_duration
            },
            'annual_average_damage_usd': total_damage / max(1, len(set(e['year'] for e in events)))
        }
    
    def _assess_flood_risk(self, events: List[Dict], region_params: Dict) -> Dict[str, Any]:
        """Assess flood risk levels and return periods."""
        if not events:
            return {
                'risk_level': 'Low',
                'return_periods': {},
                'vulnerability_assessment': 'Minimal flood activity'
            }
        
        # Calculate return periods for different severities
        years_simulated = max(e['year'] for e in events) + 1
        
        severity_counts = {'normal': 0, 'severe': 0, 'extreme': 0}
        for event in events:
            severity_counts[event['severity']] += 1
        
        return_periods = {}
        for severity, count in severity_counts.items():
            if count > 0:
                return_periods[severity] = years_simulated / count
            else:
                return_periods[severity] = float('inf')
        
        # Overall risk assessment
        flood_frequency = len(events) / years_simulated
        
        if flood_frequency > 0.4:
            risk_level = 'Very High'
        elif flood_frequency > 0.3:
            risk_level = 'High'
        elif flood_frequency > 0.2:
            risk_level = 'Medium'
        elif flood_frequency > 0.1:
            risk_level = 'Low'
        else:
            risk_level = 'Very Low'
        
        # Vulnerability factors
        vulnerability_factors = {
            'flash_flood_risk': region_params['flash_flood_risk'],
            'drainage_capacity': 'Limited' if flood_frequency > 0.3 else 'Adequate',
            'early_warning_effectiveness': 'Needs improvement' if risk_level in ['High', 'Very High'] else 'Adequate',
            'infrastructure_resilience': 'Low' if return_periods.get('severe', float('inf')) < 10 else 'Medium'
        }
        
        return {
            'risk_level': risk_level,
            'flood_frequency': flood_frequency,
            'return_periods': return_periods,
            'vulnerability_factors': vulnerability_factors,
            'adaptation_priority': 'Critical' if risk_level == 'Very High' else
                                 'High' if risk_level == 'High' else
                                 'Medium' if risk_level == 'Medium' else 'Low'
        }
    
    def optimize_water_allocation(self,
                                river_system: str,
                                flow_data: np.ndarray,
                                demands: Dict[str, float]) -> Dict[str, Any]:
        """Optimize water allocation among competing uses.
        
        Args:
            river_system: River system for allocation
            flow_data: Historical or simulated flow data
            demands: Water demands by sector
            
        Returns:
            Optimized allocation strategy
        """
        # Define optimization problem
        def objective(allocation):
            """Objective function to minimize (negative utility)."""
            # Utility weights for different sectors
            weights = {
                'irrigation': 0.4,
                'domestic': 0.3,
                'industrial': 0.2,
                'environmental': 0.1
            }
            
            total_utility = 0
            for i, sector in enumerate(demands.keys()):
                if sector in weights:
                    # Diminishing returns utility function
                    utility = weights[sector] * np.sqrt(allocation[i] / demands[sector])
                    total_utility += utility
            
            return -total_utility  # Minimize negative utility
        
        # Constraints
        def flow_constraint(allocation):
            """Total allocation cannot exceed available flow."""
            return np.mean(flow_data) - np.sum(allocation)
        
        def minimum_allocation_constraint(allocation):
            """Minimum allocation for critical sectors."""
            constraints = []
            for i, sector in enumerate(demands.keys()):
                if sector == 'domestic':
                    # Domestic water has minimum 80% allocation
                    constraints.append(allocation[i] - 0.8 * demands[sector])
                elif sector == 'environmental':
                    # Environmental flow has minimum 60% allocation
                    constraints.append(allocation[i] - 0.6 * demands[sector])
            return constraints
        
        # Initial guess (proportional allocation)
        total_demand = sum(demands.values())
        available_flow = np.mean(flow_data)
        scaling_factor = min(1.0, available_flow / total_demand)
        
        x0 = [demand * scaling_factor for demand in demands.values()]
        
        # Bounds (0 to 120% of demand for each sector)
        bounds = [(0, demand * 1.2) for demand in demands.values()]
        
        # Solve optimization
        from scipy.optimize import minimize
        
        constraints = [
            {'type': 'ineq', 'fun': flow_constraint},
            {'type': 'ineq', 'fun': lambda x: minimum_allocation_constraint(x)}
        ]
        
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        # Process results
        if result.success:
            optimal_allocation = dict(zip(demands.keys(), result.x))
            
            # Calculate performance metrics
            total_allocated = sum(optimal_allocation.values())
            allocation_efficiency = total_allocated / available_flow
            
            # Satisfaction ratios
            satisfaction_ratios = {
                sector: optimal_allocation[sector] / demand
                for sector, demand in demands.items()
            }
            
            # Reliability assessment
            reliability = self._assess_allocation_reliability(
                optimal_allocation, flow_data
            )
            
            return {
                'optimal_allocation': optimal_allocation,
                'satisfaction_ratios': satisfaction_ratios,
                'allocation_efficiency': allocation_efficiency,
                'total_allocated_m3s': total_allocated,
                'available_flow_m3s': available_flow,
                'reliability_metrics': reliability,
                'optimization_success': True
            }
        else:
            return {
                'optimization_success': False,
                'error_message': result.message,
                'fallback_allocation': dict(zip(demands.keys(), x0))
            }
    
    def _assess_allocation_reliability(self,
                                     allocation: Dict[str, float],
                                     flow_data: np.ndarray) -> Dict[str, Any]:
        """Assess reliability of water allocation strategy."""
        total_allocation = sum(allocation.values())
        
        # Calculate reliability metrics
        reliability_metrics = {
            'supply_reliability': (flow_data >= total_allocation).mean(),
            'deficit_frequency': (flow_data < total_allocation).mean(),
            'average_deficit_m3s': np.mean(np.maximum(0, total_allocation - flow_data)),
            'maximum_deficit_m3s': np.max(np.maximum(0, total_allocation - flow_data)),
            'vulnerability': np.mean(np.maximum(0, total_allocation - flow_data)) / total_allocation
        }
        
        # Seasonal reliability
        daily_flows = flow_data.reshape(-1, 365)
        monsoon_flows = daily_flows[:, 152:274].flatten()
        dry_season_flows = np.concatenate([
            daily_flows[:, :152].flatten(),
            daily_flows[:, 274:].flatten()
        ])
        
        reliability_metrics['seasonal_reliability'] = {
            'monsoon': (monsoon_flows >= total_allocation).mean(),
            'dry_season': (dry_season_flows >= total_allocation).mean()
        }
        
        # Risk assessment
        if reliability_metrics['supply_reliability'] > 0.95:
            risk_level = 'Low'
        elif reliability_metrics['supply_reliability'] > 0.85:
            risk_level = 'Medium'
        elif reliability_metrics['supply_reliability'] > 0.70:
            risk_level = 'High'
        else:
            risk_level = 'Critical'
        
        reliability_metrics['risk_level'] = risk_level
        
        return reliability_metrics