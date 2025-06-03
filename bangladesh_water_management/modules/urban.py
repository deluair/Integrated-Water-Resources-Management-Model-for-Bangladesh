"""Urban Water Management Module for Bangladesh.

This module handles municipal water supply, wastewater treatment, urban drainage,
and water infrastructure management for Bangladesh's urban areas.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from loguru import logger
from scipy.optimize import minimize
from scipy.stats import poisson, norm


class UrbanWaterManager:
    """Manages urban water systems including supply, treatment, and drainage.
    
    This class implements urban water demand modeling, infrastructure assessment,
    and service optimization for Bangladesh's cities and towns.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize urban water manager.
        
        Args:
            config: Configuration dictionary containing urban water parameters
        """
        self.config = config
        self.urban_config = config['urban']
        self.regions_config = config['regions']
        
        # Initialize urban centers data
        self.urban_centers = self._initialize_urban_centers()
        
        # Initialize infrastructure parameters
        self.infrastructure_params = self._initialize_infrastructure_parameters()
        
        # Initialize service standards
        self.service_standards = self._initialize_service_standards()
        
        logger.info("Urban Water Manager initialized")
    
    def _initialize_urban_centers(self) -> Dict[str, Dict]:
        """Initialize data for major urban centers in Bangladesh."""
        centers = {}
        
        # Dhaka Metropolitan Area
        centers['dhaka'] = {
            'population': 9500000,
            'area_km2': 1528,
            'population_density': 6217,  # per km²
            'growth_rate': 0.035,  # 3.5% annual
            'economic_status': 'High',
            'water_sources': {
                'groundwater': 0.78,  # 78% dependency
                'surface_water': 0.22
            },
            'infrastructure_age': {
                'distribution_network': 25,  # years
                'treatment_plants': 15,
                'pumping_stations': 20
            },
            'service_coverage': {
                'piped_water': 0.82,
                'sewerage': 0.25,
                'drainage': 0.65
            },
            'water_quality_issues': {
                'arsenic_contamination': 'Medium',
                'bacterial_contamination': 'High',
                'iron_manganese': 'High'
            },
            'climate_vulnerabilities': {
                'flooding': 'Very High',
                'waterlogging': 'High',
                'heat_stress': 'Medium'
            }
        }
        
        # Chittagong
        centers['chittagong'] = {
            'population': 2800000,
            'area_km2': 168,
            'population_density': 16667,
            'growth_rate': 0.028,
            'economic_status': 'High',
            'water_sources': {
                'groundwater': 0.45,
                'surface_water': 0.55
            },
            'infrastructure_age': {
                'distribution_network': 30,
                'treatment_plants': 18,
                'pumping_stations': 22
            },
            'service_coverage': {
                'piped_water': 0.75,
                'sewerage': 0.20,
                'drainage': 0.55
            },
            'water_quality_issues': {
                'salinity_intrusion': 'Medium',
                'industrial_pollution': 'High',
                'bacterial_contamination': 'Medium'
            },
            'climate_vulnerabilities': {
                'cyclones': 'High',
                'flooding': 'High',
                'salinity': 'Medium'
            }
        }
        
        # Khulna
        centers['khulna'] = {
            'population': 1200000,
            'area_km2': 59,
            'population_density': 20339,
            'growth_rate': 0.022,
            'economic_status': 'Medium',
            'water_sources': {
                'groundwater': 0.35,  # Limited due to salinity
                'surface_water': 0.65
            },
            'infrastructure_age': {
                'distribution_network': 35,
                'treatment_plants': 25,
                'pumping_stations': 28
            },
            'service_coverage': {
                'piped_water': 0.68,
                'sewerage': 0.15,
                'drainage': 0.45
            },
            'water_quality_issues': {
                'salinity_intrusion': 'Very High',
                'arsenic_contamination': 'Medium',
                'iron_manganese': 'High'
            },
            'climate_vulnerabilities': {
                'salinity_intrusion': 'Very High',
                'flooding': 'High',
                'cyclones': 'High'
            }
        }
        
        # Rajshahi
        centers['rajshahi'] = {
            'population': 850000,
            'area_km2': 96,
            'population_density': 8854,
            'growth_rate': 0.018,
            'economic_status': 'Medium',
            'water_sources': {
                'groundwater': 0.85,
                'surface_water': 0.15
            },
            'infrastructure_age': {
                'distribution_network': 28,
                'treatment_plants': 20,
                'pumping_stations': 25
            },
            'service_coverage': {
                'piped_water': 0.72,
                'sewerage': 0.18,
                'drainage': 0.50
            },
            'water_quality_issues': {
                'arsenic_contamination': 'High',
                'fluoride': 'Medium',
                'bacterial_contamination': 'Medium'
            },
            'climate_vulnerabilities': {
                'drought': 'High',
                'flooding': 'Medium',
                'heat_stress': 'High'
            }
        }
        
        # Sylhet
        centers['sylhet'] = {
            'population': 650000,
            'area_km2': 27,
            'population_density': 24074,
            'growth_rate': 0.025,
            'economic_status': 'Medium',
            'water_sources': {
                'groundwater': 0.70,
                'surface_water': 0.30
            },
            'infrastructure_age': {
                'distribution_network': 22,
                'treatment_plants': 12,
                'pumping_stations': 18
            },
            'service_coverage': {
                'piped_water': 0.78,
                'sewerage': 0.22,
                'drainage': 0.60
            },
            'water_quality_issues': {
                'iron_manganese': 'High',
                'bacterial_contamination': 'Medium',
                'turbidity': 'Medium'
            },
            'climate_vulnerabilities': {
                'flooding': 'Very High',
                'flash_floods': 'High',
                'waterlogging': 'High'
            }
        }
        
        return centers
    
    def _initialize_infrastructure_parameters(self) -> Dict[str, Dict]:
        """Initialize infrastructure parameters and standards."""
        params = {}
        
        # Water supply infrastructure
        params['water_supply'] = {
            'treatment_technologies': {
                'conventional': {
                    'capacity_range_mld': (10, 500),
                    'treatment_efficiency': 0.85,
                    'capital_cost_usd_per_mld': 2500000,
                    'operational_cost_usd_per_m3': 0.15,
                    'energy_kwh_per_m3': 0.8
                },
                'membrane_filtration': {
                    'capacity_range_mld': (5, 200),
                    'treatment_efficiency': 0.95,
                    'capital_cost_usd_per_mld': 4000000,
                    'operational_cost_usd_per_m3': 0.25,
                    'energy_kwh_per_m3': 1.2
                },
                'reverse_osmosis': {
                    'capacity_range_mld': (1, 50),
                    'treatment_efficiency': 0.98,
                    'capital_cost_usd_per_mld': 6000000,
                    'operational_cost_usd_per_m3': 0.45,
                    'energy_kwh_per_m3': 3.5
                }
            },
            'distribution_network': {
                'pipe_materials': {
                    'ductile_iron': {'lifespan_years': 80, 'cost_usd_per_km': 150000},
                    'pvc': {'lifespan_years': 50, 'cost_usd_per_km': 80000},
                    'hdpe': {'lifespan_years': 60, 'cost_usd_per_km': 100000}
                },
                'network_efficiency': 0.75,  # 25% losses
                'maintenance_cost_percent': 0.03  # 3% of capital cost annually
            },
            'storage_systems': {
                'elevated_tanks': {
                    'capacity_range_ml': (1, 50),
                    'cost_usd_per_ml': 800000,
                    'lifespan_years': 40
                },
                'ground_reservoirs': {
                    'capacity_range_ml': (5, 200),
                    'cost_usd_per_ml': 400000,
                    'lifespan_years': 50
                }
            }
        }
        
        # Wastewater infrastructure
        params['wastewater'] = {
            'treatment_technologies': {
                'activated_sludge': {
                    'treatment_efficiency': 0.85,
                    'capital_cost_usd_per_mld': 1800000,
                    'operational_cost_usd_per_m3': 0.12,
                    'energy_kwh_per_m3': 0.6
                },
                'lagoon_system': {
                    'treatment_efficiency': 0.70,
                    'capital_cost_usd_per_mld': 800000,
                    'operational_cost_usd_per_m3': 0.05,
                    'energy_kwh_per_m3': 0.1
                },
                'membrane_bioreactor': {
                    'treatment_efficiency': 0.95,
                    'capital_cost_usd_per_mld': 3500000,
                    'operational_cost_usd_per_m3': 0.20,
                    'energy_kwh_per_m3': 1.0
                }
            },
            'collection_network': {
                'sewer_pipes': {
                    'concrete': {'lifespan_years': 60, 'cost_usd_per_km': 200000},
                    'pvc': {'lifespan_years': 50, 'cost_usd_per_km': 120000}
                },
                'pumping_stations': {
                    'cost_usd_per_mld_capacity': 500000,
                    'operational_cost_usd_per_m3': 0.08
                }
            }
        }
        
        # Drainage infrastructure
        params['drainage'] = {
            'storm_drainage': {
                'design_standards': {
                    'return_period_years': 10,
                    'design_intensity_mm_per_hour': 75
                },
                'infrastructure_costs': {
                    'primary_drains_usd_per_km': 500000,
                    'secondary_drains_usd_per_km': 200000,
                    'pumping_stations_usd_per_cumec': 800000
                }
            },
            'retention_systems': {
                'detention_ponds': {
                    'cost_usd_per_m3': 50,
                    'maintenance_cost_percent': 0.02
                },
                'permeable_surfaces': {
                    'cost_usd_per_m2': 25,
                    'infiltration_rate_mm_per_hour': 15
                }
            }
        }
        
        return params
    
    def _initialize_service_standards(self) -> Dict[str, Dict]:
        """Initialize service level standards and benchmarks."""
        standards = {
            'water_supply': {
                'minimum_per_capita_lpd': 40,   # Liters per day
                'adequate_per_capita_lpd': 100,
                'optimal_per_capita_lpd': 150,
                'pressure_standards': {
                    'minimum_psi': 15,
                    'adequate_psi': 25
                },
                'quality_standards': {
                    'turbidity_ntu': 5,
                    'chlorine_residual_mg_per_l': 0.5,
                    'ph_range': (6.5, 8.5),
                    'arsenic_mg_per_l': 0.05,
                    'iron_mg_per_l': 1.0
                },
                'service_hours': {
                    'minimum_hours_per_day': 4,
                    'adequate_hours_per_day': 12,
                    'optimal_hours_per_day': 24
                }
            },
            'wastewater': {
                'collection_coverage_target': 0.80,
                'treatment_coverage_target': 0.75,
                'effluent_standards': {
                    'bod_mg_per_l': 30,
                    'cod_mg_per_l': 150,
                    'tss_mg_per_l': 30,
                    'nitrogen_mg_per_l': 15,
                    'phosphorus_mg_per_l': 2
                }
            },
            'drainage': {
                'design_return_period_years': 10,
                'maximum_ponding_hours': 2,
                'coverage_target': 0.90
            }
        }
        
        return standards
    
    def calculate_water_demand(self,
                             city: str,
                             projection_years: int = 20,
                             scenario: str = 'baseline') -> Dict[str, Any]:
        """Calculate current and projected water demand for a city.
        
        Args:
            city: Urban center name
            projection_years: Number of years to project
            scenario: Development scenario
            
        Returns:
            Water demand analysis and projections
        """
        if city not in self.urban_centers:
            raise ValueError(f"City {city} not found in urban centers database")
        
        city_data = self.urban_centers[city]
        
        # Current demand calculation
        current_demand = self._calculate_current_demand(city_data)
        
        # Future projections
        projections = self._project_future_demand(
            city_data, current_demand, projection_years, scenario
        )
        
        # Demand by sector
        sectoral_demand = self._calculate_sectoral_demand(city_data, current_demand)
        
        # Peak demand factors
        peak_factors = self._calculate_peak_factors(city_data)
        
        return {
            'city': city,
            'current_demand': current_demand,
            'projections': projections,
            'sectoral_breakdown': sectoral_demand,
            'peak_factors': peak_factors,
            'demand_drivers': {
                'population_growth': city_data['growth_rate'],
                'economic_development': city_data['economic_status'],
                'service_expansion': 1 - city_data['service_coverage']['piped_water']
            }
        }
    
    def _calculate_current_demand(self, city_data: Dict) -> Dict[str, float]:
        """Calculate current water demand for a city."""
        population = city_data['population']
        service_coverage = city_data['service_coverage']['piped_water']
        economic_status = city_data['economic_status']
        
        # Per capita consumption based on economic status and service level
        if economic_status == 'High':
            base_consumption_lpd = 120
        elif economic_status == 'Medium':
            base_consumption_lpd = 85
        else:
            base_consumption_lpd = 60
        
        # Adjust for service coverage (unserved population uses less)
        served_population = population * service_coverage
        unserved_population = population * (1 - service_coverage)
        
        served_consumption = served_population * base_consumption_lpd
        unserved_consumption = unserved_population * 25  # Basic needs only
        
        total_consumption_lpd = served_consumption + unserved_consumption
        
        # Convert to different units
        total_consumption_mld = total_consumption_lpd / 1000000  # Million liters per day
        total_consumption_m3_per_day = total_consumption_mld * 1000
        
        # Account for system losses
        network_efficiency = self.infrastructure_params['water_supply']['distribution_network']['network_efficiency']
        production_requirement_mld = total_consumption_mld / network_efficiency
        
        return {
            'total_consumption_mld': total_consumption_mld,
            'production_requirement_mld': production_requirement_mld,
            'served_population': served_population,
            'unserved_population': unserved_population,
            'per_capita_consumption_lpd': total_consumption_lpd / population,
            'system_losses_mld': production_requirement_mld - total_consumption_mld
        }
    
    def _project_future_demand(self,
                             city_data: Dict,
                             current_demand: Dict,
                             years: int,
                             scenario: str) -> Dict[str, Any]:
        """Project future water demand."""
        # Scenario-based growth factors
        scenario_factors = {
            'baseline': {'population_growth': 1.0, 'consumption_growth': 1.0, 'efficiency_improvement': 1.0},
            'high_growth': {'population_growth': 1.2, 'consumption_growth': 1.3, 'efficiency_improvement': 0.9},
            'sustainable': {'population_growth': 0.9, 'consumption_growth': 0.8, 'efficiency_improvement': 1.2}
        }
        
        factors = scenario_factors.get(scenario, scenario_factors['baseline'])
        
        # Annual projections
        projections = []
        base_population = city_data['population']
        base_growth_rate = city_data['growth_rate']
        base_demand = current_demand['production_requirement_mld']
        
        for year in range(1, years + 1):
            # Population projection
            adjusted_growth_rate = base_growth_rate * factors['population_growth']
            projected_population = base_population * (1 + adjusted_growth_rate) ** year
            
            # Per capita consumption changes
            consumption_growth = 0.02 * factors['consumption_growth']  # 2% annual increase
            consumption_factor = (1 + consumption_growth) ** year
            
            # Efficiency improvements
            efficiency_improvement = 0.01 * factors['efficiency_improvement']  # 1% annual improvement
            efficiency_factor = 1 / ((1 + efficiency_improvement) ** year)
            
            # Service coverage expansion
            current_coverage = city_data['service_coverage']['piped_water']
            target_coverage = min(0.95, current_coverage + 0.02 * year)  # 2% annual increase
            
            # Calculate projected demand
            population_factor = projected_population / base_population
            total_factor = population_factor * consumption_factor * efficiency_factor
            
            projected_demand = base_demand * total_factor
            
            projections.append({
                'year': year,
                'population': projected_population,
                'service_coverage': target_coverage,
                'demand_mld': projected_demand,
                'per_capita_lpd': (projected_demand * 1000000) / projected_population
            })
        
        # Summary statistics
        final_year = projections[-1]
        growth_summary = {
            'total_demand_growth_percent': ((final_year['demand_mld'] / base_demand) - 1) * 100,
            'population_growth_percent': ((final_year['population'] / base_population) - 1) * 100,
            'average_annual_demand_growth': ((final_year['demand_mld'] / base_demand) ** (1/years) - 1) * 100
        }
        
        return {
            'annual_projections': projections,
            'growth_summary': growth_summary,
            'scenario': scenario
        }
    
    def _calculate_sectoral_demand(self, city_data: Dict, current_demand: Dict) -> Dict[str, float]:
        """Calculate water demand by sector."""
        total_demand = current_demand['production_requirement_mld']
        
        # Sectoral distribution based on city characteristics
        if city_data['economic_status'] == 'High':
            sectors = {
                'domestic': 0.65,
                'commercial': 0.15,
                'industrial': 0.12,
                'institutional': 0.05,
                'other': 0.03
            }
        elif city_data['economic_status'] == 'Medium':
            sectors = {
                'domestic': 0.70,
                'commercial': 0.12,
                'industrial': 0.10,
                'institutional': 0.05,
                'other': 0.03
            }
        else:
            sectors = {
                'domestic': 0.75,
                'commercial': 0.08,
                'industrial': 0.08,
                'institutional': 0.06,
                'other': 0.03
            }
        
        # Calculate absolute demands
        sectoral_demand = {}
        for sector, fraction in sectors.items():
            sectoral_demand[sector] = total_demand * fraction
        
        return sectoral_demand
    
    def _calculate_peak_factors(self, city_data: Dict) -> Dict[str, float]:
        """Calculate peak demand factors."""
        # Peak factors vary by city size and characteristics
        population = city_data['population']
        
        if population > 5000000:  # Mega cities
            daily_peak_factor = 1.3
            hourly_peak_factor = 2.2
        elif population > 1000000:  # Large cities
            daily_peak_factor = 1.4
            hourly_peak_factor = 2.5
        else:  # Medium cities
            daily_peak_factor = 1.5
            hourly_peak_factor = 2.8
        
        return {
            'daily_peak_factor': daily_peak_factor,
            'hourly_peak_factor': hourly_peak_factor,
            'seasonal_variation': 1.2  # 20% seasonal variation
        }
    
    def assess_infrastructure_capacity(self,
                                     city: str,
                                     demand_projections: Dict) -> Dict[str, Any]:
        """Assess infrastructure capacity against demand projections.
        
        Args:
            city: Urban center name
            demand_projections: Water demand projections
            
        Returns:
            Infrastructure capacity assessment
        """
        city_data = self.urban_centers[city]
        
        # Current infrastructure capacity (estimated)
        current_capacity = self._estimate_current_capacity(city_data)
        
        # Capacity gaps analysis
        capacity_gaps = self._analyze_capacity_gaps(
            current_capacity, demand_projections
        )
        
        # Infrastructure condition assessment
        condition_assessment = self._assess_infrastructure_condition(city_data)
        
        # Investment requirements
        investment_needs = self._calculate_investment_needs(
            capacity_gaps, condition_assessment, city_data
        )
        
        return {
            'city': city,
            'current_capacity': current_capacity,
            'capacity_gaps': capacity_gaps,
            'condition_assessment': condition_assessment,
            'investment_needs': investment_needs,
            'priority_interventions': self._identify_priority_interventions(
                capacity_gaps, condition_assessment
            )
        }
    
    def _estimate_current_capacity(self, city_data: Dict) -> Dict[str, float]:
        """Estimate current infrastructure capacity."""
        population = city_data['population']
        service_coverage = city_data['service_coverage']
        
        # Estimate based on population and service coverage
        # These are simplified estimates - in practice would use detailed asset data
        
        # Water supply capacity
        served_population = population * service_coverage['piped_water']
        water_supply_capacity = served_population * 120 / 1000000  # 120 lpd, convert to MLD
        
        # Wastewater treatment capacity
        wastewater_capacity = served_population * service_coverage['sewerage'] * 80 / 1000000  # 80 lpd
        
        # Drainage capacity (estimated based on coverage)
        drainage_capacity_score = service_coverage['drainage']  # Relative score
        
        return {
            'water_supply_capacity_mld': water_supply_capacity,
            'wastewater_capacity_mld': wastewater_capacity,
            'drainage_capacity_score': drainage_capacity_score,
            'distribution_network_km': population / 1000 * 2,  # Rough estimate
            'sewer_network_km': population * service_coverage['sewerage'] / 1000 * 1.5
        }
    
    def _analyze_capacity_gaps(self,
                             current_capacity: Dict,
                             demand_projections: Dict) -> Dict[str, Any]:
        """Analyze capacity gaps over projection period."""
        projections = demand_projections['annual_projections']
        
        gaps = {
            'water_supply_gaps': [],
            'wastewater_gaps': [],
            'critical_years': {}
        }
        
        current_ws_capacity = current_capacity['water_supply_capacity_mld']
        current_ww_capacity = current_capacity['wastewater_capacity_mld']
        
        for projection in projections:
            year = projection['year']
            demand = projection['demand_mld']
            
            # Water supply gap
            ws_gap = max(0, demand - current_ws_capacity)
            gaps['water_supply_gaps'].append({
                'year': year,
                'demand_mld': demand,
                'capacity_mld': current_ws_capacity,
                'gap_mld': ws_gap,
                'gap_percent': (ws_gap / demand) * 100 if demand > 0 else 0
            })
            
            # Wastewater gap (assuming 80% of water supply becomes wastewater)
            ww_demand = demand * 0.8
            ww_gap = max(0, ww_demand - current_ww_capacity)
            gaps['wastewater_gaps'].append({
                'year': year,
                'demand_mld': ww_demand,
                'capacity_mld': current_ww_capacity,
                'gap_mld': ww_gap,
                'gap_percent': (ww_gap / ww_demand) * 100 if ww_demand > 0 else 0
            })
            
            # Identify critical years (when gaps first appear)
            if ws_gap > 0 and 'water_supply' not in gaps['critical_years']:
                gaps['critical_years']['water_supply'] = year
            
            if ww_gap > 0 and 'wastewater' not in gaps['critical_years']:
                gaps['critical_years']['wastewater'] = year
        
        return gaps
    
    def _assess_infrastructure_condition(self, city_data: Dict) -> Dict[str, Any]:
        """Assess current infrastructure condition."""
        infrastructure_age = city_data['infrastructure_age']
        
        # Condition scoring based on age (0-100 scale)
        def age_to_condition(age, design_life):
            if age < design_life * 0.3:
                return 90  # Excellent
            elif age < design_life * 0.5:
                return 75  # Good
            elif age < design_life * 0.7:
                return 60  # Fair
            elif age < design_life * 0.9:
                return 40  # Poor
            else:
                return 20  # Critical
        
        condition_scores = {
            'distribution_network': age_to_condition(
                infrastructure_age['distribution_network'], 50
            ),
            'treatment_plants': age_to_condition(
                infrastructure_age['treatment_plants'], 30
            ),
            'pumping_stations': age_to_condition(
                infrastructure_age['pumping_stations'], 25
            )
        }
        
        # Overall condition assessment
        overall_condition = np.mean(list(condition_scores.values()))
        
        # Maintenance backlog estimation
        maintenance_backlog = self._estimate_maintenance_backlog(
            condition_scores, city_data
        )
        
        return {
            'condition_scores': condition_scores,
            'overall_condition': overall_condition,
            'condition_category': (
                'Excellent' if overall_condition >= 80 else
                'Good' if overall_condition >= 65 else
                'Fair' if overall_condition >= 50 else
                'Poor' if overall_condition >= 35 else
                'Critical'
            ),
            'maintenance_backlog_usd': maintenance_backlog,
            'replacement_priorities': self._identify_replacement_priorities(condition_scores)
        }
    
    def _estimate_maintenance_backlog(self, condition_scores: Dict, city_data: Dict) -> float:
        """Estimate maintenance backlog costs."""
        population = city_data['population']
        
        # Rough estimates based on population and condition
        base_maintenance_cost_per_capita = 50  # USD per capita
        
        # Adjust based on condition
        overall_condition = np.mean(list(condition_scores.values()))
        condition_multiplier = 2.0 - (overall_condition / 100)  # Higher cost for poor condition
        
        backlog = population * base_maintenance_cost_per_capita * condition_multiplier
        
        return backlog
    
    def _identify_replacement_priorities(self, condition_scores: Dict) -> List[str]:
        """Identify infrastructure replacement priorities."""
        priorities = []
        
        # Sort by condition score (worst first)
        sorted_infrastructure = sorted(
            condition_scores.items(),
            key=lambda x: x[1]
        )
        
        for infrastructure, score in sorted_infrastructure:
            if score < 50:  # Poor or critical condition
                priorities.append(infrastructure)
        
        return priorities
    
    def _calculate_investment_needs(self,
                                  capacity_gaps: Dict,
                                  condition_assessment: Dict,
                                  city_data: Dict) -> Dict[str, Any]:
        """Calculate infrastructure investment needs."""
        # Capacity expansion costs
        expansion_costs = self._calculate_expansion_costs(capacity_gaps)
        
        # Rehabilitation costs
        rehabilitation_costs = self._calculate_rehabilitation_costs(
            condition_assessment, city_data
        )
        
        # Total investment needs
        total_investment = {
            'capacity_expansion': expansion_costs,
            'rehabilitation': rehabilitation_costs,
            'total': {}
        }
        
        # Sum up totals
        for category in ['water_supply', 'wastewater', 'drainage']:
            expansion = expansion_costs.get(category, 0)
            rehabilitation = rehabilitation_costs.get(category, 0)
            total_investment['total'][category] = expansion + rehabilitation
        
        total_investment['grand_total'] = sum(total_investment['total'].values())
        
        # Financing analysis
        financing_analysis = self._analyze_financing_options(
            total_investment, city_data
        )
        
        return {
            'investment_breakdown': total_investment,
            'financing_analysis': financing_analysis,
            'implementation_timeline': self._develop_implementation_timeline(
                total_investment
            )
        }
    
    def _calculate_expansion_costs(self, capacity_gaps: Dict) -> Dict[str, float]:
        """Calculate costs for capacity expansion."""
        costs = {}
        
        # Water supply expansion
        ws_gaps = capacity_gaps['water_supply_gaps']
        if ws_gaps:
            max_gap = max(gap['gap_mld'] for gap in ws_gaps)
            if max_gap > 0:
                # Use conventional treatment cost
                treatment_cost = max_gap * self.infrastructure_params['water_supply']['treatment_technologies']['conventional']['capital_cost_usd_per_mld']
                distribution_cost = max_gap * 500000  # Simplified distribution cost
                costs['water_supply'] = treatment_cost + distribution_cost
        
        # Wastewater expansion
        ww_gaps = capacity_gaps['wastewater_gaps']
        if ww_gaps:
            max_gap = max(gap['gap_mld'] for gap in ww_gaps)
            if max_gap > 0:
                treatment_cost = max_gap * self.infrastructure_params['wastewater']['treatment_technologies']['activated_sludge']['capital_cost_usd_per_mld']
                collection_cost = max_gap * 800000  # Simplified collection cost
                costs['wastewater'] = treatment_cost + collection_cost
        
        return costs
    
    def _calculate_rehabilitation_costs(self,
                                      condition_assessment: Dict,
                                      city_data: Dict) -> Dict[str, float]:
        """Calculate rehabilitation and replacement costs."""
        costs = {}
        population = city_data['population']
        
        # Simplified rehabilitation cost estimates
        condition_scores = condition_assessment['condition_scores']
        
        for infrastructure, score in condition_scores.items():
            if score < 60:  # Needs rehabilitation
                if infrastructure == 'distribution_network':
                    # Cost based on population and network extent
                    network_km = population / 1000 * 2
                    cost_per_km = 100000  # Rehabilitation cost
                    costs['water_supply'] = costs.get('water_supply', 0) + network_km * cost_per_km
                
                elif infrastructure == 'treatment_plants':
                    # Treatment plant rehabilitation
                    capacity_mld = population * 120 / 1000000
                    cost_per_mld = 1000000  # Rehabilitation cost
                    costs['water_supply'] = costs.get('water_supply', 0) + capacity_mld * cost_per_mld
        
        return costs
    
    def _analyze_financing_options(self,
                                 investment_needs: Dict,
                                 city_data: Dict) -> Dict[str, Any]:
        """Analyze financing options for infrastructure investment."""
        total_investment = investment_needs['grand_total']
        population = city_data['population']
        
        # Per capita investment
        per_capita_investment = total_investment / population
        
        # Financing options
        financing_options = {
            'government_budget': {
                'share_percent': 40,
                'amount_usd': total_investment * 0.4,
                'feasibility': 'Medium'
            },
            'development_partners': {
                'share_percent': 30,
                'amount_usd': total_investment * 0.3,
                'feasibility': 'High'
            },
            'private_sector': {
                'share_percent': 20,
                'amount_usd': total_investment * 0.2,
                'feasibility': 'Medium'
            },
            'user_charges': {
                'share_percent': 10,
                'amount_usd': total_investment * 0.1,
                'feasibility': 'High'
            }
        }
        
        # Affordability analysis
        affordability = self._assess_affordability(total_investment, city_data)
        
        return {
            'total_investment_usd': total_investment,
            'per_capita_investment_usd': per_capita_investment,
            'financing_options': financing_options,
            'affordability_analysis': affordability
        }
    
    def _assess_affordability(self, investment: float, city_data: Dict) -> Dict[str, Any]:
        """Assess affordability of infrastructure investment."""
        population = city_data['population']
        economic_status = city_data['economic_status']
        
        # Estimated per capita income by economic status
        income_estimates = {
            'High': 3000,    # USD per year
            'Medium': 1500,
            'Low': 800
        }
        
        per_capita_income = income_estimates.get(economic_status, 1500)
        per_capita_investment = investment / population
        
        # Affordability ratios
        investment_to_income_ratio = per_capita_investment / per_capita_income
        
        # Affordability assessment
        if investment_to_income_ratio < 0.5:
            affordability_level = 'High'
        elif investment_to_income_ratio < 1.0:
            affordability_level = 'Medium'
        elif investment_to_income_ratio < 2.0:
            affordability_level = 'Low'
        else:
            affordability_level = 'Very Low'
        
        return {
            'per_capita_income_usd': per_capita_income,
            'per_capita_investment_usd': per_capita_investment,
            'investment_to_income_ratio': investment_to_income_ratio,
            'affordability_level': affordability_level,
            'recommended_financing_period_years': max(10, min(25, int(investment_to_income_ratio * 10)))
        }
    
    def _develop_implementation_timeline(self, investment_needs: Dict) -> Dict[str, List]:
        """Develop implementation timeline for infrastructure investments."""
        timeline = {
            'phase_1_0_5_years': [],
            'phase_2_5_10_years': [],
            'phase_3_10_15_years': []
        }
        
        total_investment = investment_needs['grand_total']
        
        # Priority-based phasing
        if 'water_supply' in investment_needs['total']:
            ws_investment = investment_needs['total']['water_supply']
            if ws_investment > total_investment * 0.3:  # Major investment
                timeline['phase_1_0_5_years'].append('Water supply capacity expansion')
                timeline['phase_2_5_10_years'].append('Water supply network rehabilitation')
            else:
                timeline['phase_1_0_5_years'].append('Water supply improvements')
        
        if 'wastewater' in investment_needs['total']:
            timeline['phase_2_5_10_years'].append('Wastewater treatment expansion')
            timeline['phase_3_10_15_years'].append('Sewer network expansion')
        
        if 'drainage' in investment_needs['total']:
            timeline['phase_1_0_5_years'].append('Critical drainage improvements')
            timeline['phase_3_10_15_years'].append('Comprehensive drainage system')
        
        return timeline
    
    def _identify_priority_interventions(self,
                                       capacity_gaps: Dict,
                                       condition_assessment: Dict) -> List[Dict]:
        """Identify priority interventions based on gaps and conditions."""
        interventions = []
        
        # Critical capacity gaps
        if capacity_gaps['critical_years'].get('water_supply', float('inf')) <= 5:
            interventions.append({
                'intervention': 'Emergency water supply capacity expansion',
                'priority': 'Critical',
                'timeframe': 'Immediate (0-2 years)',
                'rationale': 'Water supply deficit expected within 5 years'
            })
        
        # Poor infrastructure condition
        condition_scores = condition_assessment['condition_scores']
        for infrastructure, score in condition_scores.items():
            if score < 40:  # Poor condition
                interventions.append({
                    'intervention': f'{infrastructure.replace("_", " ").title()} rehabilitation',
                    'priority': 'High',
                    'timeframe': 'Short-term (2-5 years)',
                    'rationale': f'Infrastructure in poor condition (score: {score})'
                })
        
        # Service coverage gaps
        interventions.append({
            'intervention': 'Service coverage expansion',
            'priority': 'Medium',
            'timeframe': 'Medium-term (5-10 years)',
            'rationale': 'Expand services to unserved populations'
        })
        
        return interventions