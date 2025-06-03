"""Economic Analysis Module for Bangladesh Water Management.

This module handles economic valuation, cost-benefit analysis, financial modeling,
and investment optimization for water management projects and policies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from loguru import logger
from scipy.optimize import minimize, linprog
from scipy.stats import norm, lognorm
import warnings
warnings.filterwarnings('ignore')

from bangladesh_water_management.data import DataLoader


class EconomicAnalyzer:
    """Handles economic analysis for water management systems.
    
    This class implements economic valuation methods, cost-benefit analysis,
    financial modeling, and investment optimization for water projects.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Economic Analyzer with configuration.
        
        Args:
            config: Simulation configuration dictionary
        """
        self.economic_config = config['economics']
        self.data_loader = DataLoader(config)
        logger.info("Economic Analyzer initialized.")
        
        # Initialize economic parameters
        self.economic_parameters = self._initialize_economic_parameters()
        
        # Initialize valuation methods
        self.valuation_methods = self._initialize_valuation_methods()
        
        # Initialize cost databases
        self.cost_databases = self._initialize_cost_databases()
    
    def _initialize_economic_parameters(self) -> Dict[str, Any]:
        """Initialize key economic parameters for Bangladesh."""
        return {
            'macroeconomic': {
                'gdp_per_capita_usd': 2500,  # 2023 estimate
                'gdp_growth_rate': 0.065,    # 6.5% annual
                'inflation_rate': 0.055,     # 5.5% annual
                'discount_rate': 0.08,       # 8% real discount rate
                'social_discount_rate': 0.05, # 5% for social projects
                'exchange_rate_bdt_per_usd': 110,
                'poverty_line_usd_per_day': 2.15
            },
            'water_sector': {
                'water_tariff_usd_per_m3': {
                    'domestic': 0.15,
                    'commercial': 0.25,
                    'industrial': 0.30
                },
                'wastewater_tariff_usd_per_m3': 0.08,
                'connection_fee_usd': {
                    'water': 50,
                    'sewer': 75
                },
                'willingness_to_pay_multiplier': {
                    'urban_high_income': 2.5,
                    'urban_medium_income': 1.8,
                    'urban_low_income': 1.2,
                    'rural': 1.0
                }
            },
            'health_economics': {
                'value_of_statistical_life_usd': 150000,  # Adjusted for Bangladesh
                'disability_adjusted_life_year_usd': 1500,
                'healthcare_cost_per_episode_usd': {
                    'diarrheal_disease': 25,
                    'cholera': 150,
                    'typhoid': 200,
                    'hepatitis': 300,
                    'arsenicosis': 500
                },
                'productivity_loss_per_day_usd': {
                    'urban': 8,
                    'rural': 5
                }
            },
            'environmental_economics': {
                'carbon_price_usd_per_ton': 15,
                'ecosystem_service_values_usd_per_ha_per_year': {
                    'wetlands': 2500,
                    'mangroves': 3500,
                    'rivers': 1200,
                    'groundwater_recharge': 800
                },
                'biodiversity_value_multiplier': 1.5
            },
            'agricultural_economics': {
                'crop_prices_usd_per_ton': {
                    'rice': 400,
                    'wheat': 350,
                    'jute': 600,
                    'vegetables': 300,
                    'fish': 1200
                },
                'yield_loss_cost_usd_per_ha': {
                    'drought': 300,
                    'flood': 250,
                    'salinity': 200,
                    'waterlogging': 150
                },
                'irrigation_benefit_usd_per_ha': 200
            }
        }
    
    def _initialize_valuation_methods(self) -> Dict[str, Any]:
        """Initialize economic valuation methodologies."""
        return {
            'cost_benefit_analysis': {
                'analysis_period_years': 25,
                'discount_rates': {
                    'financial': 0.08,
                    'economic': 0.05,
                    'social': 0.03
                },
                'sensitivity_parameters': {
                    'cost_variation': 0.20,  # ±20%
                    'benefit_variation': 0.30,  # ±30%
                    'discount_rate_variation': 0.02  # ±2%
                }
            },
            'willingness_to_pay': {
                'survey_methods': ['contingent_valuation', 'choice_modeling'],
                'benefit_transfer_adjustment': 0.7,  # Adjustment factor for transferred values
                'income_elasticity': 0.8  # Income elasticity of demand for water
            },
            'replacement_cost': {
                'alternative_sources': {
                    'bottled_water_usd_per_m3': 2.5,
                    'water_trucking_usd_per_m3': 1.8,
                    'private_borewell_usd_per_m3': 0.8
                }
            },
            'hedonic_pricing': {
                'property_value_premium_percent': {
                    'piped_water_access': 15,
                    'sewer_connection': 12,
                    'flood_protection': 20
                }
            }
        }
    
    def _initialize_cost_databases(self) -> Dict[str, Any]:
        """Initialize cost databases for different water infrastructure."""
        return {
            'capital_costs': {
                'water_supply': {
                    'intake_structures': {
                        'river_intake_usd_per_mld': 200000,
                        'groundwater_wellfield_usd_per_mld': 150000,
                        'reservoir_intake_usd_per_mld': 300000
                    },
                    'treatment_plants': {
                        'conventional_usd_per_mld': 2500000,
                        'membrane_filtration_usd_per_mld': 4000000,
                        'reverse_osmosis_usd_per_mld': 6000000,
                        'iron_removal_usd_per_mld': 800000,
                        'arsenic_removal_usd_per_mld': 1200000
                    },
                    'distribution': {
                        'transmission_mains_usd_per_km': 200000,
                        'distribution_network_usd_per_km': 100000,
                        'service_connections_usd_per_connection': 150,
                        'pumping_stations_usd_per_mld': 500000,
                        'storage_tanks_usd_per_ml': 800000
                    }
                },
                'wastewater': {
                    'collection': {
                        'sewer_mains_usd_per_km': 250000,
                        'house_connections_usd_per_connection': 200,
                        'pumping_stations_usd_per_mld': 600000
                    },
                    'treatment': {
                        'activated_sludge_usd_per_mld': 1800000,
                        'lagoon_system_usd_per_mld': 800000,
                        'constructed_wetlands_usd_per_mld': 600000,
                        'membrane_bioreactor_usd_per_mld': 3500000
                    }
                },
                'drainage': {
                    'storm_drains_usd_per_km': 300000,
                    'retention_ponds_usd_per_m3': 50,
                    'pumping_stations_usd_per_cumec': 800000
                },
                'flood_protection': {
                    'embankments_usd_per_km': 500000,
                    'flood_walls_usd_per_km': 1200000,
                    'early_warning_systems_usd': 2000000
                }
            },
            'operational_costs': {
                'water_supply': {
                    'treatment_chemicals_usd_per_m3': 0.05,
                    'energy_usd_per_m3': 0.08,
                    'labor_usd_per_mld_per_year': 50000,
                    'maintenance_percent_of_capex': 0.03
                },
                'wastewater': {
                    'treatment_chemicals_usd_per_m3': 0.03,
                    'energy_usd_per_m3': 0.06,
                    'labor_usd_per_mld_per_year': 40000,
                    'maintenance_percent_of_capex': 0.04
                },
                'general': {
                    'administration_percent_of_opex': 0.15,
                    'insurance_percent_of_capex': 0.005,
                    'replacement_reserve_percent_of_capex': 0.02
                }
            }
        }
    
    def conduct_cost_benefit_analysis(self,
                                    project_data: Dict[str, Any],
                                    analysis_period: int = 25) -> Dict[str, Any]:
        """Conduct comprehensive cost-benefit analysis for water project.
        
        Args:
            project_data: Project specifications and parameters
            analysis_period: Analysis period in years
            
        Returns:
            Complete cost-benefit analysis results
        """
        # Calculate project costs
        costs = self._calculate_project_costs(project_data, analysis_period)
        
        # Calculate project benefits
        benefits = self._calculate_project_benefits(project_data, analysis_period)
        
        # Perform financial analysis
        financial_analysis = self._perform_financial_analysis(
            costs, benefits, analysis_period
        )
        
        # Perform economic analysis
        economic_analysis = self._perform_economic_analysis(
            costs, benefits, analysis_period
        )
        
        # Sensitivity analysis
        sensitivity_analysis = self._perform_sensitivity_analysis(
            project_data, costs, benefits, analysis_period
        )
        
        # Risk analysis
        risk_analysis = self._perform_risk_analysis(
            project_data, costs, benefits
        )
        
        return {
            'project_summary': {
                'name': project_data.get('name', 'Water Project'),
                'type': project_data.get('type', 'Unknown'),
                'location': project_data.get('location', 'Bangladesh'),
                'analysis_period_years': analysis_period
            },
            'costs': costs,
            'benefits': benefits,
            'financial_analysis': financial_analysis,
            'economic_analysis': economic_analysis,
            'sensitivity_analysis': sensitivity_analysis,
            'risk_analysis': risk_analysis,
            'recommendations': self._generate_recommendations(
                financial_analysis, economic_analysis, risk_analysis
            )
        }
    
    def _calculate_project_costs(self,
                               project_data: Dict[str, Any],
                               analysis_period: int) -> Dict[str, Any]:
        """Calculate all project costs over analysis period."""
        project_type = project_data.get('type', 'water_supply')
        capacity = project_data.get('capacity_mld', 10)
        
        # Capital costs
        capital_costs = self._calculate_capital_costs(project_data)
        
        # Operational costs
        operational_costs = self._calculate_operational_costs(
            project_data, analysis_period
        )
        
        # Replacement costs
        replacement_costs = self._calculate_replacement_costs(
            capital_costs, analysis_period
        )
        
        # Total costs by year
        annual_costs = self._distribute_costs_annually(
            capital_costs, operational_costs, replacement_costs, analysis_period
        )
        
        return {
            'capital_costs': capital_costs,
            'operational_costs': operational_costs,
            'replacement_costs': replacement_costs,
            'annual_costs': annual_costs,
            'total_undiscounted': sum(annual_costs),
            'total_discounted': self._calculate_present_value(
                annual_costs, self.economic_parameters['macroeconomic']['discount_rate']
            )
        }
    
    def _calculate_capital_costs(self, project_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate capital costs for project components."""
        project_type = project_data.get('type', 'water_supply')
        capacity_mld = project_data.get('capacity_mld', 10)
        network_km = project_data.get('network_km', 50)
        connections = project_data.get('connections', 5000)
        
        costs = {}
        cost_db = self.cost_databases['capital_costs']
        
        if project_type == 'water_supply':
            # Treatment plant
            treatment_type = project_data.get('treatment_type', 'conventional')
            costs['treatment_plant'] = (
                capacity_mld * cost_db['water_supply']['treatment_plants'][f'{treatment_type}_usd_per_mld']
            )
            
            # Distribution network
            costs['distribution_network'] = (
                network_km * cost_db['water_supply']['distribution']['distribution_network_usd_per_km']
            )
            
            # Service connections
            costs['service_connections'] = (
                connections * cost_db['water_supply']['distribution']['service_connections_usd_per_connection']
            )
            
            # Pumping stations
            costs['pumping_stations'] = (
                capacity_mld * cost_db['water_supply']['distribution']['pumping_stations_usd_per_mld']
            )
            
            # Storage
            storage_capacity_ml = capacity_mld * 0.3  # 30% of daily capacity
            costs['storage'] = (
                storage_capacity_ml * cost_db['water_supply']['distribution']['storage_tanks_usd_per_ml']
            )
        
        elif project_type == 'wastewater':
            # Treatment plant
            treatment_type = project_data.get('treatment_type', 'activated_sludge')
            costs['treatment_plant'] = (
                capacity_mld * cost_db['wastewater']['treatment'][f'{treatment_type}_usd_per_mld']
            )
            
            # Collection network
            costs['collection_network'] = (
                network_km * cost_db['wastewater']['collection']['sewer_mains_usd_per_km']
            )
            
            # House connections
            costs['house_connections'] = (
                connections * cost_db['wastewater']['collection']['house_connections_usd_per_connection']
            )
        
        elif project_type == 'flood_protection':
            length_km = project_data.get('length_km', 10)
            protection_type = project_data.get('protection_type', 'embankments')
            costs['flood_protection'] = (
                length_km * cost_db['flood_protection'][f'{protection_type}_usd_per_km']
            )
        
        # Add contingency and project management costs
        subtotal = sum(costs.values())
        costs['contingency'] = subtotal * 0.15  # 15% contingency
        costs['project_management'] = subtotal * 0.08  # 8% project management
        costs['design_supervision'] = subtotal * 0.05  # 5% design and supervision
        
        return costs
    
    def _calculate_operational_costs(self,
                                   project_data: Dict[str, Any],
                                   analysis_period: int) -> Dict[str, List[float]]:
        """Calculate operational costs over analysis period."""
        project_type = project_data.get('type', 'water_supply')
        capacity_mld = project_data.get('capacity_mld', 10)
        utilization_factor = project_data.get('utilization_factor', 0.8)
        
        annual_volume_m3 = capacity_mld * 1000 * 365 * utilization_factor
        cost_db = self.cost_databases['operational_costs']
        
        # Annual operational cost components
        if project_type in ['water_supply', 'wastewater']:
            annual_chemicals = annual_volume_m3 * cost_db[project_type]['treatment_chemicals_usd_per_m3']
            annual_energy = annual_volume_m3 * cost_db[project_type]['energy_usd_per_m3']
            annual_labor = capacity_mld * cost_db[project_type]['labor_usd_per_mld_per_year']
            
            base_annual_cost = annual_chemicals + annual_energy + annual_labor
        else:
            base_annual_cost = capacity_mld * 100000  # Simplified for other project types
        
        # Escalate costs over time
        inflation_rate = self.economic_parameters['macroeconomic']['inflation_rate']
        
        operational_costs = {
            'chemicals': [],
            'energy': [],
            'labor': [],
            'maintenance': [],
            'administration': [],
            'total': []
        }
        
        for year in range(analysis_period):
            escalation_factor = (1 + inflation_rate) ** year
            
            if project_type in ['water_supply', 'wastewater']:
                chemicals_cost = annual_chemicals * escalation_factor
                energy_cost = annual_energy * escalation_factor
                labor_cost = annual_labor * escalation_factor
            else:
                chemicals_cost = 0
                energy_cost = base_annual_cost * 0.3 * escalation_factor
                labor_cost = base_annual_cost * 0.4 * escalation_factor
            
            maintenance_cost = base_annual_cost * 0.2 * escalation_factor
            administration_cost = base_annual_cost * 0.1 * escalation_factor
            
            operational_costs['chemicals'].append(chemicals_cost)
            operational_costs['energy'].append(energy_cost)
            operational_costs['labor'].append(labor_cost)
            operational_costs['maintenance'].append(maintenance_cost)
            operational_costs['administration'].append(administration_cost)
            operational_costs['total'].append(
                chemicals_cost + energy_cost + labor_cost + maintenance_cost + administration_cost
            )
        
        return operational_costs
    
    def _calculate_replacement_costs(self,
                                   capital_costs: Dict[str, float],
                                   analysis_period: int) -> Dict[str, List[float]]:
        """Calculate replacement costs over analysis period."""
        # Component lifespans
        component_lifespans = {
            'treatment_plant': 20,
            'distribution_network': 40,
            'service_connections': 25,
            'pumping_stations': 15,
            'storage': 30,
            'collection_network': 50,
            'house_connections': 30
        }
        
        replacement_costs = {component: [0] * analysis_period for component in capital_costs.keys()}
        
        for component, cost in capital_costs.items():
            if component in component_lifespans:
                lifespan = component_lifespans[component]
                
                # Schedule replacements
                for year in range(lifespan, analysis_period, lifespan):
                    # Escalate replacement cost
                    inflation_rate = self.economic_parameters['macroeconomic']['inflation_rate']
                    escalated_cost = cost * (1 + inflation_rate) ** year
                    replacement_costs[component][year] = escalated_cost
        
        # Calculate total replacement costs by year
        total_replacement = []
        for year in range(analysis_period):
            year_total = sum(replacement_costs[comp][year] for comp in replacement_costs.keys())
            total_replacement.append(year_total)
        
        replacement_costs['total'] = total_replacement
        
        return replacement_costs
    
    def _distribute_costs_annually(self,
                                 capital_costs: Dict[str, float],
                                 operational_costs: Dict[str, List[float]],
                                 replacement_costs: Dict[str, List[float]],
                                 analysis_period: int) -> List[float]:
        """Distribute all costs annually over analysis period."""
        annual_costs = []
        
        # Capital costs in year 0
        total_capital = sum(capital_costs.values())
        
        for year in range(analysis_period):
            year_cost = 0
            
            # Capital costs (only in construction years, assume 2-year construction)
            if year < 2:
                year_cost += total_capital * 0.5  # 50% each year
            
            # Operational costs
            year_cost += operational_costs['total'][year]
            
            # Replacement costs
            year_cost += replacement_costs['total'][year]
            
            annual_costs.append(year_cost)
        
        return annual_costs
    
    def _calculate_project_benefits(self,
                                  project_data: Dict[str, Any],
                                  analysis_period: int) -> Dict[str, Any]:
        """Calculate all project benefits over analysis period."""
        project_type = project_data.get('type', 'water_supply')
        
        # Direct benefits
        direct_benefits = self._calculate_direct_benefits(project_data, analysis_period)
        
        # Health benefits
        health_benefits = self._calculate_health_benefits(project_data, analysis_period)
        
        # Economic benefits
        economic_benefits = self._calculate_economic_benefits(project_data, analysis_period)
        
        # Environmental benefits
        environmental_benefits = self._calculate_environmental_benefits(project_data, analysis_period)
        
        # Social benefits
        social_benefits = self._calculate_social_benefits(project_data, analysis_period)
        
        # Total benefits by year
        annual_benefits = self._aggregate_annual_benefits(
            direct_benefits, health_benefits, economic_benefits,
            environmental_benefits, social_benefits, analysis_period
        )
        
        return {
            'direct_benefits': direct_benefits,
            'health_benefits': health_benefits,
            'economic_benefits': economic_benefits,
            'environmental_benefits': environmental_benefits,
            'social_benefits': social_benefits,
            'annual_benefits': annual_benefits,
            'total_undiscounted': sum(annual_benefits),
            'total_discounted': self._calculate_present_value(
                annual_benefits, self.economic_parameters['macroeconomic']['discount_rate']
            )
        }
    
    def _calculate_direct_benefits(self,
                                 project_data: Dict[str, Any],
                                 analysis_period: int) -> Dict[str, List[float]]:
        """Calculate direct benefits from improved water services."""
        project_type = project_data.get('type', 'water_supply')
        beneficiaries = project_data.get('beneficiaries', 50000)
        service_improvement = project_data.get('service_improvement', 'new_connection')
        
        benefits = {
            'water_cost_savings': [0] * analysis_period,
            'time_savings': [0] * analysis_period,
            'reliability_benefits': [0] * analysis_period,
            'quality_benefits': [0] * analysis_period
        }
        
        if project_type == 'water_supply':
            # Water cost savings (compared to alternative sources)
            alternative_cost = self.valuation_methods['replacement_cost']['alternative_sources']['bottled_water_usd_per_m3']
            project_tariff = self.economic_parameters['water_sector']['water_tariff_usd_per_m3']['domestic']
            
            annual_consumption_m3_per_person = 30  # 82 liters per day
            annual_savings_per_person = (alternative_cost - project_tariff) * annual_consumption_m3_per_person
            
            # Time savings (for water collection)
            if service_improvement == 'new_connection':
                time_saved_hours_per_day = 2  # Hours saved from not collecting water
                value_of_time_usd_per_hour = self.economic_parameters['health_economics']['productivity_loss_per_day_usd']['urban'] / 8
                annual_time_savings_per_person = time_saved_hours_per_day * 365 * value_of_time_usd_per_hour
            else:
                annual_time_savings_per_person = 0
            
            # Reliability benefits (willingness to pay for reliable service)
            wtp_multiplier = self.economic_parameters['water_sector']['willingness_to_pay_multiplier']['urban_medium_income']
            annual_reliability_benefit_per_person = project_tariff * annual_consumption_m3_per_person * (wtp_multiplier - 1)
            
            for year in range(analysis_period):
                # Ramp up benefits over first 3 years
                ramp_factor = min(1.0, (year + 1) / 3)
                
                benefits['water_cost_savings'][year] = annual_savings_per_person * beneficiaries * ramp_factor
                benefits['time_savings'][year] = annual_time_savings_per_person * beneficiaries * ramp_factor
                benefits['reliability_benefits'][year] = annual_reliability_benefit_per_person * beneficiaries * ramp_factor
        
        return benefits
    
    def _calculate_health_benefits(self,
                                 project_data: Dict[str, Any],
                                 analysis_period: int) -> Dict[str, List[float]]:
        """Calculate health benefits from improved water and sanitation."""
        project_type = project_data.get('type', 'water_supply')
        beneficiaries = project_data.get('beneficiaries', 50000)
        
        benefits = {
            'reduced_healthcare_costs': [0] * analysis_period,
            'reduced_mortality': [0] * analysis_period,
            'reduced_morbidity': [0] * analysis_period,
            'productivity_gains': [0] * analysis_period
        }
        
        # Disease reduction rates by project type
        disease_reduction_rates = {
            'water_supply': {
                'diarrheal_disease': 0.25,  # 25% reduction
                'cholera': 0.40,
                'typhoid': 0.30,
                'hepatitis': 0.20
            },
            'wastewater': {
                'diarrheal_disease': 0.35,
                'cholera': 0.60,
                'typhoid': 0.45,
                'hepatitis': 0.30
            },
            'flood_protection': {
                'diarrheal_disease': 0.15,
                'cholera': 0.25,
                'vector_borne': 0.20
            }
        }
        
        if project_type in disease_reduction_rates:
            # Baseline disease incidence (per 1000 population per year)
            baseline_incidence = {
                'diarrheal_disease': 150,
                'cholera': 5,
                'typhoid': 8,
                'hepatitis': 3
            }
            
            health_costs = self.economic_parameters['health_economics']['healthcare_cost_per_episode_usd']
            productivity_loss = self.economic_parameters['health_economics']['productivity_loss_per_day_usd']['urban']
            
            for year in range(analysis_period):
                total_healthcare_savings = 0
                total_productivity_gains = 0
                
                for disease, reduction_rate in disease_reduction_rates[project_type].items():
                    if disease in baseline_incidence:
                        # Cases prevented
                        cases_prevented = (baseline_incidence[disease] / 1000) * beneficiaries * reduction_rate
                        
                        # Healthcare cost savings
                        healthcare_savings = cases_prevented * health_costs[disease]
                        total_healthcare_savings += healthcare_savings
                        
                        # Productivity gains (assume 3 days lost per episode)
                        productivity_gain = cases_prevented * productivity_loss * 3
                        total_productivity_gains += productivity_gain
                
                benefits['reduced_healthcare_costs'][year] = total_healthcare_savings
                benefits['productivity_gains'][year] = total_productivity_gains
        
        return benefits
    
    def _calculate_economic_benefits(self,
                                   project_data: Dict[str, Any],
                                   analysis_period: int) -> Dict[str, List[float]]:
        """Calculate broader economic benefits."""
        project_type = project_data.get('type', 'water_supply')
        beneficiaries = project_data.get('beneficiaries', 50000)
        
        benefits = {
            'property_value_increase': [0] * analysis_period,
            'business_development': [0] * analysis_period,
            'agricultural_benefits': [0] * analysis_period,
            'tourism_benefits': [0] * analysis_period
        }
        
        # Property value increases
        if project_type in ['water_supply', 'wastewater']:
            households = beneficiaries / 4.5  # Average household size
            average_property_value = 15000  # USD
            
            premium_rates = self.valuation_methods['hedonic_pricing']['property_value_premium_percent']
            if project_type == 'water_supply':
                premium = premium_rates['piped_water_access'] / 100
            else:
                premium = premium_rates['sewer_connection'] / 100
            
            total_property_value_increase = households * average_property_value * premium
            
            # Distribute over first 5 years
            for year in range(min(5, analysis_period)):
                benefits['property_value_increase'][year] = total_property_value_increase / 5
        
        # Business development benefits
        if project_type == 'water_supply':
            # Assume 5% of beneficiaries start small businesses due to reliable water
            new_businesses = beneficiaries * 0.05
            annual_revenue_per_business = 2000  # USD
            
            for year in range(2, analysis_period):  # Start from year 2
                benefits['business_development'][year] = new_businesses * annual_revenue_per_business * 0.1  # 10% profit margin
        
        return benefits
    
    def _calculate_environmental_benefits(self,
                                        project_data: Dict[str, Any],
                                        analysis_period: int) -> Dict[str, List[float]]:
        """Calculate environmental benefits."""
        project_type = project_data.get('type', 'water_supply')
        
        benefits = {
            'ecosystem_services': [0] * analysis_period,
            'carbon_sequestration': [0] * analysis_period,
            'biodiversity_conservation': [0] * analysis_period,
            'pollution_reduction': [0] * analysis_period
        }
        
        # Environmental benefits vary by project type
        if project_type == 'wastewater':
            # Pollution reduction benefits
            capacity_mld = project_data.get('capacity_mld', 10)
            annual_volume_m3 = capacity_mld * 1000 * 365
            
            # Assume treatment prevents pollution equivalent to $0.10 per m3 treated
            annual_pollution_reduction_benefit = annual_volume_m3 * 0.10
            
            for year in range(analysis_period):
                benefits['pollution_reduction'][year] = annual_pollution_reduction_benefit
        
        elif project_type == 'flood_protection':
            # Ecosystem services from preserved wetlands
            protected_area_ha = project_data.get('protected_area_ha', 100)
            ecosystem_value = self.economic_parameters['environmental_economics']['ecosystem_service_values_usd_per_ha_per_year']['wetlands']
            
            annual_ecosystem_benefit = protected_area_ha * ecosystem_value
            
            for year in range(analysis_period):
                benefits['ecosystem_services'][year] = annual_ecosystem_benefit
        
        return benefits
    
    def _calculate_social_benefits(self,
                                 project_data: Dict[str, Any],
                                 analysis_period: int) -> Dict[str, List[float]]:
        """Calculate social benefits."""
        beneficiaries = project_data.get('beneficiaries', 50000)
        
        benefits = {
            'gender_benefits': [0] * analysis_period,
            'education_benefits': [0] * analysis_period,
            'equity_benefits': [0] * analysis_period
        }
        
        # Gender benefits (time savings for women)
        women_beneficiaries = beneficiaries * 0.52  # 52% women
        time_saved_hours_per_day = 1.5  # Hours saved from water collection
        value_of_time = self.economic_parameters['health_economics']['productivity_loss_per_day_usd']['urban'] / 8
        
        annual_gender_benefit = women_beneficiaries * time_saved_hours_per_day * 365 * value_of_time
        
        # Education benefits (children can attend school instead of collecting water)
        school_age_children = beneficiaries * 0.25  # 25% school age
        additional_school_days = 30  # Additional days per year
        value_per_school_day = 2  # USD value of education per day
        
        annual_education_benefit = school_age_children * additional_school_days * value_per_school_day
        
        for year in range(analysis_period):
            benefits['gender_benefits'][year] = annual_gender_benefit
            benefits['education_benefits'][year] = annual_education_benefit
        
        return benefits
    
    def _aggregate_annual_benefits(self, *benefit_categories, analysis_period: int) -> List[float]:
        """Aggregate all benefit categories into annual totals."""
        annual_benefits = [0] * analysis_period
        
        for category in benefit_categories[:-1]:  # Exclude analysis_period
            for benefit_type, values in category.items():
                for year in range(analysis_period):
                    annual_benefits[year] += values[year]
        
        return annual_benefits
    
    def _perform_financial_analysis(self,
                                  costs: Dict[str, Any],
                                  benefits: Dict[str, Any],
                                  analysis_period: int) -> Dict[str, Any]:
        """Perform financial analysis of the project."""
        discount_rate = self.economic_parameters['macroeconomic']['discount_rate']
        
        # Calculate NPV
        npv = self._calculate_npv(
            costs['annual_costs'], benefits['annual_benefits'], discount_rate
        )
        
        # Calculate IRR
        irr = self._calculate_irr(
            costs['annual_costs'], benefits['annual_benefits']
        )
        
        # Calculate benefit-cost ratio
        bcr = self._calculate_bcr(
            costs['annual_costs'], benefits['annual_benefits'], discount_rate
        )
        
        # Calculate payback period
        payback_period = self._calculate_payback_period(
            costs['annual_costs'], benefits['annual_benefits']
        )
        
        return {
            'npv_usd': npv,
            'irr_percent': irr * 100 if irr else None,
            'bcr': bcr,
            'payback_period_years': payback_period,
            'discount_rate_percent': discount_rate * 100,
            'financial_viability': 'Viable' if npv > 0 and bcr > 1 else 'Not Viable'
        }
    
    def _perform_economic_analysis(self,
                                 costs: Dict[str, Any],
                                 benefits: Dict[str, Any],
                                 analysis_period: int) -> Dict[str, Any]:
        """Perform economic analysis using social discount rate."""
        social_discount_rate = self.economic_parameters['macroeconomic']['social_discount_rate']
        
        # Calculate economic NPV
        economic_npv = self._calculate_npv(
            costs['annual_costs'], benefits['annual_benefits'], social_discount_rate
        )
        
        # Calculate economic BCR
        economic_bcr = self._calculate_bcr(
            costs['annual_costs'], benefits['annual_benefits'], social_discount_rate
        )
        
        return {
            'economic_npv_usd': economic_npv,
            'economic_bcr': economic_bcr,
            'social_discount_rate_percent': social_discount_rate * 100,
            'economic_viability': 'Economically Viable' if economic_npv > 0 and economic_bcr > 1 else 'Not Economically Viable'
        }
    
    def _perform_sensitivity_analysis(self,
                                    project_data: Dict[str, Any],
                                    costs: Dict[str, Any],
                                    benefits: Dict[str, Any],
                                    analysis_period: int) -> Dict[str, Any]:
        """Perform sensitivity analysis on key parameters."""
        base_npv = self._calculate_npv(
            costs['annual_costs'], benefits['annual_benefits'],
            self.economic_parameters['macroeconomic']['discount_rate']
        )
        
        sensitivity_results = {}
        
        # Cost sensitivity
        for variation in [-0.2, -0.1, 0.1, 0.2]:  # ±20%, ±10%
            adjusted_costs = [cost * (1 + variation) for cost in costs['annual_costs']]
            npv = self._calculate_npv(
                adjusted_costs, benefits['annual_benefits'],
                self.economic_parameters['macroeconomic']['discount_rate']
            )
            sensitivity_results[f'cost_{variation:+.0%}'] = {
                'npv_usd': npv,
                'npv_change_percent': ((npv - base_npv) / base_npv) * 100 if base_npv != 0 else 0
            }
        
        # Benefit sensitivity
        for variation in [-0.3, -0.15, 0.15, 0.3]:  # ±30%, ±15%
            adjusted_benefits = [benefit * (1 + variation) for benefit in benefits['annual_benefits']]
            npv = self._calculate_npv(
                costs['annual_costs'], adjusted_benefits,
                self.economic_parameters['macroeconomic']['discount_rate']
            )
            sensitivity_results[f'benefit_{variation:+.0%}'] = {
                'npv_usd': npv,
                'npv_change_percent': ((npv - base_npv) / base_npv) * 100 if base_npv != 0 else 0
            }
        
        # Discount rate sensitivity
        for rate_change in [-0.02, -0.01, 0.01, 0.02]:  # ±2%, ±1%
            adjusted_rate = self.economic_parameters['macroeconomic']['discount_rate'] + rate_change
            npv = self._calculate_npv(
                costs['annual_costs'], benefits['annual_benefits'], adjusted_rate
            )
            sensitivity_results[f'discount_rate_{rate_change:+.1%}'] = {
                'npv_usd': npv,
                'npv_change_percent': ((npv - base_npv) / base_npv) * 100 if base_npv != 0 else 0
            }
        
        return {
            'base_npv_usd': base_npv,
            'sensitivity_results': sensitivity_results,
            'most_sensitive_parameter': max(
                sensitivity_results.keys(),
                key=lambda k: abs(sensitivity_results[k]['npv_change_percent'])
            )
        }
    
    def _perform_risk_analysis(self,
                             project_data: Dict[str, Any],
                             costs: Dict[str, Any],
                             benefits: Dict[str, Any]) -> Dict[str, Any]:
        """Perform risk analysis using Monte Carlo simulation."""
        # Risk factors and their distributions
        risk_factors = {
            'cost_overrun': {'distribution': 'lognormal', 'mean': 1.0, 'std': 0.3},
            'benefit_shortfall': {'distribution': 'normal', 'mean': 1.0, 'std': 0.2},
            'delay_risk': {'distribution': 'uniform', 'low': 0, 'high': 2},  # Years of delay
            'demand_risk': {'distribution': 'normal', 'mean': 1.0, 'std': 0.15}
        }
        
        # Monte Carlo simulation (simplified)
        n_simulations = 1000
        npv_results = []
        
        base_discount_rate = self.economic_parameters['macroeconomic']['discount_rate']
        
        for _ in range(n_simulations):
            # Sample risk factors
            cost_multiplier = np.random.lognormal(0, 0.3)  # Lognormal distribution
            benefit_multiplier = max(0.5, np.random.normal(1.0, 0.2))  # Normal, minimum 0.5
            
            # Adjust costs and benefits
            adjusted_costs = [cost * cost_multiplier for cost in costs['annual_costs']]
            adjusted_benefits = [benefit * benefit_multiplier for benefit in benefits['annual_benefits']]
            
            # Calculate NPV for this simulation
            npv = self._calculate_npv(adjusted_costs, adjusted_benefits, base_discount_rate)
            npv_results.append(npv)
        
        # Analyze results
        npv_array = np.array(npv_results)
        
        return {
            'mean_npv_usd': np.mean(npv_array),
            'std_npv_usd': np.std(npv_array),
            'probability_positive_npv': np.mean(npv_array > 0),
            'percentile_5_npv_usd': np.percentile(npv_array, 5),
            'percentile_95_npv_usd': np.percentile(npv_array, 95),
            'risk_level': (
                'Low' if np.mean(npv_array > 0) > 0.8 else
                'Medium' if np.mean(npv_array > 0) > 0.6 else
                'High'
            )
        }
    
    def _calculate_npv(self, costs: List[float], benefits: List[float], discount_rate: float) -> float:
        """Calculate Net Present Value."""
        npv = 0
        for year, (cost, benefit) in enumerate(zip(costs, benefits)):
            net_flow = benefit - cost
            discounted_flow = net_flow / ((1 + discount_rate) ** year)
            npv += discounted_flow
        return npv
    
    def _calculate_irr(self, costs: List[float], benefits: List[float]) -> Optional[float]:
        """Calculate Internal Rate of Return."""
        # Simplified IRR calculation using Newton-Raphson method
        def npv_function(rate):
            return sum(
                (benefit - cost) / ((1 + rate) ** year)
                for year, (cost, benefit) in enumerate(zip(costs, benefits))
            )
        
        # Try different rates to find IRR
        for rate in np.arange(0.01, 0.50, 0.01):
            if abs(npv_function(rate)) < 1000:  # Close to zero
                return rate
        
        return None
    
    def _calculate_bcr(self, costs: List[float], benefits: List[float], discount_rate: float) -> float:
        """Calculate Benefit-Cost Ratio."""
        present_value_benefits = self._calculate_present_value(benefits, discount_rate)
        present_value_costs = self._calculate_present_value(costs, discount_rate)
        
        return present_value_benefits / present_value_costs if present_value_costs > 0 else 0
    
    def _calculate_payback_period(self, costs: List[float], benefits: List[float]) -> Optional[float]:
        """Calculate payback period in years."""
        cumulative_net_flow = 0
        
        for year, (cost, benefit) in enumerate(zip(costs, benefits)):
            cumulative_net_flow += (benefit - cost)
            if cumulative_net_flow > 0:
                return year + 1
        
        return None
    
    def _calculate_present_value(self, cash_flows: List[float], discount_rate: float) -> float:
        """Calculate present value of cash flows."""
        return sum(
            cash_flow / ((1 + discount_rate) ** year)
            for year, cash_flow in enumerate(cash_flows)
        )
    
    def _generate_recommendations(self,
                               financial_analysis: Dict[str, Any],
                               economic_analysis: Dict[str, Any],
                               risk_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis results."""
        recommendations = []
        
        # Financial viability
        if financial_analysis['npv_usd'] > 0:
            recommendations.append("Project is financially viable with positive NPV")
        else:
            recommendations.append("Project requires subsidies or grants for financial viability")
        
        # Economic justification
        if economic_analysis['economic_npv_usd'] > 0:
            recommendations.append("Project is economically justified with positive social returns")
        else:
            recommendations.append("Project may not be economically justified - review benefits")
        
        # Risk assessment
        if risk_analysis['risk_level'] == 'High':
            recommendations.append("High risk project - implement strong risk mitigation measures")
        elif risk_analysis['risk_level'] == 'Medium':
            recommendations.append("Moderate risk - monitor key risk factors closely")
        else:
            recommendations.append("Low risk project with high probability of success")
        
        # BCR recommendations
        if financial_analysis['bcr'] > 2:
            recommendations.append("Excellent benefit-cost ratio - high priority for implementation")
        elif financial_analysis['bcr'] > 1:
            recommendations.append("Acceptable benefit-cost ratio - consider for implementation")
        else:
            recommendations.append("Poor benefit-cost ratio - redesign or reject project")
        
        return recommendations