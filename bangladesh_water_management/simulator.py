"""Main Water Resources Simulator for Bangladesh.

This module contains the primary WaterResourcesSimulator class that orchestrates
all water management modules and provides the main interface for running simulations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from loguru import logger

from bangladesh_water_management.modules.groundwater import GroundwaterManager
from bangladesh_water_management.modules.salinity import SalinityManager
from bangladesh_water_management.modules.surface_water import SurfaceWaterManager
from bangladesh_water_management.modules.agriculture import AgriculturalWaterManager
from bangladesh_water_management.modules.urban import UrbanWaterManager
from bangladesh_water_management.modules.economic import EconomicAnalyzer
from bangladesh_water_management.modules.policy import PolicyAnalyzer
from bangladesh_water_management.data.synthetic_data import SyntheticDataGenerator
from bangladesh_water_management.utils.validation import validate_scenario_params
from bangladesh_water_management.visualization.dashboard import create_dashboard


class WaterResourcesSimulator:
    """Main simulator class for Bangladesh water resources management.
    
    This class integrates all water management modules and provides a unified
    interface for running various water resource scenarios and simulations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the water resources simulator.
        
        Args:
            config: Configuration dictionary containing simulation parameters
        """
        self.config = config
        self.data_generator = SyntheticDataGenerator(config)
        
        # Initialize all management modules
        self.groundwater_manager = GroundwaterManager(config)
        self.salinity_manager = SalinityManager(config)
        self.surface_water_manager = SurfaceWaterManager(config)
        self.agricultural_manager = AgriculturalWaterManager(config)
        self.urban_manager = UrbanWaterManager(config)
        self.economics_manager = EconomicAnalyzer(config)
        self.policy_engine = PolicyAnalyzer(config)
        
        # Initialize synthetic data
        self._initialize_data()
        
        logger.info("Water Resources Simulator initialized successfully")
    
    def _initialize_data(self):
        """Initialize synthetic data for all modules."""
        logger.info("Generating synthetic data for simulation...")

        # Get station_ids from config for hydrological data
        station_ids = self.config.get('surface_water', {}).get('major_rivers', [])
        if not station_ids:
            logger.warning("No 'major_rivers' found in surface_water config for station_ids. Hydrological data generation might be limited.")
            # Provide a default list if none are found, or handle error appropriately
            station_ids = ['default_station_1', 'default_station_2'] # Example default

        # Generate base datasets
        self.hydrological_data = self.data_generator.generate_hydrological_data(station_ids=station_ids)
        
        # For other data types, ensure locations/districts/cities are passed if needed
        # Example for water quality (assuming it needs locations like meteorological)
        locations_for_wq = self.config.get('regions', {}).get('all', ['default_location'])
        self.water_quality_data = self.data_generator.generate_water_quality_data(locations=locations_for_wq)

        # Example for demand data (assuming it needs regions/cities)
        cities_for_demand = self.config.get('regions', {}).get('urban', ['default_city'])
        self.demand_data = self.data_generator.generate_urban_water_data(cities=cities_for_demand)

        # Economic data is handled by EconomicAnalyzer or DataLoader, not directly by SyntheticDataGenerator for general simulation setup.
        # self.economic_data = self.data_generator.generate_economic_data(regions=regions_for_econ)
        
        logger.info("Synthetic data generation completed")
    
    def run_groundwater_scenario(self, 
                               region: str,
                               years: int = 10,
                               extraction_rate: float = 1.0,
                               recharge_enhancement: float = 0.0) -> Dict[str, Any]:
        """Run groundwater depletion scenario.
        
        Args:
            region: Target region (e.g., 'barind_tract', 'dhaka_metro', 'coastal_southwest')
            years: Simulation period in years
            extraction_rate: Multiplier for current extraction rates
            recharge_enhancement: Additional recharge as fraction of current
            
        Returns:
            Dictionary containing simulation results
        """
        validate_scenario_params('groundwater', {
            'region': region,
            'years': years,
            'extraction_rate': extraction_rate
        })
        
        logger.info(f"Running groundwater scenario for {region} over {years} years")
        
        # Run groundwater simulation
        gw_results = self.groundwater_manager.simulate_depletion(
            region=region,
            years=years,
            extraction_multiplier=extraction_rate,
            recharge_enhancement=recharge_enhancement,
            hydrological_data=self.hydrological_data
        )
        
        # Calculate economic impacts
        economic_impacts = self.economics_manager.calculate_groundwater_impacts(
            gw_results, region
        )
        
        # Agricultural impacts
        ag_impacts = self.agricultural_manager.assess_groundwater_impacts(
            gw_results, region
        )
        
        return {
            'groundwater_results': gw_results,
            'economic_impacts': economic_impacts,
            'agricultural_impacts': ag_impacts,
            'scenario_params': {
                'region': region,
                'years': years,
                'extraction_rate': extraction_rate,
                'recharge_enhancement': recharge_enhancement
            }
        }
    
    def run_salinity_scenario(self,
                            region: str,
                            sea_level_rise: float = 0.0,
                            cyclone_frequency: float = 1.0,
                            years: int = 10) -> Dict[str, Any]:
        """Run coastal salinity intrusion scenario.
        
        Args:
            region: Coastal region to simulate
            sea_level_rise: Sea level rise in meters
            cyclone_frequency: Multiplier for cyclone frequency
            years: Simulation period in years
            
        Returns:
            Dictionary containing simulation results
        """
        validate_scenario_params('salinity', {
            'region': region,
            'sea_level_rise': sea_level_rise,
            'cyclone_frequency': cyclone_frequency
        })
        
        logger.info(f"Running salinity scenario for {region}")
        
        # Run salinity simulation
        salinity_results = self.salinity_manager.simulate_intrusion(
            region=region,
            sea_level_rise=sea_level_rise,
            cyclone_frequency=cyclone_frequency,
            years=years,
            hydrological_data=self.hydrological_data
        )
        
        # Calculate agricultural impacts
        ag_impacts = self.agricultural_manager.assess_salinity_impacts(
            salinity_results, region
        )
        
        # Health and economic impacts
        health_impacts = self.economics_manager.calculate_health_impacts(
            salinity_results, region
        )
        
        return {
            'salinity_results': salinity_results,
            'agricultural_impacts': ag_impacts,
            'health_impacts': health_impacts,
            'scenario_params': {
                'region': region,
                'sea_level_rise': sea_level_rise,
                'cyclone_frequency': cyclone_frequency,
                'years': years
            }
        }
    
    def run_integrated_scenario(self,
                              regions: List[str],
                              years: int = 20,
                              climate_scenario: str = 'moderate',
                              policy_interventions: Optional[Dict] = None) -> Dict[str, Any]:
        """Run comprehensive integrated water management scenario.
        
        Args:
            regions: List of regions to include in simulation
            years: Simulation period in years
            climate_scenario: Climate change scenario ('conservative', 'moderate', 'severe')
            policy_interventions: Dictionary of policy interventions to test
            
        Returns:
            Comprehensive simulation results
        """
        logger.info(f"Running integrated scenario for {len(regions)} regions over {years} years")
        
        results = {}
        
        for region in regions:
            # Run all relevant scenarios for each region
            region_results = {}
            
            # Groundwater analysis
            if region in self.config['regions']['groundwater_dependent']:
                gw_scenario = self.run_groundwater_scenario(region, years)
                region_results['groundwater'] = gw_scenario
            
            # Salinity analysis for coastal regions
            if region in self.config['regions']['coastal']:
                salinity_scenario = self.run_salinity_scenario(region, years=years)
                region_results['salinity'] = salinity_scenario
            
            # Surface water analysis
            sw_results = self.surface_water_manager.analyze_availability(
                region, years, self.hydrological_data
            )
            region_results['surface_water'] = sw_results
            
            # Urban water supply analysis
            if region in self.config['regions']['urban']:
                urban_results = self.urban_manager.analyze_supply_demand(
                    region, years, self.demand_data
                )
                region_results['urban'] = urban_results
            
            results[region] = region_results
        
        # Cross-regional analysis
        cross_regional = self._analyze_cross_regional_impacts(results)
        
        # Policy simulation if interventions specified
        policy_results = None
        if policy_interventions:
            policy_results = self.policy_engine.simulate_interventions(
                policy_interventions, results
            )
        
        return {
            'regional_results': results,
            'cross_regional_analysis': cross_regional,
            'policy_results': policy_results,
            'scenario_params': {
                'regions': regions,
                'years': years,
                'climate_scenario': climate_scenario,
                'policy_interventions': policy_interventions
            }
        }
    
    def _analyze_cross_regional_impacts(self, regional_results: Dict) -> Dict[str, Any]:
        """Analyze impacts that cross regional boundaries."""
        # Analyze upstream-downstream relationships
        # Water transfer opportunities
        # Regional inequality in water access
        # Economic spillover effects
        
        return {
            'water_transfer_potential': self._calculate_transfer_potential(regional_results),
            'regional_inequality': self._assess_regional_inequality(regional_results),
            'economic_spillovers': self._calculate_economic_spillovers(regional_results)
        }
    
    def _calculate_transfer_potential(self, results: Dict) -> Dict:
        """Calculate potential for inter-regional water transfers."""
        # Simplified calculation - in reality would be much more complex
        surplus_regions = []
        deficit_regions = []
        
        for region, data in results.items():
            if 'surface_water' in data:
                balance = data['surface_water'].get('annual_balance', 0)
                if balance > 0:
                    surplus_regions.append(region)
                else:
                    deficit_regions.append(region)
        
        return {
            'surplus_regions': surplus_regions,
            'deficit_regions': deficit_regions,
            'transfer_opportunities': len(surplus_regions) * len(deficit_regions)
        }
    
    def _assess_regional_inequality(self, results: Dict) -> Dict:
        """Assess inequality in water access across regions."""
        access_scores = {}
        for region, data in results.items():
            # Calculate composite access score
            score = 0
            if 'groundwater' in data:
                score += data['groundwater']['groundwater_results'].get('sustainability_index', 0.5)
            if 'urban' in data:
                score += data['urban'].get('service_coverage', 0.5)
            access_scores[region] = score / 2
        
        scores = list(access_scores.values())
        return {
            'gini_coefficient': self._calculate_gini(scores),
            'regional_scores': access_scores,
            'most_vulnerable': min(access_scores.keys(), key=lambda k: access_scores[k]),
            'best_served': max(access_scores.keys(), key=lambda k: access_scores[k])
        }
    
    def _calculate_economic_spillovers(self, results: Dict) -> Dict:
        """Calculate economic spillover effects between regions."""
        # Simplified spillover calculation
        total_economic_impact = 0
        regional_impacts = {}
        
        for region, data in results.items():
            impact = 0
            if 'groundwater' in data and 'economic_impacts' in data['groundwater']:
                impact += data['groundwater']['economic_impacts'].get('total_cost', 0)
            if 'salinity' in data and 'health_impacts' in data['salinity']:
                impact += data['salinity']['health_impacts'].get('total_cost', 0)
            
            regional_impacts[region] = impact
            total_economic_impact += impact
        
        return {
            'total_impact': total_economic_impact,
            'regional_impacts': regional_impacts,
            'spillover_coefficient': 0.15  # Assume 15% spillover
        }
    
    def _calculate_gini(self, values: List[float]) -> float:
        """Calculate Gini coefficient for inequality measurement."""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        n = len(values)
        cumsum = np.cumsum(sorted_values)
        return (n + 1 - 2 * sum((n + 1 - i) * y for i, y in enumerate(cumsum))) / (n * sum(values))
    
    def generate_policy_recommendations(self, simulation_results: Dict) -> Dict[str, Any]:
        """Generate policy recommendations based on simulation results."""
        return self.policy_engine.generate_recommendations(simulation_results)
    
    def create_dashboard(self, results: Dict) -> str:
        """Create interactive dashboard for simulation results."""
        return create_dashboard(results, self.config)
    
    def export_results(self, results: Dict, format: str = 'csv', output_dir: str = 'outputs') -> List[str]:
        """Export simulation results to various formats."""
        from bangladesh_water_management.utils.export import export_results
        return export_results(results, format, output_dir)