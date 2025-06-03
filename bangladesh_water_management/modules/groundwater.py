"""Groundwater Management Module for Bangladesh.

This module handles groundwater depletion modeling, aquifer dynamics,
and sustainable extraction analysis for different regions of Bangladesh.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from loguru import logger
from scipy.integrate import odeint
from scipy.optimize import minimize


class GroundwaterManager:
    """Manages groundwater resources and depletion modeling.
    
    This class implements multi-layer aquifer modeling, depletion risk assessment,
    and sustainable extraction optimization for Bangladesh's groundwater systems.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize groundwater manager.
        
        Args:
            config: Configuration dictionary containing groundwater parameters
        """
        self.config = config
        self.gw_config = config['groundwater']
        self.regions_config = config['regions']
        
        # Initialize aquifer parameters for different regions
        self.aquifer_params = self._initialize_aquifer_parameters()
        
        logger.info("Groundwater Manager initialized")
    
    def _initialize_aquifer_parameters(self) -> Dict[str, Dict]:
        """Initialize aquifer parameters for different regions."""
        params = {}
        
        # Dhaka Metro - severe depletion
        params['dhaka_metro'] = {
            'layers': 3,
            'initial_depths': [20, 45, 80],  # meters below surface in 1970, 2000, 2024
            'current_depths': [240, 280, 320],  # current depths (2024)
            'transmissivity': [150, 200, 100],  # m²/day for each layer
            'storage_coefficient': [0.15, 0.08, 0.05],
            'recharge_rate': 50,  # mm/year
            'extraction_rate': 2.55e6,  # m³/day (current WASA extraction)
            'sustainable_yield': 1.2e6,  # m³/day
            'depletion_rate': 1.2  # m/year
        }
        
        # Barind Tract - moderate to severe depletion
        params['barind_tract'] = {
            'layers': 2,
            'initial_depths': [15, 35],
            'current_depths': [25, 50],
            'transmissivity': [120, 180],
            'storage_coefficient': [0.12, 0.06],
            'recharge_rate': 80,
            'extraction_rate': 800000,  # m³/day
            'sustainable_yield': 600000,
            'depletion_rate': 0.8
        }
        
        # Rajshahi - moderate depletion
        params['rajshahi'] = {
            'layers': 2,
            'initial_depths': [12, 30],
            'current_depths': [18, 42],
            'transmissivity': [100, 150],
            'storage_coefficient': [0.10, 0.07],
            'recharge_rate': 120,
            'extraction_rate': 400000,
            'sustainable_yield': 350000,
            'depletion_rate': 0.6
        }
        
        # Rangpur - moderate depletion
        params['rangpur'] = {
            'layers': 2,
            'initial_depths': [10, 25],
            'current_depths': [17, 35],
            'transmissivity': [90, 140],
            'storage_coefficient': [0.11, 0.08],
            'recharge_rate': 100,
            'extraction_rate': 300000,
            'sustainable_yield': 280000,
            'depletion_rate': 0.7
        }
        
        # Coastal regions - less groundwater dependent but salinity issues
        params['coastal_southwest'] = {
            'layers': 2,
            'initial_depths': [8, 20],
            'current_depths': [12, 28],
            'transmissivity': [60, 80],
            'storage_coefficient': [0.08, 0.05],
            'recharge_rate': 200,  # Higher due to coastal precipitation
            'extraction_rate': 150000,
            'sustainable_yield': 200000,
            'depletion_rate': 0.3,
            'salinity_affected': True
        }
        
        return params
    
    def simulate_depletion(self,
                          region: str,
                          years: int,
                          extraction_multiplier: float = 1.0,
                          recharge_enhancement: float = 0.0,
                          hydrological_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Simulate groundwater depletion over time.
        
        Args:
            region: Region to simulate
            years: Number of years to simulate
            extraction_multiplier: Multiplier for current extraction rates
            recharge_enhancement: Additional recharge as fraction of current
            hydrological_data: Historical hydrological data
            
        Returns:
            Dictionary containing simulation results
        """
        if region not in self.aquifer_params:
            raise ValueError(f"Region {region} not supported for groundwater simulation")
        
        params = self.aquifer_params[region]
        
        # Time array (monthly time steps)
        t = np.linspace(0, years, years * 12)
        
        # Initial conditions (current water levels)
        initial_levels = [-depth for depth in params['current_depths']]
        
        # Enhanced parameters
        extraction_rate = params['extraction_rate'] * extraction_multiplier
        recharge_rate = params['recharge_rate'] * (1 + recharge_enhancement)
        
        # Solve differential equation for each aquifer layer
        results = {}
        
        for layer in range(params['layers']):
            # Solve groundwater flow equation
            layer_results = self._solve_groundwater_equation(
                t, initial_levels[layer], extraction_rate / params['layers'],
                recharge_rate, params['transmissivity'][layer],
                params['storage_coefficient'][layer]
            )
            
            results[f'layer_{layer+1}'] = {
                'time': t,
                'water_levels': layer_results,
                'depths_below_surface': [-level for level in layer_results]
            }
        
        # Calculate aggregate metrics
        aggregate_results = self._calculate_aggregate_metrics(results, params, years)
        
        # Risk assessment
        risk_assessment = self._assess_depletion_risk(aggregate_results, params)
        
        # Sustainability analysis
        sustainability = self._analyze_sustainability(extraction_rate, params)
        
        return {
            'layer_results': results,
            'aggregate_metrics': aggregate_results,
            'risk_assessment': risk_assessment,
            'sustainability_analysis': sustainability,
            'simulation_params': {
                'region': region,
                'years': years,
                'extraction_multiplier': extraction_multiplier,
                'recharge_enhancement': recharge_enhancement
            }
        }
    
    def _solve_groundwater_equation(self,
                                   t: np.ndarray,
                                   initial_level: float,
                                   extraction: float,
                                   recharge: float,
                                   transmissivity: float,
                                   storage_coeff: float) -> np.ndarray:
        """Solve groundwater flow differential equation.
        
        Uses simplified Theis equation approach for confined aquifer.
        """
        def groundwater_ode(h, t):
            # Seasonal recharge variation
            month = int((t % 1) * 12)
            seasonal_factor = 3.5 if 5 <= month <= 9 else 0.2  # Monsoon vs dry season
            
            effective_recharge = recharge * seasonal_factor / 365.25  # mm/day to m/day
            extraction_per_area = extraction / 1000000  # Assume 1000 km² area
            
            # Simplified groundwater balance equation
            dhdt = (effective_recharge - extraction_per_area) / storage_coeff
            
            return dhdt
        
        # Solve ODE
        solution = odeint(groundwater_ode, initial_level, t)
        return solution.flatten()
    
    def _calculate_aggregate_metrics(self,
                                   layer_results: Dict,
                                   params: Dict,
                                   years: int) -> Dict[str, Any]:
        """Calculate aggregate metrics across all aquifer layers."""
        # Average water level across layers
        all_levels = []
        for layer_key in layer_results.keys():
            all_levels.append(layer_results[layer_key]['water_levels'])
        
        avg_levels = np.mean(all_levels, axis=0)
        avg_depths = -avg_levels
        
        # Calculate trends
        initial_depth = np.mean([params['current_depths']])
        final_depth = avg_depths[-1]
        total_depletion = final_depth - initial_depth
        annual_depletion_rate = total_depletion / years
        
        # Critical thresholds
        critical_depth = self.gw_config['critical_depth']
        time_to_critical = None
        if annual_depletion_rate > 0:
            remaining_depth = critical_depth - initial_depth
            if remaining_depth > 0:
                time_to_critical = remaining_depth / annual_depletion_rate
        
        return {
            'average_water_levels': avg_levels,
            'average_depths': avg_depths,
            'initial_depth': initial_depth,
            'final_depth': final_depth,
            'total_depletion': total_depletion,
            'annual_depletion_rate': annual_depletion_rate,
            'time_to_critical_depth': time_to_critical,
            'critical_depth_reached': final_depth >= critical_depth
        }
    
    def _assess_depletion_risk(self,
                              aggregate_results: Dict,
                              params: Dict) -> Dict[str, Any]:
        """Assess groundwater depletion risk levels."""
        annual_rate = aggregate_results['annual_depletion_rate']
        current_depth = aggregate_results['initial_depth']
        critical_depth = self.gw_config['critical_depth']
        
        # Risk categories
        if annual_rate <= 0:
            risk_level = 'Low'
            risk_score = 0.1
        elif annual_rate <= 0.5:
            risk_level = 'Moderate'
            risk_score = 0.4
        elif annual_rate <= 1.0:
            risk_level = 'High'
            risk_score = 0.7
        else:
            risk_level = 'Critical'
            risk_score = 0.9
        
        # Adjust for current depth
        depth_factor = min(current_depth / critical_depth, 1.0)
        adjusted_risk_score = risk_score * depth_factor
        
        return {
            'risk_level': risk_level,
            'risk_score': adjusted_risk_score,
            'annual_depletion_rate': annual_rate,
            'depth_factor': depth_factor,
            'mitigation_urgency': 'Immediate' if adjusted_risk_score > 0.8 else 
                                 'High' if adjusted_risk_score > 0.6 else
                                 'Medium' if adjusted_risk_score > 0.3 else 'Low'
        }
    
    def _analyze_sustainability(self,
                               extraction_rate: float,
                               params: Dict) -> Dict[str, Any]:
        """Analyze sustainability of current extraction rates."""
        sustainable_yield = params['sustainable_yield']
        sustainability_ratio = extraction_rate / sustainable_yield
        
        # Calculate required reduction
        if sustainability_ratio > 1.0:
            required_reduction = (extraction_rate - sustainable_yield) / extraction_rate
            sustainable = False
        else:
            required_reduction = 0.0
            sustainable = True
        
        # Sustainability index (0-1, where 1 is fully sustainable)
        sustainability_index = min(1.0, 1.0 / sustainability_ratio)
        
        return {
            'sustainable': sustainable,
            'sustainability_ratio': sustainability_ratio,
            'sustainability_index': sustainability_index,
            'current_extraction': extraction_rate,
            'sustainable_yield': sustainable_yield,
            'required_reduction_percent': required_reduction * 100,
            'excess_extraction': max(0, extraction_rate - sustainable_yield)
        }
    
    def optimize_extraction_strategy(self,
                                   region: str,
                                   years: int,
                                   target_sustainability: float = 0.8) -> Dict[str, Any]:
        """Optimize groundwater extraction strategy for sustainability.
        
        Args:
            region: Target region
            years: Planning horizon
            target_sustainability: Target sustainability index (0-1)
            
        Returns:
            Optimized extraction strategy
        """
        if region not in self.aquifer_params:
            raise ValueError(f"Region {region} not supported")
        
        params = self.aquifer_params[region]
        
        def objective(x):
            """Objective function: minimize deviation from target sustainability."""
            extraction_multiplier, recharge_enhancement = x
            
            # Run simulation
            results = self.simulate_depletion(
                region, years, extraction_multiplier, recharge_enhancement
            )
            
            sustainability_index = results['sustainability_analysis']['sustainability_index']
            
            # Penalty for being below target
            penalty = max(0, target_sustainability - sustainability_index) ** 2
            
            # Cost of interventions
            extraction_cost = max(0, extraction_multiplier - 1.0) * 0.1
            recharge_cost = recharge_enhancement * 0.5
            
            return penalty + extraction_cost + recharge_cost
        
        # Optimization constraints
        constraints = [
            {'type': 'ineq', 'fun': lambda x: x[0]},  # extraction_multiplier >= 0
            {'type': 'ineq', 'fun': lambda x: x[1]},  # recharge_enhancement >= 0
            {'type': 'ineq', 'fun': lambda x: 2.0 - x[0]},  # extraction_multiplier <= 2.0
            {'type': 'ineq', 'fun': lambda x: 1.0 - x[1]}   # recharge_enhancement <= 1.0
        ]
        
        # Initial guess
        x0 = [0.8, 0.2]  # Reduce extraction by 20%, increase recharge by 20%
        
        # Optimize
        result = minimize(objective, x0, method='SLSQP', constraints=constraints)
        
        if result.success:
            optimal_extraction, optimal_recharge = result.x
            
            # Run final simulation with optimal parameters
            optimal_results = self.simulate_depletion(
                region, years, optimal_extraction, optimal_recharge
            )
            
            return {
                'optimization_successful': True,
                'optimal_extraction_multiplier': optimal_extraction,
                'optimal_recharge_enhancement': optimal_recharge,
                'achieved_sustainability': optimal_results['sustainability_analysis']['sustainability_index'],
                'extraction_reduction_percent': (1 - optimal_extraction) * 100,
                'recharge_increase_percent': optimal_recharge * 100,
                'simulation_results': optimal_results
            }
        else:
            return {
                'optimization_successful': False,
                'error_message': result.message
            }
    
    def calculate_artificial_recharge_potential(self, region: str) -> Dict[str, Any]:
        """Calculate potential for artificial groundwater recharge.
        
        Args:
            region: Target region
            
        Returns:
            Artificial recharge potential analysis
        """
        if region not in self.aquifer_params:
            raise ValueError(f"Region {region} not supported")
        
        params = self.aquifer_params[region]
        
        # Estimate available surface area for recharge
        region_areas = {
            'dhaka_metro': 1500,  # km²
            'barind_tract': 8000,
            'rajshahi': 2400,
            'rangpur': 2300,
            'coastal_southwest': 5000
        }
        
        area = region_areas.get(region, 2000)
        
        # Recharge methods and their effectiveness
        recharge_methods = {
            'rainwater_harvesting': {
                'efficiency': 0.7,
                'cost_per_m3': 0.5,
                'potential_area_fraction': 0.1
            },
            'infiltration_basins': {
                'efficiency': 0.8,
                'cost_per_m3': 0.3,
                'potential_area_fraction': 0.05
            },
            'injection_wells': {
                'efficiency': 0.9,
                'cost_per_m3': 1.2,
                'potential_area_fraction': 0.02
            },
            'treated_wastewater': {
                'efficiency': 0.6,
                'cost_per_m3': 0.8,
                'potential_area_fraction': 0.03
            }
        }
        
        # Calculate potential for each method
        method_potentials = {}
        total_potential = 0
        
        for method, specs in recharge_methods.items():
            available_area = area * specs['potential_area_fraction']
            annual_rainfall = 1500  # mm (approximate for Bangladesh)
            
            potential_recharge = (
                available_area * 1000000 *  # Convert km² to m²
                annual_rainfall / 1000 *    # Convert mm to m
                specs['efficiency']
            )
            
            method_potentials[method] = {
                'potential_recharge_m3_per_year': potential_recharge,
                'available_area_km2': available_area,
                'efficiency': specs['efficiency'],
                'cost_per_m3': specs['cost_per_m3'],
                'total_annual_cost': potential_recharge * specs['cost_per_m3']
            }
            
            total_potential += potential_recharge
        
        # Compare with current deficit
        current_extraction = params['extraction_rate'] * 365.25  # Convert to annual
        sustainable_yield = params['sustainable_yield'] * 365.25
        deficit = max(0, current_extraction - sustainable_yield)
        
        deficit_coverage = min(1.0, total_potential / deficit) if deficit > 0 else 1.0
        
        return {
            'total_potential_m3_per_year': total_potential,
            'method_potentials': method_potentials,
            'current_deficit_m3_per_year': deficit,
            'deficit_coverage_ratio': deficit_coverage,
            'recommended_methods': self._recommend_recharge_methods(method_potentials),
            'implementation_priority': 'High' if deficit_coverage > 0.5 else 'Medium'
        }
    
    def _recommend_recharge_methods(self, method_potentials: Dict) -> List[str]:
        """Recommend most cost-effective recharge methods."""
        # Calculate cost-effectiveness (m³ per dollar)
        cost_effectiveness = {}
        for method, data in method_potentials.items():
            if data['total_annual_cost'] > 0:
                cost_effectiveness[method] = data['potential_recharge_m3_per_year'] / data['total_annual_cost']
        
        # Sort by cost-effectiveness
        sorted_methods = sorted(cost_effectiveness.items(), key=lambda x: x[1], reverse=True)
        
        return [method for method, _ in sorted_methods[:3]]  # Top 3 methods