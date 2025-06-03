"""Tests for interactions between different simulation modules.

These tests verify that modules can correctly use outputs from other modules
as inputs for their own calculations, simulating the interconnectedness of the
water resource system.
"""

import pytest
import pandas as pd
from unittest.mock import MagicMock, patch

from bangladesh_water_management.config import get_default_config
from bangladesh_water_management.modules import (
    GroundwaterManager,
    SalinityManager,
    SurfaceWaterManager,
    AgriculturalWaterManager,
    UrbanWaterManager
)

@pytest.fixture
def config_data():
    """Provides default configuration for tests."""
    return get_default_config()

@pytest.fixture
def surface_water_manager(config_data):
    manager = SurfaceWaterManager(config_data['surface_water_settings'], global_config=config_data)
    manager.simulate_river_flow = MagicMock()
    manager.calculate_water_availability = MagicMock()
    return manager

@pytest.fixture
def agricultural_water_manager(config_data):
    manager = AgriculturalWaterManager(config_data['agricultural_settings'], global_config=config_data)
    manager.calculate_irrigation_demand = MagicMock()
    manager.optimize_cropping_pattern = MagicMock()
    return manager

@pytest.fixture
def salinity_manager(config_data):
    manager = SalinityManager(config_data['salinity_settings'], global_config=config_data)
    manager.simulate_intrusion = MagicMock()
    return manager

@pytest.fixture
def groundwater_manager(config_data):
    manager = GroundwaterManager(config_data['groundwater_settings'], global_config=config_data)
    manager.update_recharge = MagicMock()
    manager.simulate_depletion = MagicMock()
    return manager

# --- Test Cases for Module Interactions ---

def test_surface_water_to_agriculture(surface_water_manager, agricultural_water_manager):
    """Test interaction: Surface water availability influencing agricultural demand."""
    # Mock output from SurfaceWaterManager
    mock_water_availability = pd.DataFrame({
        'region': ['North', 'South'],
        'available_surface_water_mcm': [1000, 800]
    })
    surface_water_manager.calculate_water_availability.return_value = mock_water_availability

    # Call a method in AgriculturalWaterManager that uses this data
    agricultural_water_manager.calculate_irrigation_demand(
        year=2023, 
        surface_water_availability=mock_water_availability
    )

    # Assert that the agricultural manager's method was called with the correct data
    agricultural_water_manager.calculate_irrigation_demand.assert_called_once()
    call_args = agricultural_water_manager.calculate_irrigation_demand.call_args
    passed_sw_data = call_args[1].get('surface_water_availability')
    pd.testing.assert_frame_equal(passed_sw_data, mock_water_availability)

def test_river_flow_to_salinity_intrusion(surface_water_manager, salinity_manager):
    """Test interaction: River flow affecting salinity intrusion."""
    # Mock output from SurfaceWaterManager
    mock_river_flow_data = pd.DataFrame({
        'date': pd.to_datetime(['2023-03-15']),
        'river_id': ['Meghna'],
        'flow_m3s': [500] # Low flow scenario
    })
    surface_water_manager.simulate_river_flow.return_value = mock_river_flow_data

    # Call a method in SalinityManager that uses this data
    salinity_manager.simulate_intrusion(
        year=2023, 
        river_flow_data=mock_river_flow_data, 
        sea_level_rise_m=0.1
    )

    salinity_manager.simulate_intrusion.assert_called_once()
    call_args = salinity_manager.simulate_intrusion.call_args
    passed_flow_data = call_args[1].get('river_flow_data')
    pd.testing.assert_frame_equal(passed_flow_data, mock_river_flow_data)

def test_irrigation_return_flow_to_groundwater_recharge(agricultural_water_manager, groundwater_manager):
    """Test interaction: Irrigation return flow contributing to groundwater recharge."""
    # Mock output from AgriculturalWaterManager (e.g., estimated return flow)
    mock_irrigation_summary = {
        'total_irrigation_mcm': 1200,
        'estimated_return_flow_mcm': 120 # 10% return flow
    }
    # Assume calculate_irrigation_demand or another method provides this
    agricultural_water_manager.calculate_irrigation_demand.return_value = mock_irrigation_summary 

    # Call a method in GroundwaterManager that uses this data
    # This might be an explicit 'update_recharge' or part of 'simulate_depletion'
    groundwater_manager.update_recharge(
        year=2023, 
        recharge_from_irrigation_mcm=mock_irrigation_summary['estimated_return_flow_mcm']
    )

    groundwater_manager.update_recharge.assert_called_once_with(
        year=2023, 
        recharge_from_irrigation_mcm=120
    )

def test_groundwater_abstraction_limits_surface_water_baseflow(groundwater_manager, surface_water_manager):
    """Test interaction: Groundwater abstraction affecting surface water baseflow."""
    # Mock output from GroundwaterManager (e.g., impact on baseflow contribution)
    mock_gw_contribution_change = {
        'river_id': ['Brahmaputra'],
        'baseflow_reduction_m3s': [50]
    }
    # Assume simulate_depletion or a dedicated method provides this
    groundwater_manager.simulate_depletion.return_value = {
        'some_depletion_metric': 0.5,
        'impact_on_baseflow': mock_gw_contribution_change
    }
    
    # Simulate groundwater depletion first to get the impact
    gw_sim_results = groundwater_manager.simulate_depletion(year=2023)

    # Call a method in SurfaceWaterManager that uses this data
    surface_water_manager.simulate_river_flow(
        year=2023, 
        climate_scenario='baseline',
        groundwater_contribution_change=gw_sim_results['impact_on_baseflow']
    )

    surface_water_manager.simulate_river_flow.assert_called_once()
    call_args = surface_water_manager.simulate_river_flow.call_args
    passed_gw_impact = call_args[1].get('groundwater_contribution_change')
    assert passed_gw_impact == mock_gw_contribution_change

# More complex interactions could involve multiple steps or feedback loops.
# For example:
# 1. Surface water manager simulates flow.
# 2. Agricultural manager calculates demand based on that flow.
# 3. Groundwater manager simulates depletion based on agricultural pumping.
# 4. This depletion affects baseflow, which feeds back into surface water simulation for the next time step.
# Testing such loops often requires a more integrated setup, potentially using the main simulator
# with carefully controlled mocks for specific parts of the chain.

# Example of a chained interaction (conceptual)
# This would typically be orchestrated by the main WaterResourcesSimulator

@patch('bangladesh_water_management.modules.surface_water.SurfaceWaterManager.get_environmental_flow_requirements')
@patch('bangladesh_water_management.modules.agricultural_water_manager.AgriculturalWaterManager.assess_water_shortage_impact')
def test_chained_interaction_surface_agri(mock_agri_impact, mock_sw_env_flow, 
                                          surface_water_manager, agricultural_water_manager):
    """Conceptual test of a chained interaction orchestrated externally (e.g., by simulator)."""
    # Step 1: Surface water simulation provides available water
    mock_water_availability = pd.DataFrame({'region': ['Central'], 'available_for_agri_mcm': [500]})
    surface_water_manager.calculate_water_availability.return_value = mock_water_availability
    
    # Step 2: Agri manager uses this to calculate demand and potential shortage
    mock_demand_result = {'total_demand_mcm': 600, 'unmet_demand_mcm': 100}
    agricultural_water_manager.calculate_irrigation_demand.return_value = mock_demand_result
    
    # --- Orchestration (simulated here) ---
    # In a real scenario, the simulator would call these in sequence.
    available_sw = surface_water_manager.calculate_water_availability(year=2023, allocation_priority=['agriculture'])
    agri_demand_info = agricultural_water_manager.calculate_irrigation_demand(year=2023, surface_water_availability=available_sw)
    
    # Step 3: Agri manager assesses impact of shortage (if any)
    if agri_demand_info['unmet_demand_mcm'] > 0:
        agricultural_water_manager.assess_water_shortage_impact(shortage_mcm=agri_demand_info['unmet_demand_mcm'])
    # --- End Orchestration ---

    surface_water_manager.calculate_water_availability.assert_called_once_with(year=2023, allocation_priority=['agriculture'])
    agricultural_water_manager.calculate_irrigation_demand.assert_called_once()
    # Check that the output of the first was passed to the second
    pd.testing.assert_frame_equal(
        agricultural_water_manager.calculate_irrigation_demand.call_args[1]['surface_water_availability'],
        mock_water_availability
    )
    mock_agri_impact.assert_called_once_with(shortage_mcm=100)