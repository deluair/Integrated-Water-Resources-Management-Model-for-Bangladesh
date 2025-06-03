"""Tests for the WaterResourcesSimulator (simulator.py) module.

This test suite verifies the core functionality of the WaterResourcesSimulator,
including its initialization, scenario execution, and interaction with various
water management modules (mocked for these tests).
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch, call

from bangladesh_water_management.simulator import WaterResourcesSimulator
from bangladesh_water_management.config import get_default_config, load_config

# Mock individual manager classes
# This allows us to test the simulator's orchestration logic without
# needing the full implementation of each manager.
@pytest.fixture
def mock_groundwater_manager():
    mock = MagicMock()
    mock.simulate_depletion.return_value = {"depletion_rate": 0.1, "remaining_storage": 1000}
    mock.assess_sustainability.return_value = {"sustainability_index": 0.75}
    return mock

@pytest.fixture
def mock_salinity_manager():
    mock = MagicMock()
    mock.simulate_intrusion.return_value = {"affected_area_sqkm": 500, "salinity_level_ppt": 5}
    mock.assess_impact.return_value = {"agricultural_loss_usd": 1000000}
    return mock

@pytest.fixture
def mock_surface_water_manager():
    mock = MagicMock()
    mock.simulate_river_flow.return_value = pd.DataFrame({'date': pd.to_datetime(['2024-01-01']), 'flow_m3s': [1500]})
    mock.simulate_flood_event.return_value = {"inundated_area_ha": 10000, "damage_cost_usd": 5000000}
    return mock

@pytest.fixture
def mock_agricultural_water_manager():
    mock = MagicMock()
    mock.calculate_irrigation_demand.return_value = {"total_demand_mcm": 1200}
    mock.optimize_cropping_pattern.return_value = {"optimal_pattern": {"rice": 0.6, "wheat": 0.4}}
    return mock

@pytest.fixture
def mock_urban_water_manager():
    mock = MagicMock()
    mock.calculate_urban_water_demand.return_value = {"total_demand_mld": 250}
    mock.assess_infrastructure_capacity.return_value = {"capacity_gap_mld": 50}
    return mock

@pytest.fixture
def mock_economic_analyzer():
    mock = MagicMock()
    mock.perform_cost_benefit_analysis.return_value = {"npv_usd": 15000000, "bcr": 1.5}
    return mock

@pytest.fixture
def mock_policy_analyzer():
    mock = MagicMock()
    mock.evaluate_policy_effectiveness.return_value = {"effectiveness_score": 0.8}
    return mock

@pytest.fixture
def mock_data_loader():
    mock = MagicMock()
    # Simulate loading some data that might be used by the simulator directly or passed to managers
    mock.load_data.return_value = pd.DataFrame({'year': [2024], 'population': [170_000_000]})
    return mock

@pytest.fixture
def mock_synthetic_data_generator():
    mock = MagicMock()
    mock.generate_meteorological_data.return_value = pd.DataFrame({
        'date': pd.to_datetime(['2024-01-01']), 
        'temperature': [25.0]
    })
    return mock

@pytest.fixture
@patch('bangladesh_water_management.simulator.DataLoader', autospec=True)
@patch('bangladesh_water_management.simulator.SyntheticDataGenerator', autospec=True)
@patch('bangladesh_water_management.simulator.GroundwaterManager', autospec=True)
@patch('bangladesh_water_management.simulator.SalinityManager', autospec=True)
@patch('bangladesh_water_management.simulator.SurfaceWaterManager', autospec=True)
@patch('bangladesh_water_management.simulator.AgriculturalWaterManager', autospec=True)
@patch('bangladesh_water_management.simulator.UrbanWaterManager', autospec=True)
@patch('bangladesh_water_management.simulator.EconomicAnalyzer', autospec=True)
@patch('bangladesh_water_management.simulator.PolicyAnalyzer', autospec=True)
def water_simulator(MockPolicyAnalyzer, MockEconomicAnalyzer, 
                      MockUrbanWaterManager, MockAgriculturalWaterManager, 
                      MockSurfaceWaterManager, MockSalinityManager, 
                      MockGroundwaterManager, MockSyntheticDataGen, MockDataLoad):
    """Fixture to create a WaterResourcesSimulator instance with mocked dependencies."""
    # Assign instances of mocks from parameters to the patched classes
    MockGroundwaterManager.return_value = mock_groundwater_manager()
    MockSalinityManager.return_value = mock_salinity_manager()
    MockSurfaceWaterManager.return_value = mock_surface_water_manager()
    MockAgriculturalWaterManager.return_value = mock_agricultural_water_manager()
    MockUrbanWaterManager.return_value = mock_urban_water_manager()
    MockEconomicAnalyzer.return_value = mock_economic_analyzer()
    MockPolicyAnalyzer.return_value = mock_policy_analyzer()
    MockDataLoad.return_value = mock_data_loader()
    MockSyntheticDataGen.return_value = mock_synthetic_data_generator()

    config = get_default_config()
    simulator = WaterResourcesSimulator(config_data=config)
    return simulator

# --- Test Cases ---

def test_simulator_initialization(water_simulator: WaterResourcesSimulator):
    """Test that the simulator and its managers are initialized."""
    assert water_simulator is not None
    assert water_simulator.config is not None
    assert water_simulator.logger is not None # Assuming logger is initialized

    # Check if managers are instantiated (they are mocks, so check if the mock was created)
    assert water_simulator.groundwater_manager is not None
    assert water_simulator.salinity_manager is not None
    assert water_simulator.surface_water_manager is not None
    assert water_simulator.agricultural_water_manager is not None
    assert water_simulator.urban_water_manager is not None
    assert water_simulator.economic_analyzer is not None
    assert water_simulator.policy_analyzer is not None
    assert water_simulator.data_loader is not None
    assert water_simulator.synthetic_data_generator is not None

    # Check if config is passed to managers (mocked managers' __init__ would be called)
    # This is implicitly tested by the patch setup if autospec=True is effective for __init__
    # For example, GroundwaterManager should have been called with the config
    water_simulator.GroundwaterManager.assert_called_once()
    # Can check args if needed: water_simulator.GroundwaterManager.assert_called_once_with(config=water_simulator.config['groundwater_settings'], global_config=water_simulator.config)

def test_run_simulation_all_scenarios(water_simulator: WaterResourcesSimulator):
    """Test running the main simulation loop with all scenarios enabled (default)."""
    results = water_simulator.run_simulation()
    assert results is not None
    assert isinstance(results, dict)
    
    # Check if scenario-specific methods on managers were called
    # Groundwater
    water_simulator.groundwater_manager.simulate_depletion.assert_called()
    water_simulator.groundwater_manager.assess_sustainability.assert_called()
    # Salinity
    water_simulator.salinity_manager.simulate_intrusion.assert_called()
    water_simulator.salinity_manager.assess_impact.assert_called()
    # Surface Water
    water_simulator.surface_water_manager.simulate_river_flow.assert_called()
    water_simulator.surface_water_manager.simulate_flood_event.assert_called()
    # Agriculture
    water_simulator.agricultural_water_manager.calculate_irrigation_demand.assert_called()
    # Urban
    water_simulator.urban_water_manager.calculate_urban_water_demand.assert_called()
    # Economic & Policy (if they have methods called directly in run_simulation)
    # water_simulator.economic_analyzer.some_method.assert_called()
    # water_simulator.policy_analyzer.some_method.assert_called()

    # Check if results from mocked managers are in the output
    assert "groundwater_simulation" in results
    assert results["groundwater_simulation"]["depletion_rate"] == 0.1
    assert "salinity_simulation" in results
    assert results["salinity_simulation"]["affected_area_sqkm"] == 500

def test_run_specific_scenario_groundwater_depletion(water_simulator: WaterResourcesSimulator):
    """Test running only the groundwater depletion scenario."""
    water_simulator.config['simulation_settings']['scenarios_to_run'] = ['groundwater_depletion']
    results = water_simulator.run_simulation()
    
    assert "groundwater_simulation" in results
    water_simulator.groundwater_manager.simulate_depletion.assert_called_once()
    water_simulator.groundwater_manager.assess_sustainability.assert_called_once()
    
    # Ensure other managers' main simulation methods were NOT called
    water_simulator.salinity_manager.simulate_intrusion.assert_not_called()
    water_simulator.surface_water_manager.simulate_river_flow.assert_not_called()

def test_run_scenario_with_time_loop(water_simulator: WaterResourcesSimulator):
    """Test the time loop within a scenario if applicable (e.g., groundwater depletion over time)."""
    # This test depends on how scenarios are implemented. If a scenario involves a time loop,
    # we'd check if manager methods are called multiple times or with different time steps.
    # For this example, simulate_depletion is called once per run_simulation call.
    # If it were called per year:    
    start_year = water_simulator.config['simulation_settings']['start_year']
    end_year = water_simulator.config['simulation_settings']['end_year']
    num_years = end_year - start_year + 1

    # Re-configure the mock to check calls per year if that was the design
    # For now, the mock is simple and returns a single dict.
    # If simulate_depletion was called for each year:
    # water_simulator.groundwater_manager.simulate_depletion.reset_mock()
    # water_simulator.run_simulation()
    # assert water_simulator.groundwater_manager.simulate_depletion.call_count == num_years
    # calls = [call(year=y, some_param=...) for y in range(start_year, end_year + 1)]
    # water_simulator.groundwater_manager.simulate_depletion.assert_has_calls(calls, any_order=False)
    
    # Current test: just ensure it's called as per the existing mock structure
    water_simulator.run_simulation()
    water_simulator.groundwater_manager.simulate_depletion.assert_called()

def test_load_external_data_if_needed(water_simulator: WaterResourcesSimulator):
    """Test that data loader is called if external data is configured for use."""
    # This depends on the simulator's logic for when to load data.
    # Assuming it loads some general data at the start or for specific scenarios.
    water_simulator.config['data_settings']['load_historical_data'] = True # Example flag
    water_simulator.run_simulation()
    water_simulator.data_loader.load_data.assert_called() # Check if load_data was called
    # Can be more specific: water_simulator.data_loader.load_data.assert_called_with(dataset_name='population_data', years_range=(2000,2020))

def test_generate_synthetic_data_if_needed(water_simulator: WaterResourcesSimulator):
    """Test that synthetic data generator is called if configured."""
    water_simulator.config['data_settings']['use_synthetic_data'] = True # Example flag
    water_simulator.config['data_settings']['synthetic_data_params']['meteorological']['num_sites'] = 5
    water_simulator.run_simulation()
    water_simulator.synthetic_data_generator.generate_meteorological_data.assert_called()
    # Example: check if called with specific parameters from config
    # water_simulator.synthetic_data_generator.generate_meteorological_data.assert_called_with(
    #     sites=water_simulator.config['data_settings']['synthetic_data_params']['meteorological']['sites_list_or_num'], 
    #     start_date=..., 
    #     end_date=...
    # )


def test_scenario_interdependencies(water_simulator: WaterResourcesSimulator):
    """Test if output from one scenario/manager can be input to another (conceptual)."""
    # This is a more complex integration test. Requires mocks to return specific data
    # that is then passed to other mocks.
    
    # Example: Surface water availability affects irrigation demand calculation
    mock_sw_output = pd.DataFrame({'date': pd.to_datetime(['2024-07-01']), 'available_flow_m3s': [500]})
    water_simulator.surface_water_manager.simulate_river_flow.return_value = mock_sw_output
    
    water_simulator.run_simulation()
    
    # Check if agricultural_water_manager.calculate_irrigation_demand was called
    # with data derived from mock_sw_output.
    # This requires inspecting the arguments passed to the mock.
    # For instance, if the call looks like: ag_manager.calculate_irrigation_demand(year, surface_water_data=...)
    # We can check the content of surface_water_data.
    # call_args = water_simulator.agricultural_water_manager.calculate_irrigation_demand.call_args
    # assert call_args is not None
    # passed_sw_data = call_args[1].get('surface_water_availability') # or whatever the param name is
    # pd.testing.assert_frame_equal(passed_sw_data, mock_sw_output) # If the exact df is passed
    
    # For now, just ensure methods are called as a basic check
    water_simulator.surface_water_manager.simulate_river_flow.assert_called()
    water_simulator.agricultural_water_manager.calculate_irrigation_demand.assert_called()


@patch('bangladesh_water_management.simulator.load_config')
def test_simulator_with_custom_config_path(mock_load_config):
    """Test simulator initialization with a custom config file path."""
    custom_config_data = get_default_config()
    custom_config_data['simulation_settings']['start_year'] = 2025
    mock_load_config.return_value = custom_config_data
    
    simulator = WaterResourcesSimulator(config_file_path="custom/path/to/config.yaml")
    
    mock_load_config.assert_called_once_with("custom/path/to/config.yaml")
    assert simulator.config['simulation_settings']['start_year'] == 2025

# Further tests could include:
# - Error handling (e.g., if a manager raises an exception during its simulation method).
# - More detailed checks on the content of the results dictionary.
# - Testing specific parameter effects from config on manager behavior.
# - Testing the `save_results` method if implemented in the simulator.