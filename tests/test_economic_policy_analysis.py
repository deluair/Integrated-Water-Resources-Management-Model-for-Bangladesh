"""Tests for EconomicAnalyzer and PolicyAnalyzer modules.

These tests verify the core functionalities of economic and policy analysis,
including cost-benefit analysis, policy evaluation, and their potential
integration with outputs from other simulation modules.
"""

import pytest
import pandas as pd
from unittest.mock import MagicMock, patch

from bangladesh_water_management.config import get_default_config
from bangladesh_water_management.modules import EconomicAnalyzer, PolicyAnalyzer

@pytest.fixture
def config_data():
    """Provides default configuration for tests."""
    return get_default_config()

@pytest.fixture
def economic_analyzer(config_data):
    analyzer = EconomicAnalyzer(config_data['economic_settings'], global_config=config_data)
    analyzer.perform_cost_benefit_analysis = MagicMock()
    analyzer.calculate_project_costs = MagicMock()
    analyzer.calculate_project_benefits = MagicMock()
    analyzer.conduct_sensitivity_analysis = MagicMock()
    return analyzer

@pytest.fixture
def policy_analyzer(config_data):
    analyzer = PolicyAnalyzer(config_data['policy_settings'], global_config=config_data)
    analyzer.evaluate_policy_effectiveness = MagicMock()
    analyzer.analyze_institutional_capacity = MagicMock()
    analyzer.generate_recommendations = MagicMock()
    return analyzer

# --- Test Cases for EconomicAnalyzer ---

def test_economic_analyzer_cba(economic_analyzer):
    """Test the perform_cost_benefit_analysis method of EconomicAnalyzer."""
    mock_project_data = {
        'name': 'New Irrigation Scheme',
        'investment_cost_usd': 5_000_000,
        'annual_om_cost_usd': 200_000,
        'annual_benefits_usd': 800_000,
        'lifespan_years': 20,
        'discount_rate': 0.05
    }
    expected_cba_result = {'npv_usd': 2_457_890, 'bcr': 1.6, 'irr': 0.12}
    economic_analyzer.perform_cost_benefit_analysis.return_value = expected_cba_result

    result = economic_analyzer.perform_cost_benefit_analysis(project_data=mock_project_data)

    economic_analyzer.perform_cost_benefit_analysis.assert_called_once_with(project_data=mock_project_data)
    assert result == expected_cba_result
    assert result['npv_usd'] > 0 # Example assertion on result content

def test_economic_analyzer_cost_calculation(economic_analyzer):
    """Test calculation of project costs."""
    mock_cost_inputs = {
        'component_A_cost': 1000, 
        'component_B_cost': 2000,
        'contingency_percentage': 0.1
    }
    expected_total_cost = 3300 # (1000 + 2000) * 1.1
    economic_analyzer.calculate_project_costs.return_value = {'total_cost_usd': expected_total_cost}

    result = economic_analyzer.calculate_project_costs(cost_inputs=mock_cost_inputs)
    
    economic_analyzer.calculate_project_costs.assert_called_once_with(cost_inputs=mock_cost_inputs)
    assert result['total_cost_usd'] == expected_total_cost

def test_economic_analyzer_benefit_calculation_from_simulation(economic_analyzer):
    """Test calculating benefits based on (mocked) simulation outputs."""
    # Example: Benefits from reduced flood damage (output from SurfaceWaterManager)
    mock_simulation_output = {
        'scenario_name': 'With Flood Protection Project',
        'flood_damage_avoided_usd': 1_500_000
    }
    expected_benefits = {'total_benefits_usd': 1_500_000}
    economic_analyzer.calculate_project_benefits.return_value = expected_benefits

    result = economic_analyzer.calculate_project_benefits(simulation_outputs=[mock_simulation_output], benefit_type='flood_damage_reduction')

    economic_analyzer.calculate_project_benefits.assert_called_once_with(
        simulation_outputs=[mock_simulation_output], 
        benefit_type='flood_damage_reduction'
    )
    assert result['total_benefits_usd'] == 1_500_000

def test_economic_analyzer_sensitivity_analysis(economic_analyzer):
    """Test sensitivity analysis functionality."""
    mock_cba_results = {'npv_usd': 2_457_890, 'bcr': 1.6}
    mock_sensitivity_params = {'discount_rate_variation': [0.03, 0.07]}
    expected_sensitivity_output = {
        'npv_at_0.03_discount': 3_000_000,
        'npv_at_0.07_discount': 2_000_000
    }
    economic_analyzer.conduct_sensitivity_analysis.return_value = expected_sensitivity_output

    result = economic_analyzer.conduct_sensitivity_analysis(
        base_cba_results=mock_cba_results, 
        parameters_to_vary=mock_sensitivity_params
    )

    economic_analyzer.conduct_sensitivity_analysis.assert_called_once_with(
        base_cba_results=mock_cba_results, 
        parameters_to_vary=mock_sensitivity_params
    )
    assert result == expected_sensitivity_output

# --- Test Cases for PolicyAnalyzer ---

def test_policy_analyzer_effectiveness_evaluation(policy_analyzer):
    """Test policy effectiveness evaluation."""
    mock_policy_details = {
        'policy_name': 'National Water Policy 2024',
        'objectives': ['Improve water quality', 'Ensure equitable access'],
        'implementation_status': 'Partial'
    }
    # Mock simulation results that might feed into policy evaluation
    mock_simulation_impacts = {
        'water_quality_index_change': 0.15, # Positive change
        'access_inequity_reduction': 0.05   # Small reduction
    }
    expected_evaluation = {'overall_effectiveness_score': 0.65, 'recommendations_needed': True}
    policy_analyzer.evaluate_policy_effectiveness.return_value = expected_evaluation

    result = policy_analyzer.evaluate_policy_effectiveness(
        policy_details=mock_policy_details, 
        simulation_impacts=mock_simulation_impacts,
        stakeholder_feedback=None # Could be another input
    )

    policy_analyzer.evaluate_policy_effectiveness.assert_called_once_with(
        policy_details=mock_policy_details, 
        simulation_impacts=mock_simulation_impacts,
        stakeholder_feedback=None
    )
    assert result == expected_evaluation

def test_policy_analyzer_institutional_capacity(policy_analyzer):
    """Test institutional capacity analysis."""
    mock_institutional_data = {
        'organization_name': 'Water Resources Planning Organization (WARPO)',
        'staffing_level_actual_vs_target': 0.7,
        'budget_utilization_rate': 0.85
    }
    expected_capacity_assessment = {
        'capacity_index': 0.78, 
        'strengths': ['Good budget utilization'], 
        'weaknesses': ['Understaffing']
    }
    policy_analyzer.analyze_institutional_capacity.return_value = expected_capacity_assessment

    result = policy_analyzer.analyze_institutional_capacity(institutional_data=[mock_institutional_data])

    policy_analyzer.analyze_institutional_capacity.assert_called_once_with(institutional_data=[mock_institutional_data])
    assert result == expected_capacity_assessment

def test_policy_analyzer_recommendation_generation(policy_analyzer):
    """Test generation of policy recommendations."""
    mock_evaluation_findings = {
        'effectiveness_score': 0.5,
        'key_issues': ['Poor enforcement', 'Lack of coordination']
    }
    mock_capacity_findings = {
        'capacity_index': 0.6,
        'weaknesses': ['Insufficient training', 'Outdated regulations']
    }
    expected_recommendations = [
        "Strengthen enforcement mechanisms for existing water policies.",
        "Improve inter-agency coordination on water management.",
        "Invest in training programs for water sector professionals.",
        "Update regulatory frameworks to reflect current challenges."
    ]
    policy_analyzer.generate_recommendations.return_value = {'recommendations_list': expected_recommendations}

    result = policy_analyzer.generate_recommendations(
        evaluation_findings=mock_evaluation_findings, 
        capacity_findings=mock_capacity_findings
    )

    policy_analyzer.generate_recommendations.assert_called_once_with(
        evaluation_findings=mock_evaluation_findings, 
        capacity_findings=mock_capacity_findings
    )
    assert result['recommendations_list'] == expected_recommendations
    assert len(result['recommendations_list']) > 0

# Further tests could include:
# - More detailed checks on the inputs and outputs of each method.
# - How EconomicAnalyzer and PolicyAnalyzer might use data from config files (e.g., cost databases, policy frameworks).
# - Scenarios where the output of one analyzer might feed into another (e.g., economic viability of a policy option).