"""Tests for the configuration management (config.py) module.

This test suite verifies the functionality of loading, validating, saving,
and updating configurations for the Bangladesh Water Management Model.
"""

import pytest
import os
import yaml
from pathlib import Path
from bangladesh_water_management.config import (
    load_config,
    save_config,
    validate_config,
    update_config,
    get_default_config,
    CONFIG_SCHEMA
)
from jsonschema import ValidationError

# --- Fixtures ---

@pytest.fixture
def temp_config_file(tmp_path: Path, default_config_data: dict) -> Path:
    """Creates a temporary YAML config file with default data."""
    file_path = tmp_path / "temp_config.yaml"
    with open(file_path, 'w') as f:
        yaml.dump(default_config_data, f)
    return file_path

@pytest.fixture
def minimal_valid_config() -> dict:
    """Provides a minimal but valid configuration according to the schema."""
    # This needs to be carefully constructed based on CONFIG_SCHEMA's requirements
    # For simplicity, we'll start with default and assume it's minimal enough for some tests
    # A truly minimal one would only have required fields.
    config = get_default_config() 
    # Example: if only 'simulation_settings' and 'output_settings' were top-level required:
    # return {
    #     'simulation_settings': config['simulation_settings'],
    #     'output_settings': config['output_settings']
    # }
    # For now, using the full default config as a stand-in for a valid config.
    return config

# --- Test Cases ---

def test_get_default_config(default_config_data: dict):
    """Test that default configuration is loaded and is a dictionary."""
    assert isinstance(default_config_data, dict)
    assert "simulation_settings" in default_config_data
    assert "regional_settings" in default_config_data
    assert default_config_data["simulation_settings"]["start_year"] == 2024

def test_load_config_valid_file(temp_config_file: Path, default_config_data: dict):
    """Test loading a valid configuration from a YAML file."""
    loaded_cfg = load_config(str(temp_config_file))
    assert loaded_cfg is not None
    assert isinstance(loaded_cfg, dict)
    assert loaded_cfg["simulation_settings"]["start_year"] == default_config_data["simulation_settings"]["start_year"]
    assert loaded_cfg["regional_settings"]["bangladesh"]["projection_period"] == default_config_data["regional_settings"]["bangladesh"]["projection_period"]

def test_load_config_non_existent_file(tmp_path: Path):
    """Test loading a non-existent config file returns default config."""
    non_existent_path = tmp_path / "non_existent.yaml"
    loaded_cfg = load_config(str(non_existent_path))
    default_cfg = get_default_config()
    assert loaded_cfg is not None
    assert loaded_cfg == default_cfg # Expect default config to be returned

def test_load_config_invalid_yaml(tmp_path: Path):
    """Test loading an invalid YAML file returns default config and logs error (implicitly)."""
    invalid_yaml_path = tmp_path / "invalid.yaml"
    with open(invalid_yaml_path, 'w') as f:
        f.write("simulation_settings: {start_year: 2025, end_year: 2050, time_step: monthly, scenarios: [base, climate_change]}
                 output_settings: {output_dir: results/, log_level: INFO, report_format: csv") # Malformed YAML
    
    loaded_cfg = load_config(str(invalid_yaml_path))
    default_cfg = get_default_config()
    assert loaded_cfg is not None
    assert loaded_cfg == default_cfg # Expect default config
    # Verification of logging would require capturing log output, more advanced. 

def test_save_config(tmp_path: Path, default_config_data: dict):
    """Test saving a configuration to a YAML file."""
    save_path = tmp_path / "saved_config.yaml"
    success = save_config(default_config_data, str(save_path))
    assert success
    assert save_path.exists()

    # Verify content by loading it back
    with open(save_path, 'r') as f:
        saved_data = yaml.safe_load(f)
    assert saved_data is not None
    assert saved_data["simulation_settings"]["start_year"] == default_config_data["simulation_settings"]["start_year"]

def test_save_config_invalid_path(default_config_data: dict):
    """Test saving config to an invalid path (e.g., a directory)."""
    # This test might be OS-dependent or rely on specific error handling in save_config
    # For now, assume save_config might log an error and return False.
    # A more robust test would mock os.makedirs or open to simulate failure.
    if os.name == 'nt': # Windows
        invalid_save_path = "C:\Windows\System32\config_test\cant_write_here.yaml" 
    else: # Linux/macOS
        invalid_save_path = "/root/cant_write_here.yaml"
    
    # Skip if we can't construct a reliably unwritable path for the current OS/permissions
    # This is a simplification; real permission tests are complex.
    try:
        Path(invalid_save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(invalid_save_path, 'w') as f:
            f.write("test") # Check if writable, then delete
        os.remove(invalid_save_path)
        # If we reached here, the path might be writable, so skip or adjust path
        # For this example, we'll assume save_config handles it gracefully.
    except OSError:
        # This means the parent dir is not writable, good for testing save_config's error handling
        pass 
        
    # It's hard to guarantee an unwritable path without more complex setup or mocking.
    # Let's assume save_config returns False on typical IOErrors.
    # A simple test: try saving to a path that's actually a directory.
    dir_path = Path(os.getcwd()) / "a_directory_path_for_test"
    dir_path.mkdir(exist_ok=True)
    success = save_config(default_config_data, str(dir_path))
    assert not success # Should fail to save to a directory path
    dir_path.rmdir()


def test_validate_config_valid(minimal_valid_config: dict):
    """Test validation of a known valid configuration."""
    is_valid, errors = validate_config(minimal_valid_config)
    assert is_valid
    assert errors is None

def test_validate_config_invalid_type(minimal_valid_config: dict):
    """Test validation with incorrect data type for a field."""
    invalid_cfg = minimal_valid_config.copy()
    invalid_cfg["simulation_settings"]["start_year"] = "not_a_year" # Invalid type
    is_valid, errors = validate_config(invalid_cfg)
    assert not is_valid
    assert errors is not None
    assert len(errors) > 0
    # Example check for a specific error message (depends on jsonschema output)
    assert any("'not_a_year' is not of type 'integer'" in error.message for error in errors)

def test_validate_config_missing_required_field():
    """Test validation with a missing required field."""
    invalid_cfg = {}
    # Assuming 'simulation_settings' is required at the top level by CONFIG_SCHEMA
    # If CONFIG_SCHEMA defines top-level properties like 'simulation_settings' as required.
    if "simulation_settings" in CONFIG_SCHEMA.get("required", []):
        is_valid, errors = validate_config(invalid_cfg)
        assert not is_valid
        assert errors is not None
        assert any("'simulation_settings' is a required property" in error.message for error in errors)
    else:
        # If no top-level fields are strictly required, this test needs adjustment
        # or test a nested required field.
        cfg_with_missing_nested = get_default_config()
        del cfg_with_missing_nested["simulation_settings"]["start_year"] # Assuming start_year is required
        
        is_valid, errors = validate_config(cfg_with_missing_nested)
        assert not is_valid
        assert errors is not None
        assert any("'start_year' is a required property" in error.message for error in errors)

def test_validate_config_additional_property_allowed(minimal_valid_config: dict):
    """Test validation allows additional properties if schema defines additionalProperties=True (or not False)."""
    cfg_with_extra = minimal_valid_config.copy()
    cfg_with_extra["new_unspecified_setting"] = "some_value"
    
    # This test depends on CONFIG_SCHEMA's definition of `additionalProperties`.
    # If `additionalProperties` is False (or not defined and defaults to True for subschemas), this might pass or fail.
    # For jsonschema, default for `additionalProperties` is True.
    is_valid, errors = validate_config(cfg_with_extra)
    
    # If your schema is strict (additionalProperties: False at the root or relevant part):
    # assert not is_valid
    # assert any("Additional properties are not allowed ('new_unspecified_setting' was unexpected)" in error.message for error in errors)
    # If your schema allows additional properties (default for jsonschema):
    assert is_valid
    assert errors is None

def test_update_config_simple_override(default_config_data: dict):
    """Test updating a simple configuration value."""
    updates = {"simulation_settings": {"start_year": 2030}}
    updated_cfg = update_config(default_config_data, updates)
    assert updated_cfg["simulation_settings"]["start_year"] == 2030
    # Ensure other values are not changed
    assert updated_cfg["simulation_settings"]["end_year"] == default_config_data["simulation_settings"]["end_year"]

def test_update_config_nested_override(default_config_data: dict):
    """Test updating a nested configuration value."""
    updates = {"regional_settings": {"bangladesh": {"climate_scenario": "RCP8.5"}}}
    updated_cfg = update_config(default_config_data, updates)
    assert updated_cfg["regional_settings"]["bangladesh"]["climate_scenario"] == "RCP8.5"
    # Ensure other nested values are not changed
    assert updated_cfg["regional_settings"]["bangladesh"]["population_growth_rate"] == default_config_data["regional_settings"]["bangladesh"]["population_growth_rate"]

def test_update_config_add_new_key(default_config_data: dict):
    """Test adding a new key to the configuration (if schema allows)."""
    updates = {"new_top_level_setting": {"param1": True}}
    updated_cfg = update_config(default_config_data, updates)
    assert "new_top_level_setting" in updated_cfg
    assert updated_cfg["new_top_level_setting"]["param1"]

def test_update_config_empty_updates(default_config_data: dict):
    """Test updating with an empty update dictionary does not change config."""
    updates = {}
    updated_cfg = update_config(default_config_data, updates)
    assert updated_cfg == default_config_data

def test_update_config_partial_dict(default_config_data: dict):
    """Test updating only part of a nested dictionary."""
    updates = {
        "sectoral_settings": {
            "agriculture": {"irrigation_efficiency_improvement": 0.15} # Update one, keep others
        }
    }
    original_crop_types = default_config_data["sectoral_settings"]["agriculture"]["crop_types"]
    updated_cfg = update_config(default_config_data, updates)
    
    assert updated_cfg["sectoral_settings"]["agriculture"]["irrigation_efficiency_improvement"] == 0.15
    assert updated_cfg["sectoral_settings"]["agriculture"]["crop_types"] == original_crop_types

# More tests could include:
# - Testing specific validation rules (enum, min/max values, patterns)
# - Testing behavior with different schema settings for `additionalProperties`
# - Testing `load_config` with environment variable overrides if that feature is implemented.