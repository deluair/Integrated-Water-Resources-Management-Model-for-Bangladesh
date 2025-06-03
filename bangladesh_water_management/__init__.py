"""Integrated Water Resources Management Model for Bangladesh.

A comprehensive Python-based simulation platform designed to model Bangladesh's complex water resource challenges
across coastal salinity intrusion, groundwater depletion, and freshwater scarcity while optimizing water allocation,
pricing, and infrastructure systems.
"""

__version__ = "1.0.0"

from bangladesh_water_management.simulator import WaterResourcesSimulator
from bangladesh_water_management.config import load_config

__all__ = ["WaterResourcesSimulator", "load_config"]