"""Water management modules for Bangladesh.

This package contains specialized modules for different aspects of water resource management:
- Groundwater management and depletion modeling
- Coastal salinity intrusion simulation
- Surface water allocation and management
- Agricultural water demand and productivity
- Urban water supply and infrastructure
- Water economics and pricing
- Policy simulation and analysis
"""

from .agriculture import AgriculturalWaterManager
from .economic import EconomicAnalyzer # Assuming EconomicManager is the class in economic.py
from .groundwater import GroundwaterManager
from .policy import PolicyAnalyzer # Assuming PolicyEngine is the class in policy.py
from .salinity import SalinityManager
from .surface_water import SurfaceWaterManager
from .urban import UrbanWaterManager

__all__ = [
    "AgriculturalWaterManager",
    "EconomicAnalyzer",
    "GroundwaterManager",
    "PolicyAnalyzer",
    "SalinityManager",
    "SurfaceWaterManager",
    "UrbanWaterManager",
]