"""Command Line Interface for Bangladesh Water Management Model."""

import argparse
from loguru import logger

from .config import load_config
from .simulator import WaterResourcesSimulator
from .utils.logging_config import setup_logging # Assuming setup_logging is in utils

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Run Bangladesh Water Management Model simulations.")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to a custom configuration YAML file.",
        default=None
    )
    parser.add_argument(
        "--scenario",
        type=str,
        choices=["groundwater", "salinity", "integrated", "none"],
        default="none",
        help="Type of scenario to run (or 'none' to just initialize)."
    )
    parser.add_argument(
        "--region",
        type=str,
        default="dhaka_metro", # Example default, adjust as needed
        help="Region for scenario (if applicable)."
    )
    parser.add_argument(
        "--years",
        type=int,
        default=5, # Example default
        help="Simulation years (if applicable)."
    )
    # Add more arguments here for specific scenarios as needed

    args = parser.parse_args()

    # Setup logging
    setup_logging() # Call the logging setup

    logger.info("Starting Bangladesh Water Management Model CLI...")

    try:
        config = load_config(args.config)
        logger.info(f"Configuration loaded. Using: {'default' if args.config is None else args.config}")

        simulator = WaterResourcesSimulator(config)
        logger.info("WaterResourcesSimulator initialized successfully.")

        if args.scenario == "groundwater":
            logger.info(f"Running groundwater scenario for region: {args.region}, years: {args.years}")
            results = simulator.run_groundwater_scenario(region=args.region, years=args.years)
            logger.info(f"Groundwater scenario results: {results}")
        elif args.scenario == "salinity":
            logger.info(f"Running salinity scenario for region: {args.region}, years: {args.years}")
            results = simulator.run_salinity_scenario(region=args.region, years=args.years)
            logger.info(f"Salinity scenario results: {results}")
        elif args.scenario == "integrated":
            # For integrated, you might need more specific args or a default set of regions
            regions = config.get('regions', {}).get('all', [args.region]) # Default to all regions or specified
            logger.info(f"Running integrated scenario for regions: {regions}, years: {args.years}")
            results = simulator.run_integrated_scenario(regions=regions, years=args.years) # Add other params as needed
            logger.info(f"Integrated scenario results: {results}")
        elif args.scenario == "none":
            logger.info("Simulator initialized. No scenario selected to run.")
        
        # Example: How to run a default scenario or provide options
        # This part can be expanded significantly based on requirements.
        # For now, just logs that it's ready.

        logger.info("CLI execution finished.")

    except Exception as e:
        logger.error(f"An error occurred during CLI execution: {e}")
        logger.exception(e) # Log full traceback
        # Depending on severity, you might want to sys.exit(1) here

if __name__ == "__main__":
    main() 