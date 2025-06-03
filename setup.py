from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="bangladesh-water-management",
    version="1.0.0",
    author="Water Resources Management Team",
    author_email="water.management@bangladesh.gov.bd",
    description="Integrated Water Resources Management Model for Bangladesh",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bangladesh-water-management/iwrm-model",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Hydrology",
        "Topic :: Scientific/Engineering :: GIS",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "bangladesh-water=bangladesh_water_management.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "bangladesh_water_management": [
            "data/*.csv",
            "data/*.json",
            "data/spatial/*.shp",
            "config/*.yaml",
        ],
    },
)