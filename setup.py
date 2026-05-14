"""
Setup script for Toxicity Prediction package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file) as f:
        requirements = [
            line.strip() 
            for line in f 
            if line.strip() and not line.startswith("#")
        ]

setup(
    name="toxicity-prediction",
    version="0.1.0",
    author="Toxicity Prediction Team",
    description="A comprehensive toolkit for predicting molecular toxicity using DeepChem and GNN models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MreegendraNarayan/toxicity-prediction",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=requirements,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="toxicity prediction GNN deep learning molecules chemistry",
    project_urls={
        "Bug Reports": "https://github.com/MreegendraNarayan/toxicity-prediction/issues",
        "Source": "https://github.com/MreegendraNarayan/toxicity-prediction",
    },
    entry_points={
        "console_scripts": [
            "toxicity-train=train:main",
            "toxicity-train-api=toxicity.inference.train_checkpoints:main",
        ],
    },
)
