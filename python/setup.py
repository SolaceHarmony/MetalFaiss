"""
setup.py - Package configuration for MetalFaiss
"""

from setuptools import setup, find_packages

setup(
    name="metalfaiss",
    version="0.1.0",
    description="MLX port of FAISS",
    author="MetalFaiss Team",
    packages=find_packages(),
    install_requires=[
        "mlx>=0.0.1",
    ],
    python_requires=">=3.8",
)
