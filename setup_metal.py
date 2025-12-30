"""
Simplified setup.py for Metal-only gsplat.

This version removes CUDA dependencies and only includes the Metal backend
for depth rendering on macOS.
"""

import os
import os.path as osp
from pathlib import Path

from setuptools import find_packages, setup

__version__ = None
exec(open("gsplat/version.py", "r").read())

URL = "https://github.com/nerfstudio-project/gsplat"

setup(
    name="gsplat-metal",
    version=__version__,
    description="Metal backend for Gaussian Splatting depth rendering (macOS only)",
    keywords="gaussian, splatting, metal, depth, macos",
    url=URL,
    download_url=f"{URL}/archive/gsplat-{__version__}.tar.gz",
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "jaxtyping",
        "torch>=2.0",
        "typing_extensions",
    ],
    extras_require={
        "dev": [
            "black[jupyter]==22.3.0",
            "isort==5.10.1",
            "pylint==2.13.4",
            "pytest==7.1.2",
            "pytest-xdist==2.5.0",
        ],
        "ply": [
            "plyfile",  # For loading PLY files
        ],
        "viz": [
            "pillow",  # For saving depth maps as images
            "matplotlib",  # For visualization
        ],
    },
    # No CUDA extensions - Metal backend is Python + Metal shaders only
    ext_modules=[],
    cmdclass={},
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "gsplat.metal": ["shaders/*.metal"],
    },
)
