# src/setup.py

import subprocess
import sys

def install_packages(packages):
    """
    Install a list of packages using pip.
    """
    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

if __name__ == "__main__":
    # Example usage when running this script directly
    packages_to_install = [
        "torch",
        "transformers",
        "pillow",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "kaggle",
        "faiss-cpu"
    ]
    install_packages(packages_to_install)
