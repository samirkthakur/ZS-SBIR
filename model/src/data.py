# src/data.py

import os
import subprocess

def setup_kaggle_credentials(kaggle_json_path: str):
    """
    Sets up Kaggle API credentials.
    """
    os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
    subprocess.run(["cp", kaggle_json_path, os.path.expanduser("~/.kaggle/kaggle.json")])
    os.chmod(os.path.expanduser("~/.kaggle/kaggle.json"), 0o600)

def download_kaggle_dataset(dataset_slug: str, download_path: str):
    """
    Downloads and unzips a Kaggle dataset given its slug and target path.
    """
    os.makedirs(download_path, exist_ok=True)
    subprocess.run(["kaggle", "datasets", "download", "-d", dataset_slug, "--unzip", "-p", download_path])

    # List extracted contents for verification
    print("\n--- Listing extracted contents to verify path ---")
    for root, dirs, files in os.walk(download_path):
        print(f"{root}/")
        for name in files:
            print(f"  {name}")

if __name__ == "__main__":
    # Example usage when running this script directly
    setup_kaggle_credentials("kaggle.json")
    download_kaggle_dataset("dhananjayapaliwal/fulldataset", "./data")
    download_kaggle_dataset("rishikashili/tuberlin", "./data/TUBerlin")
