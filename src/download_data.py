"""
Download Telco Customer Churn dataset from Kaggle using kagglehub.

This script:
- Downloads the dataset from blastchar/telco-customer-churn
- Saves it as data/raw/telco.csv
- Skips download if file already exists
- Handles errors gracefully
"""

import sys
from pathlib import Path
import pandas as pd


def download_dataset():
    """
    Download the Telco Customer Churn dataset from Kaggle using kagglehub.
    
    Returns:
        bool: True if download successful, False otherwise
    """
    # Set up paths
    project_root = Path(__file__).parent.parent
    data_raw_dir = project_root / "data" / "raw"
    output_file = data_raw_dir / "telco.csv"
    
    # Create data/raw directory if it doesn't exist
    data_raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if file already exists
    if output_file.exists():
        print(f"Dataset already exists at {output_file}")
        print("Skipping download. Delete the file to re-download.")
        return True
    
    print("Downloading Telco Customer Churn dataset from Kaggle...")
    
    try:
        import kagglehub
        from kagglehub import KaggleDatasetAdapter
        
        # Download and load dataset directly as pandas DataFrame
        print("Loading dataset using kagglehub...")
        
        # Try to load the CSV file directly (expected name: WA_Fn-UseC_-Telco-Customer-Churn.csv)
        # First, try the common filename
        csv_filename = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
        
        try:
            df = kagglehub.dataset_load(
                KaggleDatasetAdapter.PANDAS,
                "blastchar/telco-customer-churn",
                csv_filename
            )
        except Exception:
            # If that doesn't work, try to list files and find CSV
            print("Trying to find CSV file in dataset...")
            # List files in the dataset
            dataset_path = kagglehub.dataset_download("blastchar/telco-customer-churn")
            csv_files = list(Path(dataset_path).glob("*.csv"))
            
            if csv_files:
                csv_filename = csv_files[0].name
                print(f"Found CSV file: {csv_filename}")
                df = kagglehub.dataset_load(
                    KaggleDatasetAdapter.PANDAS,
                    "blastchar/telco-customer-churn",
                    csv_filename
                )
            else:
                raise ValueError("No CSV file found in dataset")
        
        # Save to output location
        print(f"Saving dataset to {output_file}...")
        df.to_csv(output_file, index=False)
        print(f"Dataset saved successfully! Shape: {df.shape}")
        print("Download complete!")
        
        return True
        
    except ImportError:
        print("ERROR: kagglehub not installed!")
        print("\nPlease install it with:")
        print("  pip install kagglehub[pandas-datasets]")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to download dataset: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure kagglehub is installed: pip install kagglehub[pandas-datasets]")
        print("2. You may need to authenticate with Kaggle:")
        print("   - Go to https://www.kaggle.com/account")
        print("   - Create API token (downloads kaggle.json)")
        print("   - Place kaggle.json in ~/.kaggle/kaggle.json")
        print("   - Set permissions: chmod 600 ~/.kaggle/kaggle.json")
        print("3. Verify internet connection")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = download_dataset()
    sys.exit(0 if success else 1)
