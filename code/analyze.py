# pandas to analyze and view the csv data

import pandas as pd
import os

def analyze_data(file_path):
    """
    Analyze a single CSV file and display its first 5 rows
    """
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Get basic info about the file
        print(f"\n{'='*60}")
        print(f"File: {os.path.basename(file_path)}")
        print(f"{'='*60}")
        print(f"Shape: {df.shape} (rows, columns)")
        print(f"Columns: {list(df.columns)}")
        
        # Display first 5 rows
        print(f"\nFirst 5 rows:")
        print(df.head())
        
        # Display basic statistics
        print(f"\nBasic Statistics:")
        print(df.describe())
        
    except Exception as e:
        print(f"Error reading {file_path}: {str(e)}")

def analyze_all_files(folder_path):
    """
    Analyze all CSV files in the specified folder
    """
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' does not exist!")
        return
    
    # Get all CSV files in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"No CSV files found in '{folder_path}'")
        return
    
    print(f"Found {len(csv_files)} CSV files in '{folder_path}'")
    
    # Analyze each CSV file
    for csv_file in sorted(csv_files):
        file_path = os.path.join(folder_path, csv_file)
        analyze_data(file_path)

# Main execution
if __name__ == "__main__":
    folder_path = "TestData"
    analyze_all_files(folder_path)