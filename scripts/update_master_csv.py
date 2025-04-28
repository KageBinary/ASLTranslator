import pandas as pd
import glob
import os

def update_master_csv(data_folder='data/processed', master_filename='training_data_letters_MASTER.csv'):
    """
    Merge all session CSVs into a master CSV, adding session_id.
    """
    # Find all session CSV files
    csv_files = glob.glob(os.path.join(data_folder, 'training_data_letters_*.csv'))
    
    # Filter out the master CSV itself if it already exists
    csv_files = [f for f in csv_files if not f.endswith(master_filename)]
    # Bug fix. Try this option if an error or crash occurs: csv_files = [f for f in csv_files if os.path.basename(f) != master_filename]
    
    if not csv_files:
        print("No session CSV files found.")
        return
    
    dataframes = []
    for idx, file in enumerate(sorted(csv_files)):
        df = pd.read_csv(file)
        df['session_id'] = idx  # Add session ID
        dataframes.append(df)
    
    # Merge all data
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    # Save the master CSV
    master_path = os.path.join(data_folder, master_filename)
    combined_df.to_csv(master_path, index=False)
    
    print(f"Master CSV updated! {len(combined_df)} total samples from {len(csv_files)} sessions.")
    print(f"Saved to {master_path}")

# Example: If you want to call it manually.
if __name__ == "__main__":
    update_master_csv()
