import pandas as pd
import os
import glob
import logging
import numpy as np # IMPROVEMENT: Added numpy import for kdd processing

# ==============================================================================
# ROBUST PATH CONFIGURATION
# IMPROVEMENT: The script now automatically detects the project's root directory.
# This makes it independent of where you run it from.
# ==============================================================================
try:
    # This gives the full path to the current script file (prepare_datasets.py)
    SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
    # The project root is assumed to be the directory containing this script
    PROJECT_ROOT = SCRIPT_PATH
except NameError:
    # Fallback for interactive environments like Jupyter notebooks
    PROJECT_ROOT = os.getcwd()

# Define source and destination folders relative to the project root
SOURCE_DATA_FOLDER = os.path.join(PROJECT_ROOT, "source_data") 
FINAL_DATASETS_FOLDER = os.path.join(PROJECT_ROOT, "datasets") 

# Define specific input folders
INPUT_NAB_FOLDER = os.path.join(SOURCE_DATA_FOLDER, "NAB-master")
INPUT_YAHOO_FOLDER = os.path.join(SOURCE_DATA_FOLDER, "yahoo-s5-data")
INPUT_KDD_FOLDER = os.path.join(SOURCE_DATA_FOLDER, "kdd-cup-99-data")

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] - %(message)s')

# ==============================================================================
# DATA PROCESSING FUNCTIONS (Logic is the same, but paths are now robust)
# ==============================================================================

def process_nab_folder(input_dir, output_file):
    """Selects the main Twitter dataset from the NAB folder."""
    logging.info(f"Processing NAB folder: {input_dir}")
    if not os.path.exists(input_dir):
        logging.error(f"NAB source directory not found: {input_dir}")
        return

    target_file = None
    for root, dirs, files in os.walk(input_dir):
        if "Twitter_volume_AMZN.csv" in files:
            target_file = os.path.join(root, "Twitter_volume_AMZN.csv")
            break
            
    if target_file:
        df = pd.read_csv(target_file)
        if 'label' in df.columns:
            df.rename(columns={'label': 'is_anomaly'}, inplace=True)
        df.to_csv(output_file, index=False)
        logging.info(f"Successfully created NAB dataset at: {output_file}")
    else:
        logging.error("Could not find 'Twitter_volume_AMZN.csv'. Please check the NAB folder.")

def process_yahoo_folder(input_dir, output_file):
    """Combines all individual series from the Yahoo S5 folder."""
    logging.info(f"Processing Yahoo S5 folder: {input_dir}")
    if not os.path.exists(input_dir):
        logging.error(f"Yahoo source directory not found: {input_dir}")
        return

    benchmark_folder = os.path.join(input_dir, "A1Benchmark")
    if not os.path.isdir(benchmark_folder):
        benchmark_folder = input_dir

    file_paths = glob.glob(os.path.join(benchmark_folder, "real_*.csv"))
    if not file_paths:
        logging.error(f"No 'real_*.csv' files found in {benchmark_folder}.")
        return

    combined_df = pd.concat([pd.read_csv(f) for f in file_paths], ignore_index=True)
    if 'anomaly' in combined_df.columns:
        combined_df.rename(columns={'anomaly': 'is_anomaly'}, inplace=True)
        
    combined_df.to_csv(output_file, index=False)
    logging.info(f"Successfully combined {len(file_paths)} files into Yahoo dataset at: {output_file}")

def process_kdd_folder(input_dir, output_file):
    """Processes the KDD'99 data file, adding headers and a binary label."""
    logging.info(f"Processing KDD'99 folder: {input_dir}")
    if not os.path.exists(input_dir):
        logging.error(f"KDD source directory not found: {input_dir}")
        return

    # The file is often named 'kddcup.data_10_percent' without an extension
    target_file = os.path.join(input_dir, "kddcup.data_10_percent")
    if not os.path.exists(target_file):
        logging.error(f"Could not find 'kddcup.data_10_percent' in {input_dir}.")
        return
        
    column_names = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 
        'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
        'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 
        'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 
        'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 
        'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 
        'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 
        'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 
        'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label'
    ]
    
    df = pd.read_csv(target_file, header=None, names=column_names)
    df['is_anomaly'] = df['label'].apply(lambda x: 0 if x == 'normal.' else 1)
    
    # Select only numeric columns + the new binary label for simplicity
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    df_final = df[numeric_cols + ['is_anomaly']]
    
    df_final.to_csv(output_file, index=False)
    logging.info(f"Successfully processed KDD'99 dataset at: {output_file}")


def main():
    """Main function to orchestrate the preparation of all datasets."""
    logging.info("Starting dataset preparation process...")
    
    # Create destination directory if it doesn't exist
    os.makedirs(FINAL_DATASETS_FOLDER, exist_ok=True)
    
    # Process all datasets
    process_nab_folder(INPUT_NAB_FOLDER, os.path.join(FINAL_DATASETS_FOLDER, "NAB_realTweets.csv"))
    process_yahoo_folder(INPUT_YAHOO_FOLDER, os.path.join(FINAL_DATASETS_FOLDER, "Yahoo_S5_A1.csv"))
    process_kdd_folder(INPUT_KDD_FOLDER, os.path.join(FINAL_DATASETS_FOLDER, "KDD99.csv"))
    
    logging.info("="*50)
    logging.info("All datasets have been processed. The 'datasets' folder is ready!")
    logging.info("="*50)

if __name__ == "__main__":
    main()