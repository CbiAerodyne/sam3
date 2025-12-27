import os
import glob
import argparse
import numpy as np
import pandas as pd
import igor2.binarywave

def setup_args():
    parser = argparse.ArgumentParser(description="Merge Igor Binary Wave (.ibw) files into a single CSV.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing .ibw files.")
    parser.add_argument("--output_dir", type=str, default="CB", help="Directory to export the CSV.")
    parser.add_argument("--min_speed", type=float, default=0.5, help="Minimum TruckSpeed_mps to keep rows.")
    return parser.parse_args()

def load_ibw(filepath):
    """
    Load an .ibw file and return the numpy array data.
    """
    try:
        wave = igor2.binarywave.load(filepath)
        # The structure is usually wave['wave']['wData']
        # Note: igor2 might return a structure where data is accessible. 
        # Checking commonly valid access patterns.
        if 'wave' in wave and 'wData' in wave['wave']:
            return np.array(wave['wave']['wData'])
        else:
            print(f"Warning: Unexpected structure in {filepath}. Keys: {wave.keys()}")
            return None
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def main():
    args = setup_args()
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist.")
        return

    # 1. Identify files
    ibw_files = glob.glob(os.path.join(args.input_dir, "*.ibw"))
    if not ibw_files:
        print(f"No .ibw files found in {args.input_dir}")
        return
        
    print(f"Found {len(ibw_files)} .ibw files.")

    # 2. Find Key Files
    time_file = None
    speed_file = None
    
    # We map "filename (no extension)" -> "filepath"
    file_map = {}
    
    for f in ibw_files:
        basename = os.path.basename(f)
        name_no_ext = os.path.splitext(basename)[0]
        file_map[name_no_ext] = f
        
        # Use lowercase comparison for robustness if needed, but exact per user request
        if basename == "datetimeUTC2.ibw":
            time_file = f
        if basename == "TruckSpeed_mps.ibw":
            speed_file = f

    if not time_file:
        print("Error: Critical file 'datetimeUTC2.ibw' not found in input directory.")
        return

    # 3. Load Time Index
    print(f"Loading Time Index from {time_file}...")
    time_data = load_ibw(time_file)
    if time_data is None:
        return
    
    # Ensure 1D array
    time_data = time_data.flatten()
    
    # Initialize DataFrame with Time
    df = pd.DataFrame({"datetimeUTC2": time_data})
    
    # 4. Load & Merge Other Files
    for name, filepath in file_map.items():
        if filepath == time_file:
            continue
            
        print(f"Loading {name}...")
        data = load_ibw(filepath)
        
        if data is None:
            continue
            
        # Flatten to 1D
        data = data.flatten()
        
        # Check alignment
        if len(data) != len(time_data):
            print(f"Warning: Length mismatch for {name}. Expected {len(time_data)}, got {len(data)}. Skipping.")
            continue
            
        df[name] = data

    print(f"Merged Data Shape: {df.shape}")

    # 5. Filter by Speed
    if "TruckSpeed_mps" in df.columns:
        print(f"Filtering rows where TruckSpeed_mps > {args.min_speed}...")
        original_count = len(df)
        df_filtered = df[df["TruckSpeed_mps"] > args.min_speed].copy()
        print(f"Rows retained: {len(df_filtered)} / {original_count}")
    else:
        print("Warning: 'TruckSpeed_mps' column not found. Skipping filtering.")
        df_filtered = df.copy()

    # 6. Convert Timestamps (Igor Epoch: 1904-01-01)
    if "datetimeUTC2" in df_filtered.columns:
        print("Converting Igor timestamps (1904 epoch)...")
        # 1904-01-01 is the Mac/Igor epoch
        base_date = pd.Timestamp("1904-01-01")
        # Ensure it's numeric
        df_filtered["datetimeUTC2"] = pd.to_numeric(df_filtered["datetimeUTC2"], errors='coerce')
        # Convert to timedelta and add to base date
        df_filtered["timestamp"] = base_date + pd.to_timedelta(df_filtered["datetimeUTC2"], unit='s')
        
        # Sort by timestamp
        df_filtered.sort_values("timestamp", inplace=True)
        print(f"Time range: {df_filtered['timestamp'].min()} to {df_filtered['timestamp'].max()}")
    else:
        print("Warning: datetimeUTC2 column missing, cannot calculate real timestamps.")

    # 7. Export
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "merged_igor_data.csv")
    df_filtered.to_csv(output_path, index=False)
    print(f"Successfully saved merged data to {output_path}")

if __name__ == "__main__":
    main()
