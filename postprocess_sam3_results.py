import os
import glob
import pickle
import argparse
import pandas as pd
import numpy as np

def setup_args():
    parser = argparse.ArgumentParser(description="Process SAM3 inference pickle files into metrics.")
    parser.add_argument("--results_dir", type=str, required=True, help="Directory containing inference_results_*.pkl files.")
    parser.add_argument("--output_csv", type=str, default="object_metrics.csv", help="Output CSV filename.")
    return parser.parse_args()

def calculate_direction(area_series):
    """
    Determine if object is approaching or receding based on area change.
    Simple heuristic: Linear regression slope of area vs index.
    """
    if len(area_series) < 3:
        return "unknown"
    
    x = np.arange(len(area_series))
    y = area_series.values
    
    # Simple slope
    try:
        slope, _ = np.polyfit(x, y, 1)
        if slope > 0:
            return "approaching"
        elif slope < 0:
            return "receding"
        else:
            return "stationary"
    except:
        return "error"

def main():
    args = setup_args()
    
    if not os.path.exists(args.results_dir):
        print(f"Error: Directory {args.results_dir} not found.")
        return

    pkl_files = sorted(glob.glob(os.path.join(args.results_dir, "inference_results_*.pkl")))
    if not pkl_files:
        print(f"No pickle files found in {args.results_dir}.")
        return

    print(f"Found {len(pkl_files)} pickle files. Processing...")

    all_data = []

    for pkl_path in pkl_files:
        try:
            with open(pkl_path, "rb") as f:
                chunk_data = pickle.load(f)
                
            # chunk_data is a list of frame_dicts
            # frame_dict: {'file': str, 'timestamp': float/None, 'objects': list}
            
            for frame in chunk_data:
                fname = frame['file']
                ts = frame['timestamp']
                
                for obj in frame['objects']:
                    obj_id = obj['id']
                    score = obj['score']
                    box = obj['box'] # [x, y, w, h] normalized
                    
                    # Calculate Metrics
                    area = box[2] * box[3]
                    center_x = box[0] + box[2]/2
                    center_y = box[1] + box[3]/2
                    
                    all_data.append({
                        "filename": fname,
                        "timestamp": ts,
                        "datetime": pd.to_datetime(ts, unit='s') if ts else None,
                        "object_id": obj_id,
                        "score": score,
                        "box_x": box[0],
                        "box_y": box[1],
                        "box_w": box[2],
                        "box_h": box[3],
                        "box_area": area,
                        "center_x": center_x,
                        "center_y": center_y,
                        "pkl_source": os.path.basename(pkl_path)
                    })
                    
        except Exception as e:
            print(f"Error reading {pkl_path}: {e}")

    if not all_data:
        print("No object data found in pickle files.")
        return

    df = pd.DataFrame(all_data)
    
    # Direction Analysis
    # We group by object_id (and maybe source/chunk if IDs aren't unique across chunks - 
    # SAM3 usually resets IDs per session/chunk unless configured otherwise.
    # WE ASSUME IDs are unique per chunk, but maybe not across chunks.
    # Let's trust "object_id" combined with "pkl_source" or "timestamp" continuity.
    # For now, let's treat (pkl_source, object_id) as unique track key if simple.
    # Actually, SAM3 IDs restart at 0 for each prompt/session (chunk).
    
    print("Analyzing movement direction...")
    df['direction'] = 'unknown'
    
    # Process each chunk's objects separately since IDs likely reset
    for pkl_source in df['pkl_source'].unique():
        chunk_df = df[df['pkl_source'] == pkl_source]
        
        for obj_id in chunk_df['object_id'].unique():
            obj_mask = (df['pkl_source'] == pkl_source) & (df['object_id'] == obj_id)
            obj_track = df[obj_mask].sort_values('timestamp' if df['timestamp'].notnull().any() else 'filename')
            
            direction = calculate_direction(obj_track['box_area'])
            df.loc[obj_mask, 'direction'] = direction

    # Save
    out_path = os.path.join(args.results_dir, args.output_csv)
    df.to_csv(out_path, index=False)
    
    print(f"Successfully saved metrics to {out_path}")
    print(f"Total Object Detections: {len(df)}")
    print("\nSample Data:")
    print(df[['filename', 'object_id', 'box_area', 'direction']].head())

if __name__ == "__main__":
    main()
