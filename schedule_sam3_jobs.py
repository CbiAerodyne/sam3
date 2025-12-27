import os
import argparse
import pandas as pd
import glob
import re
import subprocess
from datetime import datetime, timedelta

def setup_args():
    parser = argparse.ArgumentParser(description="Map data timestamps to video files and schedule SAM3 processing.")
    parser.add_argument("--data_csv", type=str, required=True, help="Path to the merged Igor CSV.")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing video files.")
    parser.add_argument("--output_dir", type=str, default="sam3_results", help="Base directory for outputs.")
    parser.add_argument("--gap_threshold", type=float, default=2.0, help="Max gap in seconds to merge intervals.")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt for SAM3.")
    parser.add_argument("--fps", type=float, default=3.0, help="FPS for SAM3.")
    parser.add_argument("--min_duration", type=float, default=5.0, help="Min duration (sec) to keep interval.")
    parser.add_argument("--max_duration", type=float, default=20.0, help="Max duration (sec) per job. Splits long intervals.")
    parser.add_argument("--dry_run", action="store_true", help="Print jobs without running.")
    return parser.parse_args()

def parse_video_filename(filename):
    # Expected format: webcam_test_2025.08.06_00h46m31s.mp4
    match = re.search(r"(\d{4}\.\d{2}\.\d{2})_(\d{2})h(\d{2})m(\d{2})s", filename)
    if match:
        date_str, h, m, s = match.groups()
        dt_str = f"{date_str} {h}:{m}:{s}"
        dt = datetime.strptime(dt_str, "%Y.%m.%d %H:%M:%S")
        return dt
    return None

def get_video_duration(video_path):
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps > 0:
        return frames / fps
    return 0

def main():
    args = setup_args()
    
    # 1. Load Data
    print(f"Loading data from {args.data_csv}...")
    df = pd.read_csv(args.data_csv)
    if "timestamp" not in df.columns:
        print("Error: 'timestamp' column not found in CSV.")
        return
        
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.sort_values('timestamp', inplace=True)
    
    # 2. Scan Videos
    print(f"Scanning videos in {args.video_dir}...")
    video_files = sorted(glob.glob(os.path.join(args.video_dir, "*.mp4")))
    video_index = [] 
    
    for v in video_files:
        start_dt = parse_video_filename(os.path.basename(v))
        if start_dt:
            duration = get_video_duration(v)
            if duration > 0:
                end_dt = start_dt + timedelta(seconds=duration)
                video_index.append({
                    'path': v,
                    'start': start_dt,
                    'end': end_dt,
                    'start_ts': start_dt.timestamp(),
                    'end_ts': end_dt.timestamp()
                })
                # print(f"  Found {os.path.basename(v)}: {start_dt} -> {end_dt} ({duration/60:.1f} min)")
    
    if not video_index:
        print("No valid videos found.")
        return

    # 3. Build Intervals from Data
    print("Building processing intervals...")
    timestamps = df['timestamp'].tolist()
    raw_intervals = []
    
    if not timestamps:
        print("No timestamps in data.")
        return
        
    current_start = timestamps[0]
    current_end = timestamps[0]
    
    for t in timestamps[1:]:
        gap = (t - current_end).total_seconds()
        if gap <= args.gap_threshold:
            current_end = t
        else:
            raw_intervals.append((current_start, current_end))
            current_start = t
            current_end = t
    raw_intervals.append((current_start, current_end))
    
    # Filter by duration
    intervals = []
    print(f"Filtering intervals < {args.min_duration}s...")
    for start, end in raw_intervals:
        duration = (end - start).total_seconds()
        if duration >= args.min_duration:
            intervals.append((start, end))
    
    print(f"Found {len(intervals)} valid high-speed intervals (from {len(raw_intervals)} raw).")

    # 4. Map Intervals to Videos (Splitting if needed)
    jobs = []
    
    for (req_start, req_end) in intervals:
        for vid in video_index:
            overlap_start = max(req_start, vid['start'])
            overlap_end = min(req_end, vid['end'])
            
            if overlap_start < overlap_end:
                # Iterate through overlap in max_duration chunks
                current_sub_start = overlap_start
                while current_sub_start < overlap_end:
                    current_sub_end = min(current_sub_start + timedelta(seconds=args.max_duration), overlap_end)
                    
                    job_start_offset = (current_sub_start - vid['start']).total_seconds()
                    job_end_offset = (current_sub_end - vid['start']).total_seconds()
                    
                    jobs.append({
                        'video_path': vid['path'],
                        'start_offset': job_start_offset,
                        'end_offset': job_end_offset,
                        'video_start_ts': vid['start_ts'],
                        'duration': job_end_offset - job_start_offset
                    })
                    
                    current_sub_start = current_sub_end

    print(f"Generated {len(jobs)} processing jobs.")

    # 5. Execute Jobs
    import sys
    
    for idx, job in enumerate(jobs):
        print(f"\n[Job {idx+1}/{len(jobs)}] Processing {os.path.basename(job['video_path'])}")
        print(f"  Range: {job['start_offset']:.1f}s to {job['end_offset']:.1f}s (Duration: {job['duration']:.1f}s)")
        
        cmd = [
            sys.executable, "run_sam3_video.py",
            "--video_path", job['video_path'],
            "--prompt", args.prompt,
            "--output_dir", args.output_dir,
            "--fps", str(args.fps),
            "--start_time", str(job['start_offset']),
            "--end_time", str(job['end_offset']),
            "--video_start_timestamp", str(job['video_start_ts'])
        ]
        
        if args.dry_run:
            print(f"  Command: {' '.join(cmd)}")
        else:
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"  Job failed: {e}")

if __name__ == "__main__":
    main()
