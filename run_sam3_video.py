import os
import argparse
import sys
import torch
import cv2
import glob
import numpy as np
import shutil
import math
import time
import psutil
import subprocess
from datetime import datetime
from PIL import Image
from tqdm import tqdm
from contextlib import nullcontext

# Add the repository root to sys.path
repo_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(repo_root)

def setup_args():
    parser = argparse.ArgumentParser(description="Run SAM3 on a video processing in chunks.")
    
    # Common Args
    parser.add_argument("--video_path", type=str, required=False, help="Path to the video file (Coordinator mode).")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for segmentation.")
    parser.add_argument("--output_dir", type=str, default="output_sam3", help="Directory to save visualized outputs.")
    parser.add_argument("--model_path", type=str, default="checkpoints/sam3.pt", help="Path to the SAM3 checkpoint file.")
    parser.add_argument("--chunk_duration", type=float, default=5.0, help="Duration of each chunk in minutes.")
    parser.add_argument("--fps", type=float, default=5.0, help="Target FPS for processing.")
    
    # New Targeted Processing Args
    parser.add_argument("--start_time", type=float, help="Start time (sec) for targeted processing.")
    parser.add_argument("--end_time", type=float, help="End time (sec) for targeted processing.")
    parser.add_argument("--video_start_timestamp", type=float, help="Epoch timestamp of video start for file naming.")
    
    # Worker Mode Args
    parser.add_argument("--worker_mode", action="store_true", help="Run in worker mode (internal use).")
    parser.add_argument("--frames_dir", type=str, help="Directory of frames to process (Worker mode).")
    parser.add_argument("--chunk_idx", type=int, help="Index of the chunk being processed (Worker mode).")
    parser.add_argument("--video_fps", type=float, help="Original FPS of the video (passed from Coordinator).")
    parser.add_argument("--use_bfloat16", action="store_true", help="Use bfloat16 precision for improved VRAM/performance.")
    
    return parser.parse_args()

class SystemMonitor:
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        
    def get_stats(self):
        # CPU
        cpu_percent = psutil.cpu_percent(interval=None)
        
        # RAM
        ram_info = psutil.virtual_memory()
        ram_used_gb = ram_info.used / (1024**3)
        ram_total_gb = ram_info.total / (1024**3)
        
        # GPU
        gpu_stats = "N/A"
        if torch.cuda.is_available():
            try:
                # Using torch.cuda to get memory info for device 0 (simplification)
                # Note: this is memory allocated by PyTorch, not total system VRAM usage
                allocated = torch.cuda.memory_allocated(0) / (1024**3)
                reserved = torch.cuda.memory_reserved(0) / (1024**3)
                gpu_stats = f"{allocated:.1f}GB alloc / {reserved:.1f}GB res"
            except:
                 gpu_stats = "Err"
                 
        return f"CPU: {cpu_percent:.1f}% | RAM: {ram_used_gb:.1f}/{ram_total_gb:.1f}GB | GPU: {gpu_stats}"

# ==================== Worker Logic ====================

def worker_main(args):
    # Delayed inputs to save import time in parent
    import sam3
    from sam3.model_builder import build_sam3_video_predictor
    from sam3.visualization_utils import render_masklet_frame
    
    chunk_idx = args.chunk_idx
    frames_dir = args.frames_dir
    prompt = args.prompt
    model_path = args.model_path
    output_dir = args.output_dir
    
    monitor = SystemMonitor()
    print(f"  [Worker {chunk_idx}] Process Started. PID: {os.getpid()}")
    print(f"  [Worker {chunk_idx}] Initial Stats: {monitor.get_stats()}")

    # 0. Build Predictor
    print(f"  [Worker {chunk_idx}] Building SAM3 predictor...")
    t0 = time.time()
    gpus_to_use = range(torch.cuda.device_count())
    try:
        predictor = build_sam3_video_predictor(
            gpus_to_use=gpus_to_use,
            checkpoint_path=model_path
        )
        # We rely on torch.autocast for bfloat16, manual casting caused mismatches.
            
    except Exception as e:
         print(f"  [Worker {chunk_idx}] Error building predictor: {e}")
         return
    print(f"  [Worker {chunk_idx}] Model built in {time.time() - t0:.1f}s")
    
    # Context Manager Helper
    autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if args.use_bfloat16 else nullcontext()

    # 1. Start Session
    print(f"  [Worker {chunk_idx}] Starting inference session...")
    try:
        with autocast_ctx:
            response = predictor.handle_request(
                request=dict(
                    type="start_session",
                    resource_path=frames_dir,
                )
            )
        session_id = response["session_id"]
    except Exception as e:
        print(f"  [Worker {chunk_idx}] Error starting session: {e}")
        return

    # 2. Add Prompt
    print(f"  [Worker {chunk_idx}] Adding prompt '{prompt}'...")
    try:
        with autocast_ctx:
            predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=session_id,
                    frame_index=0,
                    text=prompt,
                )
            )
    except Exception as e:
         print(f"  [Worker {chunk_idx}] Error adding prompt: {e}")
         return

    # 3. Propagate
    print(f"  [Worker {chunk_idx}] Propagating masks...")
    
    outputs_per_frame = {}
    current_iter = 0
    t_start = time.time()
    t_last_log = t_start
    
    try:
        frame_files = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
        num_frames = len(frame_files)
        
        with autocast_ctx:
            stream_generator = predictor.handle_stream_request(
                request=dict(
                    type="propagate_in_video",
                    session_id=session_id,
                )
            )
            
            for response in stream_generator:
                current_iter += 1
                outputs_per_frame[response["frame_index"]] = response["outputs"]
                
                if current_iter % 10 == 0:
                    t_now = time.time()
                    elapsed = t_now - t_last_log
                    speed = 10 / elapsed if elapsed > 0 else 0
                    stats = monitor.get_stats()
                    print(f"    [Worker {chunk_idx}] Frame {current_iter}/{num_frames} | {speed:.2f} it/s | {stats}")
                    t_last_log = t_now
                
    except Exception as e:
        print(f"  [Worker {chunk_idx}] Error during propagation: {e}")
        return

    # 4. Visualize and Save
    os.makedirs(output_dir, exist_ok=True)
    print(f"  [Worker {chunk_idx}] Saving visualizations to {output_dir}...")
    
    for local_frame_idx, outputs in outputs_per_frame.items():
        if local_frame_idx >= len(frame_files):
            continue
        
        frame_path = frame_files[local_frame_idx]
        frame_image = np.array(Image.open(frame_path))
        overlay = render_masklet_frame(frame_image, outputs, frame_idx=local_frame_idx)
        
        # Naming Logic
        out_filename = os.path.basename(frame_path) 
        
        if args.video_start_timestamp and args.video_fps and args.video_fps > 0:
            try:
                # Video Filename Frame Number (from extraction)
                original_frame_num = int(os.path.splitext(os.path.basename(frame_path))[0])
                
                # Calculate time
                time_offset_sec = original_frame_num / args.video_fps
                frame_ts = args.video_start_timestamp + time_offset_sec
                frame_dt = datetime.fromtimestamp(frame_ts)
                
                out_filename = frame_dt.strftime("%Y.%m.%d_%Hh%Mm%Ss_%f") + ".jpg"
            except Exception as e:
                print(f"Warning: Could not format timestamp for {frame_path}: {e}")

        save_path = os.path.join(output_dir, out_filename)
        Image.fromarray(overlay).save(save_path)
    
    print(f"  [Worker {chunk_idx}] Done.")


# ==================== Coordinator Logic ====================

def extract_frames_for_chunk(video_path, start_time_sec, end_time_sec, target_fps, temp_dir):
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {video_path}")
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    
    start_frame = int(start_time_sec * video_fps)
    end_frame = int(end_time_sec * video_fps)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if start_frame >= total_frames:
        cap.release()
        return False, video_fps
    
    end_frame = min(end_frame, total_frames)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frame_step = max(1, int(video_fps / target_fps))
    
    current_frame = start_frame
    saved_count = 0
    
    print(f"Extracting frames for chunk: {start_time_sec:.1f}s to {end_time_sec:.1f}s (Frames {start_frame}-{end_frame})")
    
    while current_frame < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
            
        if current_frame % frame_step == 0:
            # Save as 00000123.jpg (Video Frame Number)
            save_path = os.path.join(temp_dir, f"{current_frame:08d}.jpg")
            cv2.imwrite(save_path, frame)
            saved_count += 1
            
        current_frame += 1
    
    cap.release()
    return saved_count > 0, video_fps

def coordinator_main(args):
    if not args.video_path:
        print("Error: --video_path is required in coordinator mode.")
        return

    is_targeted = (args.start_time is not None) and (args.end_time is not None)

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {args.video_path}")
        return
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / video_fps
    cap.release()
    
    if is_targeted:
        chunks = [(args.start_time, args.end_time)]
        print(f"Processing Target Interval: {args.start_time}s to {args.end_time}s")
    else:
        print(f"Video Duration: {duration_sec:.2f}s ({total_frames} frames, {video_fps} fps)")
        chunk_duration_sec = args.chunk_duration * 60
        num_chunks = math.ceil(duration_sec / chunk_duration_sec)
        chunks = []
        for i in range(num_chunks):
            start = i * chunk_duration_sec
            end = min((i + 1) * chunk_duration_sec, duration_sec)
            chunks.append((start, end))

    temp_frames_dir = os.path.join(args.output_dir, "temp_frames_buffer")
    
    for i, (start_time, end_time) in enumerate(chunks):
        
        print(f"\nProcessing Chunk {i+1}...")
        
        has_frames, vid_fps = extract_frames_for_chunk(
            args.video_path, 
            start_time, 
            end_time, 
            args.fps, 
            temp_frames_dir
        )
        
        if has_frames:
            print(f"Launching worker process...")
            cmd = [
                sys.executable,
                __file__,
                "--worker_mode",
                "--frames_dir", temp_frames_dir,
                "--output_dir", args.output_dir,
                "--prompt", args.prompt,
                "--model_path", args.model_path,
                "--chunk_idx", str(i+1),
                "--fps", str(args.fps), 
                "--video_fps", str(vid_fps) 
            ]
            
            if args.video_start_timestamp:
                cmd.extend(["--video_start_timestamp", str(args.video_start_timestamp)])
            
            if args.use_bfloat16:
                cmd.append("--use_bfloat16")

            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Worker process failed: {e}")
        else:
            print(f"Skipping chunk {i+1} (no frames extracted).")
            
    if os.path.exists(temp_frames_dir):
        shutil.rmtree(temp_frames_dir)
        
    print(f"\nAll Done! Results saved to {args.output_dir}")

def main():
    args = setup_args()
    if args.worker_mode:
        worker_main(args)
    else:
        coordinator_main(args)

if __name__ == "__main__":
    main()
