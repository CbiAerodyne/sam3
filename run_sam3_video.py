import os
import argparse
import sys
import torch
import cv2
import glob
import numpy as np
import shutil
import math
from PIL import Image
from tqdm import tqdm

# Add the repository root to sys.path
repo_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(repo_root)

import sam3
from sam3.model_builder import build_sam3_video_predictor
from sam3.visualization_utils import render_masklet_frame

def setup_args():
    parser = argparse.ArgumentParser(description="Run SAM3 on a video processing in chunks.")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the video file.")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for segmentation (e.g., 'car').")
    parser.add_argument("--output_dir", type=str, default="output_sam3", help="Directory to save visualized outputs.")
    parser.add_argument("--model_path", type=str, default="checkpoints/sam3.pt", help="Path to the SAM3 checkpoint file.")
    parser.add_argument("--chunk_duration", type=float, default=5.0, help="Duration of each chunk in minutes.")
    parser.add_argument("--fps", type=float, default=5.0, help="Target FPS for processing.")
    return parser.parse_args()

def process_chunk(chunk_idx, frames_dir, prompt, predictor, output_dir):
    """
    Process a single chunk of frames (already saved in frames_dir).
    """
    # 1. Start Session
    print(f"  [Chunk {chunk_idx}] Starting inference session...")
    try:
        response = predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=frames_dir, # Pass the directory of frames
            )
        )
        session_id = response["session_id"]
    except Exception as e:
        print(f"  [Chunk {chunk_idx}] Error starting session: {e}")
        return

    # 2. Add Text Prompt (Always at frame 0 of the chunk, treating it as a fresh video)
    print(f"  [Chunk {chunk_idx}] Adding prompt '{prompt}' at frame 0...")
    try:
        response = predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=0,
                text=prompt,
            )
        )
    except Exception as e:
        print(f"  [Chunk {chunk_idx}] Error adding prompt: {e}")
        return

    # 3. Propagate
    print(f"  [Chunk {chunk_idx}] Propagating masks...")
    outputs_per_frame = {}
    try:
        for response in predictor.handle_stream_request(
            request=dict(
                type="propagate_in_video",
                session_id=session_id,
            )
        ):
            outputs_per_frame[response["frame_index"]] = response["outputs"]
    except Exception as e:
        print(f"  [Chunk {chunk_idx}] Error during propagation: {e}")
        return

    # 4. Visualize and Save
    frame_files = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    
    # Create a subfolder for this chunk's outputs to avoid name collisions/confusion
    # or save with original frame names if possible.
    # The user said "treat each ... as separate video files", so separate folders might be cleaner,
    # but a single unified folder is often preferred for valid outputs.
    # Let's use a unified folder but preserve the filenames we generated (which are probably 8-digit frame indices).
    
    os.makedirs(output_dir, exist_ok=True)

    print(f"  [Chunk {chunk_idx}] Saving visualizations to {output_dir}...")
    for local_frame_idx, outputs in outputs_per_frame.items():
        if local_frame_idx >= len(frame_files):
            continue
        
        frame_path = frame_files[local_frame_idx]
        original_frame_name = os.path.basename(frame_path)
        
        frame_image = np.array(Image.open(frame_path))
        overlay = render_masklet_frame(frame_image, outputs, frame_idx=local_frame_idx)
        
        save_path = os.path.join(output_dir, original_frame_name)
        Image.fromarray(overlay).save(save_path)
    
    # 5. Reset/Close Session to free memory
    predictor.handle_request(request=dict(type="close_session", session_id=session_id))
    print(f"  [Chunk {chunk_idx}] Done.")

def extract_frames_for_chunk(video_path, start_time_sec, end_time_sec, target_fps, temp_dir):
    """
    Extract frames from video_path between start_time and end_time at target_fps.
    Save them to temp_dir.
    """
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {video_path}")
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate start and end frames
    start_frame = int(start_time_sec * video_fps)
    end_frame = int(end_time_sec * video_fps)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if start_frame >= total_frames:
        cap.release()
        return False # End of video
    
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
            
        # Only save if it aligns with the step relative to the START of the video (to maintain consistent FPS spacing)
        # OR just relative to start of chunk? 
        # Using (current_frame % frame_step == 0) aligns it globally.
        if current_frame % frame_step == 0:
            # Save with frame index to keep global ordering in valid filenames
            save_path = os.path.join(temp_dir, f"{current_frame:08d}.jpg")
            cv2.imwrite(save_path, frame)
            saved_count += 1
            
        current_frame += 1
    
    cap.release()
    return saved_count > 0

def main():
    args = setup_args()
    
    # 1. Setup SAM3 Predictor
    print("Building SAM3 video predictor...")
    gpus_to_use = range(torch.cuda.device_count())
    if not gpus_to_use:
        print("Warning: No CUDA devices found.")
    
    try:
        predictor = build_sam3_video_predictor(
            gpus_to_use=gpus_to_use,
            checkpoint_path=args.model_path
        )
    except Exception as e:
         print(f"Error building predictor: {e}")
         return

    # 2. Get Video Info
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {args.video_path}")
        return
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / video_fps
    cap.release()
    
    print(f"Video Duration: {duration_sec:.2f}s ({total_frames} frames, {video_fps} fps)")
    
    chunk_duration_sec = args.chunk_duration * 60
    num_chunks = math.ceil(duration_sec / chunk_duration_sec)
    
    temp_frames_dir = os.path.join(args.output_dir, "temp_frames_buffer")
    
    # 3. Process Loops
    for i in range(num_chunks):
        start_time = i * chunk_duration_sec
        
        # NOTE: We iterate until the end of video, effectively treating each chunk as a segment.
        # But we must stop extracting when we hit the end of the chunk duration.
        end_time = min((i + 1) * chunk_duration_sec, duration_sec)
        
        print(f"\nProcessing Chunk {i+1}/{num_chunks}...")
        
        has_frames = extract_frames_for_chunk(
            args.video_path, 
            start_time, 
            end_time, 
            args.fps, 
            temp_frames_dir
        )
        
        if has_frames:
            process_chunk(i+1, temp_frames_dir, args.prompt, predictor, args.output_dir)
        else:
            print(f"Skipping chunk {i+1} (no frames extracted).")
            
    # Cleanup
    if os.path.exists(temp_frames_dir):
        shutil.rmtree(temp_frames_dir)
        
    print(f"\nAll Done! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
