import os
import argparse
import sys
import torch
import cv2
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm

# Add the repository root to sys.path
repo_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(repo_root)

import sam3
from sam3.model_builder import build_sam3_video_predictor
from sam3.visualization_utils import visualize_formatted_frame_output

def setup_args():
    parser = argparse.ArgumentParser(description="Run SAM3 on a video with a text prompt.")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the video file or directory of frames.")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for segmentation (e.g., 'car').")
    parser.add_argument("--output_dir", type=str, default="output_sam3", help="Directory to save visualized outputs.")
    parser.add_argument("--frame_index", type=int, default=0, help="Frame index to apply the text prompt.")
    parser.add_argument("--model_path", type=str, default="checkpoints/sam3.pt", help="Path to the SAM3 checkpoint file.")
    return parser.parse_args()

class VideoReader:
    def __init__(self, video_path):
        self.video_path = video_path
        self.is_video_file = os.path.isfile(video_path) and (
            video_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
        )
        self.frame_paths = []
        self.num_frames = 0
        
        if self.is_video_file:
            cap = cv2.VideoCapture(video_path)
            self.num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        elif os.path.isdir(video_path):
            self.frame_paths = sorted(glob.glob(os.path.join(video_path, "*")))
            # Filter for images
            self.frame_paths = [p for p in self.frame_paths if p.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            try:
                # integer sort
                self.frame_paths.sort(key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
            except ValueError:
                # fallback sort
                self.frame_paths.sort()
            self.num_frames = len(self.frame_paths)
        else:
             raise ValueError(f"Invalid video_path: {video_path}")

    def get_frame(self, index):
        if index < 0 or index >= self.num_frames:
            return None
            
        if self.is_video_file:
            cap = cv2.VideoCapture(self.video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ret, frame = cap.read()
            cap.release()
            if ret:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                return None
        else:
            return np.array(Image.open(self.frame_paths[index]))
            
    def __len__(self):
        return self.num_frames

def main():
    args = setup_args()
    
    # 1. Setup SAM3 Predictor
    print("Building SAM3 video predictor...")
    gpus_to_use = range(torch.cuda.device_count())
    if not gpus_to_use:
        # Fallback to CPU if no GPU (though SAM3 might require GPU, let's try to be safe)
        print("Warning: No CUDA devices found. SAM3 might fail or be extremely slow on CPU.")
    
    try:
        predictor = build_sam3_video_predictor(
            gpus_to_use=gpus_to_use,
            checkpoint_path=args.model_path
        )
    except Exception as e:
         print(f"Error building predictor: {e}")
         return

    # 2. Prepare Video Reader (Lazy Load)
    print(f"Preparing video reader for {args.video_path}...")
    try:
        video_reader = VideoReader(args.video_path)
    except Exception as e:
        print(f"Error loading video info: {e}")
        return
        
    print(f"Video has {len(video_reader)} frames.")

    # 3. Start Session
    print("Starting inference session...")
    try:
        response = predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=args.video_path,
            )
        )
        session_id = response["session_id"]
    except Exception as e:
        print(f"Error starting session: {e}")
        return

    # 4. Add Text Prompt
    print(f"Adding prompt '{args.prompt}' at frame {args.frame_index}...")
    try:
        response = predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=args.frame_index,
                text=args.prompt,
            )
        )
        # We can visualize the first frame here if we want, but let's propagate first.
    except Exception as e:
        print(f"Error adding prompt: {e}")
        return

    # 5. Propagate
    print("Propagating masks through the video...")
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
        print(f"Error during propagation: {e}")
        return
        
    # 6. Visualize and Save
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving visualizations to {args.output_dir}...")
    
    # Import render_masklet_frame locally if needed or ensure it is imported
    from sam3.visualization_utils import render_masklet_frame
    
    for frame_idx, outputs in tqdm(sorted(outputs_per_frame.items())):
        if frame_idx >= len(video_reader):
            continue
            
        frame_image = video_reader.get_frame(frame_idx)
        if frame_image is None:
            print(f"Warning: Could not read frame {frame_idx}")
            continue

        overlay = render_masklet_frame(frame_image, outputs, frame_idx=frame_idx)
        
        # Save the image
        save_path = os.path.join(args.output_dir, f"{frame_idx:05d}.jpg")
        Image.fromarray(overlay).save(save_path)
        
    print(f"Done! Check {args.output_dir} for results.")

if __name__ == "__main__":
    main()
