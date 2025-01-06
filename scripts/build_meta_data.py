import os
import sys
import cv2
import glob
import pandas as pd
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.video_frame import parser_video


def save_frame(frame, frames_dir, frame_idx):
    """Save a single frame to disk."""
    frame_filename = os.path.join(frames_dir, f"frame_{frame_idx:04d}.png")
    cv2.imwrite(frame_filename, frame)


def save_frames(frames_root, video_rel_path, frame_list, max_workers=4):
    """Save frames using multiple threads."""
    sub_dir, ext = os.path.splitext(video_rel_path)
    frames_dir = os.path.join(frames_root, sub_dir)
    os.makedirs(frames_dir, exist_ok=True)

    # Save frames using multiple threads
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for frame_idx, frame in enumerate(frame_list):
            executor.submit(save_frame, frame, frames_dir, frame_idx)


def process_video(video_path, frames_root, videos_root, max_workers):
    """Process a video: parse, save frames, and gather metadata."""
    video_rel_path = os.path.relpath(video_path, videos_root)
    
    try:
        # Load video, parse video, save frames
        frame_list, frame_count = parser_video(video_path)
        if not frame_list:
            raise Exception("Empty frame list.")

        save_frames(frames_root, video_rel_path, frame_list, max_workers)

        video_meta = {
            "frame_count": frame_count,
            "resolution": frame_list[0].shape
        }
        return video_rel_path, video_meta

    except Exception as e:
        # Return the failed video path and None as metadata
        print(f"[ERROR] Failed to process {video_path}: {e}")
        return video_rel_path, None


def save_video_info_to_csv(video_rel_paths, video_meta_info, csv_path):
    """Save the video metadata to a CSV file."""
    df = pd.DataFrame({
        "video_rel_path": video_rel_paths,
        "frame_count": [info["frame_count"] if info else None for info in video_meta_info],
        "resolution": [f"{info['resolution'][0]}x{info['resolution'][1]}" if info else None for info in video_meta_info]
    })
    
    df = df.dropna()  # Drop rows with None values (failed videos)
    df.to_csv(csv_path, index=False)
    print(f"CSV saved to {csv_path}")


def process_video_callback(result):
    """Callback function to collect the result."""
    video_rel_path, video_meta = result
    if video_meta is None:
        failed_video_paths.append(video_rel_path)  # Record the failed path
    else:
        video_rel_paths.append(video_rel_path)
        video_meta_info.append(video_meta)
    pbar.update(1)


if __name__ == "__main__":
    videos_root = "/data2/videos/youtube"
    frames_root = "/data2/Fooling3D/video_sequence"
    csv_path = "/data2/videos/meta_data/video_metadata.csv"
    failed_csv_path = "/data2/videos/meta_data/failed_videos.csv"
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
    max_workers = 10

    # Collect video paths
    video_paths = []
    for ext in video_extensions:
        video_paths += glob.glob(os.path.join(videos_root, '**', ext), recursive=True)

    # Prepare arguments for multiprocessing pool
    args = [(video_path, frames_root, videos_root, max_workers) for video_path in video_paths]

    # Non-blocking parallel video processing using apply_async
    video_rel_paths = []
    video_meta_info = []
    failed_video_paths = []

    with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 8) as pool:
        # Initialize progress bar
        with tqdm(total=len(video_paths), desc="Processing videos", unit="video") as pbar:
            for arg in args:
                pool.apply_async(process_video, args=arg, callback=process_video_callback)
            pool.close()  # Close the pool to stop adding new tasks
            pool.join()   # Wait for all tasks to complete

    # Save meta data to CSV
    save_video_info_to_csv(video_rel_paths, video_meta_info, csv_path)

    # Save failed video paths to a separate CSV
    failed_df = pd.DataFrame({"failed_video_path": failed_video_paths})
    failed_df.to_csv(failed_csv_path, index=False)
    print(f"Failed video paths saved to {failed_csv_path}")
