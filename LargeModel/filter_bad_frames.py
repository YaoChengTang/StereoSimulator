import os
import csv
import multiprocessing
# Set the multiprocessing start method to 'spawn' as required by pytorch in multiprocessing
multiprocessing.set_start_method('spawn', force=True)

import logging
import argparse
from datetime import datetime

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

from utils.utils import parser_video, save_frames
import pandas as pd

DEBUG_ENVS = os.getenv('DEBUG_ENVS', 'False').lower() == 'true'



def setup_logging(gpu_id):
    log_dir = './logs'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'{datetime.now().strftime("%Y%m%d_%H%M%S")}_GPU_{gpu_id}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(process)d - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def log_info(message, silence=True):
    if DEBUG_ENVS or not silence:
        logging.info(message)

def process_video(video_path, frames_root, videos_root, max_workers=6, meta_root="./cache"):
    """Process a video: parse, save frames, and gather metadata."""
    video_rel_path = os.path.relpath(video_path, videos_root)
    
    try:
        # Load video, parse video, save frames
        frame_list, frame_count = parser_video(video_path)
        if not frame_list:
            raise Exception("Empty frame list.")

        rel_path, abs_path = save_frames(frames_root, video_rel_path, frame_list, max_workers)

        return abs_path

    except Exception as e:
        # Log the failed video path to failure_video.csv
        with open(os.path.join(meta_root, 'failure_video.csv'), 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([video_path])
        log_info(f"[ERROR] Failed to process {video_path}: \r\n {e}", silence=False)
        return None

# Producer function to process video and save frames
def producer(video_path, frames_root, videos_root, dir_queue, max_workers=6, meta_root="./cache"):
    output_dir = process_video(video_path, frames_root, videos_root, max_workers, meta_root)
    log_info(f'Processed video {video_path}, output dir: {output_dir}')
    if output_dir:
        dir_queue.put(output_dir)

# Consumer function to judge images and save good ones to CSV
def consumer(csv_file, dir_queue, model, processor, step=1):
    while True:
        try:
            dir_path = dir_queue.get(timeout=10)  # Wait for a directory path
        except multiprocessing.queues.Empty:
            break

        with open(csv_file, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            frame_paths = []
            for root, _, files in os.walk(dir_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    frame_paths.append(file_path)
                    if len(frame_paths) == step:
                        process_frames(frame_paths, model, processor, csv_writer)
                        frame_paths = []
            if frame_paths:
                process_frames(frame_paths, model, processor, csv_writer)
        dir_queue.task_done()

def process_frames(frame_paths, model, processor, csv_writer):
    messages = [setup_prompt(frame_path) for frame_path in frame_paths]
    inputs = process_prompt(messages, model, processor)
    output_texts = inference(model, processor, inputs)
    for frame_path, output_text in zip(frame_paths, output_texts):
        log_info(f'Frame path: {frame_path}, output text: {output_text}', silence=False)
        # if judge(output_text):
        #     csv_writer.writerow([frame_path])
        # else:
        #     os.remove(frame_path)

def setup_model(model_path="/mount_points/nas/Qwen2-VL-2B-Instruct"):
    # default: Load the model on the available device(s)
    log_info(f"Start loading model and processor from {model_path}", silence=False)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype="auto", device_map="auto"
    )

    # default processer
    processor = AutoProcessor.from_pretrained(model_path)
    log_info(f"Model and processor loaded from {model_path}")

    return model, processor

def setup_prompt(image_path, text=None):
    if text is None:
        text = "Describe this image."
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"{image_path}",
                },
                {   "type": "text", 
                    "text": f"{text}"
                },
            ],
        }
    ]
    log_info(f"Setup prompt for image {image_path}")
    return messages

def process_prompt(messages, model, processor):
    # Preparation for inference
    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in messages
    ]
    log_info(f"Complete texts prompt for {len(messages)} messages")
    image_inputs, video_inputs = process_vision_info(messages)
    log_info(f"Complete image prompt for {len(messages)} messages")
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    log_info(f"Complete inputs for {len(messages)} messages")
    log_info(f"Inputs: {inputs}")
    inputs = inputs.to("cuda")
    log_info(f"Complete prompt for {len(messages)} messages")
    return inputs

def inference(model, processor, inputs):
    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    log_info(f"Generated output for {len(output_text)} messages")
    return output_text

def init_root(frames_root, meta_root):
    os.makedirs(frames_root, exist_ok=True)
    os.makedirs(meta_root, exist_ok=True)

def main(video_path_csv, gpu_id, max_queue_size, frames_root='frames', videos_root='videos', max_workers=6, num_producers=3, meta_root='./cache', step=1):
    init_root(frames_root, meta_root)
    setup_logging(gpu_id)
    log_info(f'Input parameters: video_path_csv={video_path_csv}, ' +
             f'gpu_id={gpu_id}, max_queue_size={max_queue_size}, ' +
             f'frames_root={frames_root}, videos_root={videos_root}, ' + 
             f'max_workers={max_workers}, num_producers={num_producers}, ' +
             f'meta_root={meta_root}, step={step}', silence=False)

    device = torch.cuda.current_device()
    log_info(f"Current CUDA device: {device}")
    log_info(f"Device name: {torch.cuda.get_device_name(device)}")
    log_info(f"cuda avaible: {torch.cuda.is_available()}")
    log_info(f"memory allocated: {torch.cuda.memory_allocated()}")
    log_info(f"memory cached: {torch.cuda.memory_cached()}")

    model, processor = setup_model()

    dir_queue = multiprocessing.JoinableQueue(maxsize=max_queue_size)

    video_df = pd.read_csv(video_path_csv)
    video_paths = [os.path.join(videos_root, video_rel_path) for video_rel_path in video_df['video_rel_path']]

    producers = []
    for i in range(num_producers):
        for j, video_path in enumerate(video_paths[i::num_producers]):
            producer_process = multiprocessing.Process(target=producer, args=(video_path, frames_root, videos_root, 
                                                                              dir_queue, max_workers, meta_root))
            producers.append(producer_process)
            producer_process.start()

    consumer_process = multiprocessing.Process(target=consumer, args=(os.path.join(meta_root,'good_images.csv'), 
                                                                      dir_queue, model, processor, step))
    consumer_process.start()

    for producer_process in producers:
        producer_process.join()

    dir_queue.join()
    consumer_process.terminate()

    log_info(f'Processed {len(video_paths)} videos.', silence=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process videos and filter bad frames.')
    parser.add_argument('--video_path_csv', type=str, help='Path to the CSV file containing video paths')
    parser.add_argument('--gpu_id', type=str, help='GPU ID for logging purposes')
    parser.add_argument('--max_queue_size', type=int, help='Maximum size of the queue')
    parser.add_argument('--frames_root', type=str, default="/data2/Fooling3D/video_frame_sequence", help='frames root directory')
    parser.add_argument('--videos_root', type=str, default="/data2/Fooling3D/videos", help='root directory of videos')
    parser.add_argument('--meta_root', type=str, default="/data2/Fooling3D/meta_data", help='root directory of metadata')
    parser.add_argument('--max_workers', type=int, default=6, help='maximum number of workers')
    parser.add_argument('--num_producers', type=int, default=6, help='maximum number of producers')
    parser.add_argument('--step', type=int, default=1, help='step size for processing frames')
    args = parser.parse_args()

    main(args.video_path_csv, args.gpu_id, args.max_queue_size, args.frames_root, args.videos_root, args.max_workers, args.num_producers, args.meta_root, args.step)
