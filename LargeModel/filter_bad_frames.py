import os
import sys
import csv
import logging
import argparse
import pandas as pd
from tqdm import tqdm
from datetime import datetime

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from utils.utils import parser_video, save_frames
from utils.utils import decide_save, transform_string_to_array, safe_remove


DEBUG_ENVS = os.getenv('DEBUG_ENVS', 'False').lower() == 'true'

def setup_logging(gpu_id, exp_name="EXP"):
    log_dir = './logs'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'{datetime.now().strftime("%Y%m%d_%H%M%S")}-{gpu_id}-{exp_name}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(process)d - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            # logging.StreamHandler()
        ]
    )

def log_info(message, silence=True):
    if DEBUG_ENVS or not silence:
        logging.info(message)

class VideoDataset(Dataset):
    def __init__(self, video_path_csv, frames_root, videos_root, meta_root="./cache", num_thread=4):
        self.frames_root = frames_root
        self.videos_root = videos_root
        self.meta_root   = meta_root
        self.num_thread  = num_thread

        video_df = pd.read_csv(video_path_csv)
        video_paths = [os.path.join(videos_root, video_rel_path) for video_rel_path in video_df['video_rel_path']]
        self.video_paths = video_paths
        log_info(f"Total number of videos: {len(self.video_paths)}", silence=False)

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        video_rel_path = os.path.relpath(video_path, self.videos_root)
        
        try:
            frame_list, frame_count = parser_video(video_path)
            if not frame_list:
                raise Exception("Empty frame list.")

            rel_path, abs_path = save_frames(self.frames_root, video_rel_path, frame_list, self.num_thread)
            return abs_path

        except Exception as e:
            with open(os.path.join(self.meta_root, 'failure_video.csv'), 'a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([video_path])
            log_info(f"[ERROR] Failed to process {video_path}: \r\n {e}", silence=False)
            return None

def process_frames(frame_paths, model, processor, csv_writer):
    messages = [setup_prompt(frame_path) for frame_path in frame_paths]
    inputs = process_prompt(messages, model, processor)
    output_texts = inference(model, processor, inputs)
    for frame_path, output_text in zip(frame_paths, output_texts):
        # if output_text.find("yes") != -1:
        csv_writer.writerow([frame_path+','+output_text])
    return output_texts

def setup_model(model_path="/mount_points/nas/Qwen2-VL-2B-Instruct"):
    log_info(f"Start loading model and processor from {model_path}", silence=False)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype="auto", device_map="auto"
    )
    min_pixels = 256 * 28 * 28
    max_pixels = 1024 * 28 * 28
    processor = AutoProcessor.from_pretrained(
        model_path, min_pixels=min_pixels, max_pixels=max_pixels,
    )
    # processor = AutoProcessor.from_pretrained(model_path)
    log_info(f"Model and processor loaded from {model_path}")
    return model, processor

def setup_prompt(image_path, text=None):
    if text is None:
        # text = "If any answer to the following questions is yes, return 'yes', otherwise return 'no'. Does this image feature any artistic creation of landscapes on a flat surface? Does this image contain any areas with perspective illusion? Does this image contain any optical illusion graffiti or artwork?  Are these illusion artworks complete, not semi-finished products? Does this image contain any transparent or high-reflective areas? Does this image show a display screen playing 3D objects or scenes? Does the image contain areas that make you mistake them for 3D objects? Does this image have no or little watermarks or captions that affect its quality? Is this image quality high and not too blurry? Is the image resolution larger than 600*600? Are most areas of the artistic creation or illusion not covered by an artist's body or a single/two hands from an artist?"
        text = "Reply to me in the format of a string concatenating 'yes' or 'no' with ','. Each 'yes or 'no' is an answer to each following question. " + \
               "Does this image feature any flat artistic creation of landscapes where the surface of the creation is flat and has no ups and downs? " + \
               "Does this image contain any areas with perspective illusion? " + \
               "Does this image contain any optical illusion graffiti or artwork? " + \
               "Does this image contain any transparent or high-reflective areas? " + \
               "Does this image show a display screen playing 3D objects or scenes? " + \
               "Does the image contain areas that make you mistake them for 3D objects? " + \
               "Does this image contain excessive watermarks or captions that seriously affect its quality? " + \
               "Does this image contain small watermarks or captions or on a corner? " + \
               "Is this image too blurry? " + \
               "Are most regions of the artistic creation covered by a single/two hands? " + \
               "Is this image a software interface? "
    
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

def main(args):
    init_root(args.frames_root, args.meta_root)
    setup_logging(args.gpu_id)
    log_info(f'Input parameters: exp_name={args.exp_name}, video_path_csv={args.video_path_csv}, ' +
             f'gpu_id={args.gpu_id}, batch_size={args.batch_size}, num_workers={args.num_workers}, ' +
             f'num_thread={args.num_thread}, frames_root={args.frames_root}, videos_root={args.videos_root}, ' + 
             f'meta_root={args.meta_root}, max_step={args.max_step}, num_clip={args.num_clip}, model_path={args.model_path}', silence=False)

    device = torch.cuda.current_device()
    log_info(f"Current CUDA device: {device}")
    log_info(f"Device name: {torch.cuda.get_device_name(device)}")
    log_info(f"cuda available: {torch.cuda.is_available()}")
    log_info(f"memory allocated: {torch.cuda.memory_allocated()}")
    log_info(f"memory cached: {torch.cuda.memory_reserved()}")

    model, processor = setup_model(args.model_path)

    dataset = VideoDataset(args.video_path_csv, args.frames_root, args.videos_root, args.meta_root, args.num_thread)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = os.path.join(args.meta_root, f'{timestamp}-{args.gpu_id}-{args.exp_name}-good_images.csv')

    with open(csv_file, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        total_num_videos = len(dataloader)
        for vide_idx, batch in enumerate(dataloader):
            dir_path = batch[0]
            if dir_path is None:
                log_info(f"There is a invalid dir_path !!!")
                continue

            # Collect all frame paths
            frame_paths = []
            for root, _, files in os.walk(dir_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    frame_paths.append(file_path)
            frame_paths.sort()
            log_info(f'Total number of frames to process: {len(frame_paths)} in {dir_path}', silence=False)
            # print(f'Total number of frames to process: {len(frame_paths)}')

            # Process frames
            step = min(args.max_step, len(frame_paths) // args.num_clip)
            completed = False
            for frame_idx, file_path in enumerate(tqdm(frame_paths, desc=f"Processing frames (PID: {os.getpid()}, video: {vide_idx}/{total_num_videos})")):
                if not os.path.exists(file_path):
                    continue
                
                if frame_idx % step == 0:
                    output_texts = process_frames([file_path], model, processor, csv_writer)
                    output_text  = output_texts[0]
                    log_info("-"*10 + f" frame_idx: {frame_idx}   ---   {output_text}", silence=False)

                    # arr = transform_string_to_array(output_text)
                    # res = decide_save(arr)
                    # log_info(f"arr: {arr}, res: {res}", silence=False)

                    # Delete bad frames
                    # if output_text.lower().find("yes")==-1:
                    if not decide_save(transform_string_to_array(output_text)):
                        log_info(f"Find a bad frame: frame_idx: {frame_idx}, {file_path}", silence=False)
                        start_idx = max(0, frame_idx - step // 2)
                        end_idx = min(len(frame_paths), frame_idx + step // 2)
                        files_to_remove = frame_paths[start_idx:end_idx]
                        for file_path in files_to_remove:
                            safe_remove(file_path)
                        deleted = True
                    else:
                        deleted = False

                elif not completed and frame_idx > len(frame_paths) // step * step:
                    # If step is large, there may be some valuable frames not processed in the last part of frames
                    again_start_idx = len(frame_paths) // step * step + deleted * (step // 2)
                    if len(frame_paths) - again_start_idx + 1 > args.max_step//2:
                        step = (len(frame_paths) - again_start_idx + 1) // args.num_clip
                        for frame_idx in range(again_start_idx, len(frame_paths)):
                            if frame_idx % step == 0:
                                output_texts = process_frames([frame_paths[frame_idx]], model, processor, csv_writer)
                                output_text  = output_texts[0]
                                log_info("-"*10 + f" frame_idx: {frame_idx}  ---   {output_text}", silence=False)

                                # Delete bad frames
                                # if output_text.lower().find("yes")==-1:
                                if not decide_save(transform_string_to_array(output_text)):
                                    log_info(f"Find a bad frame: frame_idx: {frame_idx}, {frame_paths[frame_idx]}", silence=False)
                                    start_idx = max(again_start_idx, frame_idx - step // 2)
                                    end_idx = min(len(frame_paths), frame_idx + step // 2)
                                    files_to_remove = frame_paths[start_idx:end_idx]
                                    for file_path in files_to_remove:
                                        safe_remove(file_path)
                    
                    # The last several frames mainly contain many watermarks or captions, so we remove them
                    remain_start_idx = again_start_idx + (len(frame_paths) - again_start_idx + 1) // step * step
                    files_to_remove = frame_paths[remain_start_idx:]
                    for file_path in files_to_remove:
                        safe_remove(file_path)
                    
                    completed = True

    log_info(f'Processed {total_num_videos} videos.', silence=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process videos and filter bad frames.')
    parser.add_argument('--exp_name', type=str, default="", help='Experiment name')
    parser.add_argument('--video_path_csv', type=str, help='Path to the CSV file containing video paths')
    parser.add_argument('--gpu_id', type=str, help='GPU ID for logging purposes')
    parser.add_argument('--batch_size', type=int, help='Batch size for DataLoader')
    parser.add_argument('--num_workers', type=int, help='number of workers for DataLoader')
    parser.add_argument('--num_thread', type=int, help='number of threads for saving frames')
    parser.add_argument('--frames_root', type=str, default="/data2/Fooling3D/video_frame_sequence", help='frames root directory')
    parser.add_argument('--videos_root', type=str, default="/data2/Fooling3D/videos", help='root directory of videos')
    parser.add_argument('--meta_root', type=str, default="/data2/Fooling3D/meta_data", help='root directory of metadata')
    parser.add_argument('--max_step', type=int, default=1, help='maximum step size for processing frames')
    parser.add_argument('--num_clip', type=int, default=1, help='expected number of clips for processing frames')
    parser.add_argument('--model_path', type=str, default="/mount_points/nas/Qwen2-VL-2B-Instruct", help='path to the model')
    args = parser.parse_args()

    main(args)
