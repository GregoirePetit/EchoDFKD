import argparse
import configparser
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import cv2
import torchvision.transforms as T
from open_clip import create_model_and_transforms, tokenize

def compute_frame_distribution(
    video_embeddings: torch.Tensor,
    prompt_embeddings: torch.Tensor,
):
    per_frame_similarities = (
        video_embeddings @ prompt_embeddings.T
    ) 
    histogram = per_frame_similarities
    histogram = histogram.detach()
    
    return histogram

# Add core and root directories to the system path
core_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.dirname(core_dir)
sys.path.append(root_dir)

# Initialize the model and preprocessing
echo_clip, _, preprocess_val = create_model_and_transforms(
    "hf-hub:mkaichristensen/echo-clip", precision="bf16", device="cuda"
)
echo_clip.eval()

def process_video_chunk(chunk):
    """Process a chunk of video frames and return their embeddings."""
    with torch.no_grad():
        chunk = torch.stack(
            [preprocess_val(T.ToPILImage()(frame)) for frame in chunk], dim=0
        ).cuda()
        chunk = chunk.to(torch.bfloat16)
        embeddings = F.normalize(echo_clip.encode_image(chunk), dim=-1)
    return embeddings


def getInfo(video, video_expanded, chunk_size=200):
    """Extract and process video embeddings, then compute similarity scores."""
    torch.cuda.empty_cache()

    def extract_embeddings(video, chunk_size):
        all_embeddings = []
        for i in range(0, len(video), chunk_size):
            chunk = video[i : i + chunk_size]
            embeddings = process_video_chunk(chunk)
            all_embeddings.append(embeddings)
        return torch.cat(all_embeddings, dim=0)

    def compute_similarity(video_embedding, video_expanded_embedding):
        prompts = tokenize(["NOTHING", "LEFT VENTRICLE"]).cuda()
        text_embeddings = F.normalize(echo_clip.encode_text(prompts), dim=-1)
        similarity = compute_frame_distribution(video_expanded_embedding, text_embeddings)
        similarity = similarity.cpu().float().numpy()
        similarity = similarity.reshape(similarity.shape[1], similarity.shape[2]).T

        prompts = tokenize(["WALL"]).cuda()
        text_embeddings = F.normalize(echo_clip.encode_text(prompts), dim=-1)
        similarity_raw = compute_frame_distribution(video_embedding, text_embeddings)
        similarity_raw = similarity_raw.reshape(similarity_raw.shape[1], similarity_raw.shape[2]).T
        return [np.mean(list(similarity_raw[i])) for i in range(similarity_raw.shape[0])], [np.mean(list(similarity[i])) for i in range(similarity.shape[0])]

    video_embedding = extract_embeddings(video, chunk_size)
    video_embedding = F.normalize(video_embedding, dim=-1).unsqueeze(0)
    video_expanded_embedding = extract_embeddings(video_expanded, chunk_size)
    video_expanded_embedding = F.normalize(video_expanded_embedding, dim=-1).unsqueeze(0)

    similarity_raw_scores, similarity_scores = compute_similarity(video_embedding, video_expanded_embedding)

    torch.cuda.empty_cache()
    return similarity_raw_scores[0] + similarity_scores[0] - similarity_scores[1]


def load_config(config_path="config_file.cf"):
    """Load configuration from a file."""
    config = configparser.ConfigParser()
    config.read(config_path)
    return config

def create_segmentations(video_path, segmentation_path):
    """Create a video from video_path, overlaying segmentation masks from segmentation_path."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    frames_expanded = frames.copy()
    masks = np.load(segmentation_path)["arr_0"]
    masks = masks.squeeze()
    masks = masks.astype(np.uint8)
    masks = (masks > 127).astype(np.uint8)
    masks_expanded = masks.copy()
    for i, frame in enumerate(frames):
        frame[masks[i] == 1] = [0, 0, 0]
    for i, frame in enumerate(frames_expanded):
        masks_expanded[i] = cv2.dilate(masks[i], np.ones((5, 5), np.uint8), iterations=1)
        frame[masks[i] == 1] = [0, 0, 0]
    
    return frames, frames_expanded

def main():
    """Main function to run inference on videos based on the provided configuration."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--xp_name", type=str, default=None, help="Experiment name.")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument(
        "--examples",
        type=str,
        default=None,
        help="Path of a file which contains list of examples on which to infer",
    )
    args = parser.parse_args()
    

    for video_path, segmentation_path in json.load(open(args.examples)):
        video, video_expanded = create_segmentations(video_path, segmentation_path)
        similarity = getInfo(video, video_expanded)
        print(f"Similarity score: {similarity}")



if __name__ == "__main__":
    main()
