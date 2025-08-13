# src/eval.py

import os
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import average_precision_score
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from transformers import CLIPProcessor, CLIPModel

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- SketchProjectionNet Definition ---
class SketchProjectionNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SketchProjectionNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

def load_clip_and_processor():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model, processor

def evaluate_sbirk(test_sketch_path, test_photo_path, split_save_path, projection_net_save_path):
    # Load category split
    with open(split_save_path, "r") as f:
        split = json.load(f)
    test_categories = split["test_categories"]

    # Load CLIP
    model, processor = load_clip_and_processor()
    clip_dim = 512

    # Load Projection Network
    projection_net = SketchProjectionNet(clip_dim, clip_dim).to(device)
    projection_net.load_state_dict(torch.load(projection_net_save_path, map_location=device))
    projection_net.eval()

    # Build gallery embeddings
    print("Precomputing gallery embeddings for test categories...")
    gallery_image_filepaths = []
    gallery_embeddings = []

    for category in tqdm(test_categories, desc="Encoding gallery"):
        cat_path = os.path.join(test_photo_path, category)
        if not os.path.isdir(cat_path):
            continue

        image_files = glob.glob(os.path.join(cat_path, '*'))
        for img_file in image_files:
            try:
                img = Image.open(img_file).convert("RGB")
                inputs = processor(images=img, return_tensors="pt").to(device)
                with torch.no_grad():
                    embed = model.get_image_features(**inputs)
                    embed = F.normalize(embed, p=2, dim=-1).cpu().numpy()
                gallery_embeddings.append(embed)
                gallery_image_filepaths.append(img_file)
            except Exception as e:
                print(f"Error processing gallery image {img_file}: {e}")

    gallery_embeddings = np.concatenate(gallery_embeddings, axis=0)

    # Evaluation loop
    print("\nRunning full benchmark on test split...")
    correct_top1 = 0
    all_aps = []
    total_sketches = 0

    for category in tqdm(test_categories, desc="Evaluating sketches"):
        cat_path = os.path.join(test_sketch_path, category)
        if not os.path.isdir(cat_path):
            continue

        sketch_files = glob.glob(os.path.join(cat_path, "*.png")) + \
                       glob.glob(os.path.join(cat_path, "*.jpg")) + \
                       glob.glob(os.path.join(cat_path, "*.jpeg"))

        for sketch_file in sketch_files:
            try:
                sketch_image = Image.open(sketch_file).convert("RGB")
                inputs = processor(images=sketch_image, return_tensors="pt").to(device)

                with torch.no_grad():
                    sketch_embed = model.get_image_features(**inputs)
                    projected_embed = projection_net(sketch_embed)
                    projected_embed = F.normalize(projected_embed, p=2, dim=-1).cpu().numpy()

                # Cosine similarity
                sims = cosine_similarity(projected_embed, gallery_embeddings)[0]
                sorted_indices = np.argsort(sims)[::-1]

                # Top-1 accuracy
                top1_image_path = gallery_image_filepaths[sorted_indices[0]]
                top1_category = os.path.basename(os.path.dirname(top1_image_path))
                if top1_category == category:
                    correct_top1 += 1

                # mAP
                true_labels = np.array([
                    1 if os.path.basename(os.path.dirname(img_path)) == category else 0
                    for img_path in gallery_image_filepaths
                ])
                ap = average_precision_score(true_labels, sims)
                all_aps.append(ap)

                total_sketches += 1

            except Exception as e:
                print(f"Error processing sketch {sketch_file}: {e}")

    # Final metrics
    rank1_acc = correct_top1 / total_sketches * 100 if total_sketches > 0 else 0
    mean_ap = np.mean(all_aps) * 100 if all_aps else 0

    print("\nBenchmark Results:")
    print(f"Total Test Sketches Evaluated: {total_sketches}")
    print(f"Rank-1 Accuracy: {rank1_acc:.2f}%")
    print(f"Mean Average Precision (mAP): {mean_ap:.2f}%")

    return rank1_acc, mean_ap

if __name__ == "__main__":
    BASE_DATA_DIR = "data/temp_extraction"
    EMBEDDINGS_DIR = "ML_Embeddings_SBIR_PyTorch"

    TEST_SKETCH_PATH = os.path.join(BASE_DATA_DIR, "256x256", "splitted_sketches", "test", "tx_000100000000")
    TEST_PHOTO_PATH = os.path.join(BASE_DATA_DIR, "256x256", "photo", "tx_000100000000")
    SPLIT_SAVE_PATH = os.path.join(EMBEDDINGS_DIR, "category_split.json")
    PROJECTION_NET_SAVE_PATH = os.path.join(EMBEDDINGS_DIR, "sketch_projection_net_category_infonce.pth")

    evaluate_sbirk(TEST_SKETCH_PATH, TEST_PHOTO_PATH, SPLIT_SAVE_PATH, PROJECTION_NET_SAVE_PATH)
