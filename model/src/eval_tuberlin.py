# src/eval_tuberlin.py

import os
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import average_precision_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
import faiss

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

def evaluate_tuberlin(sketch_path, image_path, projection_net_path, gallery_embeddings_save_path):
    # Load CLIP and processor
    model, processor = load_clip_and_processor()
    clip_dim = 512

    # Load Projection Network
    projection_net = SketchProjectionNet(clip_dim, clip_dim).to(device)
    projection_net.load_state_dict(torch.load(projection_net_path, map_location=device))
    projection_net.eval()

    # Load or build gallery embeddings
    if os.path.exists(gallery_embeddings_save_path):
        print("Loading precomputed gallery embeddings...")
        saved = np.load(gallery_embeddings_save_path, allow_pickle=True)
        gallery_embeddings = saved['embeddings']
        gallery_image_filepaths = saved['filepaths'].tolist()
        print("Loaded gallery embeddings with shape:", gallery_embeddings.shape)
    else:
        print("Precomputing gallery embeddings for TU-Berlin...")
        gallery_image_filepaths = []
        gallery_embeddings = []

        tu_berlin_categories = sorted(os.listdir(image_path))

        for category in tqdm(tu_berlin_categories, desc="Encoding gallery"):
            cat_path = os.path.join(image_path, category)
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
        np.savez(gallery_embeddings_save_path, embeddings=gallery_embeddings, filepaths=np.array(gallery_image_filepaths))
        print(f"Saved gallery embeddings to {gallery_embeddings_save_path}")

    # Build FAISS index
    index = faiss.IndexFlatIP(clip_dim)
    index.add(gallery_embeddings.astype(np.float32))

    # Evaluation
    print("\nRunning TU-Berlin benchmark with FAISS...")
    tu_berlin_categories = sorted(os.listdir(sketch_path))
    correct_top1 = 0
    all_aps = []
    total_sketches = 0
    BATCH_SIZE = 32
    MAX_K = min(1000, len(gallery_embeddings))

    for category in tqdm(tu_berlin_categories, desc="Evaluating sketches"):
        cat_path = os.path.join(sketch_path, category)
        if not os.path.isdir(cat_path):
            continue

        sketch_files = glob.glob(os.path.join(cat_path, "*.png")) + \
                       glob.glob(os.path.join(cat_path, "*.jpg")) + \
                       glob.glob(os.path.join(cat_path, "*.jpeg"))

        for i in range(0, len(sketch_files), BATCH_SIZE):
            batch_files = sketch_files[i:i+BATCH_SIZE]
            images = [Image.open(f).convert("RGB") for f in batch_files]
            inputs = processor(images=images, return_tensors="pt", padding=True).to(device)

            with torch.no_grad():
                sketch_embeds = model.get_image_features(**inputs)
                projected_embeds = projection_net(sketch_embeds)
                projected_embeds = F.normalize(projected_embeds, p=2, dim=-1).cpu().numpy()

            sims, indices = index.search(projected_embeds.astype(np.float32), k=MAX_K)

            for j, sims_j in enumerate(sims):
                sorted_indices = indices[j]
                top1_image_path = gallery_image_filepaths[sorted_indices[0]]
                top1_category = os.path.basename(os.path.dirname(top1_image_path))
                if top1_category == category:
                    correct_top1 += 1

                true_labels = np.array([
                    1 if os.path.basename(os.path.dirname(gallery_image_filepaths[idx])) == category else 0
                    for idx in sorted_indices
                ])
                if np.sum(true_labels) == 0:
                    continue
                ap = average_precision_score(true_labels, sims_j)
                all_aps.append(ap)

                total_sketches += 1

    rank1_acc = correct_top1 / total_sketches * 100 if total_sketches > 0 else 0
    mean_ap = np.mean(all_aps) * 100 if all_aps else 0

    print("\nTU-Berlin Benchmark Results:")
    print(f"Total Test Sketches Evaluated: {total_sketches}")
    print(f"Rank-1 Accuracy: {rank1_acc:.2f}%")
    print(f"Mean Average Precision (mAP): {mean_ap:.2f}%")

    return rank1_acc, mean_ap

if __name__ == "__main__":
    BASE_DATA_DIR = "data"
    EMBEDDINGS_DIR = "ML_Embeddings_SBIR_PyTorch"

    TU_BERLIN_SKETCH_PATH = os.path.join(BASE_DATA_DIR, "TUBerlin", "png_ready")
    TU_BERLIN_IMAGE_PATH = os.path.join(BASE_DATA_DIR, "TUBerlin", "ImageResized_ready")
    PROJECTION_NET_PATH = os.path.join(EMBEDDINGS_DIR, "sketch_projection_net_category_infonce.pth")
    GALLERY_EMBEDDINGS_SAVE_PATH = os.path.join(EMBEDDINGS_DIR, "gallery_embeddings.npz")

    evaluate_tuberlin(TU_BERLIN_SKETCH_PATH, TU_BERLIN_IMAGE_PATH, PROJECTION_NET_PATH, GALLERY_EMBEDDINGS_SAVE_PATH)
