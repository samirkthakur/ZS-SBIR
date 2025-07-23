import os
import glob
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
import torch.nn as nn

# --- Sketch Projection Network definition (for loading saved weights) ---
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

def generate_gallery_embeddings(
    base_data_dir="data",
    extracted_folder_name="temp_extraction",
    embeddings_dir="ML_Embeddings_SBIR_PyTorch",
    projection_net_filename="sketch_projection_net.pth",
    gallery_embeddings_filename="gallery_embeddings.npy",
    gallery_imagepaths_filename="gallery_image_filepaths.npy"
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load CLIP model
    print("\nLoading CLIP model for inference...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    print("CLIP model loaded.")

    CLIP_EMBEDDING_DIM = 512

    # Load projection network if required later (currently loaded but not used here)
    projection_net = SketchProjectionNet(CLIP_EMBEDDING_DIM, CLIP_EMBEDDING_DIM).to(device)
    projection_net_path = os.path.join(embeddings_dir, projection_net_filename)
    if os.path.exists(projection_net_path):
        projection_net.load_state_dict(torch.load(projection_net_path, map_location=device))
        projection_net.eval()
        print(f"Projection network loaded from {projection_net_path}")
    else:
        print(f"Warning: Projection network not found at {projection_net_path}. Continuing without it.")

    # Check for existing gallery embeddings
    gallery_embeddings_path = os.path.join(embeddings_dir, gallery_embeddings_filename)
    gallery_imagepaths_path = os.path.join(embeddings_dir, gallery_imagepaths_filename)

    if os.path.exists(gallery_embeddings_path) and os.path.exists(gallery_imagepaths_path):
        print(f"\nLoading existing gallery embeddings from {embeddings_dir}...")
        gallery_embeddings = np.load(gallery_embeddings_path)
        gallery_image_filepaths = np.load(gallery_imagepaths_path, allow_pickle=True)
        print(f"Loaded gallery embeddings shape: {gallery_embeddings.shape}")
        return gallery_embeddings, gallery_image_filepaths

    # Build gallery embeddings
    print("\nGallery embeddings not found. Building them now...")
    gallery_embeddings_list = []
    gallery_image_filepaths_list = []

    image_gallery_path = os.path.join(base_data_dir, extracted_folder_name, "256x256", "photo", "tx_000100000000")
    if not os.path.exists(image_gallery_path):
        raise FileNotFoundError(f"Image gallery path not found: {image_gallery_path}")

    with torch.no_grad():
        for category_name in tqdm(os.listdir(image_gallery_path), desc="Processing gallery images"):
            category_path = os.path.join(image_gallery_path, category_name)
            if not os.path.isdir(category_path):
                continue

            for image_filename in os.listdir(category_path):
                if image_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_filepath = os.path.join(category_path, image_filename)
                    try:
                        image = Image.open(image_filepath).convert("RGB")
                        inputs = processor(images=image, return_tensors="pt").to(device)
                        image_feature = model.get_image_features(**inputs).detach().cpu().numpy()
                        gallery_embeddings_list.append(image_feature)
                        gallery_image_filepaths_list.append(image_filepath)
                    except Exception as e:
                        print(f"Could not process image {image_filepath}: {e}")

    if gallery_embeddings_list:
        gallery_embeddings = np.vstack(gallery_embeddings_list)
        gallery_image_filepaths = np.array(gallery_image_filepaths_list)

        os.makedirs(embeddings_dir, exist_ok=True)
        np.save(gallery_embeddings_path, gallery_embeddings)
        np.save(gallery_imagepaths_path, gallery_image_filepaths)

        print(f"Encoded {len(gallery_embeddings_list)} gallery images. Embeddings shape: {gallery_embeddings.shape}")
        print(f"Gallery embeddings saved to {embeddings_dir}")
        return gallery_embeddings, gallery_image_filepaths
    else:
        print("No images processed in the gallery. Check your dataset contents.")
        return None, None

if __name__ == "__main__":
    generate_gallery_embeddings()
