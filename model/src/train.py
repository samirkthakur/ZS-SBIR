import os
import glob
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# --- Faster Collate Function ---
def collate_fn(batch):
    sketches, photos, labels = zip(*batch)
    sketch_inputs = processor(images=list(sketches), return_tensors="pt", padding=True)
    photo_inputs = processor(images=list(photos), return_tensors="pt", padding=True)
    return sketch_inputs['pixel_values'], photo_inputs['pixel_values'], torch.tensor(labels)


# --- Dataset Class for Category-level Training ---
class CategorySketchPhotoDataset(Dataset):
    def __init__(self, sketch_dir, photo_dir, category_list):
        self.sketch_files = []
        self.sketch_labels = []
        self.photo_files = []
        self.photo_labels = []
        self.category_to_idx = {cat: idx for idx, cat in enumerate(category_list)}

        # Load sketches
        for category in category_list:
            cat_path = os.path.join(sketch_dir, category)
            if not os.path.isdir(cat_path):
                continue
            files = glob.glob(os.path.join(cat_path, '*'))
            self.sketch_files.extend(files)
            self.sketch_labels.extend([self.category_to_idx[category]] * len(files))

        # Load photos
        for category in category_list:
            cat_path = os.path.join(photo_dir, category)
            if not os.path.isdir(cat_path):
                continue
            files = glob.glob(os.path.join(cat_path, '*'))
            self.photo_files.extend(files)
            self.photo_labels.extend([self.category_to_idx[category]] * len(files))

        # Build category index for faster sampling
        self.photo_by_category = {}
        for file, label in zip(self.photo_files, self.photo_labels):
            self.photo_by_category.setdefault(label, []).append(file)

    def __len__(self):
        return len(self.sketch_files)

    def __getitem__(self, idx):
        sketch_path = self.sketch_files[idx]
        label = self.sketch_labels[idx]
        sketch_img = Image.open(sketch_path).convert("RGB")

        # Sample a random photo from the same category
        photo_path = np.random.choice(self.photo_by_category[label])
        photo_img = Image.open(photo_path).convert("RGB")

        return sketch_img, photo_img, label

# --- Sketch Projection Network ---
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

# --- InfoNCE Loss ---
def info_nce_loss(sketch_embeds, photo_embeds, temperature=0.07, device="cpu"):
    sketch_embeds = F.normalize(sketch_embeds, p=2, dim=-1)
    photo_embeds = F.normalize(photo_embeds, p=2, dim=-1)

    logits = torch.matmul(sketch_embeds, photo_embeds.T) / temperature
    labels = torch.arange(logits.size(0)).to(device)
    loss = F.cross_entropy(logits, labels)
    return loss

# --- Main Training Function ---
def train_model(
    base_data_dir="data",
    extracted_folder_name="temp_extraction",
    embeddings_dir="ML_Embeddings_SBIR_PyTorch",
    split_filename="category_split.json",
    projection_net_save_filename="sketch_projection_net_category_infonce.pth",
    num_epochs=5,
    batch_size=32,
    lr=1e-4,
    num_workers=2
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load CLIP
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    CLIP_EMBEDDING_DIM = 512

    # Load category split
    split_path = os.path.join(embeddings_dir, split_filename)
    with open(split_path, "r") as f:
        split = json.load(f)

    train_categories = split["train_categories"]

    # Dataset and DataLoader
    train_sketch_path = os.path.join(base_data_dir, extracted_folder_name, "256x256", "splitted_sketches", "train", "tx_000100000000")
    train_photo_path = os.path.join(base_data_dir, extracted_folder_name, "256x256", "photo", "tx_000100000000")
    dataset = CategorySketchPhotoDataset(train_sketch_path, train_photo_path, train_categories)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)

    print(f"Total sketch-photo category pairs loaded: {len(dataset)}")

    # Initialize projection network
    projection_net = SketchProjectionNet(CLIP_EMBEDDING_DIM, CLIP_EMBEDDING_DIM).to(device)
    optimizer = optim.Adam(projection_net.parameters(), lr=lr)

    # Training Loop
    for epoch in range(num_epochs):
        projection_net.train()
        total_loss = 0
        for sketches, photos, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            sketches, photos = sketches.to(device), photos.to(device)

            optimizer.zero_grad()

            with torch.no_grad():
                sketch_embeds = model.get_image_features(sketches)
                photo_embeds = model.get_image_features(photos)

            projected = projection_net(sketch_embeds)
            loss = info_nce_loss(projected, photo_embeds, device=device)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} completed. Avg Loss: {avg_loss:.4f}")

    # Save model
    os.makedirs(embeddings_dir, exist_ok=True)
    save_path = os.path.join(embeddings_dir, projection_net_save_filename)
    torch.save(projection_net.state_dict(), save_path)
    print(f"Model saved at {save_path}")

if __name__ == "__main__":
    train_model()
