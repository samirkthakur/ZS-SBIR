# src/split.py

import os
import random
import json

def split_categories(
    base_data_dir: str,
    extracted_folder_name: str,
    embeddings_dir: str,
    projection_net_save_filename: str = "sketch_projection_net_category_infonce.pth",
    split_save_filename: str = "category_split.json",
    num_train: int = 100,
    seed: int = 42
):
    """
    Splits categories into train and test sets and saves the split as JSON.
    """
    random.seed(seed)

    train_sketch_path = os.path.join(
        base_data_dir,
        extracted_folder_name,
        "256x256",
        "splitted_sketches",
        "train",
        "tx_000100000000"
    )
    print("Train sketch path:", train_sketch_path)

    all_categories = sorted([
        cat for cat in os.listdir(train_sketch_path)
        if os.path.isdir(os.path.join(train_sketch_path, cat))
    ])

    print(f"Total categories found: {len(all_categories)}")

    train_categories = random.sample(all_categories, num_train)
    test_categories = [cat for cat in all_categories if cat not in train_categories]

    print(f"Train categories: {len(train_categories)}")
    print(f"Test categories: {len(test_categories)}")

    # Ensure embeddings directory exists
    os.makedirs(embeddings_dir, exist_ok=True)

    # Save split to JSON
    split_save_path = os.path.join(embeddings_dir, split_save_filename)
    with open(split_save_path, "w") as f:
        json.dump({
            "train_categories": train_categories,
            "test_categories": test_categories
        }, f, indent=2)

    print(f"Category split saved at {split_save_path}")

    # Return useful paths if needed
    return {
        "train_categories": train_categories,
        "test_categories": test_categories,
        "split_save_path": split_save_path,
        "projection_net_save_path": os.path.join(embeddings_dir, projection_net_save_filename)
    }

if __name__ == "__main__":
    BASE_DATA_DIR = "data"
    KAGGLE_EXTRACTED_FOLDER_NAME = "temp_extraction"
    EMBEDDINGS_DIR = "ML_Embeddings_SBIR_PyTorch"

    split_categories(
        base_data_dir=BASE_DATA_DIR,
        extracted_folder_name=KAGGLE_EXTRACTED_FOLDER_NAME,
        embeddings_dir=EMBEDDINGS_DIR
    )
