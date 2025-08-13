# main.py

import os
from src import data, split, train, generate_gallery_embeddings, eval_sketchy, eval_tuberlin

def main():
    # --- Global paths ---
    BASE_DATA_DIR = "data"
    KAGGLE_EXTRACTED_FOLDER_NAME = "temp_extraction"
    EMBEDDINGS_DIR = "ML_Embeddings_SBIR_PyTorch"

    # --- 1. Data download & extraction ---
    # Uses data.py main method structure
    data.setup_kaggle_credentials("kaggle.json")
    data.download_kaggle_dataset("dhananjayapaliwal/fulldataset", "./data")
    data.download_kaggle_dataset("rishikashili/tuberlin", "./data/TUBerlin")

    # --- 2. Dataset splitting ---
    split.split_categories(
        base_data_dir=BASE_DATA_DIR,
        extracted_folder_name=KAGGLE_EXTRACTED_FOLDER_NAME,
        embeddings_dir=EMBEDDINGS_DIR
    )

    # --- 3. Training ---
    train.train_model()

    # --- 4. Generate gallery embeddings (optional) ---
    generate_gallery_embeddings.generate_gallery_embeddings()

    # --- 5. Evaluate on Sketchy ---
    TEST_SKETCH_PATH = os.path.join(BASE_DATA_DIR, KAGGLE_EXTRACTED_FOLDER_NAME, "256x256", "splitted_sketches", "test", "tx_000100000000")
    TEST_PHOTO_PATH = os.path.join(BASE_DATA_DIR, KAGGLE_EXTRACTED_FOLDER_NAME, "256x256", "photo", "tx_000100000000")
    SPLIT_SAVE_PATH = os.path.join(EMBEDDINGS_DIR, "category_split.json")
    PROJECTION_NET_SAVE_PATH = os.path.join(EMBEDDINGS_DIR, "sketch_projection_net_category_infonce.pth")

    eval_sketchy.evaluate_sbirk(TEST_SKETCH_PATH, TEST_PHOTO_PATH, SPLIT_SAVE_PATH, PROJECTION_NET_SAVE_PATH)

    # --- 6. Evaluate on TU-Berlin (optional) ---
    TU_BERLIN_SKETCH_PATH = os.path.join(BASE_DATA_DIR, "TUBerlin", "png_ready")
    TU_BERLIN_IMAGE_PATH = os.path.join(BASE_DATA_DIR, "TUBerlin", "ImageResized_ready")
    GALLERY_EMBEDDINGS_SAVE_PATH = os.path.join(EMBEDDINGS_DIR, "gallery_embeddings.npz")

    eval_tuberlin.evaluate_tuberlin(
        TU_BERLIN_SKETCH_PATH,
        TU_BERLIN_IMAGE_PATH,
        PROJECTION_NET_SAVE_PATH,
        GALLERY_EMBEDDINGS_SAVE_PATH
    )

if __name__ == "__main__":
    main()
