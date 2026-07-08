from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "Universal_Tomato_Dataset"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "tomato_leaf_efficientnet_b3.pth"

IMAGE_SIZE = 320
BATCH_SIZE = 16
NUM_WORKERS = 2
SEED = 42

ACCEPT_CONFIDENCE_THRESHOLD = 0.75
UNCERTAINTY_MARGIN_THRESHOLD = 0.05
TEMPERATURE = 1.0

CLASS_NAMES = [
    "Bacterial_spot",
    "Early_blight",
    "Healthy",
    "Late_blight",
    "Leaf_Miner",
    "Leaf_Mold",
    "Magnesium_Deficiency",
    "Nitrogen_Deficiency",
    "Pottassium_Deficiency",
    "Septoria_leaf_spot",
    "Spider_mites",
    "Spotted_Wilt_Virus",
    "Target_Spot",
    "Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato_mosaic_virus",
    "powdery_mildew",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

