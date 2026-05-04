import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MNW_AI_IMAGES_DIR = os.path.join(PROJECT_ROOT, "data", "MNW", "AI_Images")
OUTPUT_DIR = PROJECT_ROOT

HF_TOKEN_ENV_KEYS = (
    "HF_TOKEN",
    "TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
    "HF_HUB_TOKEN",
)

DEFAULT_MAX_AI_SAMPLES = 17500
DEFAULT_TRAIN_BATCH_SIZE = 96
DEFAULT_EVAL_BATCH_SIZE = 64
DEFAULT_NUM_REAL_TEST_SAMPLES = 15260
DEFAULT_THRESHOLD = 0.5
DEFAULT_ROC_TARGET_FPR = 0.1
DEFAULT_ROC_SAVE_PATH = "find_roc.png"
DEFAULT_TEST_RESULTS_PATH = "test_results.npz"
DEFAULT_REPORTS_SAVE_PATH = "evaluation_reports.json"

BASELINE_CLIP_MODEL = "ViT-B/32"
BASELINE_EPOCHS = 15
BASELINE_LR = 1e-4
BASELINE_NOISE_STD = 50 / 255.0
BASELINE_MODEL_SAVE_PATH = "find_resnet50_best_ViT-B_32.pth"
BASELINE_LOG_PATH = "logs.csv"
BASELINE_CHECKPOINT_RECORD_PATH = "find_base_latest_checkpoint.txt"

CLIPBASE_CLIP_MODEL = "ViT-B/32"
CLIPBASE_EPOCHS = 15
CLIPBASE_LR = 1e-4
CLIPBASE_MODEL_SAVE_PATH = "clip_base_best_ViT-B_32.pth"
CLIPBASE_LOG_PATH = "logs_clip_base.csv"
CLIPBASE_CHECKPOINT_RECORD_PATH = "clip_base_latest_checkpoint.txt"

AUG_CLIP_MODEL = "openai/clip-vit-base-patch32"
AUG_EPOCHS = 15
AUG_LR = 1e-4
AUG_NOISE_STD = 50 / 255.0
AUG_MODEL_SAVE_PATH = "find_resnet50_best_openai_clip-vit-base-patch32_rainbow.pth"
AUG_LOG_PATH = "logs_rainbow.csv"
AUG_CHECKPOINT_RECORD_PATH = "find_rainbow_latest_checkpoint.txt"

CAUSAL_CLIP_MODEL = "openai/clip-vit-base-patch32"
CAUSAL_EPOCHS = 15
CAUSAL_LR = 1e-4
CAUSAL_NOISE_STD = 50 / 255.0
CAUSAL_MODEL_SAVE_PATH = "causal_find_resnet50_best_openai_clip-vit-base-patch32_rainbow.pth"
CAUSAL_LOG_PATH = "logs_causal_find_rainbow.csv"
CAUSAL_CHECKPOINT_RECORD_PATH = "causal_find_latest_checkpoint.txt"
