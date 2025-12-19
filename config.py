import os
from dotenv import load_dotenv
import torch

# Load environment variables from .env file
load_dotenv()

VOICES_DIR = os.getenv("VOICES_DIR")

# check if path exists
if not os.path.exists(VOICES_DIR):
    raise ValueError(f"Path {VOICES_DIR} does not exist")

if VOICES_DIR[-1] != "/":
    VOICES_DIR += "/"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
API_PORT = os.getenv("API_PORT", "5001")
API_HOST = os.getenv("API_HOST", "0.0.0.0")
SUPPORTED_VOICES = os.getenv("SUPPORTED_VOICES", "").split(",")
SUPPORTED_RESPONSE_FORMATS = ["mp3", "opus", "aac", "flac", "wav", "pcm"]
CORS_ALLOWED_ORIGIN = os.getenv("CORS_ALLOWED_ORIGIN", "*")
SEED = int(os.getenv("SEED", 0))

# if SUPPORTED_VOICES is empty, then we will use all voices in the VOICES_DIR directory
if SUPPORTED_VOICES == [""]:
    print("No voices specified, using all voices in the VOICES_DIR directory")

    SUPPORTED_VOICES = [f[:-4] for f in os.listdir(VOICES_DIR) if f.endswith(".wav")]

    print(f"Found {len(SUPPORTED_VOICES)} voices in the VOICES_DIR directory")

print(f"🚀 Running on device: {DEVICE}")

if SEED != 0:
    import utils

    utils.set_seed(SEED)  # For reproducibility
