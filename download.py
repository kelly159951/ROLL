
import os
from functools import partial

# These must be set before importing huggingface_hub through checkpoint_manager.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
os.environ.setdefault("MODEL_DOWNLOAD_TYPE", "HUGGINGFACE_HUB")
# Optional for weak access to huggingface.co:
# os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import ray
from huggingface_hub import snapshot_download

from roll.utils.checkpoint_manager import download_model, model_download_registry

model_download_registry["HUGGINGFACE_HUB"] = partial(snapshot_download, max_workers=1)

# model_name_or_path = "Qwen/Qwen3-1.7B-Base"
# model_name_or_path = download_model(model_name_or_path, local_dir="./models/"+model_name_or_path)
# print(f"load model to {model_name_or_path}")


# model_name_or_path = "Qwen/Qwen3-4B-Base"
# model_name_or_path = download_model(model_name_or_path, local_dir="./models/"+model_name_or_path)
# print(f"load model to {model_name_or_path}")

model_name_or_path = "Qwen/Qwen3-8B-Base"
model_name_or_path = download_model(model_name_or_path, local_dir="./models/Qwen3-8B-Base")
print(f"load model to {model_name_or_path}")

# model_name_or_path = "virtuoussy/Qwen2.5-7B-Instruct-RLVR"
# previous_model_download_type = os.environ.get("MODEL_DOWNLOAD_TYPE")
# os.environ["MODEL_DOWNLOAD_TYPE"] = "HUGGINGFACE_HUB"
# try:
#     model_name_or_path = download_model(model_name_or_path, local_dir="./models/Qwen2.5-7B-Instruct-RLVR")
# finally:
#     if previous_model_download_type is None:
#         os.environ.pop("MODEL_DOWNLOAD_TYPE", None)
#     else:
#         os.environ["MODEL_DOWNLOAD_TYPE"] = previous_model_download_type
# print(f"load model to {model_name_or_path}")



ray.shutdown()
