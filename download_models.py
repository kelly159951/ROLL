
import ray
from roll.utils.checkpoint_manager import download_model

model_name_or_path = "Qwen/Qwen3-1.7B-Base"
model_name_or_path = download_model(model_name_or_path, local_dir="./models/"+model_name_or_path)
print(f"load model to {model_name_or_path}")

ray.shutdown()