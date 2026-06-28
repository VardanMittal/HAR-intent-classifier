from pathlib import Path
import yaml
from src.data import loader
PROJECT_ROOT = Path(__file__).resolve().parent

with open(PROJECT_ROOT / "configs" / "config.yaml") as f:
    config = yaml.safe_load(f)

# Convert relative paths to absolute paths
for key, value in config["paths"].items():
    config["paths"][key] = PROJECT_ROOT / value


X_train, y_train, subjects_train, X_test, y_test, subjects_test = loader.load_dataset(config)

print(X_train.shape)
print(y_train.shape)
print(subjects_train.shape)
