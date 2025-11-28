import os
import shutil
from sklearn.model_selection import train_test_split
from config.config import config

def split_dataset():
    data_path = config["data_path"]
    classes = os.listdir(data_path)

    os.makedirs("data/train", exist_ok=True)
    os.makedirs("data/val", exist_ok=True)
    os.makedirs("data/test", exist_ok=True)

    for cls in classes:
        imgs = os.listdir(os.path.join(data_path, cls))

        train, temp = train_test_split(imgs, test_size=1 - config["train_split"])
        val, test = train_test_split(temp, test_size=config["test_split"] /(config["test_split"]+config["val_split"]))

        for folder, subset in zip(["train", "val", "test"], [train, val, test]):
            dest = f"data/{folder}/{cls}"
            os.makedirs(dest, exist_ok=True)
            for img in subset:
                shutil.copy(os.path.join(data_path, cls, img), os.path.join(dest, img))

if __name__ == "__main__":
    split_dataset()
