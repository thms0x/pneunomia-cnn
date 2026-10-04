import subprocess

DATA_DIR = "data/chest_xray"

def fetch_dataset():
    subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            "paultimothymooney/chest-xray-pneumonia",
            "-p",
            "data",
            "--unzip",
        ],
        capture_output=True
    )

fetch_dataset()