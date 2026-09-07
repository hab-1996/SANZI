from pathlib import Path

from huggingface_hub import snapshot_download


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "models" / "Llama-3.2-1B-Instruct"
REVISION = "9213176726f574b556790deb65791e0c5aa438b6"


def main() -> None:
    # Download Llama-3.2-1B-Instruct model
    print("Downloading Llama-3.2-1B-Instruct...")
    snapshot_download(
        repo_id="meta-llama/Llama-3.2-1B-Instruct",
        revision=REVISION,
        local_dir=MODEL_DIR,
        token=True,
    )
    print("Done!")


if __name__ == "__main__":
    main()
