from pathlib import Path

from huggingface_hub import snapshot_download


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "models" / "Qwen1.5-0.5B-Chat"
REVISION = "4d14e384a4b037942bb3f3016665157c8bcb70ea"


def main() -> None:
    # Download Qwen1.5-0.5B-Chat model
    print("Downloading Qwen1.5-0.5B-Chat...")
    snapshot_download(
        repo_id="Qwen/Qwen1.5-0.5B-Chat",
        revision=REVISION,
        local_dir=MODEL_DIR,
    )
    print("Qwen downloaded successfully!")


if __name__ == "__main__":
    main()
