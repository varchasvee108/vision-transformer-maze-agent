import torch
from core.config import Config
from models.model import MazeTransformer
from PIL import Image
import torchvision.transforms as T
from pathlib import Path
import argparse


def run_inference(image_path, checkpoint_path):
    config = Config.load(Path("config/base.toml"))

    model = MazeTransformer.load_from_checkpoint(
        checkpoint_path=checkpoint_path, config=config
    )

    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    img = Image.open(Path(image_path)).convert("RGB")
    transform = T.Compose(
        [
            T.Resize((config.input_data.image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    img = transform(img).unsqueeze(0).to(model.device)

    with torch.no_grad():
        logits = model(img)
        pred = torch.argmax(logits, dim=-1).item()

    return pred


def main():
    parser = argparse.ArgumentParser(description="Run inference on a maze image")
    parser.add_argument(
        "--image_path", type=str, required=True, help="Path to the image"
    )
    parser.add_argument(
        "--checkpoint_path", type=str, required=True, help="Path to the checkpoint"
    )
    args = parser.parse_args()

    pred = run_inference(args.image_path, args.checkpoint_path)
    print(f"Predicted action: {pred}")


if __name__ == "__main__":
    main()
