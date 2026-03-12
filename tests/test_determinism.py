import torch
import lightning as L
from models.model import MazeTransformer
from core.config import Config
from pathlib import Path


def test_determinism():
    config = Config.load(Path("config/base.toml"))
    seed = 88
    L.seed_everything(seed)
    x = torch.randn(1, 3, 132, 132)

    def get_output():
        L.seed_everything(seed)
        model = MazeTransformer(config)

        model.eval()
        with torch.no_grad():
            return model(x)

    out1 = get_output()
    out2 = get_output()

    assert torch.allclose(out1, out2, atol=1e-6), "Model is not deterministic"
    print("Model is deterministic")


if __name__ == "__main__":
    try:
        test_determinism()
    except AssertionError as e:
        print(f"Test failed: {str(e)}")
    except Exception as e:
        print(f"Test failed: {str(e)}")
