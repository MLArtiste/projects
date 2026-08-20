from pathlib import Path

import torch
from torch import nn, Tensor

from mlalib.utils import download_from_url


class ResidualBlock(nn.Module):
    """
    The ChessNet residual block

    Args:
        dim (int): The block dimension.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        out = self.norm(x)
        out = self.linear1(out)
        out = self.relu(out)
        out = self.linear2(out)
        return out + x


class ChessNet(nn.Module):
    """
    The ChessNet model

    Args:
        num_emb (int): Number of unique input tokens.
        emb_dim (int): Embedding dimension for each input token.
        num_blocks (int): Number of residual blocks.
        out_dim (int): Number of unique output tokens.
    """

    def __init__(
        self, num_emb: int, emb_dim: int, num_blocks: int, out_dim: int = 1968
    ):
        super().__init__()
        self.hidden_dim = 64 * emb_dim
        self.embedding = nn.Embedding(num_emb, emb_dim)
        self.flatten = nn.Flatten()
        self.features = nn.ModuleList(
            ResidualBlock(self.hidden_dim) for _ in range(num_blocks)
        )
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.policy_head = nn.Linear(self.hidden_dim, out_dim)

    def forward(
        self, x: Tensor, return_features: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:

        out = self.embedding(x)
        out = self.flatten(out)

        if return_features:
            activations = [out]

        for block in self.features:
            out = block(out)
            if return_features:
                activations.append(out)

        out = self.norm(out)
        out = self.policy_head(out)

        if return_features:
            activations = torch.stack(activations, dim=1)
            return out, activations

        return out


class ChessNet32_9(ChessNet):
    """
    The ChessNet32_9 model with downloadable weights.

    Args:
        weights (str or None): The weights version to load e.g 'v1.0'. If None, no pretrained
        weights are loaded. Defaults to None.
        root (str, Path or None): Optional directory to download the weights from.
    """

    WEIGHTS = {
        "v1.0": "https://huggingface.co/MLArtiste/ChessNet/resolve/main/chessnet32_9v1.pth"
    }

    def __init__(self, weights: str | None = None, root: str | Path | None = None):
        super().__init__(num_emb=1856, emb_dim=32, num_blocks=9, out_dim=1968)
        if weights is not None:
            if weights not in self.WEIGHTS:
                raise ValueError(
                    f"weights must be one of {list(self.WEIGHTS.keys())}, got {weights}"
                )
        
            path = download_from_url(self.WEIGHTS[weights], root=root)
            model_params = torch.load(path, map_location="cpu", weights_only=True)
            self.load_state_dict(model_params)
