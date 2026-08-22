import json
from pathlib import Path
import zstandard as zstd

import torch
from torch.utils.data import IterableDataset

from mlalib.utils import download_from_url


class Chesset(IterableDataset):
    """
    The Chesset dataset.

    Args:
        day (int): The day of the month to download (1-31).
        month (int): The month to download (1-12).
        val (bool): Whether to download the validation set. Defaults to False.
        root (str, Path or None): Optional directory to download the dataset to.
    """
    COUNTS_URL = (
        "https://huggingface.co/datasets/MLArtiste/Chesset/resolve/main/counts.json"
    )
    VALID_DAYS = (
        (1, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 16, 21, 22, 23, 24, 28, 30), 
        (3, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27, 28), 
        (2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 25, 26, 27, 28, 29, 31), 
        (1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 14, 15, 16, 17, 18, 19, 23, 24, 25, 28, 30), 
        (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30), 
        (1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 14, 16, 18, 19, 20, 22, 23, 25, 26, 27, 29), 
        (1, 2, 5, 6, 8, 9, 10, 11, 13, 15, 16, 18, 19, 20, 21, 23, 24, 25, 26, 28, 29, 30, 31), 
        (2, 3, 4, 5, 6, 8, 9, 11, 12, 13, 14, 15, 16, 18, 20, 21, 25, 27, 29, 30, 31), 
        (2, 3, 4, 5, 6, 8, 9, 12, 13, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29), 
        (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 29, 30, 31), 
        (1, 2, 3, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30), 
        (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31),
    )


    def __init__(
        self,
        *,
        day: int | None = None,
        month: int | None = None,
        val: bool = False,
        root: str | Path | None = None,
    ):
        if val:
            if day is not None and month is not None:
                raise ValueError("cannot specify day and month for val set")

            filename = "val_set"
        else:
            if day is None or month is None:
                raise ValueError("must specify day and month if val is False")

            if not isinstance(day, int) or not isinstance(month, int):
                raise ValueError("day and month must be integers")

            if not 1 <= month <= 12:
                raise ValueError("month must be between 1 and 12")

            if day not in self.VALID_DAYS[month - 1]:
                raise ValueError(
                    f"day {day} does not exist for month {month},"
                    f" use one of {self.VALID_DAYS[month - 1]} instead"
                )

            filename = f"25-{month:02d}-{day:02d}"

        metadata = json.loads(download_from_url(self.COUNTS_URL, root=root).read_text())
        self.count = 1000_000 if val else metadata[filename]
        url = f"https://huggingface.co/datasets/MLArtiste/Chesset/resolve/main/{filename}.fen.zstd"
        self.path = download_from_url(url, root=root)

    def __iter__(self):
        """
        Iterate over the dataset samples.

        Yields:
            tuple[Tensor, Tensor, Tensor]: FEN token IDs, target move ID, and
            boolean legal-move mask.
        """
        dctx = zstd.ZstdDecompressor()

        with (
            open(self.path, "rb") as zstd_file,
            dctx.stream_reader(zstd_file) as reader,
        ):
            while True:
                fen_bytes = reader.read(64 * 2)
                if len(fen_bytes) < 64 * 2:
                    break
                fen = torch.frombuffer(bytearray(fen_bytes), dtype=torch.uint16).long()
                target = torch.frombuffer(
                    bytearray(reader.read(2)), dtype=torch.uint16
                ).long()[0]
                num_legal = torch.frombuffer(
                    bytearray(reader.read(1)), dtype=torch.uint8
                ).item()
                legal_moves = torch.frombuffer(
                    bytearray(reader.read(num_legal * 2)), dtype=torch.uint16
                ).long()

                legal_mask = torch.zeros(1968, dtype=torch.bool)
                legal_mask[legal_moves] = True

                yield fen, target, legal_mask

    def __len__(self) -> int:
        return self.count
