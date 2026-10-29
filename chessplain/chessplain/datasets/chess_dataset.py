from pathlib import Path
import zstandard as zstd
from typing import Iterator

import torch
from torch import Tensor
from torch.utils.data import IterableDataset


class ChessDataset(IterableDataset):
    """
    The chess dataset base class.

    Args:
        paths (str, Path or list[str or Path]): The path(s) to the dataset(s) file.
        counts (int, list[int] or None): Optional number of samples in the dataset(s). Defaults to None.
    """

    def __init__(
        self,
        paths: str | Path | list[str | Path],
        counts: int | list[int] | None = None,
    ):
        self.paths = [paths] if isinstance(paths, (str, Path)) else paths

        if counts is None:
            self.counts = None

        elif isinstance(counts, int):
            if len(self.paths) != 1:
                raise ValueError("'counts' must be None or the same length as 'paths'")
            self.counts = [counts]

        else:
            if len(counts) != len(self.paths):
                raise ValueError("'counts' must be None or the same length as 'paths'")

            if any(not isinstance(c, int) for c in counts):
                raise ValueError("'counts' must be a list of integers")
            self.counts = counts

    @staticmethod
    def read_file(path: str | Path) -> Iterator[tuple[Tensor, Tensor, Tensor]]:
        """
        Iterate over the dataset samples.

        Yields:
            tuple[Tensor, Tensor, Tensor]: FEN token IDs, target move ID, and
            boolean legal-move mask.
        """
        dctx = zstd.ZstdDecompressor()

        with (
            open(path, "rb") as zstd_file,
            dctx.stream_reader(zstd_file) as reader,
        ):
            while True:
                fen_bytes = reader.read(64 * 2)
                if len(fen_bytes) < 64 * 2:
                    break

                fen = torch.frombuffer(
                    bytearray(fen_bytes),
                    dtype=torch.uint16,
                ).long()

                target = torch.frombuffer(
                    bytearray(reader.read(2)),
                    dtype=torch.uint16,
                ).long()[0]

                num_legal = torch.frombuffer(
                    bytearray(reader.read(1)),
                    dtype=torch.uint8,
                ).item()

                legal_moves = torch.frombuffer(
                    bytearray(reader.read(num_legal * 2)),
                    dtype=torch.uint16,
                ).long()

                legal_mask = torch.zeros(1968, dtype=torch.bool)
                legal_mask[legal_moves] = True

                yield fen, target, legal_mask

    def __iter__(self):
        """
        Iterate over samples of every file in the dataset.

        Yields:
            tuple[Tensor, Tensor, Tensor]: FEN token IDs, target move ID, and
            boolean legal-move mask.
        """
        for path in self.paths:
            yield from self.read_file(path)

    def __len__(self) -> int:
        if self.counts is None:
            raise TypeError("dataset length is unknown")

        return sum(self.counts)
