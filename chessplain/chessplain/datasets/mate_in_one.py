import json
from pathlib import Path

from mlalib.utils import download_from_url

from .chess_dataset import ChessDataset


class MateInOne(ChessDataset):
    """
    The MateInOne dataset.

    Args:
        month (int): The month to download (1-12).
        year (int): The year to download (2013-2017).
        val (bool): Whether to download the validation set. Defaults to False.
        root (str, Path or None): Optional directory to download the dataset to.
    """

    COUNTS_URL = (
        "https://huggingface.co/datasets/MLArtiste/mate_in_one/resolve/main/counts.json"
    )

    def __init__(
        self,
        *,
        month: int | None = None,
        year: int | None = None,
        val: bool = False,
        root: str | Path | None = None,
    ):
        if val:
            if month is not None or year is not None:
                raise ValueError("cannot specify month or year if val is True")

            filename = "val_set"
        else:
            if month is None or year is None:
                raise ValueError("must specify month and year if val is False")

            if not isinstance(month, int) or not isinstance(year, int):
                raise ValueError("month and year must be integers")

            if not 1 <= month <= 12:
                raise ValueError("month must be between 1 and 12")

            if not 13 <= year <= 17:
                raise ValueError("year must be between 13 and 17")

            yr_suffix = year % 100
            filename = f"{yr_suffix:02d}-{month:02d}"

        url = f"https://huggingface.co/datasets/MLArtiste/mate-in-one/resolve/main/{filename}.fen.zstd"
        metadata = json.loads(download_from_url(self.COUNTS_URL, root=root).read_text())
        count = 500_000 if val else metadata[filename]
        path = download_from_url(url, root=root)

        super().__init__(paths=path, counts=count)
