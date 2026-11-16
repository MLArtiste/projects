from pathlib import Path
from mlalib.utils import download_from_url


class Lichess:
    """
    The Lichess dataset.

    Args:
        month (int): The month to load (1-12).
        year (int): The year to load (2013-present).
        root (str, Path or None): Optional directory containing the folder 
        where the dataset is stored. Defaults to the current working directory.
        download (bool): Whether to download the dataset if it is not found.
        Defaults to False.
    """

    def __init__(
        self,
        month: int,
        year: int,
        root: str | Path | None = None,
        download: bool = False,
    ):
        self.root = Path(root) if root else Path.cwd()
        self.month = f"{month:02d}"
        self.year = str(year)
        self.download = download
        self._URL = (
            "https://database.lichess.org/standard/"
            f"lichess_db_standard_rated_{self.year}-{self.month}.pgn.zst"
        )

        self.path = (
            self.root
            / "lichess"
            / f"lichess_db_standard_rated_{self.year}-{self.month}.pgn.zst"
        )

        if self.path.is_file():
            return

        if self.download:
            download_from_url(self._URL, filename=self.path)

        if not self.path.is_file():
            raise FileNotFoundError(
                f"Dataset file {self.path} not found. "
                "Set download=True to download it."
            )
