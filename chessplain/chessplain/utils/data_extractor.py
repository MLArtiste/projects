import csv
import gzip
from pathlib import Path

import torch
import zstandard as zstd
from chess import pgn, Board, Move

from .tokenizer import FENTokenizer
from .uci_vocab import IDX_TO_UCI, UCI_TO_IDX


def _write_sample(
    writer, board: Board, move: Move, tokenizer: FENTokenizer, uci_to_idx: dict[str, int]
):
    """
    Write a single chess position and its target data to a compressed stream.

    Args:
        writer: Writable compressed stream to which the encoded sample is written.
        board (Board): Chess position from which the sample is generated.
        move: Move played from the current position.
        tokenizer (FENTokenizer): Tokenizer used to encode the board position.
        uci_to_idx (dict[str, int]): Mapping from UCI move notation to vocabulary
        indices.
    """
    fen = torch.tensor(
        tokenizer.encode(board.fen()),
        dtype=torch.uint16,
    )

    target = torch.tensor(
        [uci_to_idx[move.uci()]],
        dtype=torch.uint16,
    )
    legal_moves = torch.tensor(
        [uci_to_idx[m.uci()] for m in board.legal_moves],
        dtype=torch.uint16,
    )
    num_legal_moves = torch.tensor(
        [len(legal_moves)],
        dtype=torch.uint8,
    )

    writer.write(fen.numpy().tobytes())
    writer.write(target.numpy().tobytes())
    writer.write(num_legal_moves.numpy().tobytes())
    writer.write(legal_moves.numpy().tobytes())


def build_train_data(
    pgn_paths: str | Path | list[str | Path],
    output_path: str | Path,
    num_samples: int | None = None,
):
    """
    Extract training samples from gzip-compressed PGN files and write them to a
    Zstandard-compressed file.

    Args:
        pgn_paths (str, Path or list[str or Path]): Path(s) to gzip-compressed PGN files.
        output_path (str or Path): Path where the compressed dataset is written.
        num_samples (int or None): Maximum number of positions to extract.
        If None, all available positions are extracted. Defaults to None.
    """
    pgn_paths = [pgn_paths] if isinstance(pgn_paths, (str, Path)) else pgn_paths
    output_path = Path(output_path)
    tokenizer = FENTokenizer()
    num_positions = 0
    compressor = zstd.ZstdCompressor()

    with (
        open(output_path, "wb") as out_file,
        compressor.stream_writer(out_file) as writer,
    ):
        for pgn_path in pgn_paths:
            print(f"Processing {Path(pgn_path).name}...")

            with gzip.open(pgn_path, "rt", encoding="utf-8") as pgn_file:
                while True:
                    game = pgn.read_game(pgn_file)
                    if game is None:
                        break

                    board = game.board()

                    for move in game.mainline_moves():
                        _write_sample(writer, board, move, tokenizer, UCI_TO_IDX)
                        board.push(move)
                        num_positions += 1

                        if num_samples is not None and num_positions >= num_samples:
                            print("Target limit reached.")
                            break

                    if num_samples is not None and num_positions >= num_samples:
                        break

                if num_samples is not None and num_positions >= num_samples:
                    break

    print(f"Extracted {num_positions:,} positions.")


def build_val_data(
    pgn_paths: str | Path | list[str | Path],
    output_path: str | Path,
    num_samples: int | None = 1_000_000,
):
    """
    Extract unique samples from gzip-compressed PGN files
    and write them to a Zstandard-compressed file.

    Positions are considered unique using the board position without the
    halfmove and fullmove counters, together with the move played from that
    position.

    Args:
        pgn_paths (str, Path, or list[str or Path]): Path(s) to gzip-compressed
        PGN files.
        output_path (str or Path): Path where the compressed validation dataset
        is written.
        num_samples (int or None): Maximum number of unique positions to
        extract. If None, all unique positions
        are extracted. Defaults to 1,000,000.
    """

    pgn_paths = (
        [Path(pgn_paths)]
        if isinstance(pgn_paths, (str, Path))
        else [Path(path) for path in pgn_paths]
    )
    output_path = Path(output_path)

    tokenizer = FENTokenizer()
    uci_vocab = {uci: idx for idx, uci in enumerate(IDX_TO_UCI)}

    seen = set()
    num_positions = 0
    num_unique = 0

    compressor = zstd.ZstdCompressor()

    with (
        open(output_path, "wb") as out_file,
        compressor.stream_writer(out_file) as writer,
    ):
        for pgn_path in pgn_paths:
            print(f"Processing {pgn_path.name}...")

            with gzip.open(pgn_path, "rt", encoding="utf-8") as pgn_file:
                while True:
                    game = pgn.read_game(pgn_file)
                    if game is None:
                        break

                    board = game.board()

                    for move in game.mainline_moves():
                        fen = board.fen()

                        # Remove halfmove and fullmove counters
                        clean_fen = " ".join(fen.split()[:4])

                        key = (clean_fen, move.uci())

                        if key not in seen:
                            seen.add(key)
                            _write_sample(writer, board, move, tokenizer, uci_vocab)
                            num_unique += 1

                        board.push(move)
                        num_positions += 1

                        if num_samples is not None and num_unique >= num_samples:
                            print("Target limit reached.")
                            break

                    if num_samples is not None and num_unique >= num_samples:
                        break

                if num_samples is not None and num_unique >= num_samples:
                    break

    print(
        f"Extracted {num_unique:,} unique positions from {num_positions:,} positions."
    )


def build_annotated_fens(
    pgn_paths: str | Path | list[str | Path],
    output_path: str | Path,
    num_samples: int | None = None,
):
    """
    Extract annotated chess positions from gzip-compressed PGN files
    and write them to a CSV file.

    For each position with a comment, the extracted record contains the
    position, evaluation, UCI move, and SAN move. Positions are deduplicated
    using the FEN representation without the halfmove and fullmove counters.

    Args:
        pgn_paths (str, Path, or list[str or Path]): Path(s) to gzip-compressed
        PGN files.
        output_path (str or Path): Path where the CSV dataset is written.
        num_samples (int or None): Maximum number of unique annotated positions
        to extract. If None, all available annotated positions are extracted.
        Defaults to None.
    """
    pgn_paths = (
        [Path(pgn_paths)]
        if isinstance(pgn_paths, (str, Path))
        else [Path(path) for path in pgn_paths]
    )
    output_path = Path(output_path)

    records = {}

    num_positions = 0

    for pgn_path in pgn_paths:
        print(f"Processing {pgn_path.name}...")

        with gzip.open(pgn_path, "rt", encoding="utf-8") as pgn_file:
            while True:
                game = pgn.read_game(pgn_file)

                if game is None:
                    break

                board = game.board()
                node = game

                while node.variations:
                    next_node = node.variation(0)
                    num_positions += 1

                    fen = " ".join(board.fen().split()[:4])
                    san_move = next_node.san()

                    if fen not in records:
                        comment = next_node.comment

                        if comment:
                            eval_score = comment.split("/")[0].strip()

                            records[fen] = (
                                eval_score,
                                next_node.move.uci(),
                                san_move,
                            )

                            if num_samples is not None and len(records) >= num_samples:
                                print("Target limit reached.")
                                break

                    board.push(next_node.move)
                    node = next_node

                if num_samples is not None and len(records) >= num_samples:
                    break

            if num_samples is not None and len(records) >= num_samples:
                break

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["fen", "eval", "uci", "san"])

        for fen, (eval_score, uci, san) in records.items():
            writer.writerow([fen, eval_score, uci, san])

    print(
        f"Extracted {len(records):,} unique positions "
        f"from {num_positions:,} positions."
    )
