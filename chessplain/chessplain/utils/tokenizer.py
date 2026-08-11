class FENTokenizer:
    """
    Tokenizer for converting FEN strings into square-based tokens.
    """

    _squares = tuple(f"{file}{rank}" for rank in "87654321" for file in "abcdefgh")
    _square_to_idx = {sq: i for i, sq in enumerate(_squares)}

    def __init__(self):
        self.idx_to_token = self._build_vocab()
        self.vocab = {tok: idx for idx, tok in enumerate(self.idx_to_token)}

    def tokenize(self, fen: str) -> list[str]:
        """
        Tokenize a FEN string into square-based tokens.

        Args:
            fen (str): FEN string to tokenize.

        Returns:
            list[str]: List of tokens representing the 64 board squares.
        """
        board, turn, castle, en_passant = fen.split()[:4]
        tokens = []

        for c in board:
            if c == "/":
                continue
            if c.isdigit():
                tokens.extend(["_"] * int(c))
            else:
                tokens.append(c)

        self._annotate_king(tokens, castle, "K", "KQ", turn)
        self._annotate_king(tokens, castle, "k", "kq", turn)

        self._add_en_passant_info(tokens, en_passant)

        tokens = [f"{square}:{piece}" for square, piece in zip(self._squares, tokens)]

        return tokens

    def encode(self, fen: str) -> list[int]:
        """
        Encode a FEN string into a list of token IDs.

        Args:
            fen (str): FEN string to encode.

        Returns:
            list[int]: List of token IDs corresponding to the encoded FEN.
        """
        tokens = self.tokenize(fen)
        return [self.vocab[token] for token in tokens]

    def decode(self, token_ids: list[int]) -> list[str]:
        """
        Decode token IDs into their corresponding string tokens.

        Args:
            token_ids (list[int]): List of token IDs to decode.

        Returns:
            list[str]: List of decoded tokens.
        """
        return [self.idx_to_token[idx] for idx in token_ids]

    def _annotate_king(
        self,
        tokens: list[str],
        castle: str,
        king: str,
        sides: str,
        turn: str,
    ):
        """
        Annotate a king token with its turn and castling rights.

        Args:
            tokens (list[str]): List of board tokens to modify.
            castle (str): FEN castling-rights field.
            king (str): King token to annotate.
            sides (str): Castling-right identifiers to check for the king.
            turn (str): Active color from the FEN string.
        """
        idx = tokens.index(king)
        suffix = "".join(side for side in sides if side in castle)
        king += f"_{turn}"
        if suffix:
            king += f"_{suffix}"
        tokens[idx] = king

    def _add_en_passant_info(self, tokens: list[str], target: str):
        """
        Add en passant information to the relevant pawn token.

        Args:
            tokens (list[str]): List of board tokens to modify.
            target (str): FEN en passant target square, or '-' if none exists.
        """
        if target == "-":
            return
        file = target[0]
        rank = int(target[1])
        if rank == 6:
            pawn_rank = rank - 1
        else:
            pawn_rank = rank + 1

        pawn_square = f"{file}{pawn_rank}"
        idx = self._square_to_idx[pawn_square]
        tokens[idx] += "_E"

    def _build_vocab(self) -> list[str]:
        """
        Build the vocabulary for the tokenizer.

        Returns:
            list[str]: List of tokens in vocabulary index order.
        """
        vocab = []
        for square in self._squares:
            vocab.append(f"{square}:_")
            for piece in [
                "P",
                "P_E",
                "N",
                "B",
                "R",
                "Q",
                "K_w",
                "K_w_K",
                "K_w_Q",
                "K_w_KQ",
                "K_b",
                "K_b_K",
                "K_b_Q",
                "K_b_KQ",
            ]:
                vocab.append(f"{square}:{piece}")
            for piece in [
                "p",
                "p_E",
                "n",
                "b",
                "r",
                "q",
                "k_w",
                "k_w_k",
                "k_w_q",
                "k_w_kq",
                "k_b",
                "k_b_k",
                "k_b_q",
                "k_b_kq",
            ]:
                vocab.append(f"{square}:{piece}")

        return vocab
