import torch
from torch import nn

from chess import Board

from .tokenizer import FENTokenizer
from .uci_vocab import IDX_TO_UCI, UCI_TO_IDX


class ChessNetEngine:
    """
    Simple engine interface for ChessNet model.

    Args:
        chessnet (nn.Module): ChessNet model.
        device (str or torch.device): Device to run the model on. Defaults to 'cpu'.
    """

    def __init__(self, chessnet: nn.Module, device: str | torch.device = "cpu"):
        self.chessnet = chessnet.to(device)
        self.chessnet.eval()
        self.device = device
        self.tokenizer = FENTokenizer()
        self.idx_to_uci = IDX_TO_UCI
        self.uci_to_idx = UCI_TO_IDX

    def get_move(self, board: Board) -> str:
        """
        Get the engine's move for the given board.

        Args:
            board (chess.Board): Board to get move for.

        Returns:
            str: Engine's UCI move prediction.
        """
        fen = board.fen()
        tokens = self.tokenizer.encode(fen)
        token_ids = torch.tensor(tokens, device=self.device).unsqueeze(0)
        legal_moves = torch.tensor(
            [self.uci_to_idx[m.uci()] for m in board.legal_moves], device=self.device
        )
        with torch.no_grad():
            logits = self.chessnet(token_ids)
            legal_mask = torch.zeros(
                len(self.idx_to_uci),
                dtype=torch.bool,
                device=self.device,
            )
            legal_mask[legal_moves] = True
            mask_value = torch.finfo(logits.dtype).min
            logits = logits.masked_fill(~legal_mask, mask_value)
            move_idx = logits.argmax(dim=-1).item()

        return self.idx_to_uci[move_idx]
