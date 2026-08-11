import chess


def get_idx_to_uci() -> list[str]:
    """
    Returns a list of unique UCI moves.
    
    Returns:
        list[str]: List of unique UCI strings.
    """
    
    legal_uci = []
    empty_fen_list = ["8"] * 8
    pieces = "pnbrqkPNBRQK"

    for p in pieces:
        file_placement = [
            f"{p}7",
            f"1{p}6",
            f"2{p}5",
            f"3{p}4",
            f"4{p}3",
            f"5{p}2",
            f"6{p}1",
            f"7{p}",
        ]
        active_player = "b" if p.islower() else "w"
        for i in range(8):
            for ff in file_placement:
                fen = empty_fen_list.copy()
                fen[i] = ff
                fen = "/".join(fen)
                fen = f"{fen} {active_player} KQkq - 0 1"
                legal_uci.extend(
                    list(move.uci() for move in chess.Board(fen).legal_moves)
                )

    capt_promo = []

    blk_src_sq = [f"{file}2" for file in "abcdefgh"]
    wht_src_sq = [f"{file}7" for file in "abcdefgh"]

    for src in blk_src_sq:
        for s in [ord(src[0]) - 1, ord(src[0]) + 1]:
            if chr(s).isalpha() and chr(s) in "abcdefgh":
                capt_promo.append(f"{src}{chr(s)}1")

    for src in wht_src_sq:
        for s in [ord(src[0]) - 1, ord(src[0]) + 1]:
            if chr(s).isalpha() and chr(s) in "abcdefgh":
                capt_promo.append(f"{src}{chr(s)}8")

    for move in capt_promo:
        for promo in "nbrq":
            legal_uci.append(move + promo)

    legal_uci = sorted(set(legal_uci))
    return legal_uci


IDX_TO_UCI = tuple(get_idx_to_uci())
UCI_TO_IDX = {uci: idx for idx, uci in enumerate(IDX_TO_UCI)}
