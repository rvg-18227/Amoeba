from amoeba_game import AmoebaGame
from types import SimpleNamespace
from players.g2_player import Player


def create_args(m: float, d: float, A: int):
    args = SimpleNamespace()
    args.no_gui = True
    args.no_vid = True
    args.disable_logging = True
    args.disable_timeout = True
    args.log_path = None
    args.seed = 0
    args.batch_mode = True

    args.metabolism = m
    args.density = d
    args.size = A
    args.final = 10_000

    return args


metabolism = 1.0
density = 0.1
size = 9
amoeba_game = AmoebaGame(create_args(1, 0.1, 9))
amoeba_game.add_player_object(
    "G2",
    Player(
        amoeba_game.rng, amoeba_game.get_player_logger("G2"), metabolism, size * 4, ""
    ),
)
amoeba_game.play_game()
print(amoeba_game.turns)
