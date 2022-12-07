import os
import pickle
import sys
from itertools import product
from multiprocessing import Pool, cpu_count
from types import SimpleNamespace

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from amoeba_game import AmoebaGame
from players.g2_player import Player, PlayerParameters


class GameConfig:
    m: float
    d: float
    A: int

    def __init__(self, m: float, d: float, A: int):
        self.m = m
        self.d = d
        self.A = A


def create_args(game_config: GameConfig):
    args = SimpleNamespace()
    args.no_gui = True
    args.no_vid = True
    args.disable_logging = True
    args.disable_timeout = True
    args.log_path = None
    args.seed = 0
    args.batch_mode = True

    args.metabolism = game_config.m
    args.density = game_config.d
    args.size = game_config.A
    args.final = 1_000

    return args


def create_params(
    formation_threshold: float,
    teeth_gap: int,
    teeth_shift_period: int,
    one_wide_backbone: bool,
    vertical_flip_size: int,
) -> PlayerParameters:
    p = PlayerParameters()
    p.formation_threshold = formation_threshold
    p.teeth_gap = teeth_gap
    p.teeth_shift_period = teeth_shift_period
    p.one_wide_backbone = one_wide_backbone
    p.vertical_flip_size = vertical_flip_size

    return p


# tournament_metabolisms = [0.05, 0.1, 0.25, 0.4, 1.0]
tournament_metabolisms = [1.0]
# tournament_densities = [0.01, 0.05, 0.1, 0.2]
tournament_densities = [0.2]
# tournament_sizes = [3, 5, 8, 15, 25]
tournament_sizes = [5]

# formation_thresholds = [0.7, 0.8, 0.9, 0.99]
# teeth_gaps = [1, 2, 3]
# teeth_alternation_periods = [2, 4, 6, 8]
# one_wide_backbones = [True, False]
# vertical_flip_sizes = [50, 100, 200]
formation_thresholds = [0.7, 0.8, 0.9, 0.99]
teeth_gaps = [1, 2]
teeth_alternation_periods = [2]
one_wide_backbones = [True]
vertical_flip_sizes = [100]

TRIALS_PER_DATAPOINT = 1

game_configs = [
    GameConfig(m, d, A)
    for m, d, A in list(
        product(tournament_metabolisms, tournament_densities, tournament_sizes)
    )
]
player_params = [
    create_params(
        formation_threshold,
        teeth_gap,
        teeth_shift_period,
        one_wide_backbone,
        vertical_flip_size,
    )
    for formation_threshold, teeth_gap, teeth_shift_period, one_wide_backbone, vertical_flip_size in product(
        formation_thresholds,
        teeth_gaps,
        teeth_alternation_periods,
        one_wide_backbones,
        vertical_flip_sizes,
    )
]

work = list(product(game_configs, player_params)) * TRIALS_PER_DATAPOINT

# Suppress stdout
# https://stackoverflow.com/questions/2828953/silence-the-stdout-of-a-function-in-python-without-trashing-sys-stdout-and-resto
class DummyFile(object):
    def write(self, x):
        pass

    def flush(self):
        pass


def run_trial(
    args: tuple[GameConfig, PlayerParameters]
) -> tuple[GameConfig, PlayerParameters, tuple[bool, int, int]]:
    # Suppress stdout
    save_stdout = sys.stdout
    sys.stdout = DummyFile()

    game_config, player_params = args

    amoeba_game = AmoebaGame(create_args(game_config))
    amoeba_game.add_player_object(
        "G2",
        Player(
            amoeba_game.rng,
            amoeba_game.get_player_logger("G2"),
            game_config.m,
            game_config.A * 4,
            "",
            params=player_params,
        ),
    )
    result = amoeba_game.play_game()

    # Restore stdout
    sys.stdout = save_stdout

    return game_config, player_params, result


if __name__ == "__main__":
    # if os.path.exists("benchmark.pkl"):
    #     with open("benchmark.pkl", "rb") as f:
    #         results = pickle.load(f)
    # else:
    #     pool = Pool(cpu_count())
    #     results = list(tqdm(pool.imap_unordered(run_trial, work), total=len(work)))

    #     with open("benchmark.pkl", "wb+") as f:
    #         pickle.dump(results, f)
    pool = Pool(cpu_count())
    results = list(tqdm(pool.imap_unordered(run_trial, work), total=len(work)))

    df = pd.DataFrame(results, columns=["game_config", "player_params", "result"])
    df = df.apply(
        (
            lambda r: pd.Series(
                {
                    "m": r.game_config.m,
                    "d": r.game_config.d,
                    "A": r.game_config.A,
                    "formation_threshold": r.player_params.formation_threshold,
                    "teeth_gap": r.player_params.teeth_gap,
                    "teeth_shift_period": r.player_params.teeth_shift_period,
                    "one_wide_backbone": r.player_params.one_wide_backbone,
                    "vertical_flip_size": r.player_params.vertical_flip_size,
                    "completed": float(r.result[0]),
                    "final_size": r.result[1],
                    "turns": r.result[2],
                }
            )
        ),
        axis="columns",
        result_type="expand",
    )
    df = df.sort_values(["m", "d", "A"])
    df.to_csv("parameter_benchmarks.csv")
    # split_dfs = [
    #     df[df.m == config.m][df.d == config.d][df.A == config.A]
    #     for config in sorted(game_configs, key=lambda c: c.A)
    # ]
    # num_rows = len(tournament_sizes)
    # num_cols = len(game_configs) // num_rows
    # fig, axes = plt.subplots(num_rows, num_cols)
    # # plt.subplots_adjust(hspace=0.1, wspace=0.1, left=0, right=1, top=1)
    # plt.subplots_adjust(left=0, right=1, bottom=0.05, top=0.95, wspace=0.02)
    # sns.set_theme()
    # for i, compare in enumerate(split_dfs):
    #     ax = axes[i // num_cols][i % num_cols]
    #     sns.heatmap(
    #         compare.drop(["m", "d", "A"]),
    #         annot=True,
    #         fmt=".1f",
    #         ax=ax,
    #         cbar=False,
    #         yticklabels=False,
    #         xticklabels=True,
    #     )
    #     ax.set_title(
    #         f"m={compare.iloc[0].m}, d={compare.iloc[0].d}, A={compare.iloc[0].A}"
    #     )
    #     ax.tick_params(left=False)
    # plt.show()
