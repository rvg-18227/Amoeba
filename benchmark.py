import argparse
from itertools import product
import numpy as np
import pickle
import sys
from typing import Any

from amoeba_game import AmoebaGame


# -----------------------------------------------------------------------------
#   Ameoba Game Parser
# -----------------------------------------------------------------------------

ameoba_parser = argparse.ArgumentParser()

ameoba_parser.add_argument("--metabolism", "-m", type=float, default=1.0,
    help="Value between 0 and 1 (including 1) that "
        "indicates what proportion of the amoeba "
        "is allowed to retract in one turn")
ameoba_parser.add_argument("--size", "-A", type=int, default=15,
    help="length of a side of the initial amoeba square (min=3, max=50")
ameoba_parser.add_argument("--final", "-l", type=int, default=1000,
    help="the maximum number of days")
ameoba_parser.add_argument("--density", "-d", type=float, default=0.3,
    help="Density of bacteria on the map")
ameoba_parser.add_argument("--seed", "-s", type=int, default=2,
    help="Seed used by random number generator, specify 0 to "
    "use no seed and have different random behavior on each launch")
ameoba_parser.add_argument("--port", type=int, default=8080,
    help="Port to start, specify -1 to auto-assign")
ameoba_parser.add_argument("--address", "-a", type=str, default="127.0.0.1",
    help="Address")
ameoba_parser.add_argument("--no_browser", "-nb", action="store_true",
    help="Disable browser launching in GUI mode")
ameoba_parser.add_argument("--no_gui", "-ng", action="store_true",
    help="Disable GUI")
ameoba_parser.add_argument("--log_path", default="log",
    help="Directory path to dump log files, filepath if disable_logging is false")
ameoba_parser.add_argument("--disable_logging", action="store_true",
    help="Disable Logging, log_path becomes path to file")
ameoba_parser.add_argument("--disable_timeout", action="store_true",
    help="Disable timeouts for player code")
ameoba_parser.add_argument("--player", "-p", default="d",
    help="Specifying player")
ameoba_parser.add_argument("--vid_name", "-v", default="game",
    help="Naming the video file")
ameoba_parser.add_argument("--no_vid", "-nv", action="store_true",
    help="Stops generating video of the session")


# -----------------------------------------------------------------------------
#   Parser
# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument(
    "--out",
    "-o",
    type=str,
    default='res/benchmark/default.p',
    help="file path to store benchmark results as a picke file"
)

parser.add_argument(
    "--seed",
    "-s",
    type=int,
    default=5123467,
    help="integer to seed np.random for generating seeds for simulator"
)

parser.add_argument(
    "--player",
    "-p",
    type=int,
    default=4,
    help="player to benchmark"
)

parser.add_argument(
    "--max_turns",
    "-l",
    type=int,
    default=500,
    help="max number of turns for an ameoba game"
)

# -----------------------------------------------------------------------------
#   Helpers
# -----------------------------------------------------------------------------

# Suppress stdout
# https://stackoverflow.com/questions/2828953/silence-the-stdout-of-a-function-in-python-without-trashing-sys-stdout-and-resto
class DummyFile(object):
    def write(self, x):
        pass

    def flush(self):
        pass


# -----------------------------------------------------------------------------
#   Game Configuration
# -----------------------------------------------------------------------------

class GameConfig:

    base_args = ['-ng', '--no_vid']

    def _add_opt(self, opt: str, val: Any):
        self.base_args.extend([opt, str(val)])
        return self

    def add_player(self, player: int):
        self._add_opt('-p', player)
        return self

    def add_metabolism(self, metabolism: float):
        self._add_opt('-m', metabolism)
        return self

    def add_size(self, size: int):
        self._add_opt('-A', size)
        return self

    def add_density(self, density: float):
        self._add_opt('-d', density)
        return self

    def add_seed(self, seed: int):
        self._add_opt('-s', seed)
        return self

    def add_max_turns(self, max_turns: int):
        self._add_opt('-l', max_turns)
        return self

    def get_args(self) -> list:
        return self.base_args


def create_config(
    player: int,
    size: int,
    density: float,
    metabolism: float,
    seed: int,
    max_turns: int
) -> GameConfig:
    return (
        GameConfig()
            .add_player(player)
            .add_size(size)
            .add_density(density)
            .add_metabolism(metabolism)
            .add_seed(seed)
            .add_max_turns(max_turns)
    )


# -----------------------------------------------------------------------------
#   Benchmarking methods
# -----------------------------------------------------------------------------

def run(config: GameConfig) -> tuple[bool, int, float]:
    """Runs the simulator for the given game configuration, and returns
    the results.
    
    Returns:
      ok: True if ameoba reached the goal size, otherwise False.
      turns: # turns taken to complete the game or tried but failed.
      ts: cpu time in seconds taken for the game.
    """
    # suppress stdout
    save_stdout = sys.stdout
    sys.stdout = DummyFile()

    args = ameoba_parser.parse_args(config.get_args())
    game = AmoebaGame(args)

    ok = game.goal_reached
    turns = game.turns
    ts = game.end_time - game.start_time

    # restore stdout
    sys.stdout = save_stdout

    return ok, turns, ts


# -----------------------------------------------------------------------------
#   Script Entrypoint
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    args = parser.parse_args()

    np.random.seed(args.seed)
    player = args.player
    max_turns = args.max_turns
    sizes = [3, 5, 8, 15, 25]
    metabolism = [1.0, 0.4, 0.25, 0.1, 0.05]
    densities = [0.2, 0.1, 0.05, 0.01]
    seeds = np.random.randint(0, high=sys.maxsize, size=1)

    opts = [
        (player, *opt)
        for opt in product(sizes, metabolism, densities, seeds)
        if opt[0] * opt[1] > 1
    ]

    print("\n---\nresults:\n")
    print(
        "%-8s %-3s %-9s %-12s %-22s %-10s %-7s %-6s" %
        ("player", "A", "density", "metabolism", "seed", "ok", "turns", "time")
    )

    out = []
    for opt in opts:
        config = create_config(*opt, max_turns)

        ok, turns, ts = run(config)

        run_result = [*opt, ok, turns, ts]
        out.append(run_result)

        print(
            "%-8d %-3d %-9.2f %-12.2f %-22d %-10r %-7d %-6.2f" %
            tuple(run_result)
        )

    with open(args.out, "wb") as f:
        pickle.dump(out, f)
