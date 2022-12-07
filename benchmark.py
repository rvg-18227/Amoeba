import itertools
from collections import defaultdict
from types import SimpleNamespace

import pandas as pd
import ray
import time
from tqdm import tqdm

from amoeba_game import AmoebaGame

NUM_CPUS = 30  # None to use all cores

PLAYERS = [2]
METABOLISMS = [0.05, 0.1, 0.25, 0.4, 1.0]
DENSITIES = [0.01, 0.05, 0.1, 0.2]
SIZES = [3, 5, 8, 15, 25]

MAX_TURNS = 10000
SEED = 0  # Randomize behavior


@ray.remote(num_cpus=1)
def run_amoeba_game(args):
    amoeba_game = AmoebaGame(args)
    amoeba_game.start_game()
    return amoeba_game


def rayobj_to_iterator(obj_ids):
    """Convert a set of ray ObjRef to an iterator that yeilds results as they finish"""
    while obj_ids:
        done, obj_ids = ray.wait(obj_ids)
        yield ray.get(done[0])


def create_args(p:str, m: float, d: float, A: int):
    args = SimpleNamespace()
    # Constants
    args.no_gui = True
    args.no_vid = True
    args.vid_name = None
    args.disable_logging = True
    args.log_path = None
    args.disable_timeout = False
    args.batch_mode = True
    args.final = MAX_TURNS
    args.seed = SEED

    args.player = str(p)
    args.metabolism = m
    args.density = d
    args.size = A

    return args


if __name__ == "__main__":

    ray.init(num_cpus=NUM_CPUS, configure_logging=True, log_to_driver=False)  # don't print the console outputs

    # Start parallel runs. Try combinations of parameters.
    runs_list = []
    for (px, mx, dx, sx) in itertools.product(PLAYERS, METABOLISMS, DENSITIES, SIZES):
        args = create_args(px, mx, dx, sx)
        runs_list.append(run_amoeba_game.remote(args))

    # data for Pandas array
    data = defaultdict(list)
    filename = "benchmark_results.csv"
    save_interval = 2.0  # Save every N seconds
    print(f"Saving benchmark results to {filename}")
    last_save_time = time.time()

    # Progress bar to track finished ray tasks. Periodically save to a CSV file so we may get partial results.
    result_list = []  # All the returned amoeba objects. For plotting.
    for r in tqdm(rayobj_to_iterator(runs_list), total=len(runs_list), ncols=80):
        result_list.append(r)

        data["Player"].append(r.player_name)
        data["Goal Reached"].append(r.goal_reached)
        data["Num Turns"].append(r.game_end)
        data["Starting Size"].append(r.start_size)
        data["Metabolism"].append(r.metabolism)
        data["Density"].append(r.density)
        data["Total Time"].append(round(r.total_time, 2))
        data["Final Size"].append(r.amoeba_size)
        data["Goal Size"].append(r.goal_size)

        if time.time() - last_save_time > save_interval:
            df = pd.DataFrame(data=data)
            df.to_csv(filename, index=False)
