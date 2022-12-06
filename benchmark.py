import argparse
import itertools

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
class AmoebaCopy(AmoebaGame):
    """Convert to ray actor to allow remote calls.
        Limit each class instance to a single CPU.
    """
    def __init__(self, args):
        super().__init__(args)


def rayobj_to_iterator(obj_ids):
    """Convert a set of ray ObjRef to an iterator that yeilds results as they finish"""
    while obj_ids:
        done, obj_ids = ray.wait(obj_ids)
        yield ray.get(done[0])


if __name__ == "__main__":
    # NOTE: The args are overridden by the benchmark script. None of these are used.
    parser = argparse.ArgumentParser()
    parser.add_argument("--metabolism", "-m", type=float, default=1.0, help="Value between 0 and 1 (including 1) that "
                                                                            "indicates what proportion of the amoeba "
                                                                            "is allowed to retract in one turn")
    parser.add_argument("--size", "-A", type=int, default=15, help="length of a side of the initial amoeba square "
                                                                   "(min=3, max=50")
    parser.add_argument("--final", "-l", type=int, default=1000, help="the maximum number of days")
    parser.add_argument("--density", "-d", type=float, default=0.3, help="Density of bacteria on the map")
    parser.add_argument("--seed", "-s", type=int, default=2, help="Seed used by random number generator, specify 0 to "
                                                                  "use no seed and have different random behavior on "
                                                                  "each launch")
    parser.add_argument("--no_gui", "-ng", action="store_true", help="Disable GUI")
    parser.add_argument("--log_path", default="log", help="Directory path to dump log files, filepath if "
                                                          "disable_logging is false")
    parser.add_argument("--disable_logging", action="store_true", help="Disable Logging, log_path becomes path to file")
    parser.add_argument("--disable_timeout", action="store_true", help="Disable timeouts for player code")
    parser.add_argument("--player", "-p", default="d", help="Specifying player")
    parser.add_argument("--vid_name", "-v", default="game", help="Naming the video file")
    parser.add_argument("--no_vid", "-nv", action="store_true", help="Stops generating video of the session")
    parser.add_argument("--batch_mode", action="store_true", help="For running games via scripts")
    args = parser.parse_args()

    args.no_gui = True
    args.no_vid = True
    args.disable_logging = True
    args.log_path = None
    args.disable_timeout = False
    args.batch_mode = True
    args.final = MAX_TURNS
    args.seed = SEED

    ray.init(num_cpus=NUM_CPUS, configure_logging=True, log_to_driver=False)  # don't print the console outputs
    # testing = Create as many copies as cpus with identical args
    # ray_start_time = time.time()
    # actor_list = [AmoebaCopy.remote(args) for _ in range(NUM_CPUS)]
    # runs_list = [x.start_game.remote() for x in actor_list]
    # result_list = ray.get(runs_list)
    # ray_end_time = time.time()
    # print(f"Ray total time: {ray_end_time - ray_start_time}")

    # Create the ray actors from class. Try combinations of parameters.
    actor_list = []
    for (px, mx, dx, sx) in itertools.product(PLAYERS, METABOLISMS, DENSITIES, SIZES):
        args.player = str(px)
        args.metabolism = mx
        args.density = dx
        args.size = sx
        actor_list.append(AmoebaCopy.remote(args))

    # Start parallel runs
    runs_list = [x.start_game.remote() for x in actor_list]

    # Create datastructure for Pandas array
    data = {
        "Player": [],  # self.player
        "Goal Reached": [],  # self.goal_reached
        "Num Turns": [],  # self.turns
        "Starting Size": [],  # self.start_size
        "Metabolism": [],  # self.metabolism
        "Density": [],  # self.density
        "Total Time": [],  # self.total_time
        "Final Size": [],  # self.amoeba_size
        "Goal Size": [],  # self.goal_size
    }

    # Progress bar to track finished ray tasks. Periodically save to a CSV file so we may get partial results.
    filename = "benchmark_results.csv"
    save_interval = 2.0  # Save every N seconds
    print(f"Saving benchmark results to {filename}")
    last_save_time = time.time()
    result_list = []
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
