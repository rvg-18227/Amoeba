from abc import abstractmethod, ABC
import logging
import math
import os
import sys

import numpy as np

sys.path.append(os.getcwd())
from amoeba_state import AmoebaState


#------------------------------------------------------------------------------
#  Types
#------------------------------------------------------------------------------

cell = tuple[int, int]


#------------------------------------------------------------------------------
#  Helpers
#------------------------------------------------------------------------------

def find_movable_neighbor(
    x: int,
    y: int,
    amoeba_map: np.ndarray,
    bacteria: list[cell]
) -> list[cell]:

    if (x, y) in bacteria:
        return []

    out = []
    for x2, y2 in [
        # index of 4 neighboring cells
        (x, (y-1) % 100),
        (x, (y+1) % 100),
        ((x-1) % 100, y),
        ((x+1) % 100, y)
    ]:
        if amoeba_map[x2][y2] == 0:
            out.append((x2, y2))

    return out

def find_movable_cells(
    retract: list[cell],
    periphery: list[cell],
    amoeba_map: np.ndarray,
    bacteria: list[cell],
    mini: int
) -> list[cell]:

    movable = set()
    new_periphery = list(set(periphery) - set(retract))
    for i, j in new_periphery:
        nbr = find_movable_neighbor(i, j, amoeba_map, bacteria)
        for cell in nbr:
            movable.add(cell)

    movable_list = list(movable) + retract

    return movable_list[:mini]


#------------------------------------------------------------------------------
#  Movement Strategy
#------------------------------------------------------------------------------

class Strategy(ABC):

    @abstractmethod
    def move(
        self, state: AmoebaState, memory: int
    ) -> tuple[list[cell], list[cell], int]:

        pass

class RandomWalk(Strategy):
    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def move(
        self, state: AmoebaState, memory: int
    ) -> tuple[list[cell], list[cell], int]:

        mini = min(5, len(state.periphery) // 2)
        retract = [
            tuple(i)
            for i in self.rng.choice(
                state.periphery, replace=False, size=mini
            )
        ]

        movable = find_movable_cells(
            retract, state.periphery, state.amoeba_map, state.bacteria, mini)

        info = 0

        return retract, movable, info

class BucketAttack(Strategy):

    def _get_target_cells(self, size: int, cog: cell, xmax: int) -> list[cell]:
        """Returns the cells of the target shape by centering vertically on
        the y-value of Ameoba's center of gravity and placing the bucket arms
        on the column of Ameoba's rightmost cell.

        @size: current size of ameoba
        @cog: ameoba's center of gravity
        """
        # TODO:
        # from(yunlan): I think I've confused x and y coords, the target shape
        # actually has buckets facing downwards... To fix later

        buckets = math.floor((size - 2) / 4)
        arm_cells_cnt = 1 + buckets
        wall_cells_cnt = size - arm_cells_cnt

        _, y_cog = cog
        wall_top    = (y_cog + math.floor((wall_cells_cnt - 1) / 2))
        wall_bottom = (y_cog - math.ceil((wall_cells_cnt - 1) / 2))
        wall_cell_ys = np.linspace(wall_bottom, wall_top, wall_cells_cnt, True, dtype=int)

        arm_top    = (y_cog + 3 * math.floor((arm_cells_cnt - 1) / 2))
        arm_bottom = (y_cog - 3 * math.ceil((arm_cells_cnt - 1) / 2))
        arm_cell_ys = np.linspace(arm_bottom, arm_top, arm_cells_cnt, True, dtype=int)

        wall_cells = [
            ( (xmax - 1) % 100, y % 100 )
            for y in wall_cell_ys
        ]

        arm_cells = [
            ( xmax % 100, y % 100 )
            for y in arm_cell_ys
        ]

        return wall_cells + arm_cells

    def _reshape(
        self,
        curr_state: AmoebaState,
        memory: int,
        target: set[cell]
    ) -> tuple[list[cell], list[cell], int]:
        """Computes cells to retract and cells to move onto in a best effort way
        to morphy the ameoba into the target shape.

        @curr_state: current state known to the Ameoba, its periphery, etc
        @memory: 1 byte memory of ameoba
        @target: target shape - represented by a list of cells - to morph
                 ameoba into
        """
        # simple heuristic:
        # 1. find all cells we can retract and doesn't overlap with the target
        # 2. find all cells we can move onto once we retract all cells in step 1
        # 3. find and return the overlap between cells in step 2 and target cells
        #    unoccupied by our Ameoba

        retractable_cells = set(curr_state.periphery) - target
        occupiable_cells = find_movable_cells(
            list(retractable_cells), curr_state.periphery, curr_state.amoeba_map,
            curr_state.bacteria, len(retractable_cells)
        )

        ameoba_cells = list(zip(*np.where(curr_state.amoeba_map == 1)))
        unoccupied_target_cells = target - set(ameoba_cells)

        to_occupy = set(occupiable_cells).intersection(unoccupied_target_cells)

        return list(retractable_cells)[:len(to_occupy)], list(to_occupy), memory

    def move(
        self, state: AmoebaState, memory: int
    ) -> tuple[list[cell], list[cell], int]:

        # TODO: compute ameoba's center of gravity, and xmax
        size = 9
        cog = (50, 50)
        xmax = 51

        target_cells = self._get_target_cells(size, cog, xmax)
        return self._reshape(state, memory, set(target_cells))


#------------------------------------------------------------------------------
#  Group 3 Ameoba
#------------------------------------------------------------------------------

class Player:
    def __init__(
        self,
        rng: np.random.Generator,
        logger: logging.Logger,
        metabolism: float,
        goal_size: int,
        precomp_dir: str
    ) -> None:
        """Initialise the player with the basic amoeba information

        Args:
            rng: numpy random number generator, use this for same player
                 behavior across run
            logger: logger use this like logger.info("message")
            metabolism: the percentage of amoeba cells, that can move
            goal_size: the size the amoeba must reach
            precomp_dir: Directory path to store/load pre-computation
        """

        self.rng = rng
        self.logger = logger
        self.metabolism = metabolism
        self.goal_size = goal_size
        self.current_size = goal_size / 4

        self.strategies = dict(
            random_walk=RandomWalk(rng),
            bucket_attack=BucketAttack()
        )

    def move(
        self,
        last_percept: AmoebaState,
        current_percept: AmoebaState,
        info: int
    ) -> tuple[list[cell], list[cell], int]:
        """Computes and returns an amoeba movement given the current state of
        the amoeba map.

        Args:
            last_percept: contains state information after the previous move
            current_percept: contains current state information
            info: a byte (ranging from 0 to 256) to convey information from
                  the previous turn

        Returns:
            1. A list of cells on the periphery that the amoeba retracts
            2. A list of cells the retracted cells have moved to
            3. A byte of information (values range from 0 to 255) that the
               amoeba can use
        """
        for i, j in current_percept.bacteria:
            current_percept.amoeba_map[i][j] = 1
            current_percept.current_size += 1
        self.current_size = current_percept.current_size

        return self.strategies['random_walk'].move(current_percept, info)


#------------------------------------------------------------------------------
#  Unit Tests
#------------------------------------------------------------------------------

def Test_BucketAttack():
    bucket_attack = BucketAttack()
    cases = [
        dict(size=9, cog=(50,50), xmax=51,
            expected=[
                # wall
                (50, 47), (50, 48), (50, 49), (50, 50), (50, 51), (50, 52), (50, 53),
                # bucket arms
                (51, 47),                     (51, 50) 

            ])
    ]

    for tc in cases:
        got = bucket_attack._get_target_cells(tc['size'], tc['cog'], tc['xmax'])
        assert got == tc['expected'], f"expect: {tc['expected']}\nbut got: {got}"

    print("[ PASSED ] BucketAttack._get_target_cells")

    # BucketAttack e2e test
    ameoba_map = np.zeros(shape=(100, 100))
    ameoba_map[49:52, 49:52] = 1

    periphery = [
        (49, 49), (49, 50), (49, 51),
        (50, 49),           (50, 51),
        (51, 49), (51, 50), (51, 51)
    ]
    movable_cells = find_movable_cells([], periphery, ameoba_map, [], 12)

    curr_state = AmoebaState(9, ameoba_map, periphery, [], movable_cells)
    to_retract, to_occupy, _ = bucket_attack.move(curr_state, 0)

    print()
    print(f"retracted cells:      {to_retract}")
    print(f"newly occupied cells: {to_occupy}")


if __name__ == "__main__":
    Test_BucketAttack()