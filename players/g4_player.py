from abc import abstractmethod, ABC
from enum import Enum
import logging
import math
import os
import sys
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.getcwd())
from amoeba_state import AmoebaState
import constants


#------------------------------------------------------------------------------
#  Miscellaneous
#------------------------------------------------------------------------------

debug = 0
debug_fig = None


#------------------------------------------------------------------------------
#  Types
#------------------------------------------------------------------------------

cell = tuple[int, int]

class State(Enum):
    empty, ameoba, bacteria = range(3)


#------------------------------------------------------------------------------
#  Debug Helpers
#------------------------------------------------------------------------------

def visualize_reshape(
    target: list[cell], ameoba: list[cell], bacteria: list[cell],
    occupiable: list[cell], retractable: list[cell],
    retract: list[cell], extend: list[cell]):

    if not debug:
        return

    # make sure we only create one plot for debugging
    # before creating a new one, clear the old one for redrawing
    global debug_fig
    if debug_fig is None:
        debug_fig = plt.subplots(1, 2)
    else:
        fig, _ = debug_fig
        fig.clear()
        debug_fig = fig, fig.subplots(1, 2)
    
    fig, (ax1, ax2) = debug_fig

    # marker sizes
    size_s = (mpl.rcParams['lines.markersize'] + 1.5) ** 2
    size_m = (mpl.rcParams['lines.markersize'] + 2.5) ** 2
    size_l = (mpl.rcParams['lines.markersize'] + 4) ** 2

    # common: ameoba & target
    for ax in [ax1, ax2]:
        ax.plot(*list(zip(*target)), 'r.', label='target')
        ax.plot(*list(zip(*ameoba)), 'g.', label='ameoba')

    def scatter(ax: plt.Axes, pts: list[cell], **kwargs) -> None:
        if len(pts) == 0:
            return

        ax.scatter(*list(zip(*pts)), **kwargs)


    # subplot 1: occupiable, retractable
    scatter(
        ax1, occupiable, label='occupiable',
        facecolors='none', edgecolors='tab:purple', s=size_s
    )
    scatter(
        ax1, retractable, label='retractable',
        facecolors='none', edgecolors='forestgreen', marker='s', s=size_l
    )

    # subplot 2: retract, extend, bacteria
    scatter(
        ax2, retract, label='retract',
        facecolors='none', edgecolors='forestgreen', marker='s', s=size_l
    )
    scatter(
        ax2, extend, label='extend',
        facecolors='none', edgecolors='tab:purple', s=size_s
    )
    scatter(
        ax2, bacteria, label='bacteria',
        facecolors='none', edgecolors='orange', marker='s', s=size_m
    )

    # legend
    ax1.legend(
        loc='upper center', bbox_to_anchor=(0.5, 1.1),
        ncol=4, fancybox=True, shadow=True
    )
    ax2.legend(
        loc='upper center', bbox_to_anchor=(0.5, 1.1),
        ncol=5, fancybox=True, shadow=True
    )
    fig.tight_layout()

    # switch back to figure 1: ameoba simulator
    plt.figure(1)


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
        if amoeba_map[x2][y2] == State.empty.value:
            out.append((x2, y2))

    return out

def find_movable_cells(
    retract: list[cell],
    periphery: list[cell],
    amoeba_map: np.ndarray,
    bacteria: list[cell],
    n: Optional[int] = None
) -> list[cell]:

    movable = set()
    new_periphery = list(set(periphery) - set(retract))
    for i, j in new_periphery:
        nbr = find_movable_neighbor(i, j, amoeba_map, bacteria)
        for cell in nbr:
            movable.add(cell)

    movable_list = list(movable) + retract

    return (
        movable_list[:n] if n is not None
        else movable_list
    )

def retract_k(
    k: int,
    choices: list[cell],
    possible_moves: list[cell],
    state: AmoebaState
) -> list[cell]:
    """Select up to k cells to retract from choices (list of retractable cells) that
    ensures the ameoba will stay connected after retraction.

    Args:
        k: ideal number of cells to retract
        choices: cells we want to retract from
        possible_moves: cells we can extend our ameoba to
        state: the current state of the ameoba
    
    Note: This function might return n < k cells to retract because retracting any
    more cells would cause separation.
    """
    def exposure(cell) -> int:
        x, y = cell
        exposure = 0

        for xn, yn in [
            # index of 4 neighboring cells
            (x, (y-1) % 100),
            (x, (y+1) % 100),
            ((x-1) % 100, y),
            ((x+1) % 100, y)
        ]:
            if state.amoeba_map[xn][yn] == State.empty.value:
                exposure += 1

        return exposure

    sorted_choices = sorted(
        [(cell, exposure(cell)) for cell in choices],
        key=lambda x: x[1],
        reverse=True
    )

    # fast path: optimistically try if selecting topk, return if
    # it passes check_move, in the best case this reduces # invocations
    # of check_moves from O(max(len(choices), len(possible_moves))) to 1
    _k = min(k, len(choices), len(possible_moves))
    top_k = [ cell for cell, _ in sorted_choices[:_k] ]
    if check_move(top_k, possible_moves[:_k], state):
        return top_k

    # slow path:
    # further possible optimizations if performance becomes a bottleneck again:
    # 1. binary search to find the longest prefix of sorted_choices we can retract
    # 2. best effort search for up to k retractable cells:
    #    pessimistically terminate search when we fail @m check_moves in a roll
    retract, i = [], 0
    while len(retract) < k and i < len(sorted_choices):
        cell, _ = sorted_choices[i]
        _retract = retract + [cell]
        _moves = possible_moves[:len(_retract)]

        # only add a cell to retract if it doesn't cause separation
        if check_move(_retract, _moves, state):
            retract.append(cell)

        i += 1

    return retract

def check_move(
    retract: list[cell],
    move: list[cell],
    state: AmoebaState
) -> bool:

    periphery = state.periphery
    amoeba_map = state.amoeba_map
    bacteria = state.bacteria

    if not set(retract).issubset(set(periphery)):
        return False

    movable = retract[:]
    new_periphery = list(set(periphery).difference(set(retract)))
    for i, j in new_periphery:
        nbr = find_movable_neighbor(i, j, amoeba_map, bacteria)
        for x, y in nbr:
            if (x, y) not in movable:
                movable.append((x, y))

    if not set(move).issubset(set(movable)):
        return False

    amoeba = np.copy(amoeba_map)
    amoeba[amoeba < 0] = 0
    amoeba[amoeba > 0] = 1

    for i, j in retract:
        amoeba[i][j] = 0

    for i, j in move:
        amoeba[i][j] = 1

    tmp = np.where(amoeba == 1)
    result = list(zip(tmp[0], tmp[1]))
    check = np.zeros((constants.map_dim, constants.map_dim), dtype=int)

    stack = result[0:1]
    while len(stack):
        a, b = stack.pop()
        check[a][b] = 1

        if (a, (b - 1) % constants.map_dim) in result and check[a][(b - 1) % constants.map_dim] == 0:
            stack.append((a, (b - 1) % constants.map_dim))
        if (a, (b + 1) % constants.map_dim) in result and check[a][(b + 1) % constants.map_dim] == 0:
            stack.append((a, (b + 1) % constants.map_dim))
        if ((a - 1) % constants.map_dim, b) in result and check[(a - 1) % constants.map_dim][b] == 0:
            stack.append(((a - 1) % constants.map_dim, b))
        if ((a + 1) % constants.map_dim, b) in result and check[(a + 1) % constants.map_dim][b] == 0:
            stack.append(((a + 1) % constants.map_dim, b))

    return (amoeba == check).all()


#------------------------------------------------------------------------------
#  Movement Strategy
#------------------------------------------------------------------------------

class Strategy(ABC):

    def __init__(self, metabolism: float) -> None:
        self.metabolism = metabolism
        self.shifted = 1
        self.rotation = 0

    @abstractmethod
    def move(
        self, prev_state: AmoebaState, state: AmoebaState, memory: int
    ) -> tuple[list[cell], list[cell], int]:

        pass
    
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
            list(retractable_cells),
            curr_state.periphery, curr_state.amoeba_map, curr_state.bacteria
        ) 

        ameoba_cells = list(zip(*np.where(curr_state.amoeba_map == State.ameoba.value)))
        unoccupied_target_cells = target - set(ameoba_cells)
        to_occupy = set(occupiable_cells).intersection(unoccupied_target_cells)

        k = min(
            int(self.metabolism * curr_state.current_size), # max retractable cells
            len(to_occupy), len(retractable_cells)
        )
        retract = retract_k(k, list(retractable_cells), list(to_occupy), curr_state)
        extend = list(to_occupy)[:len(retract)]

        # for debugging: stop when a move causes separation
        if debug and not check_move(retract, extend, curr_state):
            print("[ G4 ] problematic move")

        # debug
        visualize_reshape(
            list(target), ameoba_cells, curr_state.bacteria,
            occupiable_cells, list(retractable_cells),
            retract, extend
        )

        return retract, extend, memory


class RandomWalk(Strategy):
    def __init__(self, metabolism: float, rng: np.random.Generator) -> None:
        super().__init__(metabolism)
        self.rng = rng

    def move(
        self, prev_state: AmoebaState, state: AmoebaState, memory: int
    ) -> tuple[list[cell], list[cell], int]:

        mini = min(5, len(state.periphery) // 2)
        retract = [
            tuple(i)
            for i in self.rng.choice(
                state.periphery, replace=False, size=mini
            )
        ]

        movable = find_movable_cells(
            retract, state.periphery, state.amoeba_map, state.bacteria, n=mini
        )

        info = 0

        return retract, movable, info

class BucketAttack(Strategy):

    def __init__(self, metabolism, bucket_width=1, shift_n=-1):
        """Initializes BucketAttack.
        
        kwargs:
          bucket_width: # cells between bucket arms, default to 1
          shift_n: shift bucket arms up/down every n turns, acceptable n value
                   is [1, 16], otherwise the shifting behavior is disabled
        """
        super().__init__(metabolism)
        self.bucket_width = bucket_width
        self.shift_enabled = shift_n >= 1 and shift_n <= 16
        self.shift_n = shift_n
    
    def _spread_vertically(
        self,
        y_center: int,
        upper_cnt: int,
        lower_cnt: int,
        step=1
    ) -> np.ndarray:
        """Returns a list of y-values by spreading around y_center.
        
        kwarg @step is used to determine the distance between spread points.
        e.g. step = 1 => spreaded pts are immediately next to one another
             step = 3 => distance between each consecutive pair of spreaded pts
                         is 2.
        """
        cnt = upper_cnt + 1 + lower_cnt
        y_top    = y_center + step * upper_cnt
        y_bottom = y_center - step * lower_cnt
        y_spreads = np.linspace(y_bottom, y_top, cnt, True, dtype=int)

        return y_spreads

    def _get_target_cells(self, size: int, cog: cell, xmax: int) -> list[cell]:
        """Returns the cells of the target shape by centering vertically on
        the y-value of Ameoba's center of gravity and placing the bucket arms
        on the column of Ameoba's rightmost cell.

        @size: current size of ameoba
        @cog: center of gravity for the target shape (only y-value used currently)
        """
        wall_cost = self.bucket_width + 1
        bucket_cost = 2 * wall_cost + 1 # 2 * wall_cost + arm_cost

        _, y_cog = cog
        buckets, orphans = divmod(size - 3, bucket_cost)
        upper_buckets, lower_buckets = math.ceil(buckets / 2), math.floor(buckets / 2)

        inner_wall_cell_ys = self._spread_vertically(
            y_cog,
            wall_cost * upper_buckets,
            wall_cost * lower_buckets
        )
        outer_wall_cell_ys = self._spread_vertically(
            y_cog,
            wall_cost * upper_buckets,
            wall_cost * lower_buckets + orphans
        )
        arm_cell_ys = self._spread_vertically(
            y_cog, upper_buckets, lower_buckets , step=wall_cost
        )

        wall_cells = (
            [ ( xmax % 100,       y % 100 ) for y in inner_wall_cell_ys ] +
            [ ( (xmax - 1) % 100, y % 100 ) for y in outer_wall_cell_ys ]
        )
        arm_cells = [ ( (xmax + 1) % 100, y % 100 ) for y in arm_cell_ys ]

        return wall_cells + arm_cells

    def _get_rectangle_target(self, size: int, cog: cell, xmax: int) -> list[cell]:
        _, y_cog = cog
        wall_length, orphans=int(size/4),size%4
        upper=y_cog+wall_length
        lower=y_cog-wall_length

        inner_wall_cell_ys = self._spread_vertically(
            y_cog,
            upper,
            lower+orphans
        )
        outer_wall_cell_ys = self._spread_vertically(
            y_cog,
            upper,
            lower
        )
        wall_cells = (
                [(xmax % 100, y % 100) for y in inner_wall_cell_ys] +
                [((xmax + 1) % 100, y % 100) for y in outer_wall_cell_ys]
        )

        return wall_cells

    def _get_cog(self, curr_state: AmoebaState) -> tuple[int, int]:
        """Compute center of gravity of current Ameoba."""
        ameoba_cells = np.array(list(zip(*np.where(curr_state.amoeba_map == State.ameoba.value))))
        cog = (round(np.average(ameoba_cells[:,0])),round(np.average(ameoba_cells[:,1])))

        return cog
    
    def _get_xmax(self, curr_state: AmoebaState) -> int:
        """Returns the x-value of the "rightmost" Ameoba cell.
        
        Note: when the ameoba's bucket arm moves from x=99 to the right, the x-value
        will wrap around to 0. In this case, xmax is 0 since that's where the bucket
        arms are. Below is a graphical illustration:

                       x=0     ...     x = 98  x = 99
                        .                 .      .
                                          .      .
                                          .      .
                        .                 .      .
                      (xmax)
        """

        # we concatenate the ameoba map along the x-axis forming a 200 * 100 map
        #               x = 0 ... x = 99 x=100 ... x=199

        ameoba_map = np.vstack((curr_state.amoeba_map, curr_state.amoeba_map))
        ameoba_xs, _ = np.where(ameoba_map == State.ameoba.value)

        # we operate a subset (x=50 -> x=149) of this new map to find @xmax by
        # searching from xmax=xmin, and increment xmax whenever we see a x-value
        # immediately larger by 1
        x_filter = (ameoba_xs >= constants.map_dim / 2) & (ameoba_xs < constants.map_dim * 1.5)
        xs = sorted(set(ameoba_xs[x_filter]))
        xmax = min(xs)
        for x in xs:
            xmax += int(x == xmax + 1)

        return xmax % constants.map_dim

    def _in_shape(self, curr_state: AmoebaState) -> bool:
        """Returns a bool indicating if our bucket arms are in shape.
        
        In our implementation, *no memory bit* is needed to store information
        of whether we have gotten into "comb" shape or not, all thanks to
        this function/heuristic.

        If we have an expected number of bucket arms, it's safe for us to
        keep marching forward, since having bucket arms in formation would
        allow us to keep capturing bacteria on our way while adjusting
        formation.

        Otherwise, we need to wait for the bucket arms to get into shape, so
        that we don't risk moving and not able to eat any bacteria along the
        way.
        """
        xmax = self._get_xmax(curr_state)
        arms_expected = 1 + math.floor((curr_state.current_size - 3) / 7)
        arms_got = len(np.where(curr_state.amoeba_map[xmax,:] == State.ameoba.value)[0])

        return arms_got >= arms_expected

    def move(
        self, prev_state: AmoebaState, state: AmoebaState, memory: int
    ) -> tuple[list[cell], list[cell], int]:

        mem = f'{memory:b}'.rjust(8, '0')
        rotation = int(mem[-4:], 2)
        shifted = int(mem[-5])

        # increment rotation counter
        rotation = (rotation + 1) % self.shift_n

        if self.shift_enabled and rotation == 0:
            shifted = shifted ^ 1
            
        # update memory
        memory = int(f'{mem[:-5]}{shifted:b}{rotation:04b}', 2)


        size = state.current_size
        # TODO: maybe not always moving horizontally?
        if shifted:
            cog = (50, 49)
        else:
            cog  = (50, 50)

        # x-value of bucket arms
        arm_xval = self._get_xmax(state)
        if not self._in_shape(state):
            arm_xval -= 1

        # compute moves
        target_cells = self._get_target_cells(size, cog, arm_xval)
        retract, extend, memory = self._reshape(state, memory, set(target_cells))

        if len(retract) > 0:
            return retract, extend, memory
        else:
            target_cells = self._get_rectangle_target(size, cog, arm_xval)
            return self._reshape(state, memory, set(target_cells))


#------------------------------------------------------------------------------
#  Group 3 Ameoba
#------------------------------------------------------------------------------

BUCKET_WIDTH = 1
SHIFT_CYCLE  = 16 


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
            random_walk=RandomWalk(metabolism, rng),
            bucket_attack=BucketAttack(
                metabolism, bucket_width=BUCKET_WIDTH, shift_n=SHIFT_CYCLE
            )
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
        # known bacteria to the ameoba will be eaten before our next move
        # update ameoba_map and current_size to reflect this
        for i, j in current_percept.bacteria:
            current_percept.amoeba_map[i][j] = State.bacteria.value
            current_percept.current_size += 1
        self.current_size = current_percept.current_size

        # TODO: dynamically select a strategy, possible factors:
        # current_size, metabolism, etc
        strategy = "bucket_attack"

        return self.strategies[strategy].move(last_percept, current_percept, info)


#------------------------------------------------------------------------------
#  Unit Tests
#------------------------------------------------------------------------------

def Test_BucketAttack():
    # NOTE(from yunlan): We are using two-layer wall now, so this test case is
    # no longer accurate, and WILL FAIL. Too lazy to write a new test case, so
    # leaving this for now.
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
    movable_cells = find_movable_cells([], periphery, ameoba_map, [])

    curr_state = AmoebaState(9, ameoba_map, periphery, [], movable_cells)
    cog = bucket_attack._get_cog(curr_state)
    print(f"cog: {cog}")
    to_retract, to_occupy, _ = bucket_attack.move(curr_state, 0)

    print()
    print(f"retracted cells:      {to_retract}")
    print(f"newly occupied cells: {to_occupy}")


if __name__ == "__main__":
    Test_BucketAttack()