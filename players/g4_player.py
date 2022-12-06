from abc import abstractmethod, ABC
from enum import Enum
import logging
import math
import os
import shutil
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

debug_png_dir = "render/debug"
debug = 0
debug_since = 50
turns = 0

if debug:
    if os.path.exists(debug_png_dir):
        shutil.rmtree(debug_png_dir)

    os.mkdir(debug_png_dir)


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

    global turns
    turns += 1

    if not debug or turns < debug_since:
        return


    axes = []
    for fig_no in [2, 3]:
        plt.figure(fig_no)
        plt.cla()
        plt.title(f"turn {turns}")
        ax = plt.gca()
        axes.append(ax)

    ax1, ax2 = axes
    
    # marker sizes
    size_xs = mpl.rcParams['lines.markersize'] / 4
    size_s = (mpl.rcParams['lines.markersize'] / 4) ** 2

    # common: ameoba
    for ax in [ax1, ax2]:
        ax.plot(*list(zip(*ameoba)), 'g.', label='ameoba', markersize=size_xs)

    def scatter(ax: plt.Axes, pts: list[cell], **kwargs) -> None:
        if len(pts) == 0:
            return

        ax.scatter(*list(zip(*pts)), **kwargs)


    # subplot 1: ameoba & target
    ax1.plot(*list(zip(*target)), 'rs', label='target', markersize=size_s, markerfacecolor='none')

    # subplot 2: retract, extend, bacteria
    scatter(
        ax2, retract, label='retract',
        facecolors='none', edgecolors='forestgreen', marker='s', s=size_s
    )
    scatter(
        ax2, extend, label='extend',
        facecolors='none', edgecolors='tab:purple', s=size_s
    )
    scatter(
        ax2, bacteria, label='bacteria',
        facecolors='none', edgecolors='orange', marker='s', s=size_s
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

    plt.figure(2)
    plt.savefig(f"render/debug/fig2_{turns}", dpi=300)
    plt.figure(3)
    plt.savefig(f"render/debug/fig3_{turns}", dpi=300)

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

    # slow path: use binary search to find the first prefix that passes
    # check_move. With a maximum ameoba size of 10,000, the number of
    # invocations to check_move is capped at around 14 ~ log_2(10,000)
    lo, hi = 0, _k
    while lo <= hi:
        mid = math.floor((lo + hi) / 2)
        prefix = [ cell for cell, _ in sorted_choices[:mid] ]

        if check_move(prefix, possible_moves[:mid], state):
            return prefix
        else:
            hi = mid - 1

    return []

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
        self.vertical_comb_center = 50

    @abstractmethod
    def move(
        self, prev_state: AmoebaState, state: AmoebaState, memory: int
    ) -> tuple[list[cell], list[cell], int]:

        pass

    def _get_cog(self, curr_state: AmoebaState) -> tuple[int, int]:
        """Compute center of gravity of current Ameoba."""
        ameoba_cells = np.array(list(zip(*np.where(curr_state.amoeba_map == State.ameoba.value))))
        cog = (round(np.average(ameoba_cells[:,0])),round(np.average(ameoba_cells[:,1])))

        return cog
    
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
        #print(ameoba_cells)
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

class BoxFarm(Strategy):

    def _make_box(self, size, top_left_corner):
        #perimeter = math.floor(size/4)
        square_length = (size // 4) + 1
        extra = size % 4
        box_cells = []
        for i in range(square_length):
            for j in range(square_length):
                if i == 0 or i == square_length - 1 or j == 0 or j == square_length - 1:
                    box_cells.append((i + top_left_corner[0], j + top_left_corner[1]))
        for i in range(extra):
            box_cells.append((top_left_corner[0], top_left_corner[1] + square_length + i))
        return box_cells

    def _sweep(self, size, ameoba_cells):
        # print('sweep')
        left_x = min(ameoba_cells[:,0])
        # print(left_x)
        left_ind = np.where(ameoba_cells[:,0]==left_x)
        # print(left_ind)
        sweep_arm = ameoba_cells[left_ind]
        # print(sweep_arm)
        # print(ameoba_cells)
        # print()
        return sweep_arm

    def _init(self, ameoba_cells, size, corner):
        square_length = math.floor(math.sqrt(size)) + 1
        minx, miny = min(ameoba_cells[:,0]), min(ameoba_cells[:,1])
        if corner == 0:
            top_right_corner = (minx+square_length-2,miny)
           # print(top_right_corner)
        else:
            top_right_corner = (corner, miny)
        box_cells = self._make_box(size, top_right_corner)
        return box_cells, top_right_corner

    def move(
        self, prev_state: AmoebaState, state: AmoebaState, memory: int
    ) -> tuple[list[cell], list[cell], int]:
        
        size = (state.current_size)
        ameoba_cells = np.array(list(zip(*np.where(state.amoeba_map == State.ameoba.value))))

        ameoba_cells_set = list(set([tuple(ti) for ti in ameoba_cells]))
        loop = True
        while loop:
            loop = False
            mem = f'{memory:b}'
            while len(mem) < 8:
                mem = '0' + mem
            initialize = int(mem[0])
            corner = int(mem[1:],2)
            if initialize == 0:
                target_cells, corner_new = self._init(ameoba_cells,size,corner)
                #print(target_cells)
                mem_corner = f'{corner_new[0]:b}'
                while len(mem_corner) < 8:
                    mem_corner = '0' + mem_corner
                mem = mem_corner
                over = len(set(ameoba_cells_set))
                if set(target_cells) == set(ameoba_cells_set):
                    initialize = 1
                    mem_initialize = f'{initialize:b}'
                    mem = mem_initialize + mem[1:]

            if initialize == 1:
                print("SWEEP")
                sweep_cells = self._sweep(size, ameoba_cells)
                sweep_cells_set = list(set([tuple(ti) for ti in sweep_cells]))
                #target_cells = list(set(ameoba_cells_set) - set(sweep_cells_set))
                target_cells, corner_new2 = self._init(ameoba_cells,size,corner)
                # print(target_cells)

                x = sweep_cells[0][0]
                res = [list(ele) for ele in target_cells]
                res = np.array(res)
             
                ind = np.where(res[:,0]<=x)
                #past_cells = np.array(list(zip(*np.where(ameoba_cells[:,0]<=x))))
                #print(past_cells)
                # print(res)

                past_cells = res[ind]
                # print(past_cells)
                past_cells_set = list(set([tuple(ti) for ti in past_cells]))
                target_cells_set = list(set([tuple(ti) for ti in target_cells]))
                target_cells = list(set(target_cells_set) - set(past_cells_set))
                # print(target_cells)
                y = min(ameoba_cells[:,1])
                square_length = (size // 4) + 1
                leftover = (size % 4)
                ready = False
                for i in range(square_length-2):
                    point = ((x+1,y+1+i))
                    if point in target_cells:
                        ready = True
                    target_cells.append(point)
                    
                for i in range(leftover):
                    point = ((x+1,y+square_length+i))
                    target_cells.append(point)
                # print(leftover)
                # print(len(past_cells))
                # print(square_length-2)
                # print((len(past_cells) - (square_length - 2) - leftover)//2)
                for i in range((len(past_cells) - (square_length-2)-leftover)//2):
                    print(i)
                    target_cells.append((corner+square_length+i,y))
                    target_cells.append((corner+square_length+i,y+square_length-1))
                # print(target_cells)
                print(target_cells)
                print(ameoba_cells)
                if (set(target_cells)) == (set(ameoba_cells_set)) or ready:#or set(target_cells) < set(ameoba_cells_set):
                    print("TESTING TESTING")
                    min_y = min(ameoba_cells[:,1])
                    min_ind = np.where(ameoba_cells[:,1]==min_y)
                    top = ameoba_cells[min_ind]
                    corner = min(ameoba_cells[:,0]) + 1
                    initialize = 0
                    mem_corner = f'{corner:b}'
                    while len(mem_corner) < 8:
                        mem_corner = '0' + mem_corner
                    mem = mem_corner
                    memory = int(mem,2)
                    loop = True
            
            # for cell in sweep_cells:
            #     if cell[1] == miny or cell[1] == maxy:
            #         target_cells.append((cell[0],cell[1]))
            #     else:
            #         target_cells.append((cell[0]+1,cell[1]))

        #target_cells = self._get_target_cells(size, ameoba_cells)

        memory = int(mem,2)
        #print(target_cells)
        return self._reshape(state, memory, set(target_cells))


class BucketAttack(Strategy):

    def __init__(self, metabolism, bucket_width=1, shift_n=-1, v_size=200):
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
        self.v_size = v_size

        # derived statistics
        self.wall_cost = self.bucket_width + 1
        self.bucket_cost = 2 * self.wall_cost + 1 # 1 is cost of bucket arm
    
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

    def _spread_horizontally(
            self,
            x_center: int,
            left_cnt: int,
            right_cnt: int,
            step=1
    ) -> np.ndarray:
        """Returns a list of x-values by spreading around x_center."""

        cnt = right_cnt + 1 + left_cnt
        x_right = x_center + step * right_cnt
        x_left = x_center - step * left_cnt
        x_spreads = np.linspace(x_left, x_right, cnt, True, dtype=int)
        return x_spreads

    def _spread_diagonally(self,size: int, cog: cell, up_or_down: int)-> list[cell]:
        # TODO
        x_cog,y_cog = cog
        cur = 0
        cur_x = x_cog
        cur_y = y_cog
        targets = []
        while cur < size:
            cur_x -= 1
            targets.append(((cur_x) % 100, cur_y % 100))
            cur += 1

            if cur < size:
                cur_y += up_or_down
                targets.append(((cur_x) % 100, cur_y % 100))
                cur += 1
            if cur_x%100 == 0:
                print("reach edge",len(targets))
        print(targets)
        return targets

    def _get_target_cells(self, size: int, cog: cell, xmax: int) -> list[cell]:
        """Returns the cells of the target shape by centering vertically on
        the y-value of Ameoba's center of gravity and placing the bucket arms
        on the column of Ameoba's rightmost cell.

        @size: current size of ameoba
        @cog: center of gravity for the target shape (only y-value used currently)
        """
        wall_cost = self.wall_cost
        bucket_cost = self.bucket_cost

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

    def _get_horizontal_comb_target_cells(self, size: int, cog: cell, ymax: int) -> list[cell]:
        """Returns the cells of the target shape by centering vertically on
        the y-value of Ameoba's center of gravity and placing the bucket arms
        on the column of Ameoba's rightmost cell.

        @size: current size of ameoba
        @cog: center of gravity for the target shape (only y-value used currently)
        """

        x_cog, _ = cog
        buckets, orphans = divmod(size - 3, 7)
        left_buckets, right_buckets = math.ceil(buckets / 2), math.floor(buckets / 2)

        inner_wall_cell_xs = self._spread_horizontally(
            x_cog,
            3 * left_buckets,
            3 * right_buckets
        )
        outer_wall_cell_xs = self._spread_horizontally(
            x_cog,
            3 * left_buckets,
            3 * right_buckets+ orphans
        )
        arm_cell_xs = self._spread_horizontally(
            x_cog, left_buckets, right_buckets , step=3
        )

        if self.shifted == 1:
            min_x = min(arm_cell_xs)
            max_x = max(arm_cell_xs)
            arm_cell_xs = [x - 1 for x in arm_cell_xs if x != min_x and x != max_x] + [max_x, min_x]
            print("SHIFTED")

        wall_cells = (
            [ ( x % 100,       ymax % 100 ) for x in inner_wall_cell_xs ] +
            [ ( x % 100,  (ymax-1) % 100 ) for x in outer_wall_cell_xs ]
        )
        arm_cells = [ ( x % 100, (ymax+1) % 100 ) for x in arm_cell_xs ]

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

    def _get_bridge_V_target_cells(self, size: int, cog: cell, xmax: int) -> list[cell]:
        _, y_cog = cog
        comb_size=min(290,size)
        bridge_size=min(200,size-comb_size)
        extra_cells_size=size-comb_size-bridge_size

        comb_targets=self._get_target_cells(comb_size,cog,xmax)

        if bridge_size>0:
            print("have bridge",size, comb_size,bridge_size)
            bridge_length,orphan=divmod(bridge_size,2)
            if xmax>50:
                if xmax-bridge_length > 50:
                    upper_bridge_cells=[(cur_x,y_cog) for cur_x in range(xmax-bridge_length,xmax)]
                    lower_bridge_cells=[(cur_x,y_cog-1) for cur_x in range(xmax-bridge_length-orphan,xmax)]
                    bridge_targets= upper_bridge_cells+lower_bridge_cells
                    return comb_targets + bridge_targets
                else:
                    bridge_length = xmax - 50
                    upper_bridge_cells = [(cur_x, y_cog) for cur_x in range(xmax - bridge_length, xmax)]
                    lower_bridge_cells = [(cur_x, y_cog - 1) for cur_x in range(xmax - bridge_length, xmax)]
                    bridge_targets = upper_bridge_cells + lower_bridge_cells
                    bridge_size = len(bridge_targets)
                    v_size = min(self.v_size, size - comb_size - bridge_size)
                    v_targets = self._get_vshape_target(v_size, (50, 50))
                    #print("Grow V", size, comb_size, bridge_size, v_size)
                    horizontal_comb_size = size - comb_size - bridge_size - v_size
                    if horizontal_comb_size > 0:
                         comb_targets = self._get_target_cells(comb_size + horizontal_comb_size, cog, xmax)
                    return comb_targets + bridge_targets + v_targets

            else:
                    left_upper_bridge_cells = [(cur_x, y_cog) for cur_x in range(0, xmax)]
                    left_lower_bridge_cells = [(cur_x, y_cog - 1) for cur_x in range(0, xmax+orphan)]

                    right_upper_bridge_cells = [(cur_x, y_cog) for cur_x in range(50, 100)]
                    right_lower_bridge_cells = [(cur_x, y_cog - 1) for cur_x in range(50,100)]
                    bridge_targets = left_upper_bridge_cells + left_lower_bridge_cells+right_upper_bridge_cells+right_lower_bridge_cells
                    #print("wrapped around bridge target:",bridge_targets)
                    #print("wrapped around comb target:",comb_targets)
                    bridge_size = len(bridge_targets)
                    v_size = min(self.v_size, size - comb_size - bridge_size)
                    v_targets = self._get_vshape_target(v_size, (50, 50))
                    horizontal_comb_size = size - comb_size - bridge_size - v_size
                    if horizontal_comb_size > 0:
                        comb_targets = self._get_target_cells(comb_size + horizontal_comb_size, cog, xmax)
                    return comb_targets + bridge_targets + v_targets


        return comb_targets

    def _get_vshape_target(self, size: int, cog: cell) -> list[cell]:
        arm_length =size//2
        #print("upper")
        upper_arm= self._spread_diagonally(arm_length, cog, 1)
        #print(f"len(upper bridge cells): {len(set(upper_arm))}")
        #print("lower")
        lower_arm= self._spread_diagonally(arm_length, cog, -1)
        #print(f"len(upper bridge cells): {len(set(lower_arm))}")
        arm_cells = upper_arm+lower_arm
        return arm_cells

    def _get_bridge_target(self, size: int, cog: cell) -> list[cell]:
        bridge_cells=[]
        x_cog,y_cog= cog
        cur_x=x_cog
        for i in range(size):
            bridge_cells.append((cur_x, y_cog))
            cur_x-=1
        return bridge_cells

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
                      (xmax
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

    def _reach_border(self, curr_state: AmoebaState) -> bool:
        _, ameoba_ys = np.where(curr_state.amoeba_map == State.ameoba.value)
        lower_bound, upper_bound = min(ameoba_ys), max(ameoba_ys)

        return abs(upper_bound - lower_bound) >= 98

    def _in_shape(self, xmax: int, curr_state: AmoebaState) -> bool:
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
        arms_got = len(np.where(curr_state.amoeba_map[xmax,:] == State.ameoba.value)[0])

        if not self._reach_border:
            arms_expected = 1 + math.floor((curr_state.current_size - 3) / self.bucket_cost)
        else:
            arms_expected = 2 * ((constants.map_dim / 2 - 1) // (self.bucket_width + 1) + 1)

        return arms_got >= arms_expected

    def move(
        self, prev_state: AmoebaState, state: AmoebaState, memory: int
    ) -> tuple[list[cell], list[cell], int]:

        # ----------------
        #  Decode Memory
        # ----------------
        mem = f'{memory:b}'.rjust(8, '0')
        old_xmax = int(mem[:7], 2)
        shifted = int(mem[-1])

        # ---------------
        #  State Update
        # ---------------
        # rotation counter & shifted
        rotation = (old_xmax + 1) % self.shift_n
        if self.shift_enabled and rotation == 0:
            shifted = shifted ^ 1

        # arm_xval
        if not self._reach_border(state):
            xmax = self._get_xmax(state)
        else:
            xmax = old_xmax + 1

        if not self._in_shape(xmax % 100, state):
            arm_xval = (xmax - 1) % 100
        else:
            arm_xval = xmax % 100

        # ----------------
        #  Update Memory
        # ----------------
        arm_xval_bin = '{0:07b}'.format(arm_xval)
        shifted_bin  = '{0:01b}'.format(shifted)
        memory = int(arm_xval_bin + shifted_bin, 2)


        size = state.current_size
        # TODO: maybe not always moving horizontally?
        if shifted:
            cog = (50, 49)
        else:
            cog  = (50, 50)

        # -----------------------
        #  compute target shape
        # -----------------------
        # TODO: if reach boder, return comb+1cell each step as the bridge + Vshape
        # need about 290 cells to reach border
        if self._reach_border(state):
            target_cells = self._get_bridge_V_target_cells(size, cog, arm_xval)
        else:
            target_cells = self._get_target_cells(size, cog, arm_xval)

        # ----------------------------------
        #  compute moves: retract & extend
        # ----------------------------------
        # TODO: need to modify normal_retract to fit the V_shape
        retract, extend, memory = self._reshape(state, memory, set(target_cells))

        if len(retract) > 0:
            return retract, extend, memory
        else:
            target_cells = self._get_rectangle_target(size, cog, arm_xval)
            return self._reshape(state, memory, set(target_cells))


#------------------------------------------------------------------------------
#  Group 4 Ameoba
#------------------------------------------------------------------------------

BUCKET_WIDTH = 1
SHIFT_CYCLE  = -1
V_SIZE = 400


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
            box_farm=BoxFarm(metabolism),
            bucket_attack=BucketAttack(
                metabolism,
                bucket_width=BUCKET_WIDTH,
                shift_n=SHIFT_CYCLE,
                v_size=V_SIZE
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
        if self.goal_size <= 64:
            strategy="box_farm"
        else:
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