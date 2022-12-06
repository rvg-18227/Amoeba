import os
import pickle
import random

import numpy as np
import numpy.typing as npt
from typing import Tuple, List
import logging
from amoeba_state import AmoebaState
import math
import time
import matplotlib.pyplot as plt
from enum import Enum
import sys
import random as rnd


# CONSTS #

MAP_DIM = 100
MAX_BASE_LEN = min(MAP_DIM, 100)
TOOTH_SPACING = 1       # 1 best
SHIFTING_FREQ = 6       # 6 best for high metabolisms, 4 better for low
SIZE_MULTIPLIER = 4     # 4 best for density = 0.1 metabolism = 0.1
MOVING_TYPE = 'center'  # 'center' best for low metabolisms - 'center_teeth_first' better for high
TWO_RAKE = True

# Best configs so far #
# m = 0.1; A = 5; s = 2 -> 1 6 4 'center' -> 202 moves
# m = 1.0; A = 5; s = 2 -> 1 6 6 'center_teeth_first' -> 74 moves


# ********* HELPER FUNCTIONS ********* #


# the following 3 methods are borrowed from G2
def map_to_coords(amoeba_map: npt.NDArray) -> list[Tuple[int, int]]:
    return list(map(tuple, np.transpose(amoeba_map.nonzero()).tolist()))


def coords_to_map(coords: list[tuple[int, int]], size=MAP_DIM) -> npt.NDArray:
    amoeba_map = np.zeros((size, size), dtype=np.int8)
    for x, y in coords:
        amoeba_map[x, y] = 1
    return amoeba_map


def show_amoeba_map(amoeba_map: npt.NDArray, retracts=[], extends=[]) -> None:
    retracts_map = coords_to_map(retracts)
    extends_map = coords_to_map(extends)

    map = np.zeros((MAP_DIM, MAP_DIM), dtype=np.int8)
    for x in range(MAP_DIM):
        for y in range(MAP_DIM):
            # transpose map for visualization as we add cells
            if retracts_map[x, y] == 1:
                map[y, x] = -1
            elif extends_map[x, y] == 1:
                map[y, x] = 2
            elif amoeba_map[x, y] == 1:
                map[y, x] = 1

    plt.rcParams["figure.figsize"] = (10, 10)
    plt.pcolormesh(map, edgecolors='k', linewidth=1)
    ax = plt.gca()
    ax.set_aspect('equal')
    # plt.savefig(f"debug/{turn}.png")
    plt.show()


# the following 2 methods are borrowed from myself from project 3
def tree_index(factors, max_factors):
    index = 0
    for i in range(len(factors)):
        n_children = 1
        for factor in max_factors[i + 1:]:
            n_children *= factor
        index += factors[i] * n_children
    return index


def tree_factors(index, max_factors):
    factors = []
    for i in range(len(max_factors)):
        n_children = 1
        for factor in max_factors[i + 1:]:
            n_children *= factor
        factor = index // n_children
        factors.append(factor)
        index %= n_children

    return factors


# search the list for the element that causes check(li) to fail and remove the element
def binary_search(li, check):
    mid = len(li) // 2

    if not check(li[:mid]):
        return binary_search(li[:mid], check) + li[mid:]
    elif not check(li[mid + 1:]):
        return li[:mid + 1] + binary_search(li[mid + 1:], check)
    elif not check(li):
        return li[:mid] + li[mid + 1:]
    else:
        return li


def binary_search_item(li, check):
    mid = len(li) // 2

    if not check(li[:mid]):
        return binary_search_item(li[:mid], check)
    elif not check(li[mid + 1:]):
        return binary_search_item(li[mid + 1:], check)
    elif not check(li):
        return li[mid]
    else:
        return None


def iter_from_middle(lst):
    try:
        middle = len(lst) // 2
        yield lst[middle]

        for shift in range(1, middle + 1):
            # order is important!
            yield lst[middle - shift]
            yield lst[middle + shift]

    except IndexError:  # occures on lst[len(lst)] or for empty list
        return


# ********* BYTE INFO ******** #

class MaxVals(Enum):
    #is_rake = 2
    x_val = 100
    tooth_shift = 2


class Memory:
    def __init__(self, byte=None, vals=None):
        if byte is not None:
            vals = get_byte_info(byte)
            self.x_val = vals[0]
            self.tooth_shift = vals[1]
        elif vals is not None:
            self.x_val = vals[0]
            self.tooth_shift = vals[1]
        else:
            self.x_val = 50
            self.tooth_shift = 1

    def get_byte(self):
        return set_byte_info([self.x_val, self.tooth_shift])

    def get_vals(self):
        return [self.x_val, self.tooth_shift]


def get_byte_info(byte: int):
    max_vals = [field.value for field in MaxVals]
    factors = tree_factors(byte, max_vals)

    return factors


def set_byte_info(values):
    max_vals = [field.value for field in MaxVals]
    byte = tree_index(values, max_vals)

    return byte


# ********* MAIN CODE ********* #

class Player:
    def __init__(self, rng: np.random.Generator, logger: logging.Logger, metabolism: float, goal_size: int,
                 precomp_dir: str) -> None:
        """Initialise the player with the basic amoeba information

            Args:
                rng (np.random.Generator): numpy random number generator, use this for same player behavior across run
                logger (logging.Logger): logger use this like logger.info("message")
                metabolism (float): the percentage of amoeba cells, that can move
                goal_size (int): the size the amoeba must reach
                precomp_dir (str): Directory path to store/load pre-computation
        """

        # precomp_path = os.path.join(precomp_dir, "{}.pkl".format(map_path))

        # # precompute check
        # if os.path.isfile(precomp_path):
        #     # Getting back the objects:
        #     with open(precomp_path, "rb") as f:
        #         self.obj0, self.obj1, self.obj2 = pickle.load(f)
        # else:
        #     # Compute objects to store
        #     self.obj0, self.obj1, self.obj2 = _

        #     # Dump the objects
        #     with open(precomp_path, 'wb') as f:
        #         pickle.dump([self.obj0, self.obj1, self.obj2], f)

        self.rng = rng
        self.logger = logger
        self.metabolism = metabolism
        self.goal_size = goal_size
        self.current_size = goal_size / 4

        self.current_size: int = None
        self.amoeba_map: npt.NDArray = None
        self.bacteria_cells: List[Tuple[int, int]] = None
        self.retractable_cells: List[Tuple[int, int]] = None
        self.extendable_cells: List[Tuple[int, int]] = None
        self.num_available_moves: int = None
        self.map_state: npt.NDArray = None

    def generate_tooth_formation(self, amoeba_size: int) -> npt.NDArray:
        formation = np.zeros((MAP_DIM, MAP_DIM), dtype=np.int8)
        center_x = MAP_DIM // 2
        center_y = MAP_DIM // 2
        spacing = TOOTH_SPACING + 1

        # find the number of complete 5-cell modules
        complete_modules = amoeba_size // 5
        # find the number of unfinished sections
        additional_sections = amoeba_size % 5 // 2
        # find whether there is an odd cell
        reserve = amoeba_size % 5 % 2

        base_len = min(complete_modules * 2 + additional_sections, MAX_BASE_LEN)
        # teeth_len = min(complete_modules, base_len//2)
        # teeth_len = min(complete_modules, base_len // spacing)
        teeth_len = base_len // spacing
        # reserve = max(reserve, amoeba_size-(base_len*2 + teeth_len))
        start_y = center_y - base_len // 2

        # add the 2-cell-wide base and teeth
        for y in range(start_y, start_y + base_len):
            formation[center_x, y%100] = 1
            formation[center_x - 1, y%100] = 1
            if y%100 % spacing == 0:
                formation[center_x + 1, y%100] = 1

        # add the teeth
        # start_modules = start_y  # +(additional_sections+1)//2
        # for y in range(start_modules, start_y + teeth_len * spacing):
        #     # if (y - start_modules) % spacing == 0:
        #     if y%100 % spacing == 0:
        #         formation[center_x + 1, y%100] = 1

        # # add the "reserve" cell at the back
        # for i in range(reserve):
        #     row = i//base_len
        #     col = i % base_len
        #     formation[center_x-2-row, start_y+col] = 1

        # show_amoeba_map(formation)
        return formation

    def generate_tworake_formation(self, amoeba_size: int, curr_x: int, shift: int) -> npt.NDArray:
        formation = np.zeros((MAP_DIM, MAP_DIM), dtype=np.int8)
        center_x = MAP_DIM // 2
        center_y = MAP_DIM // 2
        spacing = TOOTH_SPACING + 1
        # if amoeba_size > 300 and (abs(curr_x - 50) < 3 or abs(curr_x - 100) < 3):

        # PREVENT RAKES FROM COLLIDING
        d50 = lambda x: abs(x - (50 * round(x / 50)))
        # print(amoeba_size)
        if amoeba_size > 350 and d50(curr_x) < 3:
            curr_x = ((50 * round(curr_x / 50)) + 3) % 100
            teeth = False
        else:
            teeth = True

        # ADD FIRST RAKE
        # print('gtf x: ', curr_x)
        # find the number of complete 5-cell modules
        complete_modules = amoeba_size // 5
        # find the number of unfinished sections
        additional_sections = amoeba_size % 5 // 2
        # find whether there is an odd cell
        reserve = amoeba_size % 5 % 2

        base_len = min(complete_modules * 2 + additional_sections, MAX_BASE_LEN)
        # teeth_len = min(complete_modules, base_len//2)
        # teeth_len = min(complete_modules, base_len // spacing)
        teeth_len = base_len // spacing
        # reserve = max(reserve, amoeba_size-(base_len*2 + teeth_len))
        start_y = center_y - base_len // 2
        # start_x = (center_x + offset_x)%100
        start_x = curr_x

        # add the 2-cell-wide base and teeth
        cells_used = 0
        for y in range(start_y, start_y + base_len):
            formation[start_x, y % 100] = 1
            formation[(start_x - 1) % 100, y % 100] = 1
            cells_used += 2
            if y % 100 % spacing == shift:
                formation[(start_x + 1) % 100, y % 100] = 1
                cells_used += 1

        # ADD THE MIDDLE BAR
        cells_used = (formation == 1).sum()
        available = amoeba_size - cells_used

        # bar_length = min(100, available)
        if curr_x < 50:
            bar_length = min(100, available)
        else:
            bar_length = min((curr_x - 50) * 2, available)

        for offset in range(1, bar_length + 1):
            formation[(curr_x - offset) % 100, center_y] = 1

        # ADD THE SECOND RAKE
        cells_used = (formation == 1).sum()
        available = amoeba_size - cells_used

        complete_modules = available // 5
        # find the number of unfinished sections
        additional_sections = available % 5 // 2
        # find whether there is an odd cell
        reserve = available % 5 % 2

        base_len = min(complete_modules * 2 + additional_sections, MAX_BASE_LEN)
        # teeth_len = min(complete_modules, base_len//2)
        # teeth_len = min(complete_modules, base_len // spacing)
        teeth_len = base_len // spacing
        # reserve = max(reserve, amoeba_size-(base_len*2 + teeth_len))
        start_y = center_y - base_len // 2
        # start_x = (center_x + offset_x)%100

        start_x = (99 - curr_x) % 100
        for y in range(start_y, start_y + base_len):
            formation[start_x, y % 100] = 1
            formation[(start_x + 1) % 100, y % 100] = 1
            if y % 100 % spacing == shift ^ 1:
                formation[(start_x - 1) % 100, y % 100] = 1

        # DO SOMETHING WITH EXCESS CELLS
        cells_used = (formation == 1).sum()
        available = amoeba_size - cells_used
        # cube_side = math.ceil((available ** 0.5))

        # for y in iter_from_middle(list(range(0, 50)) + list(range(51, 100))):
        #     for x in iter_from_middle(range(0, 100)):
        #     # for x in range(center_x-cube_side//2, center_x-cube_side//2+cube_side):
        #         if available <= 0:
        #             break
        #         formation[x, y] = 1
        #         available -= 1
        #     if available <= 0:
        #         break

        # for x in iter_from_middle(range(0, 100)):
        #     for y in iter_from_middle(list(range(0, 50)) + list(range(51, 100))):
        #     # for x in range(center_x-cube_side//2, center_x-cube_side//2+cube_side):
        #         if available <= 0:
        #             break
        #         formation[x, y] = 1
        #         available -= 1
        #     if available <= 0:
        #         break


        # show_amoeba_map(formation)
        return formation

    # sort potential retracts based on the number of neighbors (less members -> higher priority)
    def sort_retracts(self, potential_retracts):
        ranked_cells = []
        for x, y in potential_retracts:
            neighbors = [((x-1) % 100, y), ((x+1) % 100, y), (x, (y-1) % 100), (x, (y+1) % 100)]
            score = 0
            for neighbor in neighbors:
                if self.amoeba_map[neighbor] == 1:
                    score += 1
            ranked_cells.append(((x, y), score))
        return [cell for cell, score in sorted(ranked_cells, key=lambda t: t[1])]

    def get_retracts_neighbors(self, potential_retracts):
        ranked_cells_dict = {}
        for x, y in potential_retracts:
            neighbors = [((x-1) % 100, y), ((x+1) % 100, y), (x, (y-1) % 100), (x, (y+1) % 100)]
            score = 0
            for neighbor in neighbors:
                if self.amoeba_map[neighbor] == 1:
                    score += 1
            ranked_cells_dict[(x, y)] = score
        return ranked_cells_dict

    def sort_retracts(self, retracts, ranked_cells_dict):
        retracts = sorted(retracts, key=lambda r: (ranked_cells_dict[r], -abs(r[0]-50)), reverse=True)
        return retracts

    def get_valid_neighbors(self, cell):
        x, y = cell
        neighbors = [((x - 1) % 100, y), ((x + 1) % 100, y), (x, (y - 1) % 100), (x, (y + 1) % 100)]
        valid_neighbors = []
        for neighbor in neighbors:
            if self.amoeba_map[neighbor] == 1:
                valid_neighbors.append(neighbor)
        return valid_neighbors

    def get_neighbors(self, cell):
        x, y = cell
        neighbors = [((x - 1) % 100, y), ((x + 1) % 100, y), (x, (y - 1) % 100), (x, (y + 1) % 100)]
        return neighbors

    # copied from G2
    def get_morph_moves(self, desired_amoeba: npt.NDArray) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """ Function which takes a starting amoeba state and a desired amoeba state and generates a set of retracts and extends
            to morph the amoeba shape towards the desired shape.
        """

        current_points = map_to_coords(self.amoeba_map)
        desired_points = map_to_coords(desired_amoeba)

        potential_retracts = [p for p in list(set(current_points).difference(set(desired_points))) if
                              (p in self.retractable_cells) and not any([neighbor in self.bacteria_cells for neighbor in self.get_neighbors(p)])]
        potential_extends = [p for p in list(set(desired_points).difference(set(current_points))) if
                             p in self.extendable_cells]

        # potential_retracts.sort(key=lambda pos: pos[1])
        if MOVING_TYPE == 'top_down':
            potential_extends.sort(key=lambda pos: pos[1])
        elif MOVING_TYPE == 'top_down_teeth_first':
            potential_extends.sort(key=lambda pos: (-pos[0], pos[1]))
        elif MOVING_TYPE == 'center':
            potential_extends.sort(key=lambda pos: abs(50 - pos[1]))
        elif MOVING_TYPE == 'center_teeth_first':
            potential_extends.sort(key=lambda pos: (-pos[0], abs(50-pos[1])))

        # Loop through potential extends, searching for a matching retract
        retracts = []
        extends = []
        change = True   # tracks whether anything was moved to prevent endless loop
        # potential_extends = potential_extends[::-1]
        # while potential_extends and potential_retracts and change and len(retracts) < self.num_available_moves:
        #     change = False
        #     # for potential_extend in potential_extends:
        #     #     for potential_retract in potential_retracts:
        #     for potential_retract in potential_retracts:
        #         # for potential_extend in potential_extends:
        #         if self.check_move(retracts + [potential_retract], extends + [potential_extends[-1]]):
        #             # matching retract found, add the extend and retract to our lists
        #             retracts.append(potential_retract)
        #             potential_retracts.remove(potential_retract)
        #             # extends.append(potential_extend)
        #             extends.append(potential_extends.pop())
        #             # potential_extends.remove(potential_extend)
        #             change = True
        #             break
        ranked_cells_dict = self.get_retracts_neighbors(potential_retracts)
        # potential_retracts.sort(key=lambda r: (ranked_cells_dict[r], -r[0]), reverse=True)
        # potential_retracts = self.sort_retracts(potential_retracts)
        potential_retracts = self.sort_retracts(potential_retracts, ranked_cells_dict)

        possible_moves = min(len(potential_retracts), len(potential_extends), self.num_available_moves)
        potential_extends.reverse()

        for _ in range(possible_moves):
            next_ret = potential_retracts.pop()
            retracts.append(next_ret)
            extends.append(potential_extends.pop())

            for neighbor in self.get_neighbors(next_ret):
                if neighbor in ranked_cells_dict:
                    ranked_cells_dict[neighbor] -= 1
            # potential_retracts.sort(key=lambda r: ranked_cells_dict[r], reverse=True)
            # potential_retracts.sort(key=lambda r: (ranked_cells_dict[r], -r[0]), reverse=True)
            potential_retracts = self.sort_retracts(potential_retracts, ranked_cells_dict)

        # retracts = potential_retracts[:possible_moves]
        # extends = potential_extends[:possible_moves]

        while not self.check_move(retracts, extends):
            # retracts = binary_search(retracts, lambda r: self.check_move(r, extends[:len(r)]))
            bad_retract = binary_search_item(retracts, lambda r: self.check_move(r, extends[:len(r)]))

            for neighbor in self.get_neighbors(bad_retract):
                if neighbor in ranked_cells_dict:
                    ranked_cells_dict[neighbor] += 1

            retracts.remove(bad_retract)
            extends.pop()

            # potential_retracts.sort(key=lambda r: ranked_cells_dict[r], reverse=True)
            # potential_retracts.sort(key=lambda r: (ranked_cells_dict[r], -r[0]), reverse=True)
            potential_retracts = self.sort_retracts(potential_retracts, ranked_cells_dict)

            if potential_retracts and potential_extends:
                retracts.append(potential_retracts.pop())
                extends.append(potential_extends.pop())

        # show_amoeba_map(self.amoeba_map, retracts, extends)
        # truncate to account for smaller metabolism
        # print(self.check_move(retracts, extends))
        return retracts, extends

    # adapted from amoeba game code
    def check_move(self, retracts: List[Tuple[int, int]], extends: List[Tuple[int, int]]) -> bool:
        if not set(retracts).issubset(set(self.retractable_cells)):
            return False
        movable = retracts[:]
        new_periphery = list(set(self.retractable_cells).difference(set(retracts)))
        for i, j in new_periphery:
            nbr = self.find_movable_neighbor(i, j)
            for x, y in nbr:
                if (x, y) not in movable:
                    movable.append((x, y))

        if not set(extends).issubset(set(movable)):
            return False

        amoeba = np.copy(self.map_state)
        amoeba[amoeba < 0] = 0
        amoeba[amoeba > 0] = 1

        for i, j in retracts:
            amoeba[i][j] = 0

        for i, j in extends:
            amoeba[i][j] = 1

        tmp = np.where(amoeba == 1)
        result = list(zip(tmp[0], tmp[1]))
        check = np.zeros((MAP_DIM, MAP_DIM), dtype=int)

        stack = result[0:1]
        while len(stack):
            a, b = stack.pop()
            check[a][b] = 1

            if (a, (b - 1) % MAP_DIM) in result and check[a][(b - 1) % MAP_DIM] == 0:
                stack.append((a, (b - 1) % MAP_DIM))
            if (a, (b + 1) % MAP_DIM) in result and check[a][(b + 1) % MAP_DIM] == 0:
                stack.append((a, (b + 1) % MAP_DIM))
            if ((a - 1) % MAP_DIM, b) in result and check[(a - 1) % MAP_DIM][b] == 0:
                stack.append(((a - 1) % MAP_DIM, b))
            if ((a + 1) % MAP_DIM, b) in result and check[(a + 1) % MAP_DIM][b] == 0:
                stack.append(((a + 1) % MAP_DIM, b))

        return (amoeba == check).all()

    # copied from G2
    def store_current_percept(self, current_percept: AmoebaState) -> None:
        self.current_size = current_percept.current_size
        self.amoeba_map = current_percept.amoeba_map
        self.retractable_cells = current_percept.periphery
        self.bacteria_cells = current_percept.bacteria
        self.extendable_cells = current_percept.movable_cells
        self.num_available_moves = int(np.ceil(self.metabolism * current_percept.current_size))
        self.map_state = np.copy(self.amoeba_map)
        for bacteria in self.bacteria_cells:
            self.map_state[bacteria] = 1

    def move(self, last_percept: AmoebaState, current_percept: AmoebaState, info: int) -> Tuple[
        List[Tuple[int, int]], List[Tuple[int, int]], int]:
        """Function which retrieves the current state of the amoeba map and returns an amoeba movement
            Args:
                last_percept (AmoebaState): contains state information after the previous move
                current_percept(AmoebaState): contains current state information
                info (int): byte (ranging from 0 to 256) to convey information from previous turn
            Returns:
                Tuple[List[Tuple[int, int]], List[Tuple[int, int]], int]: This function returns three variables:
                    1. A list of cells on the periphery that the amoeba retracts
                    2. A list of positions the retracted cells have moved to
                    3. A byte of information (values range from 0 to 255) that the amoeba can use
        """
        self.store_current_percept(current_percept)

        retracts = []
        moves = []
        mem = Memory(byte=info)
        
        if self.is_square(current_percept):
            mem.x_val = 50
            mem.tooth_shift = 1

        while len(retracts) == 0 and len(moves) == 0:
            if mem.tooth_shift == 0:
                offset_x = mem.x_val - MAP_DIM//2 - 1
                x_val = mem.x_val - 1
                # offset_x = mem.x_val - MAP_DIM // 2
            else:
                x_val = mem.x_val
                offset_x = mem.x_val - MAP_DIM//2

            offset_y = 0 if (mem.x_val % (SHIFTING_FREQ*2)) < SHIFTING_FREQ else -1

            if not TWO_RAKE:
                if SIZE_MULTIPLIER == 0 or self.current_size > 250:
                    target_formation = self.generate_tooth_formation(self.current_size)
                # elif SIZE_MULTIPLIER < 0:
                #     target_size = ((self.goal_size // 4 + (-SIZE_MULTIPLIER) * ((100 + offset_x) % 100)) + self.current_size) // 2
                #     target_formation = self.generate_tooth_formation(target_size)
                # elif self.current_size >= 250:
                #     target_formation = self.generate_tworake_formation(self.current_size, mem.x_val)
                else:
                    target_size = self.goal_size // 4 + SIZE_MULTIPLIER * ((100 + offset_x) % 100)  # calculate the desired size with regard
                    if target_size//self.current_size < 0.8:
                        target_size = self.current_size
                    target_formation = self.generate_tooth_formation(target_size)

                # target_formation = self.generate_tooth_formation(last_percept.current_size)
                target_formation = np.roll(target_formation, offset_x, 0)
                target_formation = np.roll(target_formation, offset_y, 1)
            else:
                target_size = self.goal_size // 4 + SIZE_MULTIPLIER * (
                            (100 + offset_x) % 100)  # calculate the desired size with regard
                # if target_size // self.current_size < 0.8:
                #     target_size = self.current_size

                target_formation = self.generate_tworake_formation(self.current_size, x_val, offset_y + 1)
                # target_formation = self.generate_tworake_formation(target_size, x_val, offset_y + 1)

            diff = np.count_nonzero(target_formation & self.amoeba_map != target_formation)
            if diff/self.current_size <= 0.15 and diff <= 30:
                retracts, moves = [], []
            else:
                retracts, moves = self.get_morph_moves(target_formation)
                # if len(retracts)/self.num_available_moves < 0.2:
                #     retracts, moves = [], []

            if len(retracts) == 0 and len(moves) == 0:
            # if random.random() < 0.5 or (len(retracts) == 0 and len(moves) == 0):
                if mem.tooth_shift == 1:
                    mem.x_val = (mem.x_val + 1) % 100
                    if mem.x_val % SHIFTING_FREQ == 0:
                        mem.tooth_shift = 0
                else:
                    mem.tooth_shift = 1
                    # mem.x_val = (mem.x_val + 1) % 100

        info = mem.get_byte()
        return retracts, moves, info

    def shift_col(self, arr, col, shift):
        arr2 = np.copy(arr)
        arr2[:, col] = np.roll(arr2[:,col], shift)
        return arr2

    def bounds(self, current_percept):
        min_x, max_x, min_y, max_y = 100, -1, 100, -1
        for y, x in current_percept.periphery:
            if y < min_y:
                min_y = y
            elif y > max_y:
                max_y = y
            if x < min_x:
                min_x = x
            elif x > max_x:
                max_x = x

        return min_x, max_x, min_y, max_y

    def is_square(self, current_percept):
        min_x, max_x, min_y, max_y = self.bounds(current_percept)
        len_x = max_x - min_x + 1
        len_y = max_y - min_y + 1
        if len_x == len_y and len_x * len_y == current_percept.current_size:
            return True
        return False

    def find_movable_cells(self, retract, periphery, amoeba_map, bacteria, mini):
        movable = []
        new_periphery = list(set(periphery).difference(set(retract)))
        for i, j in new_periphery:
            nbr = self.find_movable_neighbor(i, j, amoeba_map, bacteria)
            for x, y in nbr:
                if (x, y) not in movable:
                    movable.append((x, y))

        movable += retract

        return movable[:mini]

    # def find_movable_neighbor(self, x, y, amoeba_map, bacteria):
    #     out = []
    #     if (x, y) not in bacteria:
    #         if amoeba_map[x][(y - 1) % 100] == 0:
    #             out.append((x, (y - 1) % 100))
    #         if amoeba_map[x][(y + 1) % 100] == 0:
    #             out.append((x, (y + 1) % 100))
    #         if amoeba_map[(x - 1) % 100][y] == 0:
    #             out.append(((x - 1) % 100, y))
    #         if amoeba_map[(x + 1) % 100][y] == 0:
    #             out.append(((x + 1) % 100, y))
    #
    #     return out

    def find_movable_neighbor(self, x, y):
        out = []
        if self.map_state[x][(y - 1) % MAP_DIM] < 1:
            out.append((x, (y - 1) % MAP_DIM))
        if self.map_state[x][(y + 1) % MAP_DIM] < 1:
            out.append((x, (y + 1) % MAP_DIM))
        if self.map_state[(x - 1) % MAP_DIM][y] < 1:
            out.append(((x - 1) % MAP_DIM, y))
        if self.map_state[(x + 1) % MAP_DIM][y] < 1:
            out.append(((x + 1) % MAP_DIM, y))

        return out

