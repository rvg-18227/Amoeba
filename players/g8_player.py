import logging
import math
import os
import pickle
from enum import Enum
from typing import List, Tuple, Dict, Set

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

import constants
from amoeba_state import AmoebaState

turn = 0


# ---------------------------------------------------------------------------- #
#                               Constants                                      #
# ---------------------------------------------------------------------------- #

CENTER_X = constants.map_dim // 2
CENTER_Y = constants.map_dim // 2

COMB_SEPARATION_DIST = 4
TEETH_GAP = 3

VERTICAL_SHIFT_PERIOD = 2
VERTICAL_SHIFT_LIST = (
    (
        [0 for i in range(VERTICAL_SHIFT_PERIOD)]
        + [1 for i in range(VERTICAL_SHIFT_PERIOD)]
    )
    * (round(np.ceil(100 / (VERTICAL_SHIFT_PERIOD * 2))))
)[:100]

# ---------------------------------------------------------------------------- #
#                               Helper Functions                               #
# ---------------------------------------------------------------------------- #


def wrap_coordinates(x, y):
    return (x % constants.map_dim, y % constants.map_dim)


def map_to_coords(amoeba_map: npt.NDArray) -> List[Tuple[int, int]]:
    return list(map(tuple, np.transpose(amoeba_map.nonzero()).tolist()))


def coords_to_map(coords: List[Tuple[int, int]], size=constants.map_dim) -> npt.NDArray:
    amoeba_map = np.zeros((size, size), dtype=np.int8)
    for x, y in coords:
        amoeba_map[x, y] = 1
    return amoeba_map


def show_amoeba_map(amoeba_map: npt.NDArray, retracts=[], extends=[], title="") -> None:
    retracts_map = coords_to_map(retracts)
    extends_map = coords_to_map(extends)

    map = np.zeros((constants.map_dim, constants.map_dim), dtype=np.int8)
    for x in range(constants.map_dim):
        for y in range(constants.map_dim):
            # transpose map for visualization as we add cells
            if retracts_map[x, y] == 1:
                map[y, x] = -1
            elif extends_map[x, y] == 1:
                map[y, x] = 2
            elif amoeba_map[x, y] == 1:
                map[y, x] = 1

    plt.rcParams["figure.figsize"] = (10, 10)
    plt.pcolormesh(map, edgecolors="k", linewidth=1)
    ax = plt.gca()
    ax.set_aspect("equal")
    plt.title(title)
    # plt.savefig(f"debug/{turn}.png")
    plt.show()


# ---------------------------------------------------------------------------- #
#                                Memory Bit Mask                               #
# ---------------------------------------------------------------------------- #


class MemoryFields(Enum):
    Initialized = 0
    Translating = 1

class Status(Enum):
    Morphing = 0
    Translating = 1


def read_memory(memory: int) -> Dict[MemoryFields, bool]:
    out = {}
    for field in MemoryFields:
        value = True if (memory & (1 << field.value)) >> field.value else False
        out[field] = value
    return out


def change_memory_field(memory: int, field: MemoryFields, value: bool) -> int:
    bit = 1 if value else 0
    mask = 1 << field.value
    # Unset the bit, then or in the new bit
    return (memory & ~mask) | ((bit << field.value) & mask)


if __name__ == "__main__":
    memory = 0
    fields = read_memory(memory)
    assert fields[MemoryFields.Initialized] == False
    assert fields[MemoryFields.Translating] == False

    memory = change_memory_field(memory, MemoryFields.Initialized, True)
    fields = read_memory(memory)
    assert fields[MemoryFields.Initialized] == True
    assert fields[MemoryFields.Translating] == False

    memory = change_memory_field(memory, MemoryFields.Translating, True)
    fields = read_memory(memory)
    assert fields[MemoryFields.Initialized] == True
    assert fields[MemoryFields.Translating] == True

    memory = change_memory_field(memory, MemoryFields.Translating, False)
    fields = read_memory(memory)
    assert fields[MemoryFields.Initialized] == True
    assert fields[MemoryFields.Translating] == False

    memory = change_memory_field(memory, MemoryFields.Initialized, False)
    fields = read_memory(memory)
    assert fields[MemoryFields.Initialized] == False
    assert fields[MemoryFields.Translating] == False

def encode_byte(x, status):
    assert 0 <= x <= 100
    bit_string = format(x, "07b")
    if status == Status.Morphing:
        return int(bit_string + "0", 2)
    else:
        return int(bit_string + "1", 2)

def decode_byte(byte):
    bit_string = format(byte, "08b")
    x = int(bit_string[:7], 2)
    status = Status.Morphing if bit_string[-1] == "0" else Status.Translating
    return x, status


# ---------------------------------------------------------------------------- #
#                               Formation Class                                #
# ---------------------------------------------------------------------------- #


class Formation:
    def __init__(self, initial_formation=None) -> None:
        self.map = (
            initial_formation
            if initial_formation
            else np.zeros((constants.map_dim, constants.map_dim), dtype=np.int8)
        )

    def add_cell(self, x, y) -> None:
        self.map[x % constants.map_dim, y % constants.map_dim] = 1
        
    def get_cell(self, x, y) -> int:
        return self.map[x % constants.map_dim, y % constants.map_dim]

    def merge_formation(self, formation_map: npt.NDArray):
        self.map = np.logical_or(self.map, formation_map)


# ---------------------------------------------------------------------------- #
#                               Main Player Class                              #
# ---------------------------------------------------------------------------- #


class Player:
    def __init__(
        self,
        rng: np.random.Generator,
        logger: logging.Logger,
        metabolism: float,
        goal_size: int,
        precomp_dir: str,
    ) -> None:
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

        self.vertical_shift = 0
        self.method = 0 # Determines which strategy to use: 0 -> double comb, 1 -> V-shape for low density

        # Class accessible percept variables, written at the start of each turn
        self.current_size: int = None
        self.amoeba_map: npt.NDArray = None
        self.bacteria_cells: Set[Tuple[int, int]] = None
        self.retractable_cells: List[Tuple[int, int]] = None
        self.extendable_cells: List[Tuple[int, int]] = None
        self.num_available_moves: int = None

    ### Based on Group 2's infrastructure, but changed all implementation ####
    def generate_comb_formation(
        self, size: int, tooth_offset=0, center_x=CENTER_X, center_y=CENTER_Y, comb_idx=0
    ) -> npt.NDArray:
        formation = Formation()
        
        if size < 2:
            return formation.map

        teeth_size = min((size // 6), 48) # new tooth for every 2 backbone

        remaining_cells = size - teeth_size
        divider = min((int(remaining_cells * 0.66)), 100)
        # backbone_size = min((size - teeth_size - divider), 99)
        backbone_size = (size - teeth_size - divider)

        print("Divider size: ", divider)

        
        teeth_size = min(teeth_size, backbone_size // 2)
        

        formation.add_cell(center_x, center_y)

        for i in range(1, divider // 2): # Adding the divider
            formation.add_cell(center_x, center_y - i)
            formation.add_cell(center_x, center_y + i)
        

        if backbone_size > 96: # 48 long backbone for each side

            # Make the main backbone
            for i in range(1, 49): # Adding the two backbones
                formation.add_cell(center_x - i, center_y - tooth_offset)
            for i in range(1, 49): # Adding the two backbones
                formation.add_cell(center_x + i, center_y + tooth_offset)
            backbone_size -= 96

            # Make second backbones
            if backbone_size > 120:
                # this is excess
                for i in range(1, 49): # Adding the second backbones
                    formation.add_cell(center_x - i, center_y - tooth_offset + 49)
                for i in range(1, 49): # Adding the second backbones
                    formation.add_cell(center_x + i, center_y + tooth_offset - 49)
                # Add the teeth
                for i in range(1, 25): # Adding the right teeth
                    formation.add_cell(center_x - 2 * i, center_y - 1 - tooth_offset + 49)
                for i in range(1, 25): # Adding the left teeth
                    formation.add_cell(center_x + 2 * i, center_y + 1 + tooth_offset - 49)

                backbone_size -= 120

                # Adds excess to first backbones
                backbones = backbone_size // 96 # number of 48-long backbones each side has
                excess = backbone_size % 96

                # Adds excess to original backbone
                for b in range(2, backbones + 1):
                    for i in range(1, 49): # Adding the two backbones
                        formation.add_cell(center_x - i, center_y - tooth_offset + (b-1))
                    for i in range(1, 49): # Adding the two backbones
                        formation.add_cell(center_x + i, center_y + tooth_offset - (b-1))

                if (excess > 0):
                    # Add excess (first check if more teeth can be added)
                    need_teeth = 48 - teeth_size
                    if (need_teeth) > 0:
                        if excess <= need_teeth:
                            teeth_size += excess
                            excess = 0
                        else:
                            teeth_size += excess
                            excess -= need_teeth


                    left = excess // 2
                    right = excess - left

                    for i in range(1, left + 1): # Adding the two backbones
                        formation.add_cell(center_x - i, center_y - tooth_offset + (backbones))
                    for i in range(1, right + 1): # Adding the two backbones
                        formation.add_cell(center_x + i, center_y + tooth_offset - (backbones))

            else:
                new_teeth = backbone_size // 3
                new_backbone = backbone_size - new_teeth

                for i in range(1, new_backbone+1): # Adding the second backbones
                    formation.add_cell(center_x - i, center_y - tooth_offset + 49)
                for i in range(1, new_backbone+1): # Adding the second backbones
                    formation.add_cell(center_x + i, center_y + tooth_offset - 49)
                # Add the teeth
                for i in range(1, new_teeth+1): # Adding the right teeth
                    formation.add_cell(center_x - 2 * i, center_y - 1 - tooth_offset + 49)
                for i in range(1, new_teeth+1): # Adding the left teeth
                    formation.add_cell(center_x + 2 * i, center_y + 1 + tooth_offset - 49)

                backbone_size -= new_teeth
                backbone_size -= new_backbone
                
            

        else:
            left = backbone_size // 2
            right = backbone_size - left

            for i in range(1, left + 1): # Adding the two backbones
                formation.add_cell(center_x - i, center_y - tooth_offset)
            for i in range(1, right + 1): # Adding the two backbones
                formation.add_cell(center_x + i, center_y + tooth_offset)


        # Add the teeth
        left_teeth = teeth_size // 2
        right_teeth = teeth_size - left_teeth

        for i in range(1, left_teeth + 1): # Adding the right teeth
            formation.add_cell(center_x - 2 * i, center_y - 1 - tooth_offset)
        for i in range(1, right_teeth + 1): # Adding the left teeth
            formation.add_cell(center_x + 2 * i, center_y + 1 + tooth_offset)
            

        # global turn
        # if turn > 25:
        #     show_amoeba_map(formation.map)
        return formation.map


    ### Adopted from Group 2 ####

    def get_morph_moves(
        self, desired_amoeba: npt.NDArray
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Function which takes a starting amoeba state and a desired amoeba state and generates a set of retracts and extends
        to morph the amoeba shape towards the desired shape.
        """

        current_points = map_to_coords(self.amoeba_map)
        desired_points = map_to_coords(desired_amoeba)

        potential_retracts = [
            p
            for p in list(set(current_points).difference(set(desired_points)))
            if p in self.retractable_cells
        ]
        potential_extends = [
            p
            for p in list(set(desired_points).difference(set(current_points)))
            if p in self.extendable_cells
        ]
        potential_extends.sort(key=lambda p: p[1])


        retracts = []
        extends = []
        check_calls = 0
        for potential_extend in [p for p in potential_extends]:
            # Ensure we only move as much as possible given our current metabolism
            if len(extends) >= self.num_available_moves:
                break

            matching_retracts = list(potential_retracts)
            matching_retracts.sort(key=lambda p: math.dist(p, potential_extend))

            for i in range(len(matching_retracts)):
                retract = matching_retracts[i]
                # Matching retract found, add the extend and retract to our lists
                if self.check_move(retracts + [retract], extends + [potential_extend]):
                    check_calls += 1
                    retracts.append(retract)
                    potential_retracts.remove(retract)
                    extends.append(potential_extend)
                    potential_extends.remove(potential_extend)
                    break
                check_calls += 1
        
        return retracts, extends



    # Adapted from Group 3's implementation
    def gen_low_density_formation(self, size, center_x):
        center_y = constants.map_dim // 2
        formation = np.zeros((constants.map_dim, constants.map_dim), dtype=int)
        offsets = {(0,0), (0,1), (0,-1), (1,1), (1,-1)}
        total_cells = size - 5
        # if total_cells > 12:
        #     total_cells -= 12
        #     form_claws = True
        form_claws = True
        num_claw_cells = int(total_cells * 0.1) // 4 * 4
        total_cells -= num_claw_cells
        i = 1
        j = 2
        while total_cells > 0:
            if total_cells < 6:
                if total_cells > 1:
                    # If possible add evenly
                    offsets.update({(i,j), (i,-j)})
                    total_cells -= 2
                    i += 1
                else:
                    # Add last remaining to left arm
                    offsets.update({(i, j)})
                    total_cells -= 1
            else:
                # if there are at least 6 add 3 to each side
                offsets.update({(i, j), (i+1,j), (i+2, j), (i, -j), (i+1,-j), (i+2, -j)})
                total_cells -= 6
                i += 2
                j += 1
        # print("Final pair ", i, j)
        if form_claws:
            # print("CLAWS!")
            # for t in range(1, 7):
            #     offsets.update({(i, j-t), (i, -j+t)})
            for t in range(num_claw_cells // 4):
                offsets.update({(i+t, j-t-1), (i+t+1, j-t-1)})
                offsets.update({(i+t, -j+t+1), (i+t+1, -j+t+1)})
                #offsets.update({(i+t, j+2), (i+t, -j-2)})
            
        for i, j in offsets:
            formation[wrap_coordinates(center_x + i, center_y + j)] = 1
        # plt.clf()
        # plt.imshow(formation)
        # plt.show()
        return formation



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

    def find_movable_neighbor(
        self, x: int, y: int, amoeba_map: npt.NDArray, bacteria: Set[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        out = []
        if (x, y) not in bacteria:
            if amoeba_map[x][(y - 1) % constants.map_dim] == 0:
                out.append((x, (y - 1) % constants.map_dim))
            if amoeba_map[x][(y + 1) % constants.map_dim] == 0:
                out.append((x, (y + 1) % constants.map_dim))
            if amoeba_map[(x - 1) % constants.map_dim][y] == 0:
                out.append(((x - 1) % constants.map_dim, y))
            if amoeba_map[(x + 1) % constants.map_dim][y] == 0:
                out.append(((x + 1) % constants.map_dim, y))
        return out

    # Adapted from amoeba_game code
    def check_move(
        self, retracts: List[Tuple[int, int]], extends: List[Tuple[int, int]]
    ) -> bool:
        if not set(retracts).issubset(set(self.retractable_cells)):
            return False

        movable = set(retracts[:])
        new_periphery = list(set(self.retractable_cells).difference(set(retracts)))
        for i, j in new_periphery:
            nbr = self.find_movable_neighbor(i, j, self.amoeba_map, self.bacteria_cells)
            for x, y in nbr:
                if (x, y) not in movable:
                    movable.add((x, y))

        if not set(extends).issubset(movable):
            return False

        amoeba = np.copy(self.amoeba_map)
        amoeba[amoeba < 0] = 0
        amoeba[amoeba > 0] = 1

        for i, j in retracts:
            amoeba[i][j] = 0

        for i, j in extends:
            amoeba[i][j] = 1

        tmp = np.where(amoeba == 1)
        result = list(zip(tmp[0], tmp[1]))
        check = np.zeros((constants.map_dim, constants.map_dim), dtype=int)

        stack = result[0:1]
        result = set(result)
        while len(stack):
            a, b = stack.pop()
            check[a][b] = 1

            if (a, (b - 1) % constants.map_dim) in result and check[a][
                (b - 1) % constants.map_dim
            ] == 0:
                stack.append((a, (b - 1) % constants.map_dim))
            if (a, (b + 1) % constants.map_dim) in result and check[a][
                (b + 1) % constants.map_dim
            ] == 0:
                stack.append((a, (b + 1) % constants.map_dim))
            if ((a - 1) % constants.map_dim, b) in result and check[
                (a - 1) % constants.map_dim
            ][b] == 0:
                stack.append(((a - 1) % constants.map_dim, b))
            if ((a + 1) % constants.map_dim, b) in result and check[
                (a + 1) % constants.map_dim
            ][b] == 0:
                stack.append(((a + 1) % constants.map_dim, b))

        return (amoeba == check).all()

    def store_current_percept(self, current_percept: AmoebaState) -> None:
        self.current_size = current_percept.current_size
        self.amoeba_map = current_percept.amoeba_map
        self.retractable_cells = current_percept.periphery
        self.bacteria_cells = set(current_percept.bacteria)
        self.extendable_cells = current_percept.movable_cells
        self.num_available_moves = int(
            np.ceil(self.metabolism * current_percept.current_size)
        )

    def move(
        self, last_percept: AmoebaState, current_percept: AmoebaState, info: int
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], int]:
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
        global turn
        turn += 1

        if turn == 1:
            print("Turn #{}".format(turn))
            a = len(last_percept.bacteria)
            b = len(last_percept.periphery)

            print("Est. Density: ", (a/b))

        self.store_current_percept(current_percept)
        
        retracts = []
        moves = []

        # Two different implementations based on density and side length

        if self.method == 0: # Double comb for higher density/larger side length

            memory_fields = read_memory(info)
            if not memory_fields[MemoryFields.Initialized]:
                retracts, moves = self.get_morph_moves(
                    self.generate_comb_formation(self.current_size, 0)
                )
                if len(moves) == 0:
                    info = change_memory_field(info, MemoryFields.Initialized, True)
                    info = (50 << 1) | info
                    memory_fields = read_memory(info)

            if memory_fields[MemoryFields.Initialized]:
                print("Vertical shift: ", self.vertical_shift)
                try:
                    curr_coords = map_to_coords(self.amoeba_map)
                    top_divider = min(y for x, y in curr_coords if x == CENTER_X) # moving upward
                    bottom_divider = max(y for x, y in curr_coords if x == CENTER_X) # moving downward
                except Exception as e:
                    print(e)
                    self.method = 1
                    return None
                print("Height: ", bottom_divider - top_divider)
                real_height = bottom_divider - top_divider
                if (real_height < 98) and ( (real_height // 2) < (self.vertical_shift-5) ): # Has shifted more than divider, so converts to V-shape
                    self.method = 1
                    return None


                next_comb = self.generate_comb_formation(
                    self.current_size, self.vertical_shift, CENTER_X, CENTER_Y
                )
                # Check if current comb formation is filled
                settled = self.amoeba_map[next_comb].all()
                if not settled:
                    retracts, moves = self.get_morph_moves(next_comb)
                
                    # Actually, we have no more moves to make 
                    if len(moves) == 0:
                        settled = True

                if settled:
                    self.vertical_shift += 1

                    next_comb = self.generate_comb_formation(
                        self.current_size, self.vertical_shift, CENTER_X, CENTER_Y
                    )
                    retracts, moves = self.get_morph_moves(next_comb)
                    # info = new_backbone_col << 1 | 1

            return retracts, moves, info


        # ****************************************** SEPARATION ********************************************************

        else: # 1 = V-shape lower density strategy

            x, status = decode_byte(info)
            if turn == 1:
                prev_center_x = constants.map_dim // 2
            else:
                prev_center_x = x
            # print("PREV CENTER X: {}".format(prev_center_x))
            # print("NEXT CENTER X: {}".format((prev_center_x + 1) % constants.map_dim))
            if status == Status.Morphing:
                formation = self.gen_low_density_formation(self.current_size, prev_center_x)
                retracts, moves = self.get_morph_moves(formation)
                if len(moves) == 0:
                    # now it's time to translate
                    center_x = (prev_center_x + 1) % constants.map_dim
                    center_y = constants.map_dim // 2
                    formation = self.gen_low_density_formation(self.current_size, center_x)
                    retracts, moves = self.get_morph_moves(formation)
                    info = encode_byte(center_x, Status.Translating)
                    # if (center_x, center_y) not in retracts:
                    #     #assert False
                    #     print("Morphing to center_x = {}".format(prev_center_x))
                    #     info = encode_byte(prev_center_x, Status.Morphing)
                    #     return retracts, moves, info
                    print("Translating to center_x = {}".format(center_x))
                    return retracts, moves, info
                else:
                    # still morphing
                    print("Morphing to center_x = {}".format(prev_center_x))
                    info = encode_byte(prev_center_x, Status.Morphing)
                    return retracts, moves, info
            else:
                center_x = (prev_center_x + 1) % constants.map_dim
                center_y = constants.map_dim // 2
                formation = self.gen_low_density_formation(self.current_size, center_x)
                retracts, moves = self.get_morph_moves(formation)
                if center_x not in [x for x, _ in retracts]:
                    #assert False
                    print("Morphing to center_x = {}".format(prev_center_x))
                    info = encode_byte(prev_center_x, Status.Morphing)
                    return retracts, moves, info
                else:
                    info = encode_byte(center_x, Status.Translating)
                    print("Translating to center_x = {}".format(center_x))
                    return retracts, moves, info