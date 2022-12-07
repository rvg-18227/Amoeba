import numpy as np
import numpy.typing as npt
import logging
from typing import Tuple, List
from amoeba_state import AmoebaState
import pstats
import cProfile

# ---------------------------------------------------------------------------- #
#                               Helper Functions                               #
# ---------------------------------------------------------------------------- #

# Copied from Group 2
def map_to_coords(amoeba_map: npt.NDArray) -> list[Tuple[int, int]]:
    return list(map(tuple, np.transpose(amoeba_map.nonzero()).tolist()))

def binary_search(list, goal):
    mid = len(list) // 2

    if not goal(list[:mid]):
        return binary_search(list[:mid], goal)
    elif not goal(list[mid + 1:]):
        return binary_search(list[mid + 1:], goal)
    elif not goal(list):
        return list[mid]
    else:
        return None

def wrap_point(x, y):
    return (x % 100, y % 100)

def get_neighbors(cell):
    x, y = cell
    neighbors = [wrap_point(x - 1, y), wrap_point(x + 1, y), wrap_point(x, y + 1), wrap_point(x, y - 1)]
    return neighbors

def generate_rake(formation, move_teeth, available, spacing, x_position, center_y, reverse=False):
    complete_chunks = available // (2 * spacing + 1)
    additional_chunks = available % (2 * spacing + 1) // 2
    base_length = min(complete_chunks * 3 + additional_chunks, 100)

    start_y = center_y - base_length // 2

    if not reverse:
        start_x = x_position

        for i in range(start_y, start_y + base_length):
            formation[start_x, i % 100] = 1
            formation[(start_x - 1) % 100, i % 100] = 1
            if i % 100 % spacing == move_teeth:
                formation[(start_x + 1) % 100, i % 100] = 1
    else:
        start_x = (99 - x_position) % 100

        for i in range(start_y, start_y + base_length):
            formation[start_x, i % 100] = 1
            formation[(start_x + 1) % 100, i % 100] = 1
            if i % 100 % spacing == move_teeth ^ 1:
                formation[(start_x - 1) % 100, i % 100] = 1

    return formation

def generate_bar(formation, available, x_position, center_y):
    bar_length = min((x_position - 50) * 2, available)
    if x_position < 50: bar_length = min(100, available)
        
    for offset in range(1, bar_length + 1):
        formation[wrap_point((x_position - offset), center_y)] = 1
    
    return formation

# ---------------------------------------------------------------------------- #
#                                Info Byte Class                               #
# ---------------------------------------------------------------------------- #

class Infobyte:
    def __init__(self, byte=None, x_position=None, move_teeth=None):
        if x_position is not None:
            self.x_position = x_position
        else:
            self.x_position = 50
        
        if move_teeth is not None:
            self.move_teeth = move_teeth
        else:
            self.move_teeth = 1
        
        self.infobyte = encode_info(self.x_position, self.move_teeth)

    def set_x_position(self, x_position):
        self.x_position = x_position
        self.infobyte = encode_info(self.x_position, self.move_teeth)
    
    def set_move_teeth(self, move_teeth):
        self.move_teeth = move_teeth
        self.infobyte = encode_info(self.x_position, self.move_teeth)
    
def encode_info(move_teeth: int, x_position: int) -> int:
    """Encode the information to be sent
        Args:
            move_teeth (int): 1 bit for whether the teeth are shifting or not
            x_position (int): the desired x position of the amoeba
        Returns:
            int: the encoded information as an int
    """
    move_teeth_binary = bin(move_teeth)[2:].zfill(1)
    x_position_binary = bin(x_position)[2:].zfill(7)
    info_binstr = move_teeth_binary + x_position_binary

    return int("0b" + info_binstr, 2)

def decode_info(info: int) -> Tuple[int, int]:
    """Decode the information received
        Args:
            info (int): the information received
        Returns:
            Tuple[int, int]: move_teeth, x_position, 
    """
    info_str = bin(info)[2:].zfill(8)

    return int(info_str[0:1], 2), int(info_str[1:8], 2)

# ---------------------------------------------------------------------------- #
#                               Main Player Class                              #
# ---------------------------------------------------------------------------- #

TOOTH_SPACING = 2
SHIFTING_FREQUENCY = 6

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

        self.rng = rng
        self.logger = logger
        self.metabolism = metabolism
        self.goal_size = goal_size

        self.current_size = None
        self.amoeba_map = None
        self.bacteria_cells = None
        self.retractable_cells = None
        self.extendable_cells = None
        self.num_available_moves = None
        self.map_state = None

        # InfoByte
        self.x_position = None
        self.move_teeth = None
    
    def make_two_rakes(self, amoeba_size: int, x_position: int, move_teeth: int) -> npt.NDArray:
        formation = np.zeros((100, 100), dtype=np.int8)
        center_y = 50
        spacing = TOOTH_SPACING + 1

        stop_collision = lambda x: abs(x - (50 * round(x / 50)))

        # first rake
        if amoeba_size > 350 and stop_collision(x_position) < 3:
            x_position = ((50 * round(x_position / 50)) + 3) % 100
        formation = generate_rake(formation, move_teeth, amoeba_size, spacing, x_position, center_y)
        
        # middle bar
        cells_used = (formation == 1).sum()
        available = amoeba_size - cells_used
        formation = generate_bar(formation, available, x_position, center_y)

        # second rake
        cells_used = (formation == 1).sum()
        available = amoeba_size - cells_used
        formation = generate_rake(formation, move_teeth, available, spacing, x_position, center_y, reverse=True)
        
        return formation

    # Copied from Group 2, slightly modified
    def get_morph_moves(self, desired_amoeba: npt.NDArray) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """ Function which takes a starting amoeba state and a desired amoeba state and generates a set of retracts and extends
            to morph the amoeba shape towards the desired shape.
        """

        current_points = map_to_coords(self.amoeba_map)
        desired_points = map_to_coords(desired_amoeba)

        potential_retracts = [p for p in list(set(current_points).difference(set(desired_points))) if
                              (p in self.retractable_cells) and not any([neighbor in self.bacteria_cells for neighbor in get_neighbors(p)])]
        potential_extends = [p for p in list(set(desired_points).difference(set(current_points))) if
                             p in self.extendable_cells]
        potential_extends.sort(key=lambda pos: abs(50 - pos[1]))

        retracts = []
        extends = []

        ranked_cells_dict = {}
        for x, y in potential_retracts:
            neighbors = [((x-1) % 100, y), ((x+1) % 100, y), (x, (y-1) % 100), (x, (y+1) % 100)]
            score = 0
            for neighbor in neighbors:
                if self.amoeba_map[neighbor] == 1:
                    score += 1
            ranked_cells_dict[(x, y)] = score

        potential_retracts = sorted(potential_retracts, key=lambda x: (ranked_cells_dict[x], -abs(x[0]-50)), reverse=True)

        possible_moves = min(len(potential_retracts), len(potential_extends), self.num_available_moves)
        potential_extends.reverse()

        for _ in range(possible_moves):
            next_ret = potential_retracts.pop()
            retracts.append(next_ret)
            extends.append(potential_extends.pop())

            for neighbor in get_neighbors(next_ret):
                if neighbor in ranked_cells_dict:
                    ranked_cells_dict[neighbor] -= 1

            potential_retracts = sorted(potential_retracts, key=lambda x: (ranked_cells_dict[x], -abs(x[0]-50)), reverse=True)

        while not self.check_move(retracts, extends):
            bad_retract = binary_search(retracts, lambda r: self.check_move(r, extends[:len(r)]))

            for neighbor in get_neighbors(bad_retract):
                if neighbor in ranked_cells_dict:
                    ranked_cells_dict[neighbor] += 1

            retracts.remove(bad_retract)
            extends.pop()

            potential_retracts = sorted(potential_retracts, key=lambda x: (ranked_cells_dict[x], -abs(x[0]-50)), reverse=True)

            if potential_retracts and potential_extends:
                retracts.append(potential_retracts.pop())
                extends.append(potential_extends.pop())

        return retracts, extends

    # Copied from the simulator and modified
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
        check = np.zeros((100, 100), dtype=int)

        stack = result[0:1]
        while len(stack):
            a, b = stack.pop()
            check[a][b] = 1

            if (a, (b - 1) % 100) in result and check[a][(b - 1) % 100] == 0:
                stack.append((a, (b - 1) % 100))
            if (a, (b + 1) % 100) in result and check[a][(b + 1) % 100] == 0:
                stack.append((a, (b + 1) % 100))
            if ((a - 1) % 100, b) in result and check[(a - 1) % 100][b] == 0:
                stack.append(((a - 1) % 100, b))
            if ((a + 1) % 100, b) in result and check[(a + 1) % 100][b] == 0:
                stack.append(((a + 1) % 100, b))

        return (amoeba == check).all()

    # Copied from Group 2
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

    def move(self, last_percept, current_percept, info) -> Tuple[list, list, int]:
        self.store_current_percept(current_percept)

        retract = []
        move = []
        info_str = bin(info)[2:].zfill(8)

        # set to none to show that we don't store info across turns
        self.move_teeth, self.x_position = None, None

        self.move_teeth, self.x_position = int(info_str[0:1], 2), int(info_str[1:8], 2)
        move_teeth, x_position = self.move_teeth, self.x_position

        if self.is_square(current_percept):
            self.x_position = 50
            self.move_teeth = 1

        while len(retract) == 0 and len(move) == 0:
            if self.move_teeth == 0:
                x_position = self.x_position - 1
            else:
                x_position = self.x_position
        
            offset_y = 0 if (self.x_position % (2 * SHIFTING_FREQUENCY)) < SHIFTING_FREQUENCY else -1

            target_formation = self.make_two_rakes(self.current_size, x_position, offset_y + 1)

            diff = np.count_nonzero(target_formation & self.amoeba_map != target_formation)
            if diff / self.current_size <= 0.15 and diff <= 30:
                retract, move = [], []
            else:
                retract, move = self.get_morph_moves(target_formation)
            
            if len(retract) == 0 and len(move) == 0:
                if self.move_teeth == 1:
                    self.x_position = (self.x_position + 1) % 100
                    if self.x_position % SHIFTING_FREQUENCY == 0:
                        self.move_teeth = 0
                else:
                    self.move_teeth = 1
        
        move_teeth_binary = bin(self.move_teeth)[2:].zfill(1)
        x_position_binary = bin(self.x_position)[2:].zfill(7)
        info = int(move_teeth_binary + x_position_binary, 2)

        return retract, move, info

    def find_movable_neighbor(self, x, y):
        out = []
        if self.map_state[x][(y - 1) % 100] == 0:
            out.append((x, (y - 1) % 100))
        if self.map_state[x][(y + 1) % 100] == 0:
            out.append((x, (y + 1) % 100))
        if self.map_state[(x - 1) % 100][y] == 0:
            out.append(((x - 1) % 100, y))
        if self.map_state[(x + 1) % 100][y] == 0:
            out.append(((x + 1) % 100, y))

        return out

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
