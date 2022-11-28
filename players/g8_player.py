import os
import pickle
import numpy as np
import logging
import constants
from amoeba_state import AmoebaState
import matplotlib.pyplot as plt
from statistics import median

def wrap_coordinates(x, y):
        return x % constants.map_dim, y % constants.map_dim

def get_nonzero_coordinates(amoeba_map):
    return sorted(zip(*np.nonzero(amoeba_map.T)))


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
        self.current_size = goal_size // 4

        self.turn = 0
    
    def check_move(self, retract, move):
        if not set(retract).issubset(set(self.periphery)):
            return False

        movable = retract[:]
        new_periphery = list(set(self.periphery).difference(set(retract)))
        for i, j in new_periphery:
            nbr = self.find_movable_neighbor(i, j, self.amoeba_map, self.bacteria)
            for x, y in nbr:
                if (x, y) not in movable:
                    movable.append((x, y))

        if not set(move).issubset(set(movable)):
            return False

        amoeba = np.copy(self.amoeba_map)
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
    
    
    def gen_static_formation(self):
        formation_map = np.zeros((constants.map_dim, constants.map_dim))
        center_x, center_y = constants.map_dim // 2, constants.map_dim // 2
        width = ((self.current_size - 1) * 2) // 5
        num_teeth = min(self.current_size - 2 * width, width // 2)
    
        for i in range(0, (width // 2) + 1):
            formation_map[wrap_coordinates(center_x , center_y + i)] = 1
            formation_map[wrap_coordinates(center_x, center_y - i)] = 1
            
            formation_map[wrap_coordinates(center_x - 1, center_y + i)] = 1
            formation_map[wrap_coordinates(center_x - 1, center_y - i)] = 1
        
        for i in range(1, num_teeth + 1, 2):
            formation_map[wrap_coordinates(center_x + 1, center_y + i)] = 1
            formation_map[wrap_coordinates(center_x + 1, center_y - i)] = 1
            formation_map[wrap_coordinates(center_x + 2, center_y + i)] = 1
            formation_map[wrap_coordinates(center_x + 2, center_y - i)] = 1
        return formation_map
        
    def get_retracts_moves(self, formation_map):
        current_coordinates = get_nonzero_coordinates(self.amoeba_map)
        target_coordinates = get_nonzero_coordinates(formation_map)
        retract_candidates = [p for p in (set(current_coordinates) - set(target_coordinates)) if p in self.periphery]
        #self.movable_cells = self.find_movable_cells(retracts, self.periphery, self.amoeba_map, self.bacteria)
        move_candidates = [p for p in (set(target_coordinates) - set(current_coordinates)) if p in self.movable_cells]

        # while not self.check_move(retracts, moves) and len(retracts) > 0 and len(moves) > 0:
        #     retracts.pop()
        #     moves.pop()
        retracts = []
        moves = []        
        for m in move_candidates:
            for r in retract_candidates:
                if self.check_move(retracts + [r], moves + [m]):
                    # matching retract found, add the extend and retract to our lists
                    retracts.append(r)
                    retract_candidates.remove(r)
                    moves.append(m)
                    move_candidates.remove(m)
                    break
        bound = int(self.metabolism * self.current_size)
        s = min(5, len(self.periphery) // 2)
        if len(retracts) == 0 or len(moves) == 0:
            return [], []
    
        return retracts, moves

    
    def move(self, last_percept, current_percept, info) -> (list, list, int):
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
        self.turn += 1
        
        # Saving the current state of the map
        self.current_size = current_percept.current_size
        self.amoeba_map = current_percept.amoeba_map
        self.periphery = current_percept.periphery
        self.bacteria = current_percept.bacteria
        self.movable_cells = current_percept.movable_cells

        
        max_movable_cells = int(self.metabolism * self.current_size)
        if info == 1:
            # initial formation has formed
            # shift by 1 and move teeth
            rows = [x for x, _ in get_nonzero_coordinates(self.amoeba_map)]
            current_row = int(min(rows))
            #print(current_row)
            formation_map = self.gen_static_formation()
            dist_moved = (current_row + 1) - (constants.map_dim // 2)
            print(dist_moved)
            formation_map = np.roll(formation_map, dist_moved + 1, axis=0)
            teeth_shift = current_row % 2
            formation_map = np.roll(formation_map, teeth_shift, axis=1)
            retracts, moves = self.get_retracts_moves(formation_map)
            if len(retracts) == 0 or len(moves) == 0:
                print("No moves found")
            plt.clf()
            plt.imshow(formation_map)
            plt.savefig("dynamic_formation.png")
           
            return retracts, moves, 1
        else:
            # initial formation has not formed yet
            formation_map = self.gen_static_formation()
            plt.clf()
            plt.imshow(formation_map)
            plt.savefig('formation_map.png')
            retracts, moves = self.get_retracts_moves(formation_map)
            if len(moves) == 0:
                return [], [], 1
            else:
                return retracts, moves, 0

    
    def get_top_row(self, periphery):
        # Gets top row of amoeba to be retracted
        # For each x coord, we want the one with the lowest y coord
        top_row_vals = {}
        for (x, y) in periphery:
            if x in top_row_vals:
                if y < top_row_vals[x]:
                    top_row_vals[x] = y
            else:
                top_row_vals[x] = y

        top_row = []
        for key, val in top_row_vals.items():
            top_row.append((key, val))

        return top_row



    def find_movable_cells(self, retract, periphery, amoeba_map, bacteria):
        movable = []
        new_periphery = list(set(periphery).difference(set(retract)))
        for i, j in new_periphery:
            nbr = self.find_movable_neighbor(i, j, amoeba_map, bacteria)
            for x, y in nbr:
                if (x, y) not in movable:
                    movable.append((x, y))

        movable += retract

        return movable

    def find_movable_neighbor(self, x, y, amoeba_map, bacteria):
        out = []
        if (x, y) not in bacteria:
            if amoeba_map[x][(y - 1) % 100] == 0:
                out.append((x, (y - 1) % 100))
            if amoeba_map[x][(y + 1) % 100] == 0:
                out.append((x, (y + 1) % 100))
            if amoeba_map[(x - 1) % 100][y] == 0:
                out.append(((x - 1) % 100, y))
            if amoeba_map[(x + 1) % 100][y] == 0:
                out.append(((x + 1) % 100, y))

        return out
