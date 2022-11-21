import os
import pickle
import numpy as np
import logging
from amoeba_state import AmoebaState


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
        self.current_size = current_percept.current_size
        split = self.split_amoeba(current_percept.amoeba_map)

        num_cells = 5
        moveable_cells_backend = self.sample_backend(current_percept.amoeba_map, num_cells, split)
        
        mini = min(5, len(current_percept.periphery) * self.metabolism)
        for i, j in current_percept.bacteria:
            current_percept.amoeba_map[i][j] = 1

        
        
        
        retract = [tuple(i) for i in self.rng.choice(current_percept.periphery, replace=False, size=mini)]
        movable = self.find_movable_cells(retract, current_percept.periphery, current_percept.amoeba_map,
                                          current_percept.bacteria, mini)

        info = 0

        
        
        return retract, movable, info

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

    def split_amoeba(self, amoeba_map) -> bool:
        split = False
        amoeba_begin = False
        amoeba_end = False

        for i in range(100):
            curr_column = amoeba_map[:, i]
            value = np.max(curr_column)
            if value == 1:
                if not amoeba_begin:
                    amoeba_begin = True
                elif amoeba_end:
                    split = True
                    break
            elif value == 0:
                if amoeba_begin:
                    amoeba_end = True

        return split

    def sample_column(self, column, num_cells):
        """Function that sample a column of the amoeba map

        Args:
            column (np.ndarray): 1D numpy array of the column
            num_cells (int): number of cells to sample

        Returns:
            move_cells: list of cells to move
        """
        move_cells = []
        # prioritize even row
        for i in range(column.shape[0]):
            if len(move_cells) == num_cells:
                break
            if column[i] == 1 and i % 2 == 0:
                move_cells.append(i)
        # append odd row
        for i in range(column.shape[0]):
            if len(move_cells) == num_cells:
                break
            if column[i] == 1 and i % 2 != 0:
                move_cells.append(i)
        return move_cells
        
    
    def sample_backend(self, amoeba_map, num_cells, split=False) -> list:
        """Function that smaple the backend of the amoeba
        
        Args:
            amoeba_map (np.ndarray): 2D numpy array of the current amoeba map
            num_cells (int): number of cells to sample
            split (bool): whether the amoeba has split or not
        Returns:
            move_cells: list of cells to move
        """
        def find_move_cells(start, num_cells, amoeba_map):
            move_cells = []
            for i in range(start, 100):
                if num_cells == 0:
                    break
                curr_column = amoeba_map[:, i]
                if np.max(curr_column) == 1:
                    additional_cells = self.sample_column(curr_column, num_cells)
                    move_cells += [(j, i) for j in additional_cells]
                    num_cells -=  len(additional_cells)
            return move_cells
        
        start = 0
        if split:
            start = None
            # move pass the first chunk of amoeba
            for i in range(0, 100):
                curr_column = amoeba_map[:, i]
                if np.max(curr_column) == 0:
                    start = i
                    break
        return find_move_cells(start, num_cells, amoeba_map)
                    
                    
