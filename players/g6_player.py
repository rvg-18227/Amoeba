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
        self.logger.info(f'----------------Turn {info}-----------------')
        self.current_size = current_percept.current_size
        split, split_pt = self.split_amoeba(current_percept.amoeba_map)

        mini = int(self.current_size*self.metabolism)

        info_binary  = format(info, '04b')
        stage = int(info_binary[0])

        retract_list = self.reorganize_retract(current_percept.amoeba_map, current_percept.periphery)
        movable = self.find_movable_cells(retract_list, current_percept.periphery, current_percept.amoeba_map,
                    current_percept.bacteria)
        expand_list = self.reorganize_expand(current_percept.amoeba_map, movable)

        mini = min(mini, len(retract_list), len(expand_list))

        self.logger.info(f'retract: {retract_list}')
        self.logger.info(f'expand: {expand_list}')

        return retract_list[:mini], expand_list[:mini], info+1

    def reorganize_retract(self, amoeba_map, periphery):
        amoeba_loc = np.stack(np.where(amoeba_map == 1)).T.astype(int)
        amoeba_loc = amoeba_loc[amoeba_loc[:, 1].argsort()]
        right_side = np.min(amoeba_loc[:, 1])
        left_side = np.max(amoeba_loc[:, 1])
        retract_list = []

        for row in range(right_side, left_side+1):
            if len(retract_list) == 4:
                break

            row_array = np.where(amoeba_loc[:, 1] == row)[0]
            row_cells = amoeba_loc[row_array]
            columns = row_cells[:, 0]

            for col in columns:
                if len(retract_list) == 4:
                    break
                num_column = np.size(np.where(amoeba_loc[:, 0] == col)[0])
                if num_column > 2:
                    cell = (col, row)
                    if cell in periphery:
                        retract_list.append(cell)
                        amoeba_loc = np.delete(amoeba_loc, np.where(amoeba_loc==cell)[0], axis=0)

        return retract_list

    def reorganize_expand(self, amoeba_map, movable):
        amoeba_loc = np.stack(np.where(amoeba_map==1)).T.astype(int)
        rows, count = np.unique(amoeba_loc[:, 0], return_counts=True)

        direction = [-1, 1]
        expand_cells = []
        for i, idx in enumerate([rows.argmin(), rows.argmax()]):
            row = rows[idx]
            row_count = count[idx]
            row_cells = amoeba_loc[amoeba_loc[:, 0]==row]

            if row_count < 2:
                # expand to the right on the first/last row
                cell = row_cells[row_cells[:, 1].argmax()] # rightmost cell
                move = ((cell[0])%100, (cell[1]+1)%100)
                # if move in movable:
                expand_cells.append(move)

            # expand to an additional row
            for c_i in range(min(2, row_cells.shape[0])):
                cell = row_cells[-c_i-1]
                move = ((cell[0]+direction[i])%100, (cell[1])%100)
                # if move in movable:
                expand_cells.append(move)

        return expand_cells

    def find_movable_cells(self, retract, periphery, amoeba_map, bacteria):
        movable = []
        new_periphery = list(set(periphery).difference(set(retract)))
        for i, j in new_periphery:
            nbr = self.find_movable_neighbor(i, j, amoeba_map, bacteria)
            for x, y in nbr:
                if (x, y) not in movable:
                    movable.append((x, y))

        #movable += retract

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

    def split_amoeba(self, amoeba_map) -> bool:
        split = False
        amoeba_begin = False
        amoeba_end = False
        split_col = 0

        for i in range(100):
            curr_column = amoeba_map[:, i]
            value = np.max(curr_column)
            if value == 1:
                if not amoeba_begin:
                    amoeba_begin = True
                elif amoeba_end:
                    split = True
                    split_col = i - 1
                    break
            elif value == 0:
                if amoeba_begin:
                    amoeba_end = True

        return split, split_col

