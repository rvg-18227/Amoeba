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

        info_binary  = format(info, '04b')
        
        split, split_row = self.split_amoeba(current_percept.amoeba_map)
        amoeba_map = self.concat_map(current_percept.amoeba_map, split, split_row)
        self.logger.info(f'split_row (exclusive): {split_row}')

        stage = 0 if info < 10 else 1 # hard-coded
        if stage == 0:
            retract_list, expand_list = self.reorganize(
                amoeba_map, current_percept.periphery, current_percept.bacteria)
        elif stage == 1:
            retract_list, expand_list = self.forward(
                amoeba_map, current_percept.amoeba_map, current_percept.periphery, current_percept.bacteria, split_row)

        mini = min(int(self.current_size*self.metabolism), len(retract_list), len(expand_list))

        self.logger.info(f'retract: {retract_list}')
        self.logger.info(f'expand: {expand_list}')

        return retract_list[:mini], expand_list[:mini], info+1

    def concat_map(self, amoeba_map, split, split_row):
        amoeba_map = np.concatenate([amoeba_map, amoeba_map], axis=1)
        
        if not split:
            amoeba_map[:, 100:] = 0
        else:
            amoeba_map[:, :split_row+1] = 0
            amoeba_map[:, 100+split_row:] = 0

        return amoeba_map
    
    def forward(self, amoeba_map, amoeba_map_old, periphery, bacteria, split_row):
        retract_list = self.reorganize_retract(amoeba_map, periphery, min_num_per_col=1)
        movable = self.find_movable_cells(retract_list, periphery, amoeba_map_old, bacteria)
        expand_list = self.forward_expand(amoeba_map, movable, split_row)
        return retract_list, expand_list

    def reorganize(self, amoeba_map, periphery, bacteria):
        retract_list = self.reorganize_retract(amoeba_map, periphery)
        movable = self.find_movable_cells(retract_list, periphery, amoeba_map, bacteria)
        expand_list = self.reorganize_expand(amoeba_map, movable)
        return retract_list, expand_list

    def forward_expand(self, amoeba_map, movable, split_row):
        expand_cells = []
        movable = np.array(movable).astype(int)

        #print(np.stack(np.where(amoeba_map==1)).T.astype(int))

        # wrap around
        if split_row == 0:
            movable[movable[:, 1]==0, 1] += 100
        else:
            movable[movable[:, 1]<split_row, 1] += 100

        #print(movable)
        frontline = []
        for i in range(movable.shape[0]):
            cell = tuple(movable[i])
            if amoeba_map[cell[0], cell[1]-1] == 1:
                frontline.append(cell)
        frontline = np.array(frontline).astype(int)
        min_row = frontline[:, 1].min()

        # check if min_row is too large, if so wait
        amoeba_loc = np.stack(np.where(amoeba_map == 1)).T.astype(int)
        cols, _ = np.unique(amoeba_loc[:, 0], return_counts=True)
        min_row_all = min_row - 1
        for col in cols:
            min_row_all = min(min_row_all, amoeba_loc[amoeba_loc[:, 0]==col][:, 1].max())

        # print(frontline, min_row)
        # print(amoeba_loc, min_row_all)

        if min_row > min_row_all + 2:
            return []

        for i in range(frontline.shape[0]):
            cell = frontline[i]
            if cell[1] == min_row:
                expand_cells.append(tuple(cell%100))

        return expand_cells


    def reorganize_retract(self, amoeba_map, periphery, min_num_per_col=2):
        amoeba_loc = np.stack(np.where(amoeba_map == 1)).T.astype(int)
        amoeba_loc = amoeba_loc[amoeba_loc[:, 1].argsort()]
        top_side = np.min(amoeba_loc[:, 1])
        bottom_side = np.max(amoeba_loc[:, 1])
        retract_list = []

        for row in range(top_side, bottom_side):

            row_array = np.where(amoeba_loc[:, 1] == row)[0]
            row_cells = amoeba_loc[row_array]
            columns = np.sort(row_cells[:, 0])

            for col in columns:
                # never move bottom-most row
                max_row = amoeba_loc[amoeba_loc[:, 0]==col][:, 1].max()
                if row == max_row: 
                    continue

                num_column = np.size(np.where(amoeba_loc[:, 0] == col)[0])
                #self.logger.info(f'num_col: {num_column}')
                if num_column > min_num_per_col:
                    cell = (col%100, row%100)
                    if cell in periphery:
                        retract_list.append(cell)
                        #self.logger.info(f'cell retract: {cell}')
                        cell_idx = (amoeba_loc[:, 0] == cell[0]) * (amoeba_loc[:, 1] == cell[1])
                        #self.logger.info(f'cell idx : {np.where(cell_idx==True)[0]}')
                        amoeba_loc = np.delete(amoeba_loc, np.where(cell_idx==True)[0], axis=0)

        return retract_list

    def reorganize_expand(self, amoeba_map, movable):
        amoeba_loc = np.stack(np.where(amoeba_map==1)).T.astype(int)
        cols, count = np.unique(amoeba_loc[:, 0], return_counts=True)

        direction = [-1, 1]
        expand_cells = []
        for i, idx in enumerate([cols.argmin(), cols.argmax()]):
            col = cols[idx]
            col_count = count[idx]
            col_cells = amoeba_loc[amoeba_loc[:, 0]==col]

            if col_count < 2:
                # expand to the bottom of the first/last col
                cell = col_cells[col_cells[:, 1].argmax()] # lowest cell
                move = ((cell[0])%100, (cell[1]+1)%100)
                # if move in movable:
                expand_cells.append(move)

            # expand to an additional col
            for c_i in range(min(2, col_cells.shape[0])):
                cell = col_cells[-2:][c_i]
                move = ((cell[0]+direction[i])%100, (cell[1])%100)
                # if move in movable:
                expand_cells.append(move)

        return expand_cells


    def box_to_sweeper_retract(self, amoeba_map, periphery, mini):
        amoeba_loc = np.stack(np.where(amoeba_map == 1)).T.astype(int)
        amoeba_loc = amoeba_loc[amoeba_loc[:, 1].argsort()]
        top_side = np.min(amoeba_loc[:, 1])
        bottom_side = np.max(amoeba_loc[:, 1])
        retract_list = []

        for row in range(top_side, bottom_side - 1):
            if len(retract_list) == 2:
                break

            row_array = np.where(amoeba_loc[:, 1] == row)[0]
            row_cells = amoeba_loc[row_array]
            columns = np.sort(row_cells[:, 0])

            for col in columns:
                if len(retract_list) == 2:
                    break
                num_column = np.size(np.where(amoeba_loc[:, 0] == col)[0])
                self.logger.info(f'num_col: {num_column}')
                if num_column > 2:
                    cell = (col, row)
                    if cell in periphery:
                        retract_list.append(cell)
                        self.logger.info(f'cell retract: {cell}')
                        cell_idx = (amoeba_loc[:, 0] == cell[0]) * (amoeba_loc[:, 1] == cell[1])
                        self.logger.info(f'cell idx : {np.where(cell_idx == True)[0]}')
                        amoeba_loc = np.delete(amoeba_loc, np.where(cell_idx == True)[0], axis=0)

        return retract_list

    def box_to_sweeper_expand(self, amoeba_map, mini):

        amoeba_loc = np.stack(np.where(amoeba_map == 1)).T.astype(int)
        amoeba_loc = amoeba_loc[amoeba_loc[:, 1].argsort()]
        top_side = np.min(amoeba_loc[:, 1])
        bottom_side = np.max(amoeba_loc[:, 1])
        expand_cells = []

        max_row_length = np.NINF
        max_row = np.NINF
        for row in range(top_side, bottom_side - 1):
            row_array = np.where(amoeba_loc[:, 1] == row)[0]
            row_cells = amoeba_loc[row_array]
            row_len = len(row_cells)
            if row_len > max_row_length:
                max_row_length = row_len
                max_row = row

        row_use = np.where(amoeba_loc[:, 1] == max_row)[0]
        row_cells = amoeba_loc[row_use]
        row_cells = row_cells[row_cells[:, 0].argsort()]

        tentacle_one = row_cells[1]
        col_one = tentacle_one[0]
        tentacle_one_column = np.where(amoeba_loc[:, 0] == col_one)[0]
        tentacle_one_column = amoeba_loc[tentacle_one_column]
        tentacle_one_column_new = np.where(tentacle_one_column[:, 1] > max_row)[0]
        tentacle_one_column_new = tentacle_one_column[tentacle_one_column_new]
        tentacle_one_len = len(tentacle_one_column_new)
        #mini = 100
        if tentacle_one_len < mini:
            expand_cell = np.max(tentacle_one_column_new[:, 1])
            expand_cell = (col_one, expand_cell+1)
            expand_cells.append(expand_cell)

        tentacle_two = row_cells[-2]
        col_two = tentacle_two[0]
        tentacle_two_column = np.where(amoeba_loc[:, 0] == col_two)[0]
        tentacle_two_column = amoeba_loc[tentacle_two_column]
        tentacle_two_column_new = np.where(tentacle_two_column[:, 1] > max_row)[0]
        tentacle_two_column_new = tentacle_two_column[tentacle_two_column_new]
        tentacle_two_len = len(tentacle_two_column_new)
        # mini = 100
        if tentacle_two_len < mini:
            expand_cell = np.max(tentacle_two_column_new[:, 1])
            expand_cell = (col_two, expand_cell + 1)
            expand_cells.append(expand_cell)

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

    def split_amoeba(self, amoeba_map):
        split = False
        amoeba_begin = False
        amoeba_end = False
        split_row = 0

        for i in range(100):
            curr_row = amoeba_map[:, i]
            value = np.max(curr_row)
            if value == 1:
                if not amoeba_begin:
                    amoeba_begin = True
                elif amoeba_end:
                    split = True
                    split_row = i - 1
                    break
            elif value == 0:
                if amoeba_begin:
                    amoeba_end = True

        return split, split_row

