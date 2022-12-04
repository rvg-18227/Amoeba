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
        

        if info < 20:
            # expand
            stage = 0
        elif info >= 20 and info < 25:
            # forward
            stage = 1
        elif info >= 25:
            stage = 2
        #stage = 0 if info < 10 else 1 # hard-coded
        if stage == 0:
            retract_list, expand_list = self.reorganize(
                current_percept.amoeba_map, current_percept.periphery, current_percept.bacteria)
        elif stage == 1:
            retract_list, expand_list = self.forward(
                current_percept.amoeba_map, current_percept.periphery, current_percept.bacteria)
        else:
            expand_list = self.box_to_sweeper_expand(
                    current_percept.amoeba_map, mini)
            retract_list = self.box_to_sweeper_retract(
                    current_percept.amoeba_map, current_percept.periphery, mini)
            if stage == 2 and len(expand_list) == 0:  
                expand_list = None
                retract_list = None
            
        mini = min(mini, len(retract_list), len(expand_list))

        self.logger.info(f'retract: {retract_list}')
        self.logger.info(f'expand: {expand_list}')

        return retract_list[:mini], expand_list[:mini], info+1

    
    def forward(self, amoeba_map, periphery, bacteria):
        retract_list = self.reorganize_retract(amoeba_map, periphery, min_num_per_col=1)
        movable = self.find_movable_cells(retract_list, periphery, amoeba_map, bacteria)
        expand_list = self.forward_expand(amoeba_map, movable)
        return retract_list, expand_list

    def reorganize(self, amoeba_map, periphery, bacteria):
        retract_list = self.reorganize_retract(amoeba_map, periphery)
        movable = self.find_movable_cells(retract_list, periphery, amoeba_map, bacteria)
        expand_list = self.reorganize_expand(amoeba_map, movable)
        return retract_list, expand_list

    def forward_expand(self, amoeba_map, movable):
        amoeba_loc = np.stack(np.where(amoeba_map==1)).T.astype(int)

        expand_cells = []
        movable = np.array(movable).astype(int)

        frontline = []
        for i in range(movable.shape[0]):
            cell = tuple(movable[i])
            if amoeba_map[cell[0], cell[1]-1] == 1:
                frontline.append(cell)
        frontline = np.array(frontline).astype(int)
        min_row = frontline[:, 1].min()

        #print(frontline, min_row)

        for i in range(frontline.shape[0]):
            cell = tuple(frontline[i])
            if cell[1] == min_row:
                expand_cells.append(cell)

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
                num_column = np.size(np.where(amoeba_loc[:, 0] == col)[0])
                self.logger.info(f'num_col: {num_column}')
                if num_column > min_num_per_col:
                    cell = (col, row)
                    if cell in periphery:
                        retract_list.append(cell)
                        self.logger.info(f'cell retract: {cell}')
                        cell_idx = (amoeba_loc[:, 0] == cell[0]) * (amoeba_loc[:, 1] == cell[1])
                        self.logger.info(f'cell idx : {np.where(cell_idx==True)[0]}')
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

    def locate_tenticle(self, amoeba_map, tenticle_column, split=False):
        """Locate the moving tenticle of the amoeba
        
        Args:
            amoeba_map (np.array): The current amoeba map
            tenticle_length (int): The length of the tenticle
            
        
        Returns:
            list: The location of the cell on tenticle
        """
        # check for column that has legnth AT LEAST tenticle_length (size * metabolism)
        # and has continuous structure
        # def is_continuous(array, split=False):
        #     """Check if a given array has continuous chunk

        #     Args:
        #         array (np.array): 1D np array
        #         split (bool, optional): if the array is split

        #     Returns:
        #         bool: if the array is continuous
        #     """
        #     length = np.sum(array)
        #     if length == array.shape[0]:
        #         # Special cases: its wrap around
        #         return True
        #     if not split:
        #         start = np.argmax(array)
        #     else:
        #         start = None
        #         # find the first 1
        #         for i in range(100, 0, -1):
        #             next_i = (i - 1) % 100
        #             if array[i] == 1 and array[next_i] == 0:
        #                 start = i
        #                 break
        #     # loop from start to start + length
        #     # if every position is 1, then return True
        #     for i in range(start, (start + length)):
        #         if array[i % 100] == 0:
        #             return False
        #     return True
        
        def get_continuous_chunk(array, start):
            """Get continuous chunk of an array starting from start

            Args:
                array (np.array): 1D np array
                start (int): where to start
            """
            assert(array[start] == 1)
            chunks = []
            for i in range(100):
                if array[(start + i) % 100] == 1:
                    chunks.append((start + i) % 100)
            return chunks
                
        # ASSUMPTION: the based row is the one with the maximum number of cell
        base_row = np.argmax(np.sum(amoeba_map, axis=0))
        tenticle_start_row = (base_row + 1) % 100
        # get the chunk of the moving tenticle
        chunk = get_continuous_chunk(amoeba_map[tenticle_column, :], tenticle_start_row)
        return None
        
    def is_singular(self, amoeba_map, tenticle_column, chunks):
        """Check if the tenticle is singular

        Args:
            amoeba_map (np.array): 2d np array
            tenticle_column (int): column of the moving tenticle (0 - 99)
            chunks (list[int]): rows of the moving tenticle
        """
        # check two side of the column
        for i in chunks:
            if amoeba_map[(tenticle_column + 1) % 100, i] == 1 or amoeba_map[(tenticle_column - 1) % 100, i] == 1:
                return False
        # check tip of the column
        if amoeba_map[tenticle_column, chunks[-1] + 1] == 1:
            return False 
        return True
    
    def relocate_extra_cells(self, amoeba_map, tenticle_column, chunks):
        pass    
    
    def move_tenticle(self, amoeba_map, tenticle_column, chunks):
        pass
    
    def close_in(self, amoeba_map, tenticle_column):
        """Close in function for clashing formation

        Args:
            amoeba_map (_type_): 2d np array
            tenticle_column(int): column of the moving tenticle (0 - 99)
        """
        # locate the moving tenticle
        #tenticle_length = self.size * self.metabolism
        #moving_tenticle = None
        # possible for new cell eaten such that tenticle_length increases between moves
        #while moving_tenticle is None:
        moving_chunks = self.locate_tenticle(amoeba_map, tenticle_column)
        # check for singular column
        if self.is_singular(amoeba_map, tenticle_column, moving_chunks):
            # move the tenticle 1 step forward
            return self.move_tenticle(amoeba_map, tenticle_column, moving_chunks)
        else:
            # relocate the extra cell
            pass
