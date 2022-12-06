import os
import pickle
import numpy as np
import logging
from amoeba_state import AmoebaState
from matplotlib import pyplot as plt

EXTEND_COLOR = (np.random.rand(1,1,3) * 255).astype(int)
RETRACT_COLOR = (np.random.rand(1,1,3) * 255).astype(int)
AMOEABA_COLOR = (np.random.rand(1,1,3) * 255).astype(int)

class Drawer:
    def __init__(self):
        self.base = np.zeros((100, 100, 3))
    
    def draw(self, curr_percept, extract_coord, extend_coord):
        self.clear_graph()
        self._draw_amoeba(curr_percept)
        if len(extend_coord) != 0:
            self._draw_extend(extend_coord)
        if len(extract_coord) != 0:
            self._draw_retract(extract_coord)
        self.save()
    
    def _draw_extend(self, coord):
        x = np.array(coord)[:, 0]
        y = np.array(coord)[:, 1]
        self.base[x, y] = EXTEND_COLOR
    
    def _draw_retract(self, coord):
        x = np.array(coord)[:, 0]
        y = np.array(coord)[:, 1]
        self.base[x, y] = RETRACT_COLOR
    
    def _draw_amoeba(self, current_percept):
        coord = np.stack(np.where(current_percept.amoeba_map != 0), axis= 1)
        x = np.array(coord)[:, 0]
        y = np.array(coord)[:, 1]
        self.base[x, y] = AMOEABA_COLOR
    
    def clear_graph(self):
        self.base = np.zeros((100, 100, 3))
    
    def save(self, name='tmp.png'):
        plt.imsave(name, self.base.astype(np.uint8))

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
        self.drawer = Drawer()

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

        if info < 40:
            # expand
            stage = 0
        elif info < 70:
            # forward
            stage = 1
        elif info >= 70:
            stage = 2
        
        if stage == 0:
            if info < 20:
                retract_list, expand_list = self.reorganize(
                    amoeba_map, current_percept.periphery, current_percept.bacteria, split_row)
            else:
                retract_list, expand_list = self.init_organize(
                    amoeba_map, current_percept.periphery, current_percept.bacteria)
        elif stage == 1:
            retract_list, expand_list = self.forward(
                amoeba_map, current_percept.amoeba_map, current_percept.periphery, current_percept.bacteria, split_row)
        else:

            ##amoeba_loc = np.stack(np.where(amoeba_map == 1)).T.astype(int)
            #amoeba_loc = amoeba_loc[amoeba_loc[:, 1].argsort()]
           # bottom_side = np.max(amoeba_loc[:, 1])

            expand_list = self.box_to_sweeper_expand(
                    amoeba_map, int(self.current_size*self.metabolism))
            retract_list = self.box_to_sweeper_retract(
                    amoeba_map, current_percept.periphery, int(self.current_size*self.metabolism))
            if stage == 2 and len(retract_list) == 0:  
                # Close in
                col_one = self.find_first_tentacle(amoeba_map)
                print(col_one)
                print('close_in')
                retract_list, expand_list = self.close_in(amoeba_map)
            
            if len(retract_list) == 0:
                info = 0
                
            
        mini = min(int(self.current_size*self.metabolism), len(retract_list), len(expand_list))
        
        self.logger.info(f'retract: {retract_list[:mini]}')
        self.logger.info(f'expand: {expand_list[:mini]}')

        self.drawer.draw(current_percept, retract_list[:mini], expand_list[:mini])
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
        retract_list = self.organize_retract(amoeba_map, periphery, min_num_per_col=1)
        movable = self.find_movable_cells(retract_list, periphery, amoeba_map_old, bacteria)
        expand_list = self.forward_expand(amoeba_map, movable, split_row)
        return retract_list, expand_list

    def init_organize(self, amoeba_map, periphery, bacteria):
        retract_list = self.organize_retract(amoeba_map, periphery)
        movable = self.find_movable_cells(retract_list, periphery, amoeba_map, bacteria)
        expand_list = self.organize_expand(amoeba_map, movable)
        return retract_list, expand_list

    def reorganize(self, amoeba_map, periphery, bacteria, split_row):
        retract_list = self.reorganize_retract(amoeba_map, periphery)
        movable = self.find_movable_cells(retract_list, periphery, amoeba_map, bacteria)
        expand_list = self.reorganize_expand(amoeba_map, movable, split_row)
        print('retract:', retract_list)
        print('expand:', expand_list)
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

    def organize_retract(self, amoeba_map, periphery, min_num_per_col=2):
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

            priorize_columns = []
            # priorize columns that's about to disconnect
            for i in range(columns.shape[0]):
                cell = (columns[i], row)
                if amoeba_map[cell[0], cell[1]+1] == 0 or amoeba_map[cell[0]-1, cell[1]]+amoeba_map[cell[0]+1, cell[1]] == 0:
                    priorize_columns.append(columns[i])
            priorize_columns = sorted(priorize_columns, reverse=True) 

            for col in priorize_columns:
                # if row > bottom_side - 7: # do not retract bottom 7 rows, hardcoded
                #     continue
                cell = (col%100, row%100)
                if cell in periphery:
                    retract_list.append(cell)
                    cell_idx = (amoeba_loc[:, 0] == cell[0]) * (amoeba_loc[:, 1] == cell[1])
                    amoeba_loc = np.delete(amoeba_loc, np.where(cell_idx==True)[0], axis=0)

        return retract_list

    def reorganize_expand(self, amoeba_map, movable, split_row):
        amoeba_loc = np.stack(np.where(amoeba_map==1)).T.astype(int)
        cols, count = np.unique(amoeba_loc[:, 0], return_counts=True)

        expand_cells = []
        movable = np.array(movable).astype(int)

        # wrap around
        if split_row == 0:
            movable[movable[:, 1]==0, 1] += 100
        else:
            movable[movable[:, 1]<split_row, 1] += 100

        max_row = movable[:, 1].max()
        movable = movable[movable[:, 1]!=max_row]
        movable = movable[movable[:, 1].argsort()[::-1]]

        for i in range(movable.shape[0]):
            cell = movable[i]
            if cell[0] < 4 or cell[0] > 95 or amoeba_map[cell[0], cell[1]] == 1:
                continue
            expand_cells.append(tuple(cell%100))

        return expand_cells[:10]

    def organize_expand(self, amoeba_map, movable):
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
                if move in movable:
                    expand_cells.append(move)

            # expand to an additional col
            for c_i in range(min(2, col_cells.shape[0])):
                cell = col_cells[-2:][c_i]
                if cell[0]+direction[i] < 4 or cell[0]+direction[i] > 95:
                    continue
                move = ((cell[0]+direction[i])%100, (cell[1])%100)
                if move in movable:
                    expand_cells.append(move)

        return expand_cells


    def box_to_sweeper_retract(self, amoeba_map, periphery, mini):

        amoeba_loc = np.stack(np.where(amoeba_map == 1)).T.astype(int)
        amoeba_loc = amoeba_loc[amoeba_loc[:, 1].argsort()]
        top_side = np.min(amoeba_loc[:, 1])
        bottom_side = np.max(amoeba_loc[:, 1])
        retract_list = []


        max_row_length = np.NINF
        max_row = np.NINF
        for row in range(top_side, bottom_side+1):
            #print("here", row)
            row_array = np.where(amoeba_loc[:, 1] == row)[0]
            row_cells = amoeba_loc[row_array]
            row_len = len(row_cells)
            if row_len >= max_row_length -1:
                max_row_length = row_len
                max_row = row

        row_use = np.where(amoeba_loc[:, 1] == max_row)[0]
        row_cells = amoeba_loc[row_use]
        row_cells = row_cells[row_cells[:, 0].argsort()]

        tentacle_one = row_cells[1]
        tentacle_one = tentacle_one[0]
        tentacle_two = row_cells[-2]
        tentacle_two = tentacle_two[0]
        tentacle_three = row_cells[-3]
        tentacle_three = tentacle_three[0]

        for row in range(top_side, bottom_side):
           # print(row)
            if len(retract_list) == 3:
                break

            row_array = np.where(amoeba_loc[:, 1] == row)[0]
            row_cells = amoeba_loc[row_array]
            columns = np.sort(row_cells[:, 0])

            for col in columns:
                if len(retract_list) == 3:
                    break

               # print(max_row)
                if row >= max_row:
                    continue

                num_column = np.size(np.where(amoeba_loc[:, 0] == col)[0])

                if num_column > 1:# and col != tentacle_one and col != tentacle_two and col != tentacle_three:
                    #cell = (col, row)
                    cell = (col % 100, row % 100)
                    if cell in periphery:
                        retract_list.append(cell)
                        #self.logger.info(f'cell retract: {cell}')
                        cell_idx = (amoeba_loc[:, 0] == cell[0]) * (amoeba_loc[:, 1] == cell[1])
                        #self.logger.info(f'cell idx : {np.where(cell_idx == True)[0]}')
                        amoeba_loc = np.delete(amoeba_loc, np.where(cell_idx == True)[0], axis=0)

        print("retract sweep", retract_list)
        #print(amoeba_loc)
        #quit()
        return retract_list

    def box_to_sweeper_expand(self, amoeba_map, mini):

        amoeba_loc = np.stack(np.where(amoeba_map == 1)).T.astype(int)
        amoeba_loc = amoeba_loc[amoeba_loc[:, 1].argsort()]
        top_side = np.min(amoeba_loc[:, 1])
        bottom_side = np.max(amoeba_loc[:, 1])
        expand_cells = []

        max_row_length = np.NINF
        max_row = np.NINF
        for row in range(top_side, bottom_side+1):
            row_array = np.where(amoeba_loc[:, 1] == row)[0]
            row_cells = amoeba_loc[row_array]
            row_len = len(row_cells)

            if row_len >= max_row_length:
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
            expand_cell = np.max(tentacle_one_column[:, 1])
            expand_cell = (col_one % 100, (expand_cell + 1) % 100)
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
            expand_cell = np.max(tentacle_two_column[:, 1])
            expand_cell = (col_two % 100, (expand_cell + 1) % 100)
            expand_cells.append(expand_cell)

        tentacle_three = row_cells[-3]
        col_three = tentacle_three[0]
        tentacle_three_column = np.where(amoeba_loc[:, 0] == col_three)[0]
        tentacle_three_column = amoeba_loc[tentacle_three_column]

        tentacle_three_column_new = np.where(tentacle_three_column[:, 1] > max_row)[0]
        tentacle_three_column_new = tentacle_three_column[tentacle_three_column_new]

        tentacle_three_len = len(tentacle_three_column_new)
        # mini = 100
        if tentacle_three_len < mini:
            expand_cell = np.max(tentacle_three_column[:, 1])
            expand_cell = (col_three % 100, (expand_cell + 1) % 100)
            expand_cells.append(expand_cell)
            #quit()

        print("expand", expand_cells)
        return expand_cells

    def find_first_tentacle(self, amoeba_map):
        amoeba_loc = np.stack(np.where(amoeba_map == 1)).T.astype(int)
        amoeba_loc = amoeba_loc[amoeba_loc[:, 1].argsort()]
        top_side = np.min(amoeba_loc[:, 1])
        bottom_side = np.max(amoeba_loc[:, 1])

        max_row_length = np.NINF
        max_row = np.NINF
        for row in range(top_side, bottom_side + 1):
            row_array = np.where(amoeba_loc[:, 1] == row)[0]
            row_cells = amoeba_loc[row_array]
            row_len = len(row_cells)

            if row_len >= max_row_length:
                max_row_length = row_len
                max_row = row

        row_use = np.where(amoeba_loc[:, 1] == max_row)[0]
        row_cells = amoeba_loc[row_use]
        row_cells = row_cells[row_cells[:, 0].argsort()]

        tentacle_one = row_cells[1]
        col_one = tentacle_one[0]

        return col_one

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

    def get_continuous_chunk(self, array, start):
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
                else:
                    break
            return chunks
    
    def locate_tenticle(self, amoeba_map, tentacle_column, split=False):
        """Locate the moving tenticle of the amoeba
        
        Args:
            amoeba_map (np.array): The current amoeba map
            tenticle_length (int): The length of the tenticle
            
        
        Returns:
            list: The location of the cell on tenticle
        """
        # ASSUMPTION: the based row is the one with the maximum number of cell
        base_row = np.argmax(np.sum(amoeba_map, axis=0))
        tenticle_start_row = (base_row + 1) % 100
        # get the chunk of the moving tenticle
        chunk = self.get_continuous_chunk(amoeba_map[tentacle_column, :], tenticle_start_row)
        return chunk
        
    def is_singular(self, amoeba_map, tentacle_column, chunks):
        """Check if the tenticle is singular
        Args:
            amoeba_map (np.array): 2d np array
            tenticle_column (int): column of the moving tenticle (0 - 99)
            chunks (list[int]): rows of the moving tenticle
        """
        # check two side of the column
        for i in chunks:
            if amoeba_map[(tentacle_column + 1) % 100, i] == 1 or amoeba_map[(tentacle_column - 1) % 100, i] == 1:
                return False
        # check tip of the column
        if amoeba_map[tentacle_column, chunks[-1] + 1] == 1:
            return False 
        return True
    
    def relocate_extra_cells(self, amoeba_map, tentacle_column, chunks):
        pass    
    
    def move_tenticle(self, tentacle_column, chunks):
        retract = [(tentacle_column, i) for i in chunks]
        extend = [((tentacle_column + 1) % 100, i) for i in chunks]
        return retract, extend
    
    def is_singular_chunk(self, array):
        start = array.argmax()
        for i in range(array.sum()):
            if array[(start + i) % 100] == 0:
                return False
        return True
    
    def close_in(self, amoeba_map):
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
        start_row = np.argmax(np.sum(amoeba_map, axis=0)) + 1
        # assume no spliting
        # move right most cell to the adjacent left location
        # TODO
        # Pass in location of the opposing column
        target_column = self.find_first_tentacle(amoeba_map)
        # TODO
        # check height of the tenticle, if exceed max meta, shrink it
        extract = []
        extend = []
        for i in range(start_row, start_row+100):
            if amoeba_map[:, i].sum() == 0:
                # reach the end
                break
            if self.is_singular_chunk(amoeba_map[:, i]) and amoeba_map[:, i].argmax() == target_column:
                # singular chunk, no need to move
                continue
            row_reverse = amoeba_map[:, i][::-1]
            right_most_cell = len(row_reverse) - np.argmax(row_reverse) - 1
            for j in range(right_most_cell, -1, -1):
                curr_cell = amoeba_map[(j)%100, i]
                next_cell = amoeba_map[(j-1)%100, i]
                if curr_cell == 1 and next_cell == 0:
                    extract.append((right_most_cell, i % 100))
                    extend.append((j-1, i % 100))
                    break
        return extract, extend