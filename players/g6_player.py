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

        mini = min(5, int(self.current_size*self.metabolism))

        info_binary  = format(info, '04b')
        print(info_binary)
        is_initialized = info_binary[0]

        if is_initialized == '0':
            print(last_percept.current_size)
            print(current_percept.current_size)

            amoeba_loc = np.stack(np.where(current_percept.amoeba_map == 1)).T
            amoeba_loc = amoeba_loc[amoeba_loc[:, 1].argsort()]
            right_side = np.min(amoeba_loc[:, 0])
            left_side = np.max(amoeba_loc[:, 0])
            retract_list = []
            for column in range(right_side, left_side+1):
                if len(retract_list) == 3:
                    break

                column_array = np.where(amoeba_loc[:, 0] == column)
                num_column = np.size(column_array)

                if column % 2 != 0:
                    #odd column
                    if num_column > 3:
                        bottom_cell = np.max(column_array)
                        bottom_cell = amoeba_loc[bottom_cell]
                        retract_list.append((bottom_cell[0], bottom_cell[1]))
                else:
                    #even column
                    if num_column > 2:
                        bottom_cell = np.max(column_array)
                        bottom_cell = amoeba_loc[bottom_cell]
                        retract_list.append((bottom_cell[0], bottom_cell[1]))

            if len(retract_list) == 0:
                #in correct shape
                return [], [], 255
            else:
                return retract_list, self.teeth_extend(current_percept.amoeba_map, retract_list), 0

        else:
            retract_teeth = self.teeth_retract(current_percept.amoeba_map, 3)
            extend_teeth = self.teeth_extend(current_percept.amoeba_map, retract_teeth)
            return retract_teeth, extend_teeth, 255
        #retract = self.sample_backend(current_percept.amoeba_map, mini, split)
        '''
        for i, j in current_percept.bacteria:
            current_percept.amoeba_map[i][j] = 1

        amoeba_loc = np.stack(np.where(current_percept.amoeba_map==1)).T
        amoeba_loc = amoeba_loc[amoeba_loc[:, 1].argsort()]
        self.logger.info(f'amoeba: \n{amoeba_loc}')

        movable = self.find_movable_cells(retract, current_percept.periphery, current_percept.amoeba_map,
                                          current_percept.bacteria)
        moves = self.get_branch_tips(retract, movable, current_percept.periphery, 
                                        current_percept.amoeba_map, split, split_pt=split_pt)

        move_num = min(mini, len(retract), len(moves))
        self.logger.info(f'retract: \n{retract}')
        self.logger.info(f'moves: \n{moves}')
        return retract[:move_num], moves[:move_num], info+1
        '''

    def check_formation(self, amoeba_map):
        """
        Checks if the formation is a comb
        Ignoring last row
        """
        amoeba_loc = np.stack(np.where(amoeba_map==1)).T
        rows, count = np.unique(amoeba_loc[:, 0], return_counts=True)
        max_row = rows.max()
        for i in rows.shape[0]:
            if rows[i] == max_row:
                if rows[i] % 2 == 1 and count[i] > 3:
                    return False
                elif rows[i] % 2 == 0 and count[i] > 2:
                    return False
            else:
                if rows[i] % 2 == 1 and count[i] != 3:
                    return False
                elif rows[i] % 2 == 0 and count[i] != 2:
                    return False
        return True

    def allocate_extra(self, movable, periphery, amoeba_loc, amoeba_map, split):
        """
        Use newly eaten bacterias in the even rows to extend the comb
        """
        if split:
            split_pt = 50 # hardcoded for now
            amoeba_map = np.copy(amoeba_map)
            amoeba_map = np.concatenate([amoeba_map, amoeba_map[:, :split_pt]], axis=1)
            amoeba_map[:, :split_pt] = 0

        # Get extra cells
        amoeba_loc = np.stack(np.where(amoeba_map==1)).T
        amoeba_even = amoeba_loc[amoeba_loc[:, 0]%2==0]
        even_rows, count = np.unique(amoeba_even[:, 0], return_counts=True)
        extra = []
        for i in rows.shape[0]:
            if count[i] > 0:
                cells = amoeba_even[np.where(amoeba_even[:, 0]==rows[i])[0]]
                rightmost_cell = tuple((cells[cells[:, 1].argmax()].astype(int) % 100).tolist())
                if rightmost_cell in periphery:
                    extra.append(rightmost_cell)

        # Get Extendable cells
        expand_cells = self.expand(movable, periphery, amoeba_map)

        num = min(len(extra), len(expand_cells))
        return extra[:num], expand_cells[:num]

    def expand(self, movable, periphery, amoeba_map):
        """
        Returns locations to expand on at the bottom
        """
        amoeba_loc = np.stack(np.where(amoeba_map==1)).T
        rows, count = np.unique(amoeba_loc[:, 0], return_counts=True)
        last_row = rows.max()
        last_count = count[rows.argmax()]
        last_row_cells = amoeba_loc[amoeba_loc[:, 0]==last_row]
        expand_cells = []
        if last_row % 2 == 0:
            max_num_col = 2
        else:
            max_num_col = 3
        if last_count < max_num_col:
            # expand to the right on the last row
            cell = last_row_cells[last_row_cells[:, 1].argmax()] # rightmost cell
            move = (int(cell[0])%100, int(cell[1]+1)%100)
            if move in periphery:
                expand_cells.append(move)

        # expand to an additional row
        for c_i in range(last_row_cells.shape[0]):
            cell = c_i
            move = (int(cell[0]+1)%100, int(cell[1])%100)
            if move in periphery:
                expand_cells.append(move)

        return expand_cells

    def allocate_even_row(self, movable, periphery, amoeba_loc, amoeba_map, split):
        if split:
            split_pt = 50 # hardcoded for now
            amoeba_map = np.copy(amoeba_map)
            amoeba_map = np.concatenate([amoeba_map, amoeba_map[:, :split_pt]], axis=1)
            amoeba_map[:, :split_pt] = 0

        amoeba_loc = np.stack(np.where(amoeba_map==1)).T
        amoeba_even = amoeba_loc[amoeba_loc[:, 0]%2==0]
        leftmost_col = amoeba_even[:, 1].max()
        retracts = amoeba_even[amoeba_even[:, 1]==leftmost_col]
        retract_final = []
        extend_final = []
        for i in range(retracts.shape[0]):
            retract_cell = tuple((retracts[i].astype(int)%100).tolist())
            if not retract_cell in periphery:
                continue
            cells = amoeba_even[amoeba_even[:, 0]==retract_cell[0]]
            target_col = cells[:, 1].max() + 1
            extend_cell = (retract_cell[0], int(target_col)%100)
            if not extend_cell in movable:
                continue

            retract_final.append(retract_cell)
            extend_final.append(extend_cell)

        return retract_final, extend_final

    def get_branch_tips(self, retract, movable, periphery, amoeba_map, split, split_pt):
        """
        Get the rightmost tips of the brush branches, prioritizing shorter branches
        """
        retract = np.array(retract)
        retract_even = retract[retract[:, 0]%2==0]
        retract_even[:, 1] = (retract_even[:, 1] + 1) % 100 # check cell next to the even retraction cell
        prioritize_rows = []
        curr_col = []
        for i in range(retract_even.shape[0]):
            if amoeba_map[tuple(retract_even[i])] == 0:
                # no cell next to even retraction cell
                prioritize_rows.append(retract_even[i, 0])
                curr_col.append((retract_even[i, 1]-1) % 100)

        self.logger.info(f'prioritized rows: \n{prioritize_rows}')

        periphery = np.array(periphery)
        movable = set(movable)
        self.logger.info(f'periphery: \n{periphery}')
        if split:
        	rightmost_cells = periphery[periphery[:, 1]<=split_pt]
        	nonsplit_rows = set(periphery[:, 0].tolist()) - set(rightmost_cells[:, 0].tolist())
        	for row in nonsplit_rows:
        		rightmost_cells = np.concatenate([rightmost_cells, periphery[periphery[:, 0]==row]])
        else:
        	rightmost_cells = periphery
        rightmost_val = rightmost_cells[:, 1].max() if not split else rightmost_cells[rightmost_cells[:, 1]<=split_pt].max()+100

        moves = []
        for i in range(len(prioritize_rows)):
            row = prioritize_rows[i]
            prev_row = bool(np.where(amoeba_map[row-1, :]==1)[0].shape[0])
            next_row = bool(np.where(amoeba_map[row+1, :]==1)[0].shape[0])
            temp_move = (row, curr_col[i])
            for col in range(rightmost_val, curr_col[i]-1, -1):
                if amoeba_map[row, col%100] == 0 and \
                    (not prev_row or amoeba_map[row-1, col%100] == 1) and \
                    (not next_row or amoeba_map[row+1, col%100] == 1):
                    # check if new location connects prev row and next row
                    temp_move = (row, col%100)
                    break
            moves.append(temp_move)
        self.logger.info(f'prioritized rows:{prioritize_rows}, moves: {moves}')

        rightmost_cells = rightmost_cells[rightmost_cells[:, 0]%2==1] # keep only odd rows
        if rightmost_cells.shape[0] == 0:
            return moves

        self.logger.info(f'rightmost all: \n{rightmost_cells}')
        rightmost_cells = rightmost_cells[(-rightmost_cells[:, 1]).argsort()] # sort cells by col
        rightmost_cells = rightmost_cells[np.unique(rightmost_cells[:, 0], return_index=True)[1]] # keep rightmost cell for each row
        
        if not split:
            target_col = rightmost_cells[:, 1].max()
            left_col = rightmost_cells[:, 1].min()
            if left_col == target_col:
                target_col += 1
        else:
            target_col = rightmost_cells[rightmost_cells[:, 1]<=split_pt].max()+100
        self.logger.info(f'rightmost unique: \n{rightmost_cells}')
        for i in range(rightmost_cells.shape[0]):
            col = rightmost_cells[i, 1]
            col = col if not split or col > split_pt else col+100
            if col < target_col:
                move = (rightmost_cells[i, 0], (col+1)%100)
                self.logger.info(f'move - movable: \n{move}, {move in movable}')
                if move in movable:
                    moves.append(move)

        return moves


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

    def sample_column(self, column, num_cells, odd=True):
        """Function that sample a column of the amoeba map

        Args:
            column (np.ndarray): 1D numpy array of the column
            num_cells (int): number of cells to sample
            odd (bool): whether to sample odd rows

        Returns:
            move_cells: list of cells to move
        """
        move_cells = []
        if odd:
            # sample odd rows
            for i in range(column.shape[0]):
                if len(move_cells) == num_cells:
                    break
                if column[i] == 1 and i % 2 != 0:
                    move_cells.append(i)
        else:
            # sample even row
            for i in range(column.shape[0]):
                if len(move_cells) == num_cells:
                    break
                if column[i] == 1 and i % 2 == 0:
                    move_cells.append(i)
        return move_cells
        
                    
    def teeth_retract(self, amoeba_map, num_cells, split=False) -> list:
        """Function that sample the teeth row of the amoeba
        
        Args:
            amoeba_map (np.ndarray): 2D numpy array of the current amoeba map
            num_cells (int): number of cells to sample
            split (bool): whether the amoeba has split or not
        Returns:
            move_cells: list of cells to move
        """
        def find_move_cells(start, num_cells, amoeba_map):
            for i in range(start, 100):
                curr_column = amoeba_map[:, i]
                if np.max(curr_column) == 1:
                    sample_column_idx = i
                    break
            sample_column = amoeba_map[:, sample_column_idx]
            return [(j, i) for j in self.sample_column(sample_column, num_cells, odd=True)]
        
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
    
    def teeth_extend(self, amoeba_map, retract, split=False):
        start = 0
        amoeba_map_vec = (amoeba_map.sum(axis=0) != 0).astype(int)
        if split:
            start = None
            # move pass the first chunk of amoeba
            for i in range(0, 100):
                if amoeba_map_vec[i] == 0:
                    start = i
                    break
        for i in range(start, 100):
            if amoeba_map_vec[i] == 1:
                start = i
                break
        # find the first column of the amoeba
        for i in range(start, 100):
            if amoeba_map_vec[(i+1) % 100] == 0:
                first_column = i
                break
        # return one before the first column of amoeba
        return [(i, (first_column+1) % 100) for i, _ in retract]
        
