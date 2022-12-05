import os
import pickle
import numpy as np
import logging
from amoeba_state import AmoebaState
import constants

from typing import Tuple, List
import numpy.typing as npt
import math

MAP_LENGTH = 100

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

        self.amoeba_map = None
        self.periphery = None
        self.bacteria = None
        self.movable_cells = None
        self.num_available_moves = 0
        self.static_center = [50, 50]

        self.turn = 0
    
    # Adapted from G2 aka from amoeba_game.py
    def check_move(
        self, retracts: List[Tuple[int, int]], extends: List[Tuple[int, int]]
    ) -> bool:
        if not set(retracts).issubset(self.periphery):
            return False

        movable = retracts[:]
        new_periphery = list(self.periphery.difference(set(retracts)))
        for i, j in new_periphery:
            nbr = self.find_movable_neighbor(i, j, self.amoeba_map, self.bacteria)
            for x, y in nbr:
                if (x, y) not in movable:
                    movable.append((x, y))

        if not set(extends).issubset(set(movable)):
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
        # a cell is on the periphery if it borders (orthogonally) a 
        # cell that is not occupied by the amoeba
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

    # Find shape given size of anoemba, in the form of a list of offsets from center
    def get_desired_shape(self, shape=1):
        # Assume base shape given size is always > 5
        offsets = {(0,0), (0,1), (0,-1), (1,1), (1,-1)}
        total_cells = self.current_size-5
        if shape == 0:
            i = 2
            j = 1
            while total_cells > 0:
                if total_cells < 6:
                    if total_cells > 1:
                        # If possible add evenly
                        offsets.update({(i,j), (-i,j)})
                        total_cells-=2
                        j+=1
                    else:
                        # Add last remaining to left arm
                        offsets.update({(i, j)})
                        total_cells-=1
                else:
                    # if there are at least 6 add 3 to each side
                    offsets.update({(i, j), (i,j+1), (i, j+2), (-i, j), (-i,j+1), (-i, j+2)})
                    total_cells -= 6
                    i+=1
                    j+=2
        elif shape == 1:
            j = 2
            step = 0
            while total_cells > 0:
                if step % 8 == 0:
                    offsets.add((1, j))
                    total_cells-=1
                    if total_cells > 0:
                        offsets.add((1, -j))
                elif step % 8 == 1:
                    offsets.add((2, j))
                    total_cells-=1
                    if total_cells > 0:
                        offsets.add((2, -j))
                elif step % 8 == 2:
                    offsets.add((3, j))
                    total_cells-=1
                    if total_cells > 0:
                        offsets.add((3, -j))
                    j += 1
                elif step % 8 == 3:
                    offsets.add((1, j))
                    total_cells-=1
                    if total_cells > 0:
                        offsets.add((1, -j))
                elif step % 8 == 4 or step % 8 == 5:
                    offsets.add((0, j))
                    total_cells-=1
                    if total_cells > 0:
                        offsets.add((0, -j))
                    j += 1
                elif step % 8 == 6:
                    offsets.add((0, j))
                    total_cells-=1
                    if total_cells > 0:
                        offsets.add((0, -j))
                elif step % 8 == 7:
                    offsets.add((1, j))
                    total_cells-=1
                    if total_cells > 0:
                        offsets.add((1, -j))
                    j += 1

                step += 1
        return offsets

    def get_center_point(self, current_percept, info) -> int:
        if info: # initialized
            min_x = 100
            for x, y in self.periphery:
                if y == 50:
                    min_x = min(min_x, x)
            
            return (min_x, 50) # 51?
        else:
            return (50, 50) # 51, 51? need to check later

    
    def map_to_coords(self, amoeba_map: npt.NDArray) -> set[Tuple[int, int]]:
        # borrowed from group 2
        return set(list(map(tuple, np.transpose(amoeba_map.nonzero()).tolist())))

    def offset_to_absolute(self, offsets:set[Tuple[int]], center_point:Tuple[int]) -> set[Tuple[int]]:
        absolute_cords = set()
        for offset in offsets:
            absolute_cords.add(((center_point[0] + offset[0]) % MAP_LENGTH, (center_point[1] + offset[1]) % MAP_LENGTH))
        
        return absolute_cords

    def morph(self, offsets:set, center_point:Tuple[int]):
        # adapted from group 2
        cur_ameoba_points = self.map_to_coords(self.amoeba_map)
        desired_ameoba_points = self.offset_to_absolute(offsets, center_point)

        potential_retracts = list(self.periphery.intersection((cur_ameoba_points.difference(desired_ameoba_points))))
        potential_extends = list(self.movable_cells.intersection(desired_ameoba_points.difference(cur_ameoba_points)))

        # Loop through potential extends, searching for a matching retract
        retracts = []
        extends = []
        for potential_extend in [p for p in potential_extends]:
            # Ensure we only move as much as possible given our current metabolism
            if len(extends) >= self.num_available_moves:
                break

            matching_retracts = list(potential_retracts)

            # 1
            matching_retracts.sort(key=lambda p: math.dist(p, potential_extend))

            # 2
            # slows down dramatically...sometimes better/worse score
            '''neighbors = {}
            for retract in matching_retracts:
                get_neighbors = self.find_movable_neighbor(retract[0], retract[1], 
                    self.amoeba_map, self.bacteria)
                neighbors[retract] = len(get_neighbors)

            matching_retracts = list(dict(sorted(neighbors.items(), key=lambda x: x[1])).keys())'''

            for i in range(len(matching_retracts)):
                retract = matching_retracts[i]
                # Matching retract found, add the extend and retract to our lists
                if self.check_move(retracts + [retract], extends + [potential_extend]):
                    retracts.append(retract)
                    potential_retracts.remove(retract)
                    extends.append(potential_extend)
                    potential_extends.remove(potential_extend)
                    break
        
        # TODO: extra bacteria handling??

        return retracts, extends

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

        self.current_size = current_percept.current_size
        self.amoeba_map = current_percept.amoeba_map
        self.periphery = set(current_percept.periphery)
        self.bacteria = current_percept.bacteria
        self.movable_cells = set(current_percept.movable_cells)
        self.num_available_moves = int(np.ceil(self.metabolism * self.current_size))

        # cur_ameoba_points = self.map_to_coords(self.amoeba_map)
        # desired_ameoba_points = self.offset_to_absolute(desired_shape_offsets, self.static_center)

        # potential_retracts = list(self.periphery.intersection((cur_ameoba_points.difference(desired_ameoba_points))))

        

        # if self.turn < 50:
        #     center_point = self.get_center_point(current_percept, 0)
        
        ### PARSE INFO BYTE ###
        info_bin = format(info, '08b')
        info_first_bit = info_bin[0]    # first bit of the info byte
        info_L7_bits = info_bin[1:]     # last 7 bits of the info byte
        info_L7_int = int(info_L7_bits, 2)  # info_L7_int holds int value of last 7 bits (stores coordinate)


        ### GET DESIRED OFFSETS FOR CURRENT MORPH ###
        desired_shape_offsets = self.get_desired_shape()


        ### INCREMENT CENTER POINT PHASE ###
        # move amoeba: x_cord is info_L7_int because initial info_L7_int val is 0, indicating initialization/building phase
        init_phase = info_L7_int == 0
        x_cord = info_L7_int - 1

        # move under these 2 conditions
        # 1: end of initialization phase
        if init_phase:
            x_cord = 50

            if self.in_formation(desired_shape_offsets, [x_cord, 50]):
                init_phase = False
                x_cord = 51
        
        # 2: not in initialization phase, and in formation
        elif self.in_formation(desired_shape_offsets, [x_cord, 50], err=0.2):
            x_cord += 1
            x_cord %= 100


        ### MORPH PHASE ###
        center_point = [x_cord, 50]
        retracts, moves = self.morph(desired_shape_offsets, center_point)

        # catch error (if moves == 0, no move was made, so we should step back until we can move)
        if len(moves) == 0:
            while len(moves) == 0:
                x_cord = ((x_cord + 100) - 1) % 100
                center_point = [x_cord, 50]
                retracts, moves = self.morph(desired_shape_offsets, center_point)
            x_cord = ((x_cord + 100) - 1) % 100


        ### INFO BYTE ###
        # first bit == nothing right now
        # 0 == initialization
        # 1 - 100 => 0 - 99 == x_cord
        if init_phase:
            info_L7_bits = format(0, '07b')
        else:
            info_L7_bits = format(x_cord + 1, '07b')
        
        info_bin = info_first_bit + info_L7_bits
        info = int(info_bin, 2)

        return retracts, moves, info
    

    def in_formation(self, desired_shape_offsets, cur_center, err=0.0) -> bool:
        cur_ameoba_points = self.map_to_coords(self.amoeba_map)
        desired_ameoba_points = self.offset_to_absolute(desired_shape_offsets, cur_center)

        num_potential_retracts = len(self.periphery.intersection((cur_ameoba_points.difference(desired_ameoba_points))))
        num_total_periphery = len(cur_ameoba_points)

        return (num_potential_retracts / num_total_periphery) <= err


    # Adapted from G2 aka from amoeba_game.py
    def check_move(
        self, retracts: List[Tuple[int, int]], extends: List[Tuple[int, int]]
    ) -> bool:
        if not set(retracts).issubset(self.periphery):
            return False

        movable = retracts[:]
        new_periphery = list(self.periphery.difference(set(retracts)))
        for i, j in new_periphery:
            nbr = self.find_movable_neighbor(i, j, self.amoeba_map, self.bacteria)
            for x, y in nbr:
                if (x, y) not in movable:
                    movable.append((x, y))

        if not set(extends).issubset(set(movable)):
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

    def find_movable_cells(self, retract, periphery, amoeba_map, bacteria, mini):
        # sort periphery by y coord
        '''sorted_periphery = sorted(periphery, key=lambda x: x[1])
        lowest_y = sorted_periphery[0][1]
        highest_y = sorted_periphery[-1][1]
        print(lowest_y, highest_y)'''

        movable = []
        new_periphery = list(set(periphery).difference(set(retract)))
        for i, j in new_periphery:
            nbr = self.find_movable_neighbor(i, j, amoeba_map, bacteria)
            for x, y in nbr:
                movable.append((x, y))

        movable += retract

        return movable[:mini]

    def find_movable_neighbor(self, x, y, amoeba_map, bacteria):
        # a cell is on the periphery if it borders (orthogonally) a cell that is not occupied by the amoeba
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
