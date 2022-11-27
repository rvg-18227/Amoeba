import os
import pickle
import numpy as np
import logging
from amoeba_state import AmoebaState
from copy import deepcopy


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
        self.starting_width = int(self.current_size**0.5)

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
        mini = min(5, len(current_percept.periphery) // 2)
        for i, j in current_percept.bacteria:
            current_percept.amoeba_map[i][j] = 1
            current_percept.current_size += 1

        print(self.bottom_quadrant()) 
        
    
    def circular_formation_precomputation(self, current_percept):
        """
            Goal: Precomputation 
            1. define a top arc between 135deg and 45 deg, precompute points across
            2. define a second arc and sweep from 225deg to 315 deg
            3. define a third arc and sweep from 315deg to 45deg
            4. define a fourth arc and sweep from 135deg to 225deg
        """
        raise NotImplementedError
    
    def top_quadrant(self):
        """
            Goal: Precomputation 
            1. define a top arc between 135deg and 45 deg, precompute points across
            2. define a second arc and sweep from 225deg to 315 deg
            3. define a third arc and sweep from 315deg to 45deg
            4. define a fourth arc and sweep from 135deg to 225deg
        """
        quadractic_formation = list()
        x, y = (50,50)
        while (0,0) not in quadractic_formation:
            quadractic_formation.append((x, y))
            quadractic_formation.append((x - 1, y))
            x, y = (x - 1, y - 1)
        quadractic_formation.remove((-1, 0)) # removes (0, -1)
        x, y = (51, 50)
        
        while (99, 1) not in quadractic_formation:
            quadractic_formation.append((x, y))
            quadractic_formation.append((x, y - 1))
            x, y = (x + 1, y - 1)
            print((x, y))
        quadractic_formation.append((99, 0))

        # filling row from left to right
        col = 1
        for row in range(0, 50):
            col = col + 1
            for i in range(col, 100 - col + 1):
                if (row, i) not in quadractic_formation:
                    quadractic_formation.append((row, i))
                else:
                    break
        
        return quadractic_formation

    def rigth_quadrant(self):
        """
            Goal: Precomputation 
            1. define a top arc between 135deg and 45 deg, precompute points across
            2. define a second arc and sweep from 225deg to 315 deg
            3. define a third arc and sweep from 315deg to 45deg
            4. define a fourth arc and sweep from 135deg to 225deg
        """
        quadractic_formation = list()
        x, y = (49, 51)
        while (0,99) not in quadractic_formation:
            quadractic_formation.append((x, y))
            quadractic_formation.append((x - 1, y))
            x, y = (x - 1, y + 1)
        
        x, y = (51, 51)
        
        while (99, 99) not in quadractic_formation:
            quadractic_formation.append((x, y))
            quadractic_formation.append((x + 1, y))
            x, y = (x + 1, y + 1)
        
        quadractic_formation.pop(-1)
        
        # filling row from left to right
        col = 1
        for row in range(99, 50, -1):
            col += 1
            # print((col, row) in quadractic_formation)
            # continue
            for i in range(col, 100 - col + 1):
                if (i, row) not in quadractic_formation:
                    quadractic_formation.append((i, row))
                else:
                    break
        
        return quadractic_formation

    def bottom_quadrant(self):
        """
            Goal: Precomputation 
            1. define a top arc between 135deg and 45 deg, precompute points across
            2. define a second arc and sweep from 225deg to 315 deg
            3. define a third arc and sweep from 315deg to 45deg
            4. define a fourth arc and sweep from 135deg to 225deg
        """
        # filling row from left to right
        
        quadractic_formation = list()
        
        # end_row = 51
        break_row = 99
        start_row = 1
        for col in range(99,51,-1):
            for row in range(start_row, 99):
                if row == break_row:
                    break
                else:
                    quadractic_formation.append((col, row))
            break_row = break_row - 1
            start_row = start_row + 1
        return quadractic_formation

    def vertical_point(self, current_percept, retractable):
        
        move = []
        # upward-left move
        top_reached = self.min_distance_to(0, 0, current_percept.periphery) == (0, 0)
        while retractable and not top_reached:
            x, y = retractable[0]
            x_start, y_start = self.min_distance_to(0, 0, current_percept.periphery)
            move.append(x_start - 1, y_start - 1)
            top_reached = x_start - 1 < 1 or y_start -1 < 1
        
        # downward-from-left move




    def min_distance_to(self, x_target, y_target, periphery):
        """
        x_target, y_target:
            => (0,0) = 0
            => (0,1)
        """
        move_bit = {}
        min_x, min_y = float('inf'), float('inf')
        for (x, y) in periphery:
            distance = abs(x_target - x) + abs(y_target - y)
            if distance < abs(min_x - x_target) + abs(min_y - y_target):
                min_x, min_y = x, y
        return (min_x, min_y)
    # def immobile_bacteria(self, current_percept):
    # # To do need to ensure wall bacteria are considered
    #     immobile_bateria = []
    #     for (x, y) in current_percept.bacteria:
    #         movable_nei = self.find_movable_neighbor(x, y, current_percept.amoeba_map, current_percept.bacteria)
    #         if (x, y) not in movable_nei
    #         nei_in_periphery = [i in current_percept.periphery for i in movable_nei]
    #         if len(movable_nei) == 3 and:

    # def find_neighbor_bacteria(self, x, y, current_percept):
    #     nei_bacteria = []
    #     for (x, y) in current_percept.bacteria:
    #         if current_percept.amoeba_map[x][(y - 1) %100] in current_percept.bacteria:
    #             nei_bacteria.append((x, (y - 1) %100))
    #         if current_percept.amoeba_map[(x - 1) % 100][y] in current_percept.bacteria:
    #             nei_bacteria.append(((x - 1) % 100, y))
    #         if current_percept.amoba_map[x][(y + 1) % 100] in current_percept.bacteria:
    
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

    def retractable_y_down(self, ys, periphery, amoeba_map):
        copy_amoeba_map = deepcopy(amoeba_map)
        retract = []
        ideal_ys = [y+1 for y in ys]
        for (x, y) in periphery:
            if y not in ideal_ys and self._is_internal(x, y, periphery, copy_amoeba_map):
                copy_amoeba_map[x][y] = 0
                retract.append((x, y))

        return retract

    def move_y_down(self, ys, periphery, amoeba_map, movable):
        to_move = []
        ideal_ys = [y+1 for y in ys]

        # for (x, y) in movable:
        #     if y in ideal_ys and amoeba_map[x][y-1] == 1:
        #         to_move.append((x, y))

        for y in ideal_ys:
            for x in range(100):
                if amoeba_map[x][y] == 0 and (x, y) in movable:
                    to_move.append((x, y))

        return to_move

    def _dist_to_center(self, x, y):
        return (x - 50) ** 2 + (y - 50) ** 2

    def get_neighbors(self, x, y, amoeba_map):
        neighbors = [(x - 1, y), (x + 1, y), (x, y + 1), (x, y - 1)]
        return [n for n in neighbors if amoeba_map[n[0]][n[1]] == 1]

    def _breaks_amoeba(self, x, y, periphery, amoeba_map):
        # check that all amoeba cells are connected
        isolated_neighbors = self.get_neighbors(x, y, amoeba_map)
        queue = [isolated_neighbors[0]]
        copy_amoeba_map = deepcopy(amoeba_map)
        copy_amoeba_map[x][y] = 0
        visited = set()
        to_visit_isolated_connections = set(isolated_neighbors)
        while len(queue) > 0:
            cur_x, cur_y = queue.pop(0)
            # if(copy_amoeba_map[cur_x][cur_y] == 0):
            #     print(cur_x, cur_y) #TODO: fix, this still prints
            if (cur_x, cur_y) in visited:
                continue
            if (cur_x, cur_y) in to_visit_isolated_connections:
                to_visit_isolated_connections.remove((cur_x, cur_y))
            visited.add((cur_x, cur_y))
            neighbors = self.get_neighbors(cur_x, cur_y, copy_amoeba_map)
            queue.extend(neighbors)
            if len(to_visit_isolated_connections) == 0:
                return False

        return len(visited - set([(i, j) for i, row in enumerate(copy_amoeba_map) for j, cell in enumerate(row) if cell != 0])) > 0 or len(visited) != len(set([(i, j) for i, row in enumerate(copy_amoeba_map) for j, cell in enumerate(row) if cell != 0]))
        
    def _is_internal(self, x, y, periphery, amoeba_map):

        north = [(x, i) in periphery for i in range(y+1, 101)]
        south = [(x, i) in periphery for i in range(y-1, -1, -1)]
        east = [(i, y) in periphery for i in range(x+1, 101)]
        west = [(i, y) in periphery for i in range(x-1, -1, -1)]
        return all([any(north), any(south), any(east), any(west)]) and not self._breaks_amoeba(x, y,  periphery, amoeba_map)

    def retractable_farm_cells(self, periphery, amoeba_map, ys):
        copy_amoeba_map = deepcopy(amoeba_map)
        possible_incision_points = [(x, y) for x, y in periphery if (x == 49 or x == 51 or x == 50) and y not in ys]
        incision_points = []
        internal_points = []

        for x, y in possible_incision_points:
            if self._breaks_amoeba(x, y, periphery, copy_amoeba_map):
                continue
            copy_amoeba_map[x][y] = 0
            incision_points.append((x, y))
        periphery = sorted(periphery, key=lambda x: self._dist_to_center(x[0], x[1]))
        periphery = [p for p in periphery if p not in incision_points]
        for x, y in periphery:
            if self._is_internal(x, y, periphery, copy_amoeba_map) and y not in ys:
                copy_amoeba_map[x][y] = 0
                internal_points.append((x, y))

        internal_points = sorted(internal_points, key=lambda x: self._dist_to_center(x[0], x[1]))
        # print("incision_points", incision_points)
        # print("internal_points", internal_points)
        return incision_points + internal_points

    def center_is_hollowed(self, amoeba_map):
        n_rows = 2
        x_start = 50 - self.starting_width // 2
        x_end = 50 + self.starting_width // 2
        y_start = 50 - self.starting_width // 2
        y_end = 50 + self.starting_width // 2
        total_filled = sum([sum(row[x_start:x_end]) for row in amoeba_map[y_start:y_end]])
        return total_filled <= self.starting_width * n_rows + (self.starting_width*0.5)

    def get_y_of_sweep(self, amoeba_map):
        # only start checking when orig size is cleared out (except 3 lines percent)
        # for that width, find the continuous line of 1s
        #move down if you can move all, otherwise half?
        # always move down, TODO: how to detect when at the bottom
        # how to detect when to start?
        # maybe when 1 reaches 50, start another
        # always move down then clear
        x_start = 50 - self.starting_width // 2
        x_end = 50 + self.starting_width // 2
        ys = []
        for y in range(0, 100):
            if amoeba_map[50][y] == 1 or amoeba_map[49][y] == 1 or amoeba_map[51][y] == 1:
                ys.append(y)
            # if sum([amoeba_map[x][y] for x in range(x_start, x_end)]) == x_end - x_start:
            #     ys.append(y)
        return ys

    def moveable_cells(self, retract, periphery, amoeba_map, bacteria):
        movable = []
        new_periphery = list(set(periphery).difference(set(retract)))
        for i, j in new_periphery:
            nbr = self.find_movable_neighbor(i, j, amoeba_map, bacteria)
            for x, y in nbr:
                if (x, y) not in movable:
                    movable.append((x, y))
        return movable

