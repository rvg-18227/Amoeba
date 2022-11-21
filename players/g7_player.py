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

        # retract = [tuple(i) for i in self.rng.choice(current_percept.periphery, replace=False, size=mini)]
        
        farm_is_moving = False
        ys = self.get_y_of_sweep(current_percept.amoeba_map)
        farm_is_moving = len(ys) <= 2 and self.center_is_hollowed(current_percept.amoeba_map)
        ys = ys if len(ys) <= 2 else [50]
        if len(ys) == 1 and ys[0] == 50:
            ys.append(40)#TODO get top of amoeba)
        retract = self.retractable_farm_cells(current_percept.periphery, current_percept.amoeba_map, ys)
        movable = self.moveable_cells(retract, current_percept.periphery, current_percept.amoeba_map,
                                          current_percept.bacteria)
        movable = sorted([cell for cell in movable if not self._is_internal(cell[0], cell[1], current_percept.periphery, current_percept.amoeba_map)], key=lambda x: self._dist_to_center(x[0], x[1]))
        # print("retract", retract)
        movable = movable[:len(retract)]
        info = 0
        return retract, movable, info

    def _dist_to_center(self, x, y):
        return (x - 50) ** 2 + (y - 50) ** 2

    def get_neighbors(self, x, y, amoeba_map):
        neighbors = [(x - 1, y), (x + 1, y), (x, y + 1), (x, y - 1)]
        return [n for n in neighbors if amoeba_map[n[0]][n[1]] == 1]

    def _breaks_amoeba(self, x, y, periphery, amoeba_map):
        # check that all amoeba cells are connected
        queue = [periphery[0] if periphery[0][0] != x and periphery[0][1] != y else periphery[1]]
        copy_amoeba_map = deepcopy(amoeba_map)
        copy_amoeba_map[x][y] = 0
        visited = set()
        while len(queue) > 0:
            cur_x, cur_y = queue.pop(0)
            if(amoeba_map[cur_x][cur_y] == 0):
                print(cur_x, cur_y)
            if (cur_x, cur_y) in visited:
                continue
            visited.add((cur_x, cur_y))
            neighbors = self.get_neighbors(cur_x, cur_y, copy_amoeba_map)
            queue.extend(neighbors)

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
        cutoff_percent = 0.7
        x_start = 50 - self.starting_width // 2
        x_end = 50 + self.starting_width // 2
        y_start = 50 - self.starting_width // 2
        y_end = 50 + self.starting_width // 2
        total_filled = sum([sum(row[x_start:x_end]) for row in amoeba_map[y_start:y_end]])
        return total_filled <= (self.starting_width ** 2) * cutoff_percent

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
            if amoeba_map[50][y] == 1:
                ys.append(y)
            # if sum([amoeba_map[x][y] for x in range(x_start, x_end)]) == self.starting_width:
            #     print(y)
                # ys.append(y)
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
