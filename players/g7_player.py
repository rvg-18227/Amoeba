import os
import pickle
import numpy as np
import logging
from matplotlib import pyplot as plt
from copy import deepcopy

# ---------------------------------------------------------------------------- #
#                               Helper Functions                               #
# ---------------------------------------------------------------------------- #

def wrap_point(x, y):
    '''
    Wrap the point around the grid
    '''
    return (x % 100, y % 100)

def get_neighbors(x, y, amoeba_map):
    neighbors = [wrap_point(x - 1, y), wrap_point(x + 1, y), wrap_point(x, y + 1), wrap_point(x, y - 1)]
    return [n for n in neighbors if amoeba_map[n[0]][n[1]] == 1]

def breaks_amoeba(point, amoeba_map):
    '''
    Returns whether or not the given point breaks the amoeba

    :param point: The point to check
    :param amoeba_map: The amoeba map
    :return: True if the point breaks the amoeba, False otherwise
    '''
    x, y = point
    # check that all amoeba cells are connected
    isolated_neighbors = get_neighbors(x, y, amoeba_map)
    queue = [isolated_neighbors[0]] #todso heapq??
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
        neighbors = get_neighbors(cur_x, cur_y, copy_amoeba_map)
        queue.extend(neighbors)
        if len(to_visit_isolated_connections) == 0:
            return False

    return len(visited - set([(i, j) for i, row in enumerate(copy_amoeba_map) for j, cell in enumerate(row) if cell != 0])) > 0 or len(visited) != len(set([(i, j) for i, row in enumerate(copy_amoeba_map) for j, cell in enumerate(row) if cell != 0]))

# ---------------------------------------------------------------------------- #
#                               Formation Classes                              #
# ---------------------------------------------------------------------------- #

class Formation:
    def __init__(self):
        self.phase = 0

    def update(self, phase):
        '''
        Update the formation based on the current info
        Must be called every turn to maintain the rules of the game
        '''
        self.phase = phase

    def get_all_retractable_points(self, goalFormation, state):
        '''
        Returns a list of all points that can be retracted that won't affect the goal formation
        
        :param goalFormation: The goal formation
        :param state: The current state
        :return: A list of all points that can be retracted
        '''
        canRetract = []
        amoebaMap = state.amoeba_map
        periphery = state.periphery
        for point in periphery:
            if point not in goalFormation:
                canRetract.append(point)

        return canRetract

    def get_moveable_points(self, moveablePoints, goalFormation, state):
        '''
        Returns a list of all points to move to to achieve the goal formation

        :param moveablePoints: The points that can be moved to
        :param goalFormation: The goal formation
        :param state: The current state
        :return: A list of all points to move to
        '''
        toMove = []
        amoebaMap = state.amoeba_map
        periphery = state.periphery
        for point in moveablePoints:
            if point in goalFormation:
                toMove.append(point)
        return toMove

    def get_n_moves(self, allRetracable, pointsToMoveTo, state, n_cells_can_move):
        ''' 
        Returns the points to retract and move so that len(pointsToMoveTo) == len(pointsToRetract)

        :param allRetracable: A list of all points that can be retracted
        :param pointsToMoveTo: A list of all points that need to be moved to
        :param state: The current state
        :param n_cells_can_move: The number of cells that can move based on the metabolism
        :return: A tuple of the points to retract and the points to move to
        '''
        #TODO: do this smartly
        amoebaMapCopy = deepcopy(state.amoeba_map)
        allValidRetracable = []
        for i, point in enumerate(allRetracable):
            if not breaks_amoeba(point, amoebaMapCopy):
                allValidRetracable.append(point)
                amoebaMapCopy[point[0]][point[1]] = 0

        allValidRetracable = allValidRetracable[:n_cells_can_move]
        pointsToMoveTo = pointsToMoveTo[:n_cells_can_move]

        if len(allValidRetracable) > len(pointsToMoveTo):
            return allValidRetracable[:len(pointsToMoveTo)], pointsToMoveTo
        elif len(allValidRetracable) < len(pointsToMoveTo):
            return allValidRetracable, pointsToMoveTo[:len(allValidRetracable)]
        else:
            return allValidRetracable, pointsToMoveTo

    def get_next_formation_points(self, state):
        '''
        Returns the next formation points

        :param state: The current state
        :return: A list of the next formation points
        '''
        raise NotImplementedError("Must be implemented by subclass")

class RakeFormation(Formation):

    def get_next_formation_points(self, state):
        nCells = sum([sum(row) for row in state.amoeba_map])
        amoebaMap = state.amoeba_map
        #TODO what to do when wrap all the way around?? phase 2, 3, 4...
        #TODO: change ordering of moveable points
        #TODO: change ordering of retractable points, maybe based on distance to center of formation?
        if self.phase == 0:#go forward
            xOffset, yOffset = self._get_current_xy(amoebaMap)
            return self._get_formation(xOffset+1, yOffset, state, nCells)
        elif self.phase == 1:# move down (teeth), not implemented!!!!!
            xOffset, yOffset = self._get_current_xy(amoebaMap)
            #TODO: merge with next move for those that are already in position?
            # for each x, if they all in the right place, move to the next x
            return self._get_formation(xOffset+1, yOffset, state, nCells)
        raise NotImplementedError

    def _get_current_xy(self, amoebaMap):
        '''
        Returns the current x and y offsets of the amoeba
        Assumes already in starting formation or moving

        :param amoebaMap: The amoeba map
        :return: A tuple of the current x and y
        '''
        #TODO: make this more accurate
        xs_with_cells = [i for i, row in enumerate(amoebaMap) for j, cell in enumerate(row) if cell != 0]
        xOffset = min(xs_with_cells)
        if 99 in xs_with_cells:
            for i in reversed(range(100)):
                if i in xs_with_cells:
                    xOffset = i
                else:
                    break

        ys_with_cells = [j for i, row in enumerate(amoebaMap) for j, cell in enumerate(row) if cell != 0]
        yOffset = min(ys_with_cells)
        if 99 in ys_with_cells:
            for i in reversed(range(100)):
                if i in ys_with_cells:
                    yOffset = i
                else:
                    break

        return xOffset, yOffset

    def _get_formation(self, x, yOffset, state, nCells):
        '''
        Returns the formation points for the given x and yOffset

        :param x: The x coordinate
        :param yOffset: The yOffset
        :param state: The current state
        :param nCells: The number of cells
        :return: A list of the formation points
        '''
        nChunks = nCells // 7
        formation = []
        for i in range(nChunks):
            formation += self._generate_chunk(x, yOffset)
            yOffset += 3
    
        # Add extra cells
        formation += self._generate_chunk(x, yOffset)[:nCells % 7]

        return formation

    def _generate_chunk(self, xOffset, yOffset):
        '''
        Generates a chunk of the formation
        |1|2|3|
        |4|5|
        |6|7|

        :param xOffset: The xOffset
        :param yOffset: The yOffset
        :return: A list of the chunk points
        '''
        chunk = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (0, 2), (1, 2)]
        return [wrap_point(x + xOffset, y + yOffset) for x, y in chunk]

class SpaceCurveFormation(Formation):

    def get_next_formation_points(self, state):
        raise NotImplementedError

# ---------------------------------------------------------------------------- #
#                               Main Player Class                              #
# ---------------------------------------------------------------------------- #

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

        self.formation = RakeFormation()


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
        n_cells_can_move = int(self.metabolism * self.current_size)
        

        nAdjacentBacteria = 0
        for i, j in current_percept.bacteria:
            nAdjacentBacteria += 1
            current_percept.amoeba_map[i][j] = 1

        phase, count, info = self.decode_info(info)
        # update byte of info
        BACTERIA_RATIO = 0.001 #TODO, maybe based on size of total amoeba and size of periphery??
        percent_bacteria = nAdjacentBacteria / len(current_percept.periphery)
        # print("percent_bacteria", percent_bacteria)
        count += 1 if percent_bacteria > BACTERIA_RATIO else -1
        count = max(0, count)
        count = min(7, count)

        # if high density, use space filling curve
        # if count >= 6:
        #     self.formation == SpaceCurveFormation()

        self.formation.update(phase)
        goalFormation = self.formation.get_next_formation_points(current_percept)

        allRetractable = self.formation.get_all_retractable_points(goalFormation, current_percept)

        allMovable = self.find_movable_cells(allRetractable, current_percept.periphery, current_percept.amoeba_map, current_percept.bacteria)
        toMove = self.formation.get_moveable_points(allMovable, goalFormation, current_percept)

        retract, movable = self.formation.get_n_moves(allRetractable, toMove, current_percept, n_cells_can_move)
        
        if len(retract) == 0 and len(movable) == 0 and phase == 0:
            phase = 1
            return self.move(last_percept, current_percept, self.encode_info(1, count, info))

        info = self.encode_info(phase, count, info)

        # print("phase", phase)
        return retract, movable, info

    def find_movable_cells(self, retract, periphery, amoeba_map, bacteria):
        '''
        Finds the cells that can be moved to given the retract
        :param retract: list of cells to retract
        :param periphery: list of cells on the periphery
        :param amoeba_map: map of the amoeba
        :param bacteria: list of bacteria
        :return: list of cells that can be moved
        '''
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

    def encode_info(self, phase: int, count: int, info: int) -> int:
        """Encode the information to be sent
            Args:
                phase (int): 2 bits for the current phase of the amoeba
                count (int): 3 bits for the current count of the running density
                info (int): 3 bits other info, still TODO
            Returns:
                int: the encoded information as an int
        """
        assert phase < 4
        info = 0
        info_str = "{:02b}{:03b}{:03b}".format(phase, count, info)

        return int(info_str, 2)

    def decode_info(self, info: int) -> (int, int, int):
        """Decode the information received
            Args:
                info (int): the information received
            Returns:
                Tuple[int, int, int]: phase, count, info, the decoded information as a tuple
        """
        info_str = "{0:b}".format(info).zfill(8)

        return int(info_str[0:2], 2), int(info_str[2:5], 2), int(info_str[5:8], 2)



# UNIT TESTS
class TestAmoeba():
    def __init__(self):
        self.amoeba_map = [[0 for _ in range(100)] for _ in range(100)]
        self.size = 10
        startX = 50 - self.size // 2
        startY = 50 - self.size // 2
        for i in range(startX, startX + self.size):
            for j in range(startY, startY + self.size):
                self.amoeba_map[i][j] = 1
def show_formation_test():
    formation = RakeFormation()
    formation.update(1)
    points = formation.get_next_formation_points(TestAmoeba())
    x, y = zip(*points)
    plt.scatter(x, y)
    plt.xticks(range(min(x), max(y)+1))
    plt.savefig("formation.png")

if __name__ == '__main__':
    show_formation_test()