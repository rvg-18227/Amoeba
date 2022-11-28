import os
import pickle
import numpy as np
import logging
from matplotlib import pyplot as plt
from copy import deepcopy

# ---------------------------------------------------------------------------- #
#                               Helper Functions                               #
# ---------------------------------------------------------------------------- #
def plot_points_helper(points):
    '''Visualize points'''
    x, y = zip(*points)
    plt.scatter(x, y)
    plt.xticks(range(min(x), max(y)+1))
    plt.savefig("formation.png")

def wrapped_range(start, end, step=1):
    '''
    Returns a range that wraps around the grid
    '''
    if start < end:
        return list(range(start, end, step))
    else:
        return list(range(start, 100, step)) + list(range(0, end, step))

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

def remove_duplicates(points):
    validPoints = []
    for i, point in enumerate(points):
        if point not in validPoints:
            validPoints.append(point)
    return validPoints
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
        # TODO: make this work? moveablePoints.sort(key=lambda point: self._dist_btwn_points(point, self._center_of_formation(goalFormation)))
        for point in moveablePoints:
            if point in goalFormation:
                toMove.append(point)
        return toMove

    def _dist_btwn_points(self, point1, point2):
        '''
        Returns the distance between two points

        :param point1: The first point
        :param point2: The second point
        :return: The distance between the two points
        '''
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def _center_of_formation(self, formation):
        '''
        Returns the center of the formation

        :param formation: The formation
        :return: The center of the formation
        '''
        return (sum([point[0] for point in formation]) / len(formation), sum([point[1] for point in formation]) / len(formation))

    def get_n_moves(self, allRetracable, pointsToMoveTo, state, n_cells_can_move):
        ''' 
        Returns the points to retract and move so that len(pointsToMoveTo) == len(pointsToRetract)

        :param allRetracable: A list of all points that can be retracted
        :param pointsToMoveTo: A list of all points that need to be moved to
        :param state: The current state
        :param n_cells_can_move: The number of cells that can move based on the metabolism
        :return: A tuple of the points to retract and the points to move to
        '''
        amoebaMapCopy = deepcopy(state.amoeba_map)
        moveDups = [point for point in pointsToMoveTo if pointsToMoveTo.count(point) > 1]
        validPointsToMoveTo = [point for i, point in enumerate(pointsToMoveTo) if point not in moveDups and pointsToMoveTo.index(point) == i]
        allValidRetracable = []

        #make n passes? does this work
        for j in range(2):
            for i, point in enumerate(allRetracable):
                if point not in allValidRetracable and not breaks_amoeba(point, amoebaMapCopy):
                    allValidRetracable.append(point)
                    amoebaMapCopy[point[0]][point[1]] = 0

        allValidRetracable = allValidRetracable[:n_cells_can_move]
        validPointsToMoveTo = validPointsToMoveTo[:n_cells_can_move]

        if len(allValidRetracable) > len(validPointsToMoveTo):
            return allValidRetracable[:len(validPointsToMoveTo)], validPointsToMoveTo
        elif len(allValidRetracable) < len(validPointsToMoveTo):
            return allValidRetracable, validPointsToMoveTo[:len(allValidRetracable)]
        else:
            return allValidRetracable, validPointsToMoveTo

    def get_next_formation_points(self, state):
        '''
        Returns the next formation points

        :param state: The current state
        :return: A list of the next formation points
        '''
        raise NotImplementedError("Must be implemented by subclass")

    def get_phase(self, phase, state, retract, movable):
        '''
        Returns the current phase
        
        :param phase: The current phase
        :param state: The current state
        :param retract: The points to retract
        :param movable: The points to move to
        :return: The current phase
        '''
        raise NotImplementedError("Must be implemented by subclass")

class RakeFormation(Formation):
    def __init__(self):
        self.allPoints = []
        for y in range(50):
            self.allPoints.extend([(x, 50-y) for x in range(100)])
            self.allPoints.extend([(x, 50+y) for x in range(100)])

    def get_phase(self, phase, state, retract, movable):
        nCells = sum([sum(row) for row in state.amoeba_map])
        xStart, xEnd, yStart, yEnd = self._get_current_xy(state.amoeba_map)
        emptyCols = self._get_empty_cols_between(xStart, xEnd, state.amoeba_map)
        if phase == 0:
            phase = 0
        elif phase == 1:
            phase = 0
        if nCells > 466+6 and (phase == 1 or phase == 0):
            return 2
        elif phase == 2 and len(emptyCols) >= 90:
            return 3
        elif phase == 3 and  len(emptyCols) <= 6:
            return 2
        return phase

    def get_next_formation_points(self, state):
        nCells = sum([sum(row) for row in state.amoeba_map])
        amoebaMap = state.amoeba_map
        amoebaPoints = [(i, j) for i, row in enumerate(amoebaMap) for j, cell in enumerate(row) if cell == 1]

        #TODO: change ordering of moveable points
        #TODO: change ordering of retractable points, maybe based on distance to center of formation? mostly matters at the beginning
        if self.phase == 0:
            xStart, xEnd, yStart, yEnd = self._get_current_xy(amoebaMap)
            xOffset, yOffset = xStart, yStart

            previousPoints = self._get_formation(xOffset, yOffset, state, nCells)\
                + [(xOffset+i, 50) for i in range(0, 8)]\
                + self._get_formation(xOffset+8, yOffset, state, nCells)

            previousPoints = remove_duplicates(previousPoints)[:nCells]
            totalCorrectPoints = sum([1 for point in previousPoints if point in amoebaPoints])
            # print(xStart, xEnd, yStart, yEnd)
            # print("totalCorrectPoints: ", totalCorrectPoints)
            # print(len(previousPoints))
            if totalCorrectPoints < len(previousPoints)*0.99:#
                # print("Using prev formation")
                return previousPoints

            idealPoints = self._get_formation(xOffset+1, yOffset, state, nCells)\
                + [(xOffset+i, 50) for i in range(1, 9)]\
                + self._get_formation(xOffset+9, yOffset, state, nCells)

            idealPoints = remove_duplicates(idealPoints)
            return idealPoints
        elif self.phase == 1:
            xStart, xEnd, yStart, yEnd = self._get_current_xy(amoebaMap)
            xOffset, yOffset = xStart, yStart

            previousPoints = self._get_formation(xOffset, yOffset, state, nCells)\
                + [(xOffset+i, 50) for i in range(0, 8)]\
                + self._get_formation(xOffset+8, yOffset, state, nCells)

            previousPoints = remove_duplicates(previousPoints)[:nCells]
            totalCorrectPoints = sum([1 for point in previousPoints if point in amoebaPoints])
            # print(xStart, xEnd, yStart, yEnd)
            # print("totalCorrectPoints: ", totalCorrectPoints)
            # print(len(previousPoints))
            if totalCorrectPoints < len(previousPoints)*0.99:#
                # print("Using prev formation")
                return previousPoints

            idealPoints = self._get_formation(xOffset+1, yOffset, state, nCells)\
                + [(xOffset+i, 50) for i in range(1, 9)]\
                + self._get_formation(xOffset+9, yOffset, state, nCells)

            idealPoints = remove_duplicates(idealPoints)
            return idealPoints
        elif self.phase == 2:
            xStart, xEnd, yStart, yEnd = self._get_current_xy(amoebaMap)
            xOffset, yOffset = xStart, yStart #self._get_midpoint(yStart, yEnd)

            previousPoints = self._get_formation(xStart, yOffset, state, nCells)\
                    + [(i, 50) for i in wrapped_range(xStart, xEnd-2)]\
                    + self._get_formation(xEnd-2, yOffset, state, nCells)\
                    + [(i, 50) for i in wrapped_range(0, 100)]\

            previousPoints = remove_duplicates(previousPoints)[:nCells]
            totalCorrectPoints = sum([1 for point in previousPoints if point in amoebaPoints])
            # print(xStart, xEnd, yStart, yEnd)
            # print("totalCorrectPoints: ", totalCorrectPoints)
            # print(len(previousPoints))
            if totalCorrectPoints < len(previousPoints)*0.99:#
                # print("Using prev formation")
                previousPoints += self.allPoints
                previousPoints = remove_duplicates(previousPoints)[:nCells]
                return previousPoints
            idealPoints = self._get_formation(xStart-1, yOffset, state, nCells)\
                    + [(i, 50) for i in wrapped_range(xStart, xEnd-1)]\
                    + self._get_formation(xEnd-1, yOffset, state, nCells)\
                    + [(i, 50) for i in wrapped_range(0, 100)]\
                    + self.allPoints
            return idealPoints
        elif self.phase == 3:
            xStart, xEnd, yStart, yEnd = self._get_current_xy(amoebaMap)
            xOffset, yOffset = xStart, yStart #self._get_midpoint(yStart, yEnd)

            previousPoints = self._get_formation(xStart, yOffset, state, nCells)\
                    + [(i, 50) for i in wrapped_range(xStart, xEnd-2)]\
                    + self._get_formation(xEnd-2, yOffset, state, nCells)\
                    + [(i, 50) for i in wrapped_range(0, 100)]\

            previousPoints = remove_duplicates(previousPoints)[:nCells]
            totalCorrectPoints = sum([1 for point in previousPoints if point in amoebaPoints])
            # print(xStart, xEnd, yStart, yEnd)
            # print("totalCorrectPoints: ", totalCorrectPoints)
            # print(len(previousPoints))
            if totalCorrectPoints < len(previousPoints)*0.99:
                #TODO dont include all points in prevPoint and this calculation
                # print("Using prev formation")
                previousPoints += self.allPoints
                previousPoints = remove_duplicates(previousPoints)[:nCells]
                return previousPoints

            idealPoints = self._get_formation(xStart+1, yOffset, state, nCells)\
                    + [(i, 50) for i in wrapped_range(xStart+1, xEnd-3)]\
                    + self._get_formation(xEnd-3, yOffset, state, nCells)\
                    + [(i, 50) for i in wrapped_range(0, 100)]\
                    + self.allPoints

            return idealPoints
            
            return idealPoints


        # Can have 4 phases (we use 2 bits of info)
        # Phase 0: get into formation/go forward
        # Phase 1: move down 1
        # Phase 2: 2 lines, move outwards
        # Phase 3: 2 lines move inwards
        raise NotImplementedError

    def _get_midpoint(self, start, end):
        if end < start:
            #TODO: 44 -> 3 = midpt of (100-44 + 3-0)//2 = 28
            return (100 - start + end - 0) // 2
        return (start + end) // 2

    def _get_current_xy(self, amoebaMap):
        '''
        Returns the current x and y offsets of the amoeba
        Assumes already in starting formation or moving

        :param amoebaMap: The amoeba map
        :return: A tuple of the start and end x and y
        '''
        #TODO change this all to be based on num cells in that row/col to account for random bacteria additions?
        # calculate n Cells then use that to calculate expected len of amoeba, use that to get the start end cols
        rowLens = [sum(amoebaMap[i]) for i in range(100)]
        cuttoff = max(rowLens) * 0.9
        rowsWithEntireLength = [x for x in range(100) if sum(amoebaMap[x]) >= cuttoff]

        cell_xs = [i for i, row in enumerate(amoebaMap) for j, cell in enumerate(row) if cell != 0]

        contiguousX = []
        cell_xs = sorted(list(set(cell_xs)))
        i = min(cell_xs)
        for x in cell_xs:
            if x != i:
                contiguousX = cell_xs[i:] + contiguousX
                break
            contiguousX.append(x)
            i += 1

        cell_ys = [j for i, row in enumerate(amoebaMap) for j, cell in enumerate(row) if cell != 0]
        contiguousY = []
        cell_ys = sorted(list(set(cell_ys)))
        i = min(cell_ys)
        for y in cell_ys:
            if y != i:
                contiguousY = cell_ys[i:] + contiguousY
                break
            contiguousY.append(y)
            i += 1
        contiguousXCopy = deepcopy(contiguousX)
        for x in contiguousXCopy:
            if x not in rowsWithEntireLength:
                contiguousX.remove(x)
            else:
                break
        contiguousXCopy = deepcopy(list(reversed(contiguousX)))
        for x in contiguousXCopy:
            if x not in rowsWithEntireLength:
                contiguousX.remove(x)
            else:
                break
        contiguousX.append((contiguousX[-1] + 1)%100)


        return contiguousX[0], contiguousX[-1], contiguousY[0], contiguousY[-1]

    def _get_empty_cols_between(self, start, end, amoebaMap):
        '''
        Returns the empty cols between the start and end

        :param start: The start cols (inclusive)
        :param end: The end cols (inclusive)
        :param amoebaMap: The amoeba map
        :return: list of indices of empty cols
        '''
        nCells = sum([sum(row) for row in amoebaMap])
        expectedLen = min(100, (3 * min(nCells // 7, 33)) % 100)
        emptyCols = []

        for i in wrapped_range(start, end+1):
            if sum(amoebaMap[i]) <= (3 * expectedLen/4):
                emptyCols.append(i)
        if len(emptyCols) == 0:
            print("no empty cols")
            return []

        continuousSeqs = {}
        for i in range(len(emptyCols)):
            if i == 0:
                continuousSeqs[emptyCols[i]] = 1
            else:
                if emptyCols[i] % 100 == (emptyCols[i-1] + 1) % 100:
                    continuousSeqs[emptyCols[i]] = continuousSeqs[emptyCols[i-1]] + 1
                else:
                    continuousSeqs[emptyCols[i]] = 1
                    
        longestSeq = []
        maxVal = max(continuousSeqs.values())
        if maxVal <= 1:
            print("No continuous seqs")
            return []
        idxOfMax = [k for k, v in continuousSeqs.items() if v == maxVal]
        modifiedIdxs = list(reversed(continuousSeqs.keys()))
        modifiedIdxs = modifiedIdxs[modifiedIdxs.index(idxOfMax[0]):]

        for i in modifiedIdxs:
            if i in continuousSeqs:
                longestSeq.insert(0, i)
            if continuousSeqs[i] == 1:
                break

        return longestSeq

    def _get_formation(self, x, yOffset, state, nCells):
        '''
        Returns the formation points for the given x and yOffset

        :param x: The x coordinate
        :param yOffset: The yOffset
        :param state: The current state
        :param nCells: The number of cells
        :return: A list of the formation points
        '''
        nChunks = min(nCells // 7, 33)
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

    def __init__(self):
        self.all_points = [(49, 50), (50, 50), (50, 49), (49, 49), (48, 50), (48, 49), (49, 51), (49, 48), (50, 51), (50, 48), (51, 50), (51, 49), (51, 51), (51, 48), (48, 48), (48, 51), (52, 51), (51, 47), (47, 48), (48, 52), (53, 51), (51, 46), (46, 48), (48, 53), (54, 51), (51, 45), (45, 48), (48, 54), (54, 52), (52, 45), (45, 47), (47, 54), (54, 53), (53, 45), (45, 46), (46, 54), (54, 54), (54, 45), (45, 45), (45, 54), (53, 54), (54, 46), (46, 45), (45, 53), (52, 54), (54, 47), (47, 45), (45, 52), (51, 54), (54, 48), (48, 45), (45, 51), (51, 55), (55, 48), (48, 44), (44, 51), (51, 56), (56, 48), (48, 43), (43, 51), (51, 57), (57, 48), (48, 42), (42, 51), (51, 58), (58, 48), (48, 41), (41, 51), (51, 59), (59, 48), (48, 40), (40, 51), (51, 60), (60, 48), (48, 39), (39, 51), (52, 60), (60, 47), (47, 39), (39, 52), (53, 60), (60, 46), (46, 39), (39, 53), (54, 60), (60, 45), (45, 39), (39, 54), (54, 59), (59, 45), (45, 40), (40, 54), (54, 58), (58, 45), (45, 41), (41, 54), (54, 57), (57, 45), (45, 42), (42, 54), (55, 57), (57, 44), (44, 42), (42, 55), (56, 57), (57, 43), (43, 42), (42, 56), (57, 57), (57, 42), (42, 42), (42, 57), (57, 58), (58, 42), (42, 41), (41, 57), (57, 59), (59, 42), (42, 40), (40, 57), (57, 60), (60, 42), (42, 39), (39, 57), (58, 60), (60, 41), (41, 39), (39, 58), (59, 60), (60, 40), (40, 39), (39, 59), (60, 60), (60, 39), (39, 39), (39, 60), (60, 59), (59, 39), (39, 40), (40, 60), (60, 58), (58, 39), (39, 41), (41, 60), (60, 57), (57, 39), (39, 42), (42, 60), (60, 56), (56, 39), (39, 43), (43, 60), (60, 55), (55, 39), (39, 44), (44, 60), (60, 54), (54, 39), (39, 45), (45, 60), (59, 54), (54, 40), (40, 45), (45, 59), (58, 54), (54, 41), (41, 45), (45, 58), (57, 54), (54, 42), (42, 45), (45, 57), (57, 53), (53, 42), (42, 46), (46, 57), (57, 52), (52, 42), (42, 47), (47, 57), (57, 51), (51, 42), (42, 48), (48, 57), (58, 51), (51, 41), (41, 48), (48, 58), (59, 51), (51, 40), (40, 48), (48, 59), (60, 51), (51, 39), (39, 48), (48, 60), (61, 51), (51, 38), (38, 48), (48, 61), (62, 51), (51, 37), (37, 48), (48, 62), (63, 51), (51, 36), (36, 48), (48, 63), (63, 52), (52, 36), (36, 47), (47, 63), (63, 53), (53, 36), (36, 46), (46, 63), (63, 54), (54, 36), (36, 45), (45, 63), (64, 54), (54, 35), (35, 45), (45, 64), (65, 54), (54, 34), (34, 45), (45, 65), (66, 54), (54, 33), (33, 45), (45, 66), (66, 53), (53, 33), (33, 46), (46, 66), (66, 52), (52, 33), (33, 47), (47, 66), (66, 51), (51, 33), (33, 48), (48, 66), (67, 51), (51, 32), (32, 48), (48, 67), (68, 51), (51, 31), (31, 48), (48, 68), (69, 51), (51, 30), (30, 48), (48, 69), (70, 51), (51, 29), (29, 48), (48, 70), (71, 51), (51, 28), (28, 48), (48, 71), (72, 51), (51, 27), (27, 48), (48, 72), (72, 52), (52, 27), (27, 47), (47, 72), (72, 53), (53, 27), (27, 46), (46, 72), (72, 54), (54, 27), (27, 45), (45, 72), (71, 54), (54, 28), (28, 45), (45, 71), (70, 54), (54, 29), (29, 45), (45, 70), (69, 54), (54, 30), (30, 45), (45, 69), (69, 55), (55, 30), (30, 44), (44, 69), (69, 56), (56, 30), (30, 43), (43, 69), (69, 57), (57, 30), (30, 42), (42, 69), (70, 57), (57, 29), (29, 42), (42, 70), (71, 57), (57, 28), (28, 42), (42, 71), (72, 57), (57, 27), (27, 42), (42, 72), (72, 58), (58, 27), (27, 41), (41, 72), (72, 59), (59, 27), (27, 40), (40, 72), (72, 60), (60, 27), (27, 39), (39, 72), (71, 60), (60, 28), (28, 39), (39, 71), (70, 60), (60, 29), (29, 39), (39, 70), (69, 60), (60, 30), (30, 39), (39, 69), (68, 60), (60, 31), (31, 39), (39, 68), (67, 60), (60, 32), (32, 39), (39, 67), (66, 60), (60, 33), (33, 39), (39, 66), (66, 59), (59, 33), (33, 40), (40, 66), (66, 58), (58, 33), (33, 41), (41, 66), (66, 57), (57, 33), (33, 42), (42, 66), (65, 57), (57, 34), (34, 42), (42, 65), (64, 57), (57, 35), (35, 42), (42, 64), (63, 57), (57, 36), (36, 42), (42, 63), (63, 58), (58, 36), (36, 41), (41, 63), (63, 59), (59, 36), (36, 40), (40, 63), (63, 60), (60, 36), (36, 39), (39, 63), (63, 61), (61, 36), (36, 38), (38, 63), (63, 62), (62, 36), (36, 37), (37, 63), (63, 63), (63, 36), (36, 36), (36, 63), (63, 64), (64, 36), (36, 35), (35, 63), (63, 65), (65, 36), (36, 34), (34, 63), (63, 66), (66, 36), (36, 33), (33, 63), (64, 66), (66, 35), (35, 33), (33, 64), (65, 66), (66, 34), (34, 33), (33, 65), (66, 66), (66, 33), (33, 33), (33, 66), (66, 65), (65, 33), (33, 34), (34, 66), (66, 64), (64, 33), (33, 35), (35, 66), (66, 63), (63, 33), (33, 36), (36, 66), (67, 63), (63, 32), (32, 36), (36, 67), (68, 63), (63, 31), (31, 36), (36, 68), (69, 63), (63, 30), (30, 36), (36, 69), (70, 63), (63, 29), (29, 36), (36, 70), (71, 63), (63, 28), (28, 36), (36, 71), (72, 63), (63, 27), (27, 36), (36, 72), (72, 64), (64, 27), (27, 35), (35, 72), (72, 65), (65, 27), (27, 34), (34, 72), (72, 66), (66, 27), (27, 33), (33, 72), (71, 66), (66, 28), (28, 33), (33, 71), (70, 66), (66, 29), (29, 33), (33, 70), (69, 66), (66, 30), (30, 33), (33, 69), (69, 67), (67, 30), (30, 32), (32, 69), (69, 68), (68, 30), (30, 31), (31, 69), (69, 69), (69, 30), (30, 30), (30, 69), (70, 69), (69, 29), (29, 30), (30, 70), (71, 69), (69, 28), (28, 30), (30, 71), (72, 69), (69, 27), (27, 30), (30, 72), (72, 70), (70, 27), (27, 29), (29, 72), (72, 71), (71, 27), (27, 28), (28, 72), (72, 72), (72, 27), (27, 27), (27, 72), (71, 72), (72, 28), (28, 27), (27, 71), (70, 72), (72, 29), (29, 27), (27, 70), (69, 72), (72, 30), (30, 27), (27, 69), (68, 72), (72, 31), (31, 27), (27, 68), (67, 72), (72, 32), (32, 27), (27, 67), (66, 72), (72, 33), (33, 27), (27, 66), (66, 71), (71, 33), (33, 28), (28, 66), (66, 70), (70, 33), (33, 29), (29, 66), (66, 69), (69, 33), (33, 30), (30, 66), (65, 69), (69, 34), (34, 30), (30, 65), (64, 69), (69, 35), (35, 30), (30, 64), (63, 69), (69, 36), (36, 30), (30, 63), (63, 70), (70, 36), (36, 29), (29, 63), (63, 71), (71, 36), (36, 28), (28, 63), (63, 72), (72, 36), (36, 27), (27, 63), (62, 72), (72, 37), (37, 27), (27, 62), (61, 72), (72, 38), (38, 27), (27, 61), (60, 72), (72, 39), (39, 27), (27, 60), (59, 72), (72, 40), (40, 27), (27, 59), (58, 72), (72, 41), (41, 27), (27, 58), (57, 72), (72, 42), (42, 27), (27, 57), (57, 71), (71, 42), (42, 28), (28, 57), (57, 70), (70, 42), (42, 29), (29, 57), (57, 69), (69, 42), (42, 30), (30, 57), (58, 69), (69, 41), (41, 30), (30, 58), (59, 69), (69, 40), (40, 30), (30, 59), (60, 69), (69, 39), (39, 30), (30, 60), (60, 68), (68, 39), (39, 31), (31, 60), (60, 67), (67, 39), (39, 32), (32, 60), (60, 66), (66, 39), (39, 33), (33, 60), (60, 65), (65, 39), (39, 34), (34, 60), (60, 64), (64, 39), (39, 35), (35, 60), (60, 63), (63, 39), (39, 36), (36, 60), (59, 63), (63, 40), (40, 36), (36, 59), (58, 63), (63, 41), (41, 36), (36, 58), (57, 63), (63, 42), (42, 36), (36, 57), (57, 64), (64, 42), (42, 35), (35, 57), (57, 65), (65, 42), (42, 34), (34, 57), (57, 66), (66, 42), (42, 33), (33, 57), (56, 66), (66, 43), (43, 33), (33, 56), (55, 66), (66, 44), (44, 33), (33, 55), (54, 66), (66, 45), (45, 33), (33, 54), (54, 65), (65, 45), (45, 34), (34, 54), (54, 64), (64, 45), (45, 35), (35, 54), (54, 63), (63, 45), (45, 36), (36, 54), (53, 63), (63, 46), (46, 36), (36, 53), (52, 63), (63, 47), (47, 36), (36, 52), (51, 63), (63, 48), (48, 36), (36, 51), (51, 64), (64, 48), (48, 35), (35, 51), (51, 65), (65, 48), (48, 34), (34, 51), (51, 66), (66, 48), (48, 33), (33, 51), (51, 67), (67, 48), (48, 32), (32, 51), (51, 68), (68, 48), (48, 31), (31, 51), (51, 69), (69, 48), (48, 30), (30, 51), (52, 69), (69, 47), (47, 30), (30, 52), (53, 69), (69, 46), (46, 30), (30, 53), (54, 69), (69, 45), (45, 30), (30, 54), (54, 70), (70, 45), (45, 29), (29, 54), (54, 71), (71, 45), (45, 28), (28, 54), (54, 72), (72, 45), (45, 27), (27, 54), (53, 72), (72, 46), (46, 27), (27, 53), (52, 72), (72, 47), (47, 27), (27, 52), (51, 72), (72, 48), (48, 27), (27, 51), (51, 73), (73, 48), (48, 26), (26, 51), (51, 74), (74, 48), (48, 25), (25, 51), (51, 75), (75, 48), (48, 24), (24, 51), (51, 76), (76, 48), (48, 23), (23, 51), (51, 77), (77, 48), (48, 22), (22, 51), (51, 78), (78, 48), (48, 21), (21, 51), (52, 78), (78, 47), (47, 21), (21, 52), (53, 78), (78, 46), (46, 21), (21, 53), (54, 78), (78, 45), (45, 21), (21, 54), (54, 77), (77, 45), (45, 22), (22, 54), (54, 76), (76, 45), (45, 23), (23, 54), (54, 75), (75, 45), (45, 24), (24, 54), (55, 75), (75, 44), (44, 24), (24, 55), (56, 75), (75, 43), (43, 24), (24, 56), (57, 75), (75, 42), (42, 24), (24, 57), (58, 75), (75, 41), (41, 24), (24, 58), (59, 75), (75, 40), (40, 24), (24, 59), (60, 75), (75, 39), (39, 24), (24, 60), (60, 76), (76, 39), (39, 23), (23, 60), (60, 77), (77, 39), (39, 22), (22, 60), (60, 78), (78, 39), (39, 21), (21, 60), (59, 78), (78, 40), (40, 21), (21, 59), (58, 78), (78, 41), (41, 21), (21, 58), (57, 78), (78, 42), (42, 21), (21, 57), (57, 79), (79, 42), (42, 20), (20, 57), (57, 80), (80, 42), (42, 19), (19, 57), (57, 81), (81, 42), (42, 18), (18, 57), (58, 81), (81, 41), (41, 18), (18, 58), (59, 81), (81, 40), (40, 18), (18, 59), (60, 81), (81, 39), (39, 18), (18, 60), (60, 82), (82, 39), (39, 17), (17, 60), (60, 83), (83, 39), (39, 16), (16, 60), (60, 84), (84, 39), (39, 15), (15, 60), (59, 84), (84, 40), (40, 15), (15, 59), (58, 84), (84, 41), (41, 15), (15, 58), (57, 84), (84, 42), (42, 15), (15, 57), (56, 84), (84, 43), (43, 15), (15, 56), (55, 84), (84, 44), (44, 15), (15, 55), (54, 84), (84, 45), (45, 15), (15, 54), (54, 83), (83, 45), (45, 16), (16, 54), (54, 82), (82, 45), (45, 17), (17, 54), (54, 81), (81, 45), (45, 18), (18, 54), (53, 81), (81, 46), (46, 18), (18, 53), (52, 81), (81, 47), (47, 18), (18, 52), (51, 81), (81, 48), (48, 18), (18, 51), (51, 82), (82, 48), (48, 17), (17, 51), (51, 83), (83, 48), (48, 16), (16, 51), (51, 84), (84, 48), (48, 15), (15, 51), (51, 85), (85, 48), (48, 14), (14, 51), (51, 86), (86, 48), (48, 13), (13, 51), (51, 87), (87, 48), (48, 12), (12, 51), (52, 87), (87, 47), (47, 12), (12, 52), (53, 87), (87, 46), (46, 12), (12, 53), (54, 87), (87, 45), (45, 12), (12, 54), (54, 88), (88, 45), (45, 11), (11, 54), (54, 89), (89, 45), (45, 10), (10, 54), (54, 90), (90, 45), (45, 9), (9, 54), (53, 90), (90, 46), (46, 9), (9, 53), (52, 90), (90, 47), (47, 9), (9, 52), (51, 90), (90, 48), (48, 9), (9, 51), (51, 91), (91, 48), (48, 8), (8, 51), (51, 92), (92, 48), (48, 7), (7, 51), (51, 93), (93, 48), (48, 6), (6, 51), (51, 94), (94, 48), (48, 5), (5, 51), (51, 95), (95, 48), (48, 4), (4, 51), (51, 96), (96, 48), (48, 3), (3, 51), (52, 96), (96, 47), (47, 3), (3, 52), (53, 96), (96, 46), (46, 3), (3, 53), (54, 96), (96, 45), (45, 3), (3, 54), (54, 95), (95, 45), (45, 4), (4, 54), (54, 94), (94, 45), (45, 5), (5, 54), (54, 93), (93, 45), (45, 6), (6, 54), (55, 93), (93, 44), (44, 6), (6, 55), (56, 93), (93, 43), (43, 6), (6, 56), (57, 93), (93, 42), (42, 6), (6, 57), (57, 94), (94, 42), (42, 5), (5, 57), (57, 95), (95, 42), (42, 4), (4, 57), (57, 96), (96, 42), (42, 3), (3, 57), (58, 96), (96, 41), (41, 3), (3, 58), (59, 96), (96, 40), (40, 3), (3, 59), (60, 96), (96, 39), (39, 3), (3, 60), (60, 95), (95, 39), (39, 4), (4, 60), (60, 94), (94, 39), (39, 5), (5, 60), (60, 93), (93, 39), (39, 6), (6, 60), (60, 92), (92, 39), (39, 7), (7, 60), (60, 91), (91, 39), (39, 8), (8, 60), (60, 90), (90, 39), (39, 9), (9, 60), (59, 90), (90, 40), (40, 9), (9, 59), (58, 90), (90, 41), (41, 9), (9, 58), (57, 90), (90, 42), (42, 9), (9, 57), (57, 89), (89, 42), (42, 10), (10, 57), (57, 88), (88, 42), (42, 11), (11, 57), (57, 87), (87, 42), (42, 12), (12, 57), (58, 87), (87, 41), (41, 12), (12, 58), (59, 87), (87, 40), (40, 12), (12, 59), (60, 87), (87, 39), (39, 12), (12, 60), (61, 87), (87, 38), (38, 12), (12, 61), (62, 87), (87, 37), (37, 12), (12, 62), (63, 87), (87, 36), (36, 12), (12, 63), (64, 87), (87, 35), (35, 12), (12, 64), (65, 87), (87, 34), (34, 12), (12, 65), (66, 87), (87, 33), (33, 12), (12, 66), (66, 88), (88, 33), (33, 11), (11, 66), (66, 89), (89, 33), (33, 10), (10, 66), (66, 90), (90, 33), (33, 9), (9, 66), (65, 90), (90, 34), (34, 9), (9, 65), (64, 90), (90, 35), (35, 9), (9, 64), (63, 90), (90, 36), (36, 9), (9, 63), (63, 91), (91, 36), (36, 8), (8, 63), (63, 92), (92, 36), (36, 7), (7, 63), (63, 93), (93, 36), (36, 6), (6, 63), (63, 94), (94, 36), (36, 5), (5, 63), (63, 95), (95, 36), (36, 4), (4, 63), (63, 96), (96, 36), (36, 3), (3, 63), (64, 96), (96, 35), (35, 3), (3, 64), (65, 96), (96, 34), (34, 3), (3, 65), (66, 96), (96, 33), (33, 3), (3, 66), (66, 95), (95, 33), (33, 4), (4, 66), (66, 94), (94, 33), (33, 5), (5, 66), (66, 93), (93, 33), (33, 6), (6, 66), (67, 93), (93, 32), (32, 6), (6, 67), (68, 93), (93, 31), (31, 6), (6, 68), (69, 93), (93, 30), (30, 6), (6, 69), (69, 94), (94, 30), (30, 5), (5, 69), (69, 95), (95, 30), (30, 4), (4, 69), (69, 96), (96, 30), (30, 3), (3, 69), (70, 96), (96, 29), (29, 3), (3, 70), (71, 96), (96, 28), (28, 3), (3, 71), (72, 96), (96, 27), (27, 3), (3, 72), (72, 95), (95, 27), (27, 4), (4, 72), (72, 94), (94, 27), (27, 5), (5, 72), (72, 93), (93, 27), (27, 6), (6, 72), (72, 92), (92, 27), (27, 7), (7, 72), (72, 91), (91, 27), (27, 8), (8, 72), (72, 90), (90, 27), (27, 9), (9, 72), (71, 90), (90, 28), (28, 9), (9, 71), (70, 90), (90, 29), (29, 9), (9, 70), (69, 90), (90, 30), (30, 9), (9, 69), (69, 89), (89, 30), (30, 10), (10, 69), (69, 88), (88, 30), (30, 11), (11, 69), (69, 87), (87, 30), (30, 12), (12, 69), (70, 87), (87, 29), (29, 12), (12, 70), (71, 87), (87, 28), (28, 12), (12, 71), (72, 87), (87, 27), (27, 12), (12, 72), (72, 86), (86, 27), (27, 13), (13, 72), (72, 85), (85, 27), (27, 14), (14, 72), (72, 84), (84, 27), (27, 15), (15, 72), (72, 83), (83, 27), (27, 16), (16, 72), (72, 82), (82, 27), (27, 17), (17, 72), (72, 81), (81, 27), (27, 18), (18, 72), (71, 81), (81, 28), (28, 18), (18, 71), (70, 81), (81, 29), (29, 18), (18, 70), (69, 81), (81, 30), (30, 18), (18, 69), (69, 82), (82, 30), (30, 17), (17, 69), (69, 83), (83, 30), (30, 16), (16, 69), (69, 84), (84, 30), (30, 15), (15, 69), (68, 84), (84, 31), (31, 15), (15, 68), (67, 84), (84, 32), (32, 15), (15, 67), (66, 84), (84, 33), (33, 15), (15, 66), (65, 84), (84, 34), (34, 15), (15, 65), (64, 84), (84, 35), (35, 15), (15, 64), (63, 84), (84, 36), (36, 15), (15, 63), (63, 83), (83, 36), (36, 16), (16, 63), (63, 82), (82, 36), (36, 17), (17, 63), (63, 81), (81, 36), (36, 18), (18, 63), (64, 81), (81, 35), (35, 18), (18, 64), (65, 81), (81, 34), (34, 18), (18, 65), (66, 81), (81, 33), (33, 18), (18, 66), (66, 80), (80, 33), (33, 19), (19, 66), (66, 79), (79, 33), (33, 20), (20, 66), (66, 78), (78, 33), (33, 21), (21, 66), (65, 78), (78, 34), (34, 21), (21, 65), (64, 78), (78, 35), (35, 21), (21, 64), (63, 78), (78, 36), (36, 21), (21, 63), (63, 77), (77, 36), (36, 22), (22, 63), (63, 76), (76, 36), (36, 23), (23, 63), (63, 75), (75, 36), (36, 24), (24, 63), (64, 75), (75, 35), (35, 24), (24, 64), (65, 75), (75, 34), (34, 24), (24, 65), (66, 75), (75, 33), (33, 24), (24, 66), (67, 75), (75, 32), (32, 24), (24, 67), (68, 75), (75, 31), (31, 24), (24, 68), (69, 75), (75, 30), (30, 24), (24, 69), (69, 76), (76, 30), (30, 23), (23, 69), (69, 77), (77, 30), (30, 22), (22, 69), (69, 78), (78, 30), (30, 21), (21, 69), (70, 78), (78, 29), (29, 21), (21, 70), (71, 78), (78, 28), (28, 21), (21, 71), (72, 78), (78, 27), (27, 21), (21, 72), (72, 77), (77, 27), (27, 22), (22, 72), (72, 76), (76, 27), (27, 23), (23, 72), (72, 75), (75, 27), (27, 24), (24, 72), (73, 75), (75, 26), (26, 24), (24, 73), (74, 75), (75, 25), (25, 24), (24, 74), (75, 75), (75, 24), (24, 24), (24, 75), (75, 76), (76, 24), (24, 23), (23, 75), (75, 77), (77, 24), (24, 22), (22, 75), (75, 78), (78, 24), (24, 21), (21, 75), (76, 78), (78, 23), (23, 21), (21, 76), (77, 78), (78, 22), (22, 21), (21, 77), (78, 78), (78, 21), (21, 21), (21, 78), (78, 77), (77, 21), (21, 22), (22, 78), (78, 76), (76, 21), (21, 23), (23, 78), (78, 75), (75, 21), (21, 24), (24, 78), (79, 75), (75, 20), (20, 24), (24, 79), (80, 75), (75, 19), (19, 24), (24, 80), (81, 75), (75, 18), (18, 24), (24, 81), (82, 75), (75, 17), (17, 24), (24, 82), (83, 75), (75, 16), (16, 24), (24, 83), (84, 75), (75, 15), (15, 24), (24, 84), (84, 76), (76, 15), (15, 23), (23, 84), (84, 77), (77, 15), (15, 22), (22, 84), (84, 78), (78, 15), (15, 21), (21, 84), (83, 78), (78, 16), (16, 21), (21, 83), (82, 78), (78, 17), (17, 21), (21, 82), (81, 78), (78, 18), (18, 21), (21, 81), (81, 79), (79, 18), (18, 20), (20, 81), (81, 80), (80, 18), (18, 19), (19, 81), (81, 81), (81, 18), (18, 18), (18, 81), (82, 81), (81, 17), (17, 18), (18, 82), (83, 81), (81, 16), (16, 18), (18, 83), (84, 81), (81, 15), (15, 18), (18, 84), (84, 82), (82, 15), (15, 17), (17, 84), (84, 83), (83, 15), (15, 16), (16, 84), (84, 84), (84, 15), (15, 15), (15, 84), (83, 84), (84, 16), (16, 15), (15, 83), (82, 84), (84, 17), (17, 15), (15, 82), (81, 84), (84, 18), (18, 15), (15, 81), (80, 84), (84, 19), (19, 15), (15, 80), (79, 84), (84, 20), (20, 15), (15, 79), (78, 84), (84, 21), (21, 15), (15, 78), (78, 83), (83, 21), (21, 16), (16, 78), (78, 82), (82, 21), (21, 17), (17, 78), (78, 81), (81, 21), (21, 18), (18, 78), (77, 81), (81, 22), (22, 18), (18, 77), (76, 81), (81, 23), (23, 18), (18, 76), (75, 81), (81, 24), (24, 18), (18, 75), (75, 82), (82, 24), (24, 17), (17, 75), (75, 83), (83, 24), (24, 16), (16, 75), (75, 84), (84, 24), (24, 15), (15, 75), (75, 85), (85, 24), (24, 14), (14, 75), (75, 86), (86, 24), (24, 13), (13, 75), (75, 87), (87, 24), (24, 12), (12, 75), (76, 87), (87, 23), (23, 12), (12, 76), (77, 87), (87, 22), (22, 12), (12, 77), (78, 87), (87, 21), (21, 12), (12, 78), (78, 88), (88, 21), (21, 11), (11, 78), (78, 89), (89, 21), (21, 10), (10, 78), (78, 90), (90, 21), (21, 9), (9, 78), (77, 90), (90, 22), (22, 9), (9, 77), (76, 90), (90, 23), (23, 9), (9, 76), (75, 90), (90, 24), (24, 9), (9, 75), (75, 91), (91, 24), (24, 8), (8, 75), (75, 92), (92, 24), (24, 7), (7, 75), (75, 93), (93, 24), (24, 6), (6, 75), (75, 94), (94, 24), (24, 5), (5, 75), (75, 95), (95, 24), (24, 4), (4, 75), (75, 96), (96, 24), (24, 3), (3, 75), (76, 96), (96, 23), (23, 3), (3, 76), (77, 96), (96, 22), (22, 3), (3, 77), (78, 96), (96, 21), (21, 3), (3, 78), (78, 95), (95, 21), (21, 4), (4, 78), (78, 94), (94, 21), (21, 5), (5, 78), (78, 93), (93, 21), (21, 6), (6, 78), (79, 93), (93, 20), (20, 6), (6, 79), (80, 93), (93, 19), (19, 6), (6, 80), (81, 93), (93, 18), (18, 6), (6, 81), (81, 94), (94, 18), (18, 5), (5, 81), (81, 95), (95, 18), (18, 4), (4, 81), (81, 96), (96, 18), (18, 3), (3, 81), (82, 96), (96, 17), (17, 3), (3, 82), (83, 96), (96, 16), (16, 3), (3, 83), (84, 96), (96, 15), (15, 3), (3, 84), (84, 95), (95, 15), (15, 4), (4, 84), (84, 94), (94, 15), (15, 5), (5, 84), (84, 93), (93, 15), (15, 6), (6, 84), (84, 92), (92, 15), (15, 7), (7, 84), (84, 91), (91, 15), (15, 8), (8, 84), (84, 90), (90, 15), (15, 9), (9, 84), (83, 90), (90, 16), (16, 9), (9, 83), (82, 90), (90, 17), (17, 9), (9, 82), (81, 90), (90, 18), (18, 9), (9, 81), (81, 89), (89, 18), (18, 10), (10, 81), (81, 88), (88, 18), (18, 11), (11, 81), (81, 87), (87, 18), (18, 12), (12, 81), (82, 87), (87, 17), (17, 12), (12, 82), (83, 87), (87, 16), (16, 12), (12, 83), (84, 87), (87, 15), (15, 12), (12, 84), (85, 87), (87, 14), (14, 12), (12, 85), (86, 87), (87, 13), (13, 12), (12, 86), (87, 87), (87, 12), (12, 12), (12, 87), (88, 87), (87, 11), (11, 12), (12, 88), (89, 87), (87, 10), (10, 12), (12, 89), (90, 87), (87, 9), (9, 12), (12, 90), (90, 88), (88, 9), (9, 11), (11, 90), (90, 89), (89, 9), (9, 10), (10, 90), (90, 90), (90, 9), (9, 9), (9, 90), (89, 90), (90, 10), (10, 9), (9, 89), (88, 90), (90, 11), (11, 9), (9, 88), (87, 90), (90, 12), (12, 9), (9, 87), (87, 91), (91, 12), (12, 8), (8, 87), (87, 92), (92, 12), (12, 7), (7, 87), (87, 93), (93, 12), (12, 6), (6, 87), (87, 94), (94, 12), (12, 5), (5, 87), (87, 95), (95, 12), (12, 4), (4, 87), (87, 96), (96, 12), (12, 3), (3, 87), (88, 96), (96, 11), (11, 3), (3, 88), (89, 96), (96, 10), (10, 3), (3, 89), (90, 96), (96, 9), (9, 3), (3, 90), (90, 95), (95, 9), (9, 4), (4, 90), (90, 94), (94, 9), (9, 5), (5, 90), (90, 93), (93, 9), (9, 6), (6, 90), (91, 93), (93, 8), (8, 6), (6, 91), (92, 93), (93, 7), (7, 6), (6, 92), (93, 93), (93, 6), (6, 6), (6, 93), (93, 94), (94, 6), (6, 5), (5, 93), (93, 95), (95, 6), (6, 4), (4, 93), (93, 96), (96, 6), (6, 3), (3, 93), (94, 96), (96, 5), (5, 3), (3, 94), (95, 96), (96, 4), (4, 3), (3, 95), (96, 96), (96, 3), (3, 3), (3, 96), (96, 95), (95, 3), (3, 4), (4, 96), (96, 94), (94, 3), (3, 5), (5, 96), (96, 93), (93, 3), (3, 6), (6, 96), (96, 92), (92, 3), (3, 7), (7, 96), (96, 91), (91, 3), (3, 8), (8, 96), (96, 90), (90, 3), (3, 9), (9, 96), (95, 90), (90, 4), (4, 9), (9, 95), (94, 90), (90, 5), (5, 9), (9, 94), (93, 90), (90, 6), (6, 9), (9, 93), (93, 89), (89, 6), (6, 10), (10, 93), (93, 88), (88, 6), (6, 11), (11, 93), (93, 87), (87, 6), (6, 12), (12, 93), (94, 87), (87, 5), (5, 12), (12, 94), (95, 87), (87, 4), (4, 12), (12, 95), (96, 87), (87, 3), (3, 12), (12, 96), (96, 86), (86, 3), (3, 13), (13, 96), (96, 85), (85, 3), (3, 14), (14, 96), (96, 84), (84, 3), (3, 15), (15, 96), (96, 83), (83, 3), (3, 16), (16, 96), (96, 82), (82, 3), (3, 17), (17, 96), (96, 81), (81, 3), (3, 18), (18, 96), (95, 81), (81, 4), (4, 18), (18, 95), (94, 81), (81, 5), (5, 18), (18, 94), (93, 81), (81, 6), (6, 18), (18, 93), (93, 82), (82, 6), (6, 17), (17, 93), (93, 83), (83, 6), (6, 16), (16, 93), (93, 84), (84, 6), (6, 15), (15, 93), (92, 84), (84, 7), (7, 15), (15, 92), (91, 84), (84, 8), (8, 15), (15, 91), (90, 84), (84, 9), (9, 15), (15, 90), (89, 84), (84, 10), (10, 15), (15, 89), (88, 84), (84, 11), (11, 15), (15, 88), (87, 84), (84, 12), (12, 15), (15, 87), (87, 83), (83, 12), (12, 16), (16, 87), (87, 82), (82, 12), (12, 17), (17, 87), (87, 81), (81, 12), (12, 18), (18, 87), (88, 81), (81, 11), (11, 18), (18, 88), (89, 81), (81, 10), (10, 18), (18, 89), (90, 81), (81, 9), (9, 18), (18, 90), (90, 80), (80, 9), (9, 19), (19, 90), (90, 79), (79, 9), (9, 20), (20, 90), (90, 78), (78, 9), (9, 21), (21, 90), (89, 78), (78, 10), (10, 21), (21, 89), (88, 78), (78, 11), (11, 21), (21, 88), (87, 78), (78, 12), (12, 21), (21, 87), (87, 77), (77, 12), (12, 22), (22, 87), (87, 76), (76, 12), (12, 23), (23, 87), (87, 75), (75, 12), (12, 24), (24, 87), (88, 75), (75, 11), (11, 24), (24, 88), (89, 75), (75, 10), (10, 24), (24, 89), (90, 75), (75, 9), (9, 24), (24, 90), (91, 75), (75, 8), (8, 24), (24, 91), (92, 75), (75, 7), (7, 24), (24, 92), (93, 75), (75, 6), (6, 24), (24, 93), (93, 76), (76, 6), (6, 23), (23, 93), (93, 77), (77, 6), (6, 22), (22, 93), (93, 78), (78, 6), (6, 21), (21, 93), (94, 78), (78, 5), (5, 21), (21, 94), (95, 78), (78, 4), (4, 21), (21, 95), (96, 78), (78, 3), (3, 21), (21, 96), (96, 77), (77, 3), (3, 22), (22, 96), (96, 76), (76, 3), (3, 23), (23, 96), (96, 75), (75, 3), (3, 24), (24, 96), (96, 74), (74, 3), (3, 25), (25, 96), (96, 73), (73, 3), (3, 26), (26, 96), (96, 72), (72, 3), (3, 27), (27, 96), (95, 72), (72, 4), (4, 27), (27, 95), (94, 72), (72, 5), (5, 27), (27, 94), (93, 72), (72, 6), (6, 27), (27, 93), (93, 71), (71, 6), (6, 28), (28, 93), (93, 70), (70, 6), (6, 29), (29, 93), (93, 69), (69, 6), (6, 30), (30, 93), (94, 69), (69, 5), (5, 30), (30, 94), (95, 69), (69, 4), (4, 30), (30, 95), (96, 69), (69, 3), (3, 30), (30, 96), (96, 68), (68, 3), (3, 31), (31, 96), (96, 67), (67, 3), (3, 32), (32, 96), (96, 66), (66, 3), (3, 33), (33, 96), (96, 65), (65, 3), (3, 34), (34, 96), (96, 64), (64, 3), (3, 35), (35, 96), (96, 63), (63, 3), (3, 36), (36, 96), (95, 63), (63, 4), (4, 36), (36, 95), (94, 63), (63, 5), (5, 36), (36, 94), (93, 63), (63, 6), (6, 36), (36, 93), (93, 64), (64, 6), (6, 35), (35, 93), (93, 65), (65, 6), (6, 34), (34, 93), (93, 66), (66, 6), (6, 33), (33, 93), (92, 66), (66, 7), (7, 33), (33, 92), (91, 66), (66, 8), (8, 33), (33, 91), (90, 66), (66, 9), (9, 33), (33, 90), (90, 65), (65, 9), (9, 34), (34, 90), (90, 64), (64, 9), (9, 35), (35, 90), (90, 63), (63, 9), (9, 36), (36, 90), (89, 63), (63, 10), (10, 36), (36, 89), (88, 63), (63, 11), (11, 36), (36, 88), (87, 63), (63, 12), (12, 36), (36, 87), (87, 64), (64, 12), (12, 35), (35, 87), (87, 65), (65, 12), (12, 34), (34, 87), (87, 66), (66, 12), (12, 33), (33, 87), (87, 67), (67, 12), (12, 32), (32, 87), (87, 68), (68, 12), (12, 31), (31, 87), (87, 69), (69, 12), (12, 30), (30, 87), (88, 69), (69, 11), (11, 30), (30, 88), (89, 69), (69, 10), (10, 30), (30, 89), (90, 69), (69, 9), (9, 30), (30, 90), (90, 70), (70, 9), (9, 29), (29, 90), (90, 71), (71, 9), (9, 28), (28, 90), (90, 72), (72, 9), (9, 27), (27, 90), (89, 72), (72, 10), (10, 27), (27, 89), (88, 72), (72, 11), (11, 27), (27, 88), (87, 72), (72, 12), (12, 27), (27, 87), (86, 72), (72, 13), (13, 27), (27, 86), (85, 72), (72, 14), (14, 27), (27, 85), (84, 72), (72, 15), (15, 27), (27, 84), (84, 71), (71, 15), (15, 28), (28, 84), (84, 70), (70, 15), (15, 29), (29, 84), (84, 69), (69, 15), (15, 30), (30, 84), (83, 69), (69, 16), (16, 30), (30, 83), (82, 69), (69, 17), (17, 30), (30, 82), (81, 69), (69, 18), (18, 30), (30, 81), (81, 70), (70, 18), (18, 29), (29, 81), (81, 71), (71, 18), (18, 28), (28, 81), (81, 72), (72, 18), (18, 27), (27, 81), (80, 72), (72, 19), (19, 27), (27, 80), (79, 72), (72, 20), (20, 27), (27, 79), (78, 72), (72, 21), (21, 27), (27, 78), (77, 72), (72, 22), (22, 27), (27, 77), (76, 72), (72, 23), (23, 27), (27, 76), (75, 72), (72, 24), (24, 27), (27, 75), (75, 71), (71, 24), (24, 28), (28, 75), (75, 70), (70, 24), (24, 29), (29, 75), (75, 69), (69, 24), (24, 30), (30, 75), (76, 69), (69, 23), (23, 30), (30, 76), (77, 69), (69, 22), (22, 30), (30, 77), (78, 69), (69, 21), (21, 30), (30, 78), (78, 68), (68, 21), (21, 31), (31, 78), (78, 67), (67, 21), (21, 32), (32, 78), (78, 66), (66, 21), (21, 33), (33, 78), (77, 66), (66, 22), (22, 33), (33, 77), (76, 66), (66, 23), (23, 33), (33, 76), (75, 66), (66, 24), (24, 33), (33, 75), (75, 65), (65, 24), (24, 34), (34, 75), (75, 64), (64, 24), (24, 35), (35, 75), (75, 63), (63, 24), (24, 36), (36, 75), (76, 63), (63, 23), (23, 36), (36, 76), (77, 63), (63, 22), (22, 36), (36, 77), (78, 63), (63, 21), (21, 36), (36, 78), (79, 63), (63, 20), (20, 36), (36, 79), (80, 63), (63, 19), (19, 36), (36, 80), (81, 63), (63, 18), (18, 36), (36, 81), (81, 64), (64, 18), (18, 35), (35, 81), (81, 65), (65, 18), (18, 34), (34, 81), (81, 66), (66, 18), (18, 33), (33, 81), (82, 66), (66, 17), (17, 33), (33, 82), (83, 66), (66, 16), (16, 33), (33, 83), (84, 66), (66, 15), (15, 33), (33, 84), (84, 65), (65, 15), (15, 34), (34, 84), (84, 64), (64, 15), (15, 35), (35, 84), (84, 63), (63, 15), (15, 36), (36, 84), (84, 62), (62, 15), (15, 37), (37, 84), (84, 61), (61, 15), (15, 38), (38, 84), (84, 60), (60, 15), (15, 39), (39, 84), (84, 59), (59, 15), (15, 40), (40, 84), (84, 58), (58, 15), (15, 41), (41, 84), (84, 57), (57, 15), (15, 42), (42, 84), (83, 57), (57, 16), (16, 42), (42, 83), (82, 57), (57, 17), (17, 42), (42, 82), (81, 57), (57, 18), (18, 42), (42, 81), (81, 58), (58, 18), (18, 41), (41, 81), (81, 59), (59, 18), (18, 40), (40, 81), (81, 60), (60, 18), (18, 39), (39, 81), (80, 60), (60, 19), (19, 39), (39, 80), (79, 60), (60, 20), (20, 39), (39, 79), (78, 60), (60, 21), (21, 39), (39, 78), (77, 60), (60, 22), (22, 39), (39, 77), (76, 60), (60, 23), (23, 39), (39, 76), (75, 60), (60, 24), (24, 39), (39, 75), (75, 59), (59, 24), (24, 40), (40, 75), (75, 58), (58, 24), (24, 41), (41, 75), (75, 57), (57, 24), (24, 42), (42, 75), (76, 57), (57, 23), (23, 42), (42, 76), (77, 57), (57, 22), (22, 42), (42, 77), (78, 57), (57, 21), (21, 42), (42, 78), (78, 56), (56, 21), (21, 43), (43, 78), (78, 55), (55, 21), (21, 44), (44, 78), (78, 54), (54, 21), (21, 45), (45, 78), (77, 54), (54, 22), (22, 45), (45, 77), (76, 54), (54, 23), (23, 45), (45, 76), (75, 54), (54, 24), (24, 45), (45, 75), (75, 53), (53, 24), (24, 46), (46, 75), (75, 52), (52, 24), (24, 47), (47, 75), (75, 51), (51, 24), (24, 48), (48, 75), (76, 51), (51, 23), (23, 48), (48, 76), (77, 51), (51, 22), (22, 48), (48, 77), (78, 51), (51, 21), (21, 48), (48, 78), (79, 51), (51, 20), (20, 48), (48, 79), (80, 51), (51, 19), (19, 48), (48, 80), (81, 51), (51, 18), (18, 48), (48, 81), (81, 52), (52, 18), (18, 47), (47, 81), (81, 53), (53, 18), (18, 46), (46, 81), (81, 54), (54, 18), (18, 45), (45, 81), (82, 54), (54, 17), (17, 45), (45, 82), (83, 54), (54, 16), (16, 45), (45, 83), (84, 54), (54, 15), (15, 45), (45, 84), (84, 53), (53, 15), (15, 46), (46, 84), (84, 52), (52, 15), (15, 47), (47, 84), (84, 51), (51, 15), (15, 48), (48, 84), (85, 51), (51, 14), (14, 48), (48, 85), (86, 51), (51, 13), (13, 48), (48, 86), (87, 51), (51, 12), (12, 48), (48, 87), (88, 51), (51, 11), (11, 48), (48, 88), (89, 51), (51, 10), (10, 48), (48, 89), (90, 51), (51, 9), (9, 48), (48, 90), (90, 52), (52, 9), (9, 47), (47, 90), (90, 53), (53, 9), (9, 46), (46, 90), (90, 54), (54, 9), (9, 45), (45, 90), (89, 54), (54, 10), (10, 45), (45, 89), (88, 54), (54, 11), (11, 45), (45, 88), (87, 54), (54, 12), (12, 45), (45, 87), (87, 55), (55, 12), (12, 44), (44, 87), (87, 56), (56, 12), (12, 43), (43, 87), (87, 57), (57, 12), (12, 42), (42, 87), (87, 58), (58, 12), (12, 41), (41, 87), (87, 59), (59, 12), (12, 40), (40, 87), (87, 60), (60, 12), (12, 39), (39, 87), (88, 60), (60, 11), (11, 39), (39, 88), (89, 60), (60, 10), (10, 39), (39, 89), (90, 60), (60, 9), (9, 39), (39, 90), (90, 59), (59, 9), (9, 40), (40, 90), (90, 58), (58, 9), (9, 41), (41, 90), (90, 57), (57, 9), (9, 42), (42, 90), (91, 57), (57, 8), (8, 42), (42, 91), (92, 57), (57, 7), (7, 42), (42, 92), (93, 57), (57, 6), (6, 42), (42, 93), (93, 58), (58, 6), (6, 41), (41, 93), (93, 59), (59, 6), (6, 40), (40, 93), (93, 60), (60, 6), (6, 39), (39, 93), (94, 60), (60, 5), (5, 39), (39, 94), (95, 60), (60, 4), (4, 39), (39, 95), (96, 60), (60, 3), (3, 39), (39, 96), (96, 59), (59, 3), (3, 40), (40, 96), (96, 58), (58, 3), (3, 41), (41, 96), (96, 57), (57, 3), (3, 42), (42, 96), (96, 56), (56, 3), (3, 43), (43, 96), (96, 55), (55, 3), (3, 44), (44, 96), (96, 54), (54, 3), (3, 45), (45, 96), (95, 54), (54, 4), (4, 45), (45, 95), (94, 54), (54, 5), (5, 45), (45, 94), (93, 54), (54, 6), (6, 45), (45, 93), (93, 53), (53, 6), (6, 46), (46, 93), (93, 52), (52, 6), (6, 47), (47, 93), (93, 51), (51, 6), (6, 48), (48, 93), (94, 51), (51, 5), (5, 48), (48, 94), (95, 51), (51, 4), (4, 48), (48, 95), (96, 51), (51, 3), (3, 48), (48, 96)]

    def get_next_formation_points(self, state):
        return self.all_points
    
    def get_phase(self, phase, state, retract, movable):
        return 0

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

        phase, count, isMoving, info = self.decode_info(info)
        # update byte of info
        BACTERIA_RATIO = 0.001 #TODO, maybe based on size of total amoeba and size of periphery??
        percent_bacteria = nAdjacentBacteria / len(current_percept.periphery)
        # print("percent_bacteria", percent_bacteria)
        count += 1 if percent_bacteria > BACTERIA_RATIO else -1
        count = max(0, count)
        count = min(7, count)
        #TODO: maybe once count at 7, just always use SpaceCurveFormation?
        # maybe just use 1 bit based on initial bacteria ratio?
        # all bacteria instantly run away
        # when is SFC better (0.3? 0.1?)?

        # if high density, use space filling curve
        # if count >= 6 or enough cells:
        #     self.formation == SpaceCurveFormation()

        self.formation.update(phase)
        goalFormation = self.formation.get_next_formation_points(current_percept)
        nCells = sum([sum(row) for row in current_percept.amoeba_map])
        firstCells = remove_duplicates(goalFormation)[:nCells]
        # plot_points_helper(firstCells)
        allRetractable = self.formation.get_all_retractable_points(firstCells, current_percept)

        allMovable = self.find_movable_cells(allRetractable, current_percept.periphery, current_percept.amoeba_map, current_percept.bacteria)
        toMove = self.formation.get_moveable_points(allMovable, firstCells, current_percept)

        retract, movable = self.formation.get_n_moves(allRetractable, toMove, current_percept, n_cells_can_move)
        
        phase = self.formation.get_phase(phase, current_percept, retract, movable)

        print("Phase", phase)
        if len(retract) == 0 and len(movable) == 0:
            print("No moves")
        #     return self.move(last_percept, current_percept, self.encode_info(phase, count, 0, info))
        # else:
        #     isMoving = 1

        info = self.encode_info(phase, count, isMoving, info)

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

    def encode_info(self, phase: int, count: int, isMoving: int, info: int) -> int:
        """Encode the information to be sent
            Args:
                phase (int): 2 bits for the current phase of the amoeba
                count (int): 3 bits for the current count of the running density
                isMoving (int): 1 bit for whether the amoeba is getting into position or not
                info (int): 2 bits other info, still TODO
            Returns:
                int: the encoded information as an int
        """
        assert phase < 4
        info = 0
        info_str = "{:02b}{:03b}{:01b}{:02b}".format(phase, count, isMoving, info)

        return int(info_str, 2)

    def decode_info(self, info: int) -> (int, int, int, int):
        """Decode the information received
            Args:
                info (int): the information received
            Returns:
                Tuple[int, int, int, int]: phase, count, isMoving, info, the decoded information as a tuple
        """
        info_str = "{0:b}".format(info).zfill(8)

        return int(info_str[0:2], 2), int(info_str[2:5], 2), int(info_str[5:6], 2), int(info_str[6:8], 2)



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