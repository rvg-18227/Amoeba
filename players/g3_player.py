import logging

import numpy as np

from amoeba_state import AmoebaState


cell = tuple[int, int]


class Player:
    def __init__(
        self,
        rng: np.random.Generator,
        logger: logging.Logger,
        metabolism: float,
        goal_size: int,
        precomp_dir: str
    ) -> None:
        """Initialise the player with the basic amoeba information

        Args:
            rng: numpy random number generator, use this for same player
                 behavior across run
            logger: logger use this like logger.info("message")
            metabolism: the percentage of amoeba cells, that can move
            goal_size: the size the amoeba must reach
            precomp_dir: Directory path to store/load pre-computation
        """

        self.rng = rng
        self.logger = logger
        self.metabolism = metabolism
        self.goal_size = goal_size
        self.current_size = goal_size / 4

    def move(
        self,
        last_percept: AmoebaState,
        current_percept: AmoebaState,
        info: int
    ) -> tuple[list[cell], list[cell], int]:
        """Computes and returns an amoeba movement given the current state of
        the amoeba map.

        Args:
            last_percept: contains state information after the previous move
            current_percept: contains current state information
            info: a byte (ranging from 0 to 256) to convey information from
                  the previous turn

        Returns:
            1. A list of cells on the periphery that the amoeba retracts
            2. A list of cells the retracted cells have moved to
            3. A byte of information (values range from 0 to 255) that the
               amoeba can use
        """
        self.current_size = current_percept.current_size
        mini = min(5, len(current_percept.periphery) // 2)
        for i, j in current_percept.bacteria:
            current_percept.amoeba_map[i][j] = 1

        retract = [
            tuple(i)
            for i in self.rng.choice(
                current_percept.periphery, replace=False, size=mini
            )
        ]

        movable = self.find_movable_cells(
            retract, current_percept.periphery,
            current_percept.amoeba_map, current_percept.bacteria, mini)

        info = 0

        return retract, movable, info

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