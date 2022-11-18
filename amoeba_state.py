class AmoebaState:
    def __init__(self, current_size, amoeba_map, periphery, bacteria, movable_cells):
        """
            Args:
                current_size (int): current size of the amoeba
                amoeba_map (numpy array): 2D array that represents the state of the board known to the amoeba
                periphery (List[Tuple[int, int]]: list of cells on the periphery of the amoeba
                bacteria (List[Tuple[int, int]]: list of bacteria known to the amoeba
                movable_cells (List[Tuple[int, int]]: list of movable positions given the current amoeba state
        """
        self.current_size = current_size
        self.amoeba_map = amoeba_map
        self.periphery = periphery
        self.bacteria = bacteria
        self.movable_cells = movable_cells
