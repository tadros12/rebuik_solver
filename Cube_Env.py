import numpy as np
import random


class RubiksCube2x2:
    FACES = ["U", "L", "F", "R", "B", "D"]
    COLORS = {0: "white", 1: "orange", 2: "green", 3: "red", 4: "blue", 5: "yellow"}
    MOVES = ["U", "U'", "L", "L'", "F", "F'", "R", "R'", "B", "B'", "D", "D'"]
    
    ## cubie positions and their colors, keeps track of where each piece is
    CUBIES = [
        [0, 2, 1],  ## UFL: white, green, orange
        [0, 2, 3],  ## UFR: white, green, red
        [0, 4, 1],  ## UBL: white, blue, orange
        [0, 4, 3],  ## UBR: white, blue, red
        [5, 2, 1],  ## DFL: yellow, green, orange
        [5, 2, 3],  ## DFR: yellow, green, red
        [5, 4, 1],  ## DBL: yellow, blue, orange
        [5, 4, 3],  ## DBR: yellow, blue, red
    ]
    
    ## how each move shuffles cubies and their orientations
    MOVE_PERMS = {
        "U": ([0, 1, 3, 2], [0, 0, 0, 0]),
        "U'": ([0, 2, 3, 1], [0, 0, 0, 0]),
        "L": ([0, 4, 6, 2], [1, 2, 1, 2]),
        "L'": ([0, 2, 6, 4], [2, 1, 2, 1]),
        "F": ([0, 1, 5, 4], [1, 2, 1, 2]),
        "F'": ([0, 4, 5, 1], [2, 1, 2, 1]),
        "R": ([1, 3, 7, 5], [1, 2, 1, 2]),
        "R'": ([1, 5, 7, 3], [2, 1, 2, 1]),
        "B": ([2, 3, 7, 6], [1, 2, 1, 2]),
        "B'": ([2, 6, 7, 3], [2, 1, 2, 1]),
        "D": ([4, 6, 7, 5], [0, 0, 0, 0]),
        "D'": ([4, 5, 7, 6], [0, 0, 0, 0]),
    }
    
    ## maps cubie positions to sticker indices
    STICKER_MAP = {
        (0, 0): 0, (0, 1): 8, (0, 2): 4,
        (1, 0): 1, (1, 1): 9, (1, 2): 12,
        (2, 0): 2, (2, 1): 16, (2, 2): 5,
        (3, 0): 3, (3, 1): 17, (3, 2): 13,
        (4, 0): 20, (4, 1): 10, (4, 2): 6,
        (5, 0): 21, (5, 1): 11, (5, 2): 14,
        (6, 0): 22, (6, 1): 18, (6, 2): 7,
        (7, 0): 23, (7, 1): 19, (7, 2): 15,
    }
    
    INVERSE_STICKER_MAP = {v: k for k, v in STICKER_MAP.items()}
    
    def __init__(self):
        ## initialize a fresh cube
        self.reset()
        self.solved_state = self.get_solved_state()
    
    def get_solved_state(self):
        ## what a solved cube looks like: all cubies in place, no rotations
        return np.arange(8, dtype=np.uint8), np.zeros(8, dtype=np.uint8)
    
    def reset(self):
        ## set cube back to solved
        self.cubie_positions, self.cubie_orientations = self.get_solved_state()
        return self.get_sticker_state()
    
    def apply_move(self, move):
        if move not in self.MOVE_PERMS:
            raise ValueError(f"Move {move} is no good!")
        
        affected_positions, ori_changes = self.MOVE_PERMS[move]
        affected_positions = np.array(affected_positions, dtype=np.uint8)
        ori_changes = np.array(ori_changes, dtype=np.uint8)
        
        ## shuffle cubie positions around
        self.cubie_positions[affected_positions] = self.cubie_positions[affected_positions[[1, 2, 3, 0]]]
        ## tweak orientations based on the move
        self.cubie_orientations[affected_positions] = (self.cubie_orientations[affected_positions] + ori_changes) % 3
        
        return self.get_sticker_state()
    
    def get_inverse_move(self, move):
        ## flips U to U' and vice versa
        return move[:-1] if move.endswith("'") else move + "'"
    
    def scramble(self, num_moves, avoid_redundant=True, seed=None):
        if seed is not None:
            random.seed(seed)
        
        moves = []
        last_face = None
        
        ## make random moves, avoid repeating same face if we want
        for _ in range(num_moves):
            available_moves = self.MOVES.copy()
            if avoid_redundant and last_face:
                available_moves = [m for m in available_moves if m[0] != last_face]
            move = random.choice(available_moves)
            self.apply_move(move)
            moves.append(move)
            last_face = move[0]
        
        if seed is not None:
            random.seed()  ## reset seed so we don't mess up other stuff
        
        return moves
    
    def is_solved(self):
        solved_positions, solved_orientations = self.get_solved_state()
        ## check if everything's in the right spot and orientation
        return np.array_equal(self.cubie_positions, solved_positions) and \
               np.array_equal(self.cubie_orientations, solved_orientations)
    
    def get_state(self):
        ## just grab the current sticker state
        return self.get_sticker_state()
    
    def set_state(self, state):
        if isinstance(state, tuple):
            state = np.array(state, dtype=np.uint8)
        if not isinstance(state, np.ndarray) or state.shape != (24,) or not np.all(np.isin(state, range(6))):
            raise ValueError("State needs to be 24 numbers from 0-5")
        
        ## convert sticker state to cubie positions and orientations
        positions, orientations = self._sticker_to_cubie_state(state)
        self.cubie_positions = positions
        self.cubie_orientations = orientations
    
    def get_sticker_state(self):
        sticker_state = np.zeros(24, dtype=np.uint8)
        pos_face_pairs = np.array(list(self.STICKER_MAP.keys()), dtype=np.uint8)
        sticker_indices = np.array(list(self.STICKER_MAP.values()), dtype=np.uint8)
        
        ## figure out what color goes where based on cubie positions and orientations
        cubie_ids = self.cubie_positions[pos_face_pairs[:, 0]]
        orientations = self.cubie_orientations[pos_face_pairs[:, 0]]
        face_indices = pos_face_pairs[:, 1]
        rotated_face_indices = (face_indices + orientations) % 3
        colors = np.array(self.CUBIES, dtype=np.uint8)[cubie_ids, rotated_face_indices]
        
        sticker_state[sticker_indices] = colors
        return sticker_state
    
    def _sticker_to_cubie_state(self, sticker_state):
        positions = np.zeros(8, dtype=np.uint8)
        orientations = np.zeros(8, dtype=np.uint8)
        cubie_stickers = {}
        
        ## group stickers by cubie position
        for sticker_idx, color in enumerate(sticker_state):
            if sticker_idx in self.INVERSE_STICKER_MAP:
                pos, face_idx = self.INVERSE_STICKER_MAP[sticker_idx]
                if pos not in cubie_stickers:
                    cubie_stickers[pos] = {}
                cubie_stickers[pos][face_idx] = color
        
        ## match sticker colors to cubie IDs and orientations
        for pos, stickers in cubie_stickers.items():
            if len(stickers) != 3:
                raise ValueError(f"Position {pos} has {len(stickers)} stickers, needs 3")
            colors = np.array([stickers[face_idx] for face_idx in range(3)], dtype=np.uint8)
            
            found = False
            for cubie_id, base_colors in enumerate(self.CUBIES):
                base_colors = np.array(base_colors, dtype=np.uint8)
                for ori in range(3):
                    rotated_colors = np.roll(base_colors, ori)
                    if np.array_equal(rotated_colors, colors):
                        positions[pos] = cubie_id
                        orientations[pos] = ori
                        found = True
                        break
                if found:
                    break
            if not found:
                raise ValueError(f"No cubie matches colors {colors} at position {pos}")
        
        return positions, orientations
    
    def get_compact_state(self):
        ## pack state into a tuple for Q-table
        return (tuple(self.cubie_positions.tolist()), tuple(self.cubie_orientations.tolist()))
    
    def explore_next_states(self, states):
        batch_size = states.shape[0]
        num_actions = len(self.MOVES)
        next_states = np.zeros((batch_size, num_actions, 24), dtype=np.uint8)
        goals = np.zeros((batch_size, num_actions), dtype=bool)
        
        ## try every move for each state and see what happens
        for i in range(batch_size):
            original_positions = self.cubie_positions.copy()
            original_orientations = self.cubie_orientations.copy()
            self.set_state(states[i])
            
            for j, move in enumerate(self.MOVES):
                self.apply_move(move)
                next_states[i, j] = self.get_sticker_state()
                goals[i, j] = self.is_solved()
                self.cubie_positions = original_positions.copy()
                self.cubie_orientations = original_orientations.copy()
            
            self.cubie_positions = original_positions
            self.cubie_orientations = original_orientations
        
        return next_states, goals
    
    def get_move_index(self, move):
        ## get index of a move
        return self.MOVES.index(move)
    
    def get_move_from_index(self, index):
        ## get move from index
        return self.MOVES[index]