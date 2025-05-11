import numpy as np
from heapq import heappop, heappush
import time
from Cube_Env import RubiksCube2x2


class Node:
    def __init__(self, state, parent, parent_move, depth, cost, is_solved):
        self.state = state  ## current cube state
        self.parent = parent  ## node we came from
        self.parent_move = parent_move  ## move that got us here
        self.depth = depth  ## how many moves deep
        self.cost = cost  ## cost for A* search
        self.is_solved = is_solved  ## is this a solved cube?
    
    def __hash__(self):
        ## hash state for quick lookup
        return hash(tuple(self.state))

class HeuristicFunction:
    def __call__(self, states):
        env = RubiksCube2x2()
        h_values = []
        ## estimate how far each state is from solved
        for state in states:
            env.set_state(state)
            misplaced = np.sum(env.cubie_positions != np.arange(8))
            misoriented = np.sum(env.cubie_orientations != 0)
            h_values.append(misplaced + misoriented)
        return np.array(h_values)

def batchedWeightedAStarSearch(scramble, depth_weight, num_parallel, env, heuristic_fn, max_search_itr, verbose=True, queue=None):
    open_nodes = []
    closed_nodes = {}
    
    ## start with the scrambled state
    root = Node(scramble, None, None, 0, 0, env.is_solved())
    heappush(open_nodes, (root.cost, id(root), root))
    closed_nodes[hash(root)] = root
    
    search_itr = 1
    num_nodes_generated = 1
    is_solved = False
    solved_node = None
    best_h_value = float('inf')
    start_time = time.time()
    
    ## keep searching until we solve it or give up
    while not is_solved and search_itr <= max_search_itr and open_nodes:
        start_iter_time = time.time()
        num_get = min(len(open_nodes), num_parallel)
        curr_nodes = []
        
        ## grab some nodes to explore
        for _ in range(num_get):
            if not open_nodes:
                break
            node = heappop(open_nodes)[2]
            curr_nodes.append(node)
            if node.is_solved:
                is_solved = True
                solved_node = node
                break
        
        if is_solved:
            break
        
        curr_states = np.array([node.state for node in curr_nodes])
        if not curr_states.size:
            break
        
        ## try all possible next moves
        children_states, goals = env.explore_next_states(curr_states)
        children = []
        depths = []
        
        for i in range(len(curr_nodes)):
            parent = curr_nodes[i]
            for j in range(len(children_states[i])):
                if np.any(children_states[i][j]):
                    action = env.get_move_from_index(j)
                    depths.append(parent.depth + 1)
                    children.append(
                        Node(
                            children_states[i][j],
                            parent,
                            action,
                            parent.depth + 1,
                            0,
                            goals[i][j],
                        )
                    )
        
        nodes_to_add_idx = []
        for i, child in enumerate(children):
            child_hash = hash(child)
            if child_hash in closed_nodes:
                ## update node if we found a shorter path
                if closed_nodes[child_hash].depth > child.depth:
                    found = closed_nodes.pop(child_hash)
                    found.depth = child.depth
                    found.parent = child.parent
                    found.parent_move = child.parent_move
                    children[i] = found
                    nodes_to_add_idx.append(i)
            else:
                closed_nodes[child_hash] = child
                nodes_to_add_idx.append(i)
        
        if not nodes_to_add_idx:
            search_itr += 1
            continue
        
        children = [children[i] for i in nodes_to_add_idx]
        depths = np.array([depths[i] for i in nodes_to_add_idx])
        children_states = np.array([child.state for child in children])
        
        ## use heuristic to prioritize promising moves
        h_value = heuristic_fn(children_states)
        best_h_value = min(h_value) if h_value.size else best_h_value
        costs = h_value + depth_weight * depths
        
        for cost, child in zip(costs, children):
            child.cost = cost
            heappush(open_nodes, (child.cost, id(child), child))
        
        num_nodes_generated += len(children)
        if verbose:
            print(f"Iter {search_itr} | Best H: {best_h_value:.2f} | Time: {time.time() - start_iter_time:.2f}s")
        
        search_itr += 1
    
    search_time = time.time() - start_time
    
    if is_solved:
        moves = []
        node = solved_node
        ## backtrack to get the solution
        while node.depth > 0:
            moves.append(node.parent_move)
            node = node.parent
        moves = moves[::-1]
    else:
        moves = None
    
    if queue:
        queue.put((moves, num_nodes_generated, search_itr, is_solved, search_time))
    else:
        return moves, num_nodes_generated, search_itr, is_solved, search_time