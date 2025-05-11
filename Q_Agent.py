import numpy as np
import random
import h5py
from datetime import datetime
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Rectangle
from tqdm import tqdm
from collections import defaultdict
import time
import copy
import logging
from logging.handlers import RotatingFileHandler
from Search_Utils import (HeuristicFunction, batchedWeightedAStarSearch)
from Cube_Env import RubiksCube2x2

log_handler = RotatingFileHandler("rubik_agent.log", maxBytes=10*1024*1024, backupCount=5)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[log_handler, logging.StreamHandler()]
)
logger = logging.getLogger("RubikRL")

class RubikQLearningAgent:
    def __init__(self, env, epsilon=1.0, epsilon_decay=0.9999, epsilon_min=0.02,
                 alpha=0.1, gamma=0.99, max_search_depth=11):
        self.env = env
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.alpha = alpha
        self.gamma = gamma
        self.max_search_depth = max_search_depth
        self.q_table = defaultdict(lambda: np.zeros(len(env.MOVES)))
        self.visit_counts = defaultdict(lambda: np.zeros(len(env.MOVES)))
        self.episode_rewards = []
        self.episode_steps = []
        self.solved_count = 0
        self.total_episodes = 0
        self.heuristic = HeuristicFunction()
    
    def select_action(self, state, exploration=True):
        if exploration and random.random() < self.epsilon:
            ## randomly explore sometimes
            return random.randint(0, len(self.env.MOVES) - 1)
        
        state_tuple = tuple(state[0]) + tuple(state[1])
        if state_tuple not in self.q_table and exploration:
            ## if we don't know this state, try a search
            return self._search_best_move(state)
        
        ## pick the move with the best Q-value
        return np.argmax(self.q_table[state_tuple])
    
    def _search_best_move(self, state):
        cube = RubiksCube2x2()
        cube.set_state(self.env.get_sticker_state())
        
        ## use A* to find a good move
        moves, _, _, is_solved, _ = batchedWeightedAStarSearch(
            scramble=cube.get_sticker_state(),
            depth_weight=1.0,
            num_parallel=1,
            env=self.env,
            heuristic_fn=self.heuristic,
            max_search_itr=self.max_search_depth,
            verbose=False
        )
        
        if is_solved and moves:
            move = moves[0]
            return self.env.get_move_index(move) if move in self.env.MOVES else random.randint(0, len(self.env.MOVES) - 1)
        
        ## fallback to random move
        return random.randint(0, len(self.env.MOVES) - 1)
    
    def update_q(self, state, action, reward, next_state, done):
        state_tuple = tuple(state[0]) + tuple(state[1])
        next_state_tuple = tuple(next_state[0]) + tuple(next_state[1])
        
        self.visit_counts[state_tuple][action] += 1
        
        ## update Q-value based on reward and next state
        if not done:
            best_next_action = np.argmax(self.q_table[next_state_tuple])
            td_target = reward + self.gamma * self.q_table[next_state_tuple][best_next_action]
        else:
            td_target = reward
        
        self.q_table[state_tuple][action] += self.alpha * (td_target - self.q_table[state_tuple][action])
    
    def update_epsilon(self):
        ## slowly reduce exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def train(self, num_episodes, max_steps_per_episode=150, scramble_lengths=None, log_interval=100):
        if scramble_lengths is None:
            scramble_lengths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        
        start_time = time.time()
        solved_lengths = {length: 0 for length in scramble_lengths}
        attempts_lengths = {length: 0 for length in scramble_lengths}
        
        ## keep track of where we are in the curriculum
        curriculum_state = getattr(self, 'curriculum_state', {
            'current_length': 1,
            'length_index': 0,
            'last_episode': 0,
            'solve_rates': {length: 0.0 for length in scramble_lengths},
            'solve_counts': {length: 0 for length in scramble_lengths},
            'attempt_counts': {length: 0 for length in scramble_lengths}
        })
        
        if curriculum_state['current_length'] not in scramble_lengths:
            curriculum_state['current_length'] = scramble_lengths[0]
            curriculum_state['length_index'] = 0
        
        for episode in tqdm(range(curriculum_state['last_episode'], num_episodes)):
            self.total_episodes += 1
            scramble_length = curriculum_state['current_length']
            
            self.env.reset()
            self.env.scramble(scramble_length)
            state = self.env.get_compact_state()
            
            episode_reward = 0
            curriculum_state['attempt_counts'][scramble_length] += 1
            attempts_lengths[scramble_length] += 1
            
            for step in range(max_steps_per_episode):
                action = self.select_action(state)
                self.env.apply_move(self.env.MOVES[action])
                next_state = self.env.get_compact_state()
                
                solved = self.env.is_solved()
                
                ## reward system: big bonus for solving, small penalties otherwise
                if solved:
                    reward = 10.0 + max(0, (max_steps_per_episode - step)/max_steps_per_episode * 10)
                    self.solved_count += 1
                    solved_lengths[scramble_length] += 1
                    curriculum_state['solve_counts'][scramble_length] += 1
                elif step == max_steps_per_episode - 1:
                    reward = -1.0
                else:
                    reward = -0.1
                    state_array = self.env.get_sticker_state().reshape(6, 4)
                    solved_faces = sum(np.all(state_array[face] == face) for face in range(6))
                    reward += solved_faces * 0.05
                
                self.update_q(state, action, reward, next_state, solved)
                
                episode_reward += reward
                state = next_state
                
                if solved:
                    break
            
            self.update_epsilon()
            self.episode_rewards.append(episode_reward)
            self.episode_steps.append(step + 1)
            
            if (episode + 1) % log_interval == 0:
                avg_reward = np.mean(self.episode_rewards[-log_interval:])
                avg_steps = np.mean(self.episode_steps[-log_interval:])
                solve_rate = self.solved_count / log_interval if episode >= log_interval else 0
                self.solved_count = 0
                
                logger.info(f"Ep {episode+1}/{num_episodes} | Reward: {avg_reward:.2f} | "
                           f"Steps: {avg_steps:.2f} | Solve: {solve_rate:.2f} | "
                           f"Eps: {self.epsilon:.4f} | Scramble: {curriculum_state['current_length']}")
                
                for length in scramble_lengths:
                    solves = curriculum_state['solve_counts'][length]
                    attempts = max(1, curriculum_state['attempt_counts'][length])
                    curriculum_state['solve_rates'][length] = solves / attempts
                    logger.info(f"Length {length}: {solves}/{attempts} = {curriculum_state['solve_rates'][length]:.2f}")
                    curriculum_state['solve_counts'][length] = 0
                    curriculum_state['attempt_counts'][length] = 0
                    solved_lengths[length] = 0
                    attempts_lengths[length] = 0
                
                current_rate = curriculum_state['solve_rates'][curriculum_state['current_length']]
                if current_rate > 0.7 and curriculum_state['length_index'] < len(scramble_lengths) - 1:
                    curriculum_state['length_index'] += 1
                    curriculum_state['current_length'] = scramble_lengths[curriculum_state['length_index']]
                    logger.info(f"Moving up to length {curriculum_state['current_length']} (Rate: {current_rate:.2f})")
            
        
        total_time = time.time() - start_time
        logger.info(f"Done training, took {total_time:.2f}s")
        
        curriculum_state['last_episode'] = num_episodes
        self.save("rubik_final.h5", curriculum_state)
        self.curriculum_state = curriculum_state
        
        self.plot_training_progress()
        self.plot_value_function_heatmap(max_states=20)
    
    def _get_scramble_length(self, curriculum_state):
        ## just grab the current scramble length
        return curriculum_state['current_length']
    
    def evaluate(self, num_episodes=100, scramble_lengths=None):
        if scramble_lengths is None:
            scramble_lengths = list(range(1, 12))
        
        results = {length: {"solved": 0, "total": 0} for length in scramble_lengths}
        total_steps = []
        
        current_epsilon = self.epsilon
        self.epsilon = 0.0  ## no exploration during eval
        
        logger.info("Checking how good this agent is...")
        
        for length in scramble_lengths:
            logger.info(f"Testing length {length}...")
            
            for _ in tqdm(range(num_episodes)):
                self.env.reset()
                self.env.scramble(length)
                state = self.env.get_compact_state()
                
                solved = False
                steps = 0
                
                ## try solving within a reasonable number of steps
                while steps < length * 5 + 20:
                    action = self.select_action(state, exploration=False)
                    self.env.apply_move(self.env.MOVES[action])
                    state = self.env.get_compact_state()
                    steps += 1
                    
                    if self.env.is_solved():
                        solved = True
                        break
                
                results[length]["total"] += 1
                if solved:
                    results[length]["solved"] += 1
                    total_steps.append(steps)
        
        self.epsilon = current_epsilon
        
        success_rates = {}
        for length in scramble_lengths:
            success_rate = results[length]["solved"] / results[length]["total"]
            success_rates[length] = success_rate
            logger.info(f"Length {length}: {success_rate:.2f} ({results[length]['solved']}/{results[length]['total']})")
        
        if total_steps:
            avg_steps = np.mean(total_steps)
            logger.info(f"Avg steps to solve: {avg_steps:.2f}")
        
        return success_rates
    
    def save(self, filepath, curriculum_state=None):
        temp_filepath = f"{filepath}.temp"
        
        ## only save non-zero Q-values to save space
        q_table_filtered = {k: v for k, v in self.q_table.items() if np.any(v != 0)}
        visit_counts_filtered = {k: v for k, v in self.visit_counts.items() if np.any(v != 0)}
        
        q_keys = list(q_table_filtered.keys())
        q_values = np.array([q_table_filtered[k] for k in q_keys], dtype=np.float16)
        q_key_lengths = []
        q_flattened_keys_list = []
        
        for key in q_keys:
            try:
                if not (isinstance(key, tuple) and len(key) == 16):
                    logger.warning(f"Bad Q-table key: {key}. Skipping.")
                    continue
                flattened_key = np.array(key, dtype=np.int8)
                q_flattened_keys_list.append(flattened_key)
                q_key_lengths.append(len(flattened_key))
            except Exception as e:
                logger.error(f"Error processing Q-table key {key}: {e}. Skipping.")
                continue
        
        q_flattened_keys = np.concatenate(q_flattened_keys_list) if q_key_lengths else np.array([], dtype=np.int8)
        
        logger.info(f"Q-table: {len(q_key_lengths)} keys, lengths: {set(q_key_lengths)}")
        
        vc_keys = list(visit_counts_filtered.keys())
        vc_values = np.array([visit_counts_filtered[k] for k in vc_keys], dtype=np.float16)
        vc_key_lengths = []
        vc_flattened_keys_list = []
        
        for key in vc_keys:
            try:
                if not (isinstance(key, tuple) and len(key) == 16):
                    logger.warning(f"Bad visit counts key: {key}. Skipping.")
                    continue
                flattened_key = np.array(key, dtype=np.int8)
                vc_flattened_keys_list.append(flattened_key)
                vc_key_lengths.append(len(flattened_key))
            except Exception as e:
                logger.error(f"Error processing visit counts key {key}: {e}. Skipping.")
                continue
        
        vc_flattened_keys = np.concatenate(vc_flattened_keys_list) if vc_key_lengths else np.array([], dtype=np.int8)
        
        logger.info(f"Visit counts: {len(vc_key_lengths)} keys, lengths: {set(vc_key_lengths)}")
        
        ## save everything to an HDF5 file
        with h5py.File(temp_filepath, 'w') as f:
            q_group = f.create_group('q_table')
            q_group.create_dataset('values', data=q_values, compression='gzip', compression_opts=1)
            q_group.create_dataset('flattened_keys', data=q_flattened_keys, compression='gzip', compression_opts=1)
            q_group.create_dataset('key_lengths', data=q_key_lengths)
            
            vc_group = f.create_group('visit_counts')
            vc_group.create_dataset('values', data=vc_values, compression='gzip', compression_opts=1)
            vc_group.create_dataset('flattened_keys', data=vc_flattened_keys, compression='gzip', compression_opts=1)
            vc_group.create_dataset('key_lengths', data=vc_key_lengths)
            
            params = f.create_group('parameters')
            params.create_dataset('epsilon', data=self.epsilon)
            params.create_dataset('alpha', data=self.alpha)
            params.create_dataset('gamma', data=self.gamma)
            params.create_dataset('total_episodes', data=self.total_episodes)
            params.create_dataset('max_search_depth', data=self.max_search_depth)
            
            metrics = f.create_group('metrics')
            metrics.create_dataset('episode_rewards', data=np.array(self.episode_rewards))
            metrics.create_dataset('episode_steps', data=np.array(self.episode_steps))
            
            if curriculum_state is not None:
                curriculum_group = f.create_group('curriculum_state')
                curriculum_group.attrs['current_length'] = curriculum_state['current_length']
                curriculum_group.attrs['length_index'] = curriculum_state['length_index']
                curriculum_group.attrs['last_episode'] = curriculum_state['last_episode']
                lengths = list(curriculum_state['solve_rates'].keys())
                solve_rates = [curriculum_state['solve_rates'][length] for length in lengths]
                solve_counts = [curriculum_state['solve_counts'][length] for length in lengths]
                attempt_counts = [curriculum_state['attempt_counts'][length] for length in lengths]
                curriculum_group.create_dataset('lengths', data=lengths)
                curriculum_group.create_dataset('solve_rates', data=solve_rates)
                curriculum_group.create_dataset('solve_counts', data=solve_counts)
                curriculum_group.create_dataset('attempt_counts', data=attempt_counts)
        
        if os.path.exists(filepath):
            os.remove(filepath)
        os.rename(temp_filepath, filepath)
        
        logger.info(f"Saved agent to {filepath}")

    @classmethod
    def load(cls, filepath, env):
        if not os.path.exists(filepath):
            logger.info(f"No checkpoint at {filepath}. Starting fresh.")
            return None, None
        
        logger.info(f"Loading from {filepath}...")
        
        ## create a new agent with default params
        agent = cls(
            env=env,
            epsilon=1.0,
            epsilon_decay=0.999,
            epsilon_min=0.02,
            alpha=0.1,
            gamma=0.99,
            max_search_depth=7
        )
        
        try:
            with h5py.File(filepath, 'r') as f:
                q_group = f['q_table']
                q_values = q_group['values'][:]
                q_flattened_keys = q_group['flattened_keys'][:]
                q_key_lengths = q_group['key_lengths'][:]
                
                q_keys = []
                start_idx = 0
                for length in q_key_lengths:
                    end_idx = start_idx + length
                    if length == 16:
                        key = tuple(q_flattened_keys[start_idx:end_idx])
                        q_keys.append(key)
                    else:
                        logger.warning(f"Bad Q-table key length {length}. Skipping.")
                    start_idx = end_idx
                
                agent.q_table = defaultdict(lambda: np.zeros(len(env.MOVES), dtype=np.float32))
                for i, key in enumerate(q_keys):
                    agent.q_table[key] = q_values[i].astype(np.float32)
                
                vc_group = f['visit_counts']
                vc_values = vc_group['values'][:]
                vc_flattened_keys = vc_group['flattened_keys'][:]
                vc_key_lengths = vc_group['key_lengths'][:]
                
                vc_keys = []
                start_idx = 0
                for length in vc_key_lengths:
                    end_idx = start_idx + length
                    if length == 16:
                        key = tuple(vc_flattened_keys[start_idx:end_idx])
                        vc_keys.append(key)
                    else:
                        logger.warning(f"Bad visit counts key length {length}. Skipping.")
                    start_idx = end_idx
                
                agent.visit_counts = defaultdict(lambda: np.zeros(len(env.MOVES), dtype=np.float32))
                for i, key in enumerate(vc_keys):
                    agent.visit_counts[key] = vc_values[i].astype(np.float32)
                
                params = f['parameters']
                agent.epsilon = float(params['epsilon'][()])
                agent.alpha = float(params['alpha'][()])
                agent.gamma = float(params['gamma'][()])
                agent.total_episodes = int(params['total_episodes'][()])
                agent.max_search_depth = int(params['max_search_depth'][()])
                
                metrics = f['metrics']
                agent.episode_rewards = list(metrics['episode_rewards'][:])
                agent.episode_steps = list(metrics['episode_steps'][:])
                
                curriculum_state = {}
                if 'curriculum_state' in f:
                    curriculum_group = f['curriculum_state']
                    curriculum_state['current_length'] = int(curriculum_group.attrs['current_length'])
                    curriculum_state['length_index'] = int(curriculum_group.attrs['length_index'])
                    curriculum_state['last_episode'] = int(curriculum_group.attrs['last_episode'])
                    if 'lengths' in curriculum_group:
                        lengths = curriculum_group['lengths'][:]
                        solve_rates = curriculum_group['solve_rates'][:]
                        solve_counts = curriculum_group['solve_counts'][:]
                        attempt_counts = curriculum_group['attempt_counts'][:]
                        curriculum_state['solve_rates'] = {int(length): float(rate) for length, rate in zip(lengths, solve_rates)}
                        curriculum_state['solve_counts'] = {int(length): int(count) for length, count in zip(lengths, solve_counts)}
                        curriculum_state['attempt_counts'] = {int(length): int(count) for length, count in zip(lengths, attempt_counts)}
            
            logger.info(f"Loaded {len(q_keys)} Q-table states and {len(vc_keys)} visit count states")
            return agent, curriculum_state
        except Exception as e:
            logger.error(f"Load failed {filepath}: {e}. Starting with empty agent.")
            return agent, None

    def plot_training_progress(self):
        if not self.episode_rewards:
            return

        episodes = np.arange(1, len(self.episode_rewards) + 1)
        rewards = np.array(self.episode_rewards)
        steps = np.array(self.episode_steps)

        def moving_average(x, w=50):
            ## smooth out the data for clearer plots
            return np.convolve(x, np.ones(w)/w, mode='valid') if len(x) >= w else x

        plt.style.use("default")
        fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        smoothed_rewards = moving_average(rewards)
        smoothed_steps = moving_average(steps)
        x_smoothed = episodes[-len(smoothed_rewards):]

        ## plot rewards over time
        axs[0].plot(episodes, rewards, color='skyblue', alpha=0.4, label='Reward (Raw)')
        axs[0].plot(x_smoothed, smoothed_rewards, color='blue', linewidth=2, label='Reward (Smoothed)')
        axs[0].set_title('Training Rewards')
        axs[0].set_ylabel('Reward')
        axs[0].legend()
        axs[0].grid(True, linestyle='--', alpha=0.5)
        axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))

        ## plot steps per episode
        axs[1].plot(episodes, steps, color='lightgreen', alpha=0.4, label='Steps (Raw)')
        axs[1].plot(x_smoothed, smoothed_steps, color='green', linewidth=2, label='Steps (Smoothed)')
        axs[1].set_title('Steps per Episode')
        axs[1].set_xlabel('Episode')
        axs[1].set_ylabel('Steps')
        axs[1].legend()
        axs[1].grid(True, linestyle='--', alpha=0.5)
        axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))

        plt.tight_layout()
        plt.savefig("training_progress.png", dpi=300)
        plt.close()
        logger.info("Saved training progress to training_progress.png")

    def plot_value_function_heatmap(self, max_states=40):
        if not self.q_table:
            logger.info("No Q-table data for heatmap.")
            return

        keys = list(self.q_table.keys())[:max_states]
        q_values = np.array([self.q_table[k] for k in keys])
        moves = self.env.MOVES

        ## create a heatmap of Q-values
        fig, ax = plt.subplots(figsize=(min(18, 1.5*len(moves)), max(6, 0.4 * len(keys))))
        cmap = plt.get_cmap('coolwarm')
        cax = ax.imshow(q_values, cmap=cmap, aspect='auto', interpolation='nearest')

        ax.set_xticks(np.arange(len(moves)))
        ax.set_xticklabels(moves, rotation=45, ha='right')
        ax.set_yticks(np.arange(len(keys)))
        ax.set_yticklabels([f"State {i+1}" for i in range(len(keys))])
        ax.set_title("Q-Value Heatmap")

        fig.colorbar(cax, ax=ax).set_label("Q-Value")

        for i in range(len(keys)):
            for j in range(len(moves)):
                val = q_values[i, j]
                ax.text(j, i, f"{val:.2f}", ha='center', va='center',
                        color='white' if val > np.mean(q_values) else 'black')

        plt.tight_layout()
        plt.savefig("value_function_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Saved heatmap to value_function_heatmap.png")

    def solve(self, scramble_sequence=None, max_steps=100, visualize=False):
        self.env.reset()
        if scramble_sequence:
            for move in scramble_sequence:
                try:
                    self.env.apply_move(move)
                except ValueError as e:
                    logger.error(f"Bad move in scramble: {e}")
                    return False, 0, []
        state = self.env.get_compact_state()
        solution = []
        for step in range(max_steps):
            action = self.select_action(state, exploration=False)
            move = self.env.MOVES[action]
            try:
                self.env.apply_move(move)
            except ValueError as e:
                logger.error(f"Bad move during solve: {e}")
                break
            solution.append(move)
            state = self.env.get_compact_state()
            if self.env.is_solved():
                if visualize:
                    self.plot_solution(solution, scramble_sequence)
                return True, step + 1, solution
        return False, max_steps, solution

    def plot_solution(self, solution, scramble_sequence=None, initial_state=None, show=False, save=True):
        states = []
        solved_cube = copy.deepcopy(self.env)
        solved_cube.reset()
        states.append((solved_cube.get_sticker_state().copy(), "Solved"))

        if scramble_sequence:
            scrambled_cube = copy.deepcopy(solved_cube)
            scramble_moves = []
            for move in scramble_sequence:
                scrambled_cube.apply_move(move)
                scramble_moves.append(move)
            states.append((scrambled_cube.get_sticker_state().copy(), f"Scrambled\n({' '.join(scramble_moves)})"))

        solution_cube = copy.deepcopy(self.env)
        solution_cube.reset()

        if initial_state is not None:
            solution_cube.set_state(initial_state)
        elif scramble_sequence:
            for move in scramble_sequence:
                solution_cube.apply_move(move)

        states.append((solution_cube.get_sticker_state().copy(), "Initial State"))

        for i, move in enumerate(solution):
            solution_cube.apply_move(move)
            states.append((solution_cube.get_sticker_state().copy(), f"Step {i+1}: {move}"))

        ## create subplots for each state
        fig, axs = plt.subplots(1, len(states), figsize=(4*len(states), 5))
        if len(states) == 1:
            axs = [axs]

        for i, (state, title) in enumerate(states):
            self._plot_cube_state(state, title, axs[i])

        plt.tight_layout()
        plt.subplots_adjust(left=0.03, right=0.97, top=0.92, bottom=0.08)

        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"solution_steps_{timestamp}.png"
            plt.savefig(filename, dpi=300)
            logger.info(f"Saved viz to: {filename}")

        if show:
            plt.show()
        else:
            plt.close()

    def _plot_cube_state(self, state, title, ax):
        color_map = {
            0: "#FFFFFF",  # White
            1: "#FFA500",  # Orange
            2: "#008000",  # Green
            3: "#FF0000",  # Red
            4: "#0000FF",  # Blue
            5: "#FFFF00",  # Yellow
        }

        sticker_positions = {
            ## Up face (index 0) - top middle
            (0, 0): (1, 2),
            (0, 1): (2, 2),
            (0, 2): (1, 1),
            (0, 3): (2, 1),
            ## Left face (index 1) - middle left
            (1, 0): (0, 0),
            (1, 1): (0, -1),
            (1, 2): (-1, 0),
            (1, 3): (-1, -1),
            ## Front face (index 2) - middle center
            (2, 0): (1, 0),
            (2, 1): (2, 0),
            (2, 2): (1, -1),
            (2, 3): (2, -1),
            ## Right face (index 3) - middle right
            (3, 0): (4, 0),
            (3, 1): (3, 0),
            (3, 2): (3, -1),
            (3, 3): (4, -1),
            ## Back face (index 4) - middle far right
            (4, 0): (5, 0),
            (4, 1): (6, 0),
            (4, 2): (5, -1),
            (4, 3): (6, -1),
            ## Down face (index 5) - bottom middle
            (5, 0): (1, -2),
            (5, 1): (2, -2),
            (5, 2): (1, -3),
            (5, 3): (2, -3),
        }

        ## putting some face labels on faces positions within the plot
        face_labels = {
            0: (2, 2, "U"),
            1: (0, 0, "L"),
            2: (2, 0, "F"),
            3: (4, 0, "R"),
            4: (6, 0, "B"),
            5: (2, -2, "D"),
        }

        ax.set_aspect('equal')
        ax.axis('off')

        ## draw each sticker on the cube
        for face in range(6):
            for sticker in range(4):
                color_idx = state[face * 4 + sticker]
                color = color_map[color_idx]
                x, y = sticker_positions[(face, sticker)]
                rect = Rectangle((x, y), 1, 1, facecolor=color, edgecolor="black", linewidth=1.5)
                ax.add_patch(rect)

        ## add labels for each face
        for face, (x, y, label) in face_labels.items():
            ax.text(
                x, y, label,
                ha='center', va='center',
                fontsize=12, fontweight='bold',
                bbox=dict(
                    boxstyle="circle,pad=0.25",
                    facecolor='white',
                    edgecolor='black',
                    linewidth=1.5,
                    alpha=0.9
                )
            )

        ax.set_xlim(-1.5, 7.5)
        ax.set_ylim(-4.0, 4.0)
        ax.set_title(title, fontsize=11, fontweight='bold', pad=3, loc='left', x=0.32)