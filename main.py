from Q_Agent import *

if __name__ == "__main__":
    env = RubiksCube2x2()
    agent, curriculum_state = RubikQLearningAgent.load("rubik_agent_final.h5", env)
    if agent is None:
        ## no saved agent, make a new one
        agent = RubikQLearningAgent(
            env=env,
            epsilon=1.0,
            epsilon_decay=0.99995,
            epsilon_min=0.02,
            alpha=0.2,
            gamma=0.99,
            max_search_depth=11
        )
    if curriculum_state:
        agent.curriculum_state = curriculum_state
        agent.curriculum_state['last_episode'] = 0


    # agent.train(num_episodes=100000, max_steps_per_episode=150, scramble_lengths=list(range(1, 12)), log_interval=1000)
    agent.evaluate(num_episodes=100)
    scramble = env.scramble(5)
    solved, steps, solution = agent.solve(scramble, visualize=True)
    logger.info(f"Solved: {solved}, Steps: {steps}, Solution: {' '.join(solution)}")