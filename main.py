import numpy as np
from solar_environment import SolarEnv
from performance_evaluator import PerformanceEvaluator
from ddpg.agent import Agent
from utils import plotLearning
from data_handler import data_loader, increase_data_resolution

if __name__ == '__main__':
    solar_data = data_loader(path='data/solar_irradation_Izmir_2017_2019.csv')
    solar_data = increase_data_resolution(solar_data)
    env = SolarEnv()
    evaluator = PerformanceEvaluator(env)
    agent = Agent(alpha=0.000025,
                  beta=0.00025, 
                  input_dims=env.observation_space.shape, 
                  tau=0.001, 
                  env=env,
                  batch_size=64,
                  layer1_size=50, 
                  layer2_size=50, 
                  n_actions=env.action_space.shape[0])
    
    
    np.random.seed(0)
    score_history = []
    for i in range(len(solar_data)):
        obs = env.reset(solar_daily_data=solar_data[i])
        episode_length = len(env.solar_radiation_profile)
        evaluator.reset()
        done = False
        score = 0
        while not done:
            act = agent.choose_action(obs)
            # print(f"action: {act}")
            new_state, reward, done, info = env.step(act, episode_length)
            # print(new_state, reward)
            evaluator.collect_data(i, obs, act, reward, info)
            agent.remember(obs, act, reward, new_state, int(done))
            agent.learn()
            score += reward
            obs = new_state
        average_score = score / episode_length
        score_history.append(average_score)
        print('episode ', i, 'score %.2f' % average_score,
          'trailing 100 games avg %.3f' % np.mean(score_history[-100:]))
        # if i==49:
        #     evaluator.plot_results()
    
    filename = 'LunarLander-alpha000025-beta00025-400-300.png'
    plotLearning(score_history, filename, window=100)
