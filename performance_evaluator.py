import matplotlib.pyplot as plt
import numpy as np

class PerformanceEvaluator:
    def __init__(self, env):
        """
        Initialize the PerformanceEvaluator with the environment.

        Args:
            env (gym.Env): The environment instance (e.g., MicrogridEnv).
        """
        self.env = env
        self.reset()

    def reset(self):
        """Reset all data collections."""
        self.time_steps = []
        # self.htf_outlet_temp = []
        # self.net_demand_history = []
        # self.energy_cost_history = []
        # self.penalty_history = []
        self.reward_history = []
        self.cumulative_reward = []
        # self.battery_rate_history = []
        # self.renewable_utilization_history = []
        # self.demand_history = []
        # self.renewable_gen_history = []
        self.total_reward = 0

    def collect_data(self, step, state, action, reward, info):
        """
        Collect data at each time step during the simulation.

        Args:
            step (int): The current time step.
            state (np.ndarray): The current state of the environment.
            action (np.ndarray): The action taken by the agent.
            reward (float): The reward received after taking the action.
            info (dict): Additional information from the environment.
        """
        self.total_reward += reward

        # Append data to lists
        self.time_steps.append(step)
        # self.htf_outlet_temp.append(info['soc'])
        # self.net_demand_history.append(info['net_demand'])
        # self.energy_cost_history.append(info['energy_cost'])
        # self.penalty_history.append(info['penalty'])
        self.reward_history.append(reward)
        self.cumulative_reward.append(self.total_reward)
        # self.battery_rate_history.append(action[0] * self.env.max_battery_rate)
        # self.renewable_utilization_history.append(action[1] * state[1])  # state[1] is renewable_gen
        # self.demand_history.append(state[2])  # state[2] is demand
        # self.renewable_gen_history.append(state[1])  # state[1] is renewable_gen

    def plot_results(self):
        """Plot the collected data to evaluate the agent's performance."""
        # Convert lists to numpy arrays for plotting
        time_steps = np.array(self.time_steps)
        soc_history = np.array(self.soc_history)
        net_demand_history = np.array(self.net_demand_history)
        energy_cost_history = np.array(self.energy_cost_history)
        penalty_history = np.array(self.penalty_history)
        reward_history = np.array(self.reward_history)
        cumulative_reward = np.array(self.cumulative_reward)
        battery_rate_history = np.array(self.battery_rate_history)
        renewable_utilization_history = np.array(self.renewable_utilization_history)
        demand_history = np.array(self.demand_history)
        renewable_gen_history = np.array(self.renewable_gen_history)

        # Plotting
        fig, axs = plt.subplots(7, 1, figsize=(12, 30), sharex=True)

        # State of Charge over time
        axs[0].plot(time_steps, soc_history, label='State of Charge (SOC)', color='blue')
        axs[0].set_ylabel('SOC')
        axs[0].legend()
        axs[0].grid(True)

        # Battery Charge/Discharge Rate over time
        axs[1].plot(time_steps, battery_rate_history, label='Battery Rate', color='orange')
        axs[1].set_ylabel('Battery Rate (kW)')
        axs[1].legend()
        axs[1].grid(True)

        # Renewable Utilization over time
        axs[2].plot(time_steps, renewable_utilization_history, label='Renewable Utilization', color='green')
        axs[2].set_ylabel('Renewable Utilization (kW)')
        axs[2].legend()
        axs[2].grid(True)

        # Demand and Renewable Generation over time
        axs[3].plot(time_steps, demand_history, label='Demand', color='red')
        axs[3].plot(time_steps, renewable_gen_history, label='Renewable Generation', color='cyan')
        axs[3].set_ylabel('Power (kW)')
        axs[3].legend()
        axs[3].grid(True)

        # Net Demand over time
        axs[4].plot(time_steps, net_demand_history, label='Net Demand', color='purple')
        axs[4].set_ylabel('Net Demand (kW)')
        axs[4].legend()
        axs[4].grid(True)

        # Energy Cost over time
        axs[5].plot(time_steps, energy_cost_history, label='Energy Cost', color='brown')
        axs[5].set_ylabel('Cost ($)')
        axs[5].legend()
        axs[5].grid(True)

        # Cumulative Reward over time
        axs[6].plot(time_steps, cumulative_reward, label='Cumulative Reward', color='black')
        axs[6].set_xlabel('Time Step')
        axs[6].set_ylabel('Cumulative Reward')
        axs[6].legend()
        axs[6].grid(True)

        plt.tight_layout()
        plt.show()