import gym
from gym import spaces
import numpy as np
from CoolProp.CoolProp import PropsSI
from math import fabs
from energy_models.solar_collector import calc_solar_outlet_temp
V_DOT_MIN = 0.00005
V_DOT_MAX = 10

class SolarEnv(gym.Env):
    """
    Custom Environment for PTSC collector
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(SolarEnv, self).__init__()

        # Action space: [HTF volumetric flow rate (0.00001 to 0.1)]
        self.action_space = spaces.Box(
            low=np.array([V_DOT_MIN]), 
            high=np.array([V_DOT_MAX]), 
            dtype=np.float32
        )

        # Observation space: [Outlet Temperature]
        self.observation_space = spaces.Box(
            low=np.array([500]), 
            high=np.array([600]), 
            dtype=np.float32
        )

        # Initialize state
        self.state = None
        self.time_step = 0
        # self.max_time_steps = 12  # Simulate a day in hours

        # # Microgrid parameters
        # self.battery_capacity = 100.0  # kWh
        # self.max_battery_rate = 50.0   # kW
        # self.charge_efficiency = 0.95
        # self.discharge_efficiency = 0.95
        self.initial_temp = 400         
        self.t_htf_in = 500
        self.p_htf_in = 1e5
        self.t_ambient = 300
        self.HTF_FLUID = "INCOMP::S800"  #'INCOMP::TVP1'
        self.target_htf_outlet_temp = 510


        # Load and renewable profiles
        # self.solar_radiation_profile = self._generate_solar_profile()
        
    def _generate_solar_profile(self, data):
        # Hourly (daily) solar radiation intensity (W/m2)
        # price = np.array([302, 335, 370, 428, 537, 588, 576, 503, 475, 395, 355, 330])
        solar_data = np.array(data)
        return solar_data
    
    def reset(self, solar_daily_data):
        self.solar_radiation_profile = self._generate_solar_profile(data=solar_daily_data)
        self.state = np.array([self.initial_temp], dtype=np.float32)
        self.time_step = 0
        return self.state
    
    def step(self, action, episode_length):
        volumetric_flow_rate = action[0]
        volumetric_flow_rate = np.clip(volumetric_flow_rate, V_DOT_MIN, V_DOT_MAX)
        m_dot_htf = PropsSI('D', 'P', self.p_htf_in, 'T', self.t_htf_in, self.HTF_FLUID) * volumetric_flow_rate
        # print(f"v_dot: {volumetric_flow_rate}, m_dot: {m_dot_htf}")
        t_htf_out, _, _ = calc_solar_outlet_temp(
            theta=0, 
            v_wind=20, 
            t_in=self.t_htf_in, 
            p_in=self.p_htf_in, 
            t_amb=self.t_ambient, 
            m_dot_htf=m_dot_htf,
            g_b=self.solar_radiation_profile[self.time_step]
        )

        # print(f"HTF outlet temperature: {t_htf_out}")

        reward = -fabs(t_htf_out - self.target_htf_outlet_temp)
        if volumetric_flow_rate == V_DOT_MIN or volumetric_flow_rate == V_DOT_MAX:
            reward -= 100
        self.time_step += 1
        done = self.time_step >= episode_length
        if not done:
            self.state = np.array([t_htf_out], dtype=np.float32)
        else:
            self.state = np.array([self.initial_temp], dtype=np.float32)

        info = {}

        return self.state, reward, done, info