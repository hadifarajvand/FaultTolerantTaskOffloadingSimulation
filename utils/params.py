# params.py
from utils.configuration import ConfigLoader
import itertools
from scipy.stats import norm

config = ConfigLoader()

class params:
    model_summary = "ddpg model"
    SCENARIO_TYPE = config.get('SCENARIO_TYPE')
    Permutation_Number = config.get('Permutation_Number')
    NUM_EDGE_SERVERS = config.get('NUM_EDGE_SERVERS')
    NUM_CLOUD_SERVERS = config.get('NUM_CLOUD_SERVERS')
    serverNo = NUM_EDGE_SERVERS + NUM_CLOUD_SERVERS
    alpha_edge = (None, None)
    alpha_cloud = (None, None)
    taskno = config.get('taskno')
    total_episodes = config.get('total_episodes')
    num_states = config.get('num_states')
    num_actions = None  # Will be set after class definition
    std_dev = config.get('std_dev')
    activation_function = config.get('activation_function')
    critic_lr = config.get('critic_lr')
    actor_lr = config.get('actor_lr')
    gamma = config.get('gamma')
    tau = config.get('tau')
    buffer_capacity = config.get('buffer_capacity')
    batch_size = config.get('batch_size')
    TASK_ARRIVAL_RATE = config.get('TASK_ARRIVAL_RATE')
    rsu_to_edge_profile = config.get('rsu_to_edge_profile')
    rsu_to_cloud_profile = config.get('rsu_to_cloud_profile')
    AGENT_TYPE = config.get('AGENT_TYPE', 'ddpg')

    @staticmethod
    def compute_Alpha():
        FAILURE_RATES = config.get('FAILURE_RATES')
        taskno = config.get('taskno')
        Low_demand = config.get('Low_demand')
        High_demand = config.get('High_demand')
        mean = (Low_demand + High_demand) / 2
        std = (High_demand - Low_demand) / 6
        def get_failure_rate_interval(prob_interval):
            lower_percentile_value = norm.ppf(1 - float(prob_interval[0]), loc=mean, scale=std)
            upper_percentile_value = norm.ppf(1 - float(prob_interval[1]), loc=mean, scale=std)
            return (1 / lower_percentile_value, 1 / upper_percentile_value)
        def get_alpha(failure_rates):
            return ((failure_rates[1] - failure_rates[0]) / taskno, failure_rates[1])
        alpha = {'edge': {'homogeneous': {}, 'heterogeneous': {}}, 'cloud': {'homogeneous': {}, 'heterogeneous': {}}}
        for loc in ['edge', 'cloud']:
            for scenario in ['homogeneous', 'heterogeneous']:
                for level in ['low', 'high', 'med']:
                    prob_interval = FAILURE_RATES[loc][scenario][level]
                    rates = get_failure_rate_interval(prob_interval)
                    alpha[loc][scenario][level] = get_alpha(rates)
        return alpha
    Alpha = compute_Alpha()

    @staticmethod
    def compute_num_actions():
        serverNo = params.NUM_EDGE_SERVERS + params.NUM_CLOUD_SERVERS
        return (serverNo * serverNo) + (serverNo * (serverNo - 1)) // 2

    @staticmethod
    def get_agent_class():
        if params.AGENT_TYPE == 'ddpg':
            from agent.ddpg import ddpgModel
            return ddpgModel
        elif params.AGENT_TYPE == 'td3':
            from agent.td3 import td3Model
            return td3Model
        elif params.AGENT_TYPE == 'a2c':
            from agent.a2c import a2cModel
            return a2cModel
        else:
            raise ValueError(f"Unknown AGENT_TYPE: {params.AGENT_TYPE}")

# Set num_actions after class definition
params.num_actions = params.compute_num_actions()

    
    
    