#configuration.py

import itertools
from scipy.stats import norm

class parameters:
    SCENARIO_TYPE = "homogeneous"  # Set to "homogeneous" or "heterogeneous"
    Permutation_Number = 1  # 1..24

    NUM_EDGE_SERVERS = 6  # 7,8
    NUM_CLOUD_SERVERS = 2  # 3,2
    serverNo = NUM_EDGE_SERVERS + NUM_CLOUD_SERVERS  # 10
    #Alpha = 1  # 0.005  for adjusting failure rate of server adaptively according to queue size of server


    failure_model_weight = 1  # Weight for "Permanent" failures model probability (0.9 means high)

    taskno = 200
    total_episodes = 300  # 100

    #epsilon=0.25
    alpha=0.1
    
    std_dev = 0.25
    critic_lr = 0.001   #0.002  
    actor_lr = 0.0003 #  0.0005  # 0.0005
    gamma = 0.85 # 0.9
    tau = 0.005 #0.01
    buffer_capacity = 100000 #50000
    batch_size = 256 #64
    activation_function ="softmax"  # # tanh(-1,1) sigmoid = (0,1), softmax= (0,1)

    TASK_ARRIVAL_RATE = 0.5 # Task arrival time, 0.1, 0.2
    TASK_SIZE_RANGE = (10, 100)  # heter
    
    # COMPUTATION_DEMAND_RANGE = (1, 100)
    Low_demand, High_demand = 1, 100

    rsu_to_edge_profile = {
        "bandwidth": 1,  # 100, 1000, 100000
        "propagation_delay": 0
    }

    rsu_to_cloud_profile = {
        "bandwidth": 8,  # 10, 100
        "propagation_delay": 1  # max penalty failure (<)
    }
    # Edge
    INITIAL_FAILURE_PROB_LOW_EDGE = 0.0001
    INITIAL_FAILURE_PROB_HIGH_EDGE = 0.79  # Initial high value
    INITIAL_FAILURE_PROB_MED_EDGE = 0.55
    HOMOGENEOUS_INTERVAL_EDGE = 0.10  # Interval length
    HETEROGENEOUS_INTERVAL_EDGE = 0.20  # Interval length
    EDGE_PROCESSING_FREQ_RANGE = (10, 15)  # combined range for both scenarios
    # Cloud
    INITIAL_FAILURE_PROB_LOW_CLOUD = 1e-6
    INITIAL_FAILURE_PROB_HIGH_CLOUD = 7.9e-6 # Initial high value
    INITIAL_FAILURE_PROB_MED_CLOUD = 5.5e-6
    HOMOGENEOUS_INTERVAL_CLOUD = 1e-6  # Interval length
    HETEROGENEOUS_INTERVAL_CLOUD = 2e-6  # Interval length
    CLOUD_PROCESSING_FREQ_RANGE = (30, 60)  # combined range for both scenarios 30-90
    '''
    STATES = {
        "S1": ("Low", 1 - failure_model_weight),  # 1-failure_model_weight means "Low" Permanent probability
        "S2": ("High", 1 - failure_model_weight),  # 1-failure_model_weight means "Low" Permanent probability
        "S3": ("Low", failure_model_weight),  # failure_model_weight means "High" Permanent probability
        "S4": ("High", failure_model_weight)  # failure_model_weight means "High" Permanent probability
    }
    '''
    STATES = {
        "S1": ("Low", 1 - failure_model_weight),  
        "S2": ("High", 1 - failure_model_weight),  
        "S3": ("Med", 1 - failure_model_weight)
        
    }

    @staticmethod
    def generate_state_permutations():
        permutations = list(itertools.permutations(parameters.STATES.keys()))
        return permutations

    @staticmethod
    def compute_failure_probabilities():
        failure_probs = {
            'edge': {
                'homogeneous': {
                    'low': (parameters.INITIAL_FAILURE_PROB_LOW_EDGE, parameters.INITIAL_FAILURE_PROB_LOW_EDGE + parameters.HOMOGENEOUS_INTERVAL_EDGE),
                    'high': (parameters.INITIAL_FAILURE_PROB_HIGH_EDGE, parameters.INITIAL_FAILURE_PROB_HIGH_EDGE + parameters.HOMOGENEOUS_INTERVAL_EDGE),
                    'med': (parameters.INITIAL_FAILURE_PROB_MED_EDGE, parameters.INITIAL_FAILURE_PROB_MED_EDGE + parameters.HOMOGENEOUS_INTERVAL_EDGE)
                },
                'heterogeneous': {
                    'low': (parameters.INITIAL_FAILURE_PROB_LOW_EDGE, parameters.INITIAL_FAILURE_PROB_LOW_EDGE + parameters.HETEROGENEOUS_INTERVAL_EDGE),
                    'high': (parameters.INITIAL_FAILURE_PROB_HIGH_EDGE, parameters.INITIAL_FAILURE_PROB_HIGH_EDGE + parameters.HETEROGENEOUS_INTERVAL_EDGE),
                    'med': (parameters.INITIAL_FAILURE_PROB_MED_EDGE, parameters.INITIAL_FAILURE_PROB_MED_EDGE + parameters.HETEROGENEOUS_INTERVAL_EDGE)

                }
            },
            'cloud': {
                'homogeneous': {
                    'low': (parameters.INITIAL_FAILURE_PROB_LOW_CLOUD, parameters.INITIAL_FAILURE_PROB_LOW_CLOUD + parameters.HOMOGENEOUS_INTERVAL_CLOUD),
                    'high': (parameters.INITIAL_FAILURE_PROB_HIGH_CLOUD, parameters.INITIAL_FAILURE_PROB_HIGH_CLOUD + parameters.HOMOGENEOUS_INTERVAL_CLOUD),
                    'med': (parameters.INITIAL_FAILURE_PROB_MED_CLOUD, parameters.INITIAL_FAILURE_PROB_MED_CLOUD + parameters.HOMOGENEOUS_INTERVAL_CLOUD)
                },
                'heterogeneous': {
                    'low': (parameters.INITIAL_FAILURE_PROB_LOW_CLOUD, parameters.INITIAL_FAILURE_PROB_LOW_CLOUD + parameters.HETEROGENEOUS_INTERVAL_CLOUD),
                    'high': (parameters.INITIAL_FAILURE_PROB_HIGH_CLOUD, parameters.INITIAL_FAILURE_PROB_HIGH_CLOUD + parameters.HETEROGENEOUS_INTERVAL_CLOUD),
                    'med': (parameters.INITIAL_FAILURE_PROB_MED_CLOUD, parameters.INITIAL_FAILURE_PROB_MED_CLOUD + parameters.HETEROGENEOUS_INTERVAL_CLOUD)
                }
            }
        }
        return failure_probs

    @staticmethod
    def compute_failure_rates():
        failure_probs = parameters.compute_failure_probabilities()

        mean= (parameters.Low_demand + parameters.High_demand) / 2  # Mean at the midpoint of the interval
        std = (parameters.High_demand - parameters.Low_demand) / 6  # Standard deviation chosen to fit the interval
        
        def get_failure_rate_interval(prob_interval):
            lower_percentile_value = norm.ppf(1 - prob_interval[0], loc=mean, scale=std)
            upper_percentile_value = norm.ppf(1 - prob_interval[1], loc=mean, scale=std)
            return (1 / lower_percentile_value, 1 / upper_percentile_value)

        failure_rates = {
            'edge': {
                'homogeneous': {
                    'low': get_failure_rate_interval(failure_probs['edge']['homogeneous']['low']),
                    'high': get_failure_rate_interval(failure_probs['edge']['homogeneous']['high']),
                    'med': get_failure_rate_interval(failure_probs['edge']['homogeneous']['med'])
                },
                'heterogeneous': {
                    'low': get_failure_rate_interval(failure_probs['edge']['heterogeneous']['low']),
                    'high': get_failure_rate_interval(failure_probs['edge']['heterogeneous']['high']),
                    'med': get_failure_rate_interval(failure_probs['edge']['heterogeneous']['med'])
                }
            },
            'cloud': {
                'homogeneous': {
                    'low': get_failure_rate_interval(failure_probs['cloud']['homogeneous']['low']),
                    'high': get_failure_rate_interval(failure_probs['cloud']['homogeneous']['high']),
                    'med': get_failure_rate_interval(failure_probs['cloud']['homogeneous']['med'])
                },
                'heterogeneous': {
                    'low': get_failure_rate_interval(failure_probs['cloud']['heterogeneous']['low']),
                    'high': get_failure_rate_interval(failure_probs['cloud']['heterogeneous']['high']),
                    'med': get_failure_rate_interval(failure_probs['cloud']['heterogeneous']['med'])
                }
            }
        }
        return failure_rates

    def compute_Alpha():

        failure_rates = parameters.compute_failure_rates()
        alpha = {
            'edge': {
                'homogeneous': {
                    'low': ((failure_rates['edge']['homogeneous']['low'][1]-failure_rates['edge']['homogeneous']['low'][0])/parameters.taskno,failure_rates['edge']['homogeneous']['low'][1]),
                    'high': ((failure_rates['edge']['homogeneous']['high'][1]-failure_rates['edge']['homogeneous']['high'][0])/parameters.taskno,failure_rates['edge']['homogeneous']['high'][1]),
                    'med': ((failure_rates['edge']['homogeneous']['med'][1]-failure_rates['edge']['homogeneous']['med'][0])/parameters.taskno,failure_rates['edge']['homogeneous']['med'][1])

                },
                'heterogeneous': {
                    'low': ((failure_rates['edge']['heterogeneous']['low'][1]-failure_rates['edge']['heterogeneous']['low'][0])/parameters.taskno,failure_rates['edge']['heterogeneous']['low'][1]),
                    'high': ((failure_rates['edge']['heterogeneous']['high'][1]-failure_rates['edge']['heterogeneous']['high'][0])/parameters.taskno,failure_rates['edge']['heterogeneous']['high'][1]),
                    'med': ((failure_rates['edge']['heterogeneous']['med'][1]-failure_rates['edge']['heterogeneous']['med'][0])/parameters.taskno,failure_rates['edge']['heterogeneous']['med'][1])

                }
            },
            'cloud': {
                'homogeneous': {
                    'low': ((failure_rates['cloud']['homogeneous']['low'][1]-failure_rates['cloud']['homogeneous']['low'][0])/parameters.taskno,failure_rates['cloud']['homogeneous']['low'][1]),
                    'high': ((failure_rates['cloud']['homogeneous']['high'][1]-failure_rates['cloud']['homogeneous']['high'][0])/parameters.taskno,failure_rates['cloud']['homogeneous']['high'][1]),
                    'med': ((failure_rates['cloud']['homogeneous']['med'][1]-failure_rates['cloud']['homogeneous']['med'][0])/parameters.taskno,failure_rates['cloud']['homogeneous']['med'][1])

                },
                'heterogeneous': {
                    'low': ((failure_rates['cloud']['heterogeneous']['low'][1]-failure_rates['cloud']['heterogeneous']['low'][0])/parameters.taskno,failure_rates['cloud']['heterogeneous']['low'][1]),
                    'high': ((failure_rates['cloud']['heterogeneous']['high'][1]-failure_rates['cloud']['heterogeneous']['high'][0])/parameters.taskno,failure_rates['cloud']['heterogeneous']['high'][1]),
                    'med': ((failure_rates['cloud']['heterogeneous']['med'][1]-failure_rates['cloud']['heterogeneous']['med'][0])/parameters.taskno,failure_rates['cloud']['heterogeneous']['med'][1])

                }
            }
        }
        return alpha
    
