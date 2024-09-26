# params.py
from configuration import parameters

class params:
    # Define scenario type
    model_summary="ddpg model"
    SCENARIO_TYPE = parameters.SCENARIO_TYPE
    Permutation_Number=parameters.Permutation_Number
    NUM_EDGE_SERVERS =parameters.NUM_EDGE_SERVERS
    NUM_CLOUD_SERVERS = parameters.NUM_CLOUD_SERVERS
    serverNo = NUM_EDGE_SERVERS + NUM_CLOUD_SERVERS  
    Alpha = parameters.compute_Alpha()
    alpha_edge=(None,None)
    alpha_cloud=(None,None)
    taskno = parameters.taskno
    total_episodes = parameters.total_episodes
    num_states = 4 * serverNo + 2  #  for each server: load (x), frequency of server(x), primary Failure Rate(x), backup Failure Rate(x) + task profile: task_size + computation demand (2)
    num_actions = (serverNo*serverNo)+(serverNo*(serverNo-1))//2 # 145
    std_dev = parameters.std_dev  
    #min_std_dev = 0.05
    #decay_rate = 0.99
    activation_function=parameters.activation_function
    critic_lr = parameters.critic_lr 
    actor_lr = parameters.actor_lr 
    gamma = parameters.gamma
    tau = parameters.tau 
    buffer_capacity = parameters.buffer_capacity
    batch_size = parameters.batch_size
    TASK_ARRIVAL_RATE = parameters.TASK_ARRIVAL_RATE
    rsu_to_edge_profile = parameters.rsu_to_edge_profile
    rsu_to_cloud_profile = parameters.rsu_to_cloud_profile

    
    
    