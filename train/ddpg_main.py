#ddpg_main.py
import sys
import os

# Add project root to sys.path for local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agent.ddpg import ddpgModel, Buffer
from utils.params import params
from train.mainLoop import MainLoop


# Function to run your simulation
def run_simulation():
    std_dev = params.std_dev
    critic_lr = params.critic_lr
    actor_lr = params.actor_lr
    gamma = params.gamma
    tau = params.tau
    ac_func=params.activation_function

    dm=ddpgModel(params.num_states,params.num_actions,std_dev,critic_lr,actor_lr,gamma,tau,ac_func) 
    params.model_summary=dm.model_summary()
    buffer = Buffer(dm,params.buffer_capacity, params.batch_size)

    ml=MainLoop(dm, buffer, params.total_episodes, params.taskno, params.num_states, params.num_actions)
    ml.EP() 

    # Save parameters and logs after simulation ends
    # Remove any usage of save_params_and_logs in this file


# Main script to run all simulations
def main():
    scenario_types = ["heterogeneous"] #, "homogeneous", "heterogeneous"
    permutation_numbers = [1] #  for range run: 1 to 24 (1,25) /////  [1,7,13,19] seprate permutation
    for scenario_type in scenario_types:
        for permutation_number in permutation_numbers:
            print(f"##################   Running simulation for {scenario_type} scenario, permutation {permutation_number}  ####################")
            
            # Set the scenario type and permutation number
            params.SCENARIO_TYPE = scenario_type
            params.Permutation_Number = permutation_number
            if permutation_number==1:
                failure="low" 
            elif permutation_number==3:
                failure="high"
            else:
                failure="med"
            params.alpha_edge=params.Alpha['edge'][scenario_type][failure]
            params.alpha_cloud=params.Alpha['cloud'][scenario_type][failure]

            # Run the simulation for the current scenario and permutation
            run_simulation()

if __name__ == "__main__":
    main()