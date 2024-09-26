#generate_server_and_task_parameters.py


import pandas as pd
import random
from configuration import parameters
import numpy as np

from scipy.stats import truncnorm


NUM_EDGE_SERVERS = parameters.NUM_EDGE_SERVERS
NUM_CLOUD_SERVERS = parameters.NUM_CLOUD_SERVERS
state_permutations = parameters.generate_state_permutations()

def generate_processing_frequencies():
    edge_frequencies = [round(random.uniform(*parameters.EDGE_PROCESSING_FREQ_RANGE), 2) for _ in range(NUM_EDGE_SERVERS)]
    cloud_frequencies = [round(random.uniform(*parameters.CLOUD_PROCESSING_FREQ_RANGE), 2) for _ in range(NUM_CLOUD_SERVERS)]
    return edge_frequencies, cloud_frequencies

def generate_server_info_per_permutation(edge_frequencies, cloud_frequencies, scenario_type, filename):
    failure_rates = parameters.compute_failure_rates()
    

    for perm_num, permutation in enumerate(state_permutations, start=1):
        server_info = []
        columns = ['Server_ID', 'Server_Type', 'Processing_Frequency']
        for state in permutation:
            columns.extend([f'Failure_Rate_{state}', f'Failure_Model_{state}'])

        for i in range(1, NUM_EDGE_SERVERS + 1):
            server_id = str(i)
            server_type = "Edge"
            processing_frequency = edge_frequencies[i-1]
            row = [server_id, server_type, processing_frequency]
            for state in permutation:
                state_type = parameters.STATES[state][0].lower()
                failure_model = "Permanent" if random.random() < parameters.STATES[state][1] else "Transient"
                
                if scenario_type == "homogeneous":
                    failure_rate_interval = failure_rates['edge']['homogeneous'][state_type]
                else:
                    failure_rate_interval = failure_rates['edge']['heterogeneous'][state_type]
                
                failure_rate = round(random.uniform(*failure_rate_interval), 6)  # Increased precision
                row.extend([failure_rate, failure_model])
            server_info.append(row)

        for i in range(1, NUM_CLOUD_SERVERS + 1):
            server_id = str(i + NUM_EDGE_SERVERS)
            server_type = "Cloud"
            processing_frequency = cloud_frequencies[i-1]
            row = [server_id, server_type, processing_frequency]
            for state in permutation:
                state_type = parameters.STATES[state][0].lower()
                failure_model = "Permanent" if random.random() < parameters.STATES[state][1] else "Transient"
                
                if scenario_type == "homogeneous":
                    failure_rate_interval = failure_rates['cloud']['homogeneous'][state_type]
                else:
                    failure_rate_interval = failure_rates['cloud']['heterogeneous'][state_type]
                
                failure_rate = round(random.uniform(*failure_rate_interval), 6)  # Increased precision
                row.extend([failure_rate, failure_model])
            server_info.append(row)

        server_df = pd.DataFrame(server_info, columns=columns)
        sheet_name = f'{scenario_type.capitalize()}_Permutation_{perm_num}'
        with pd.ExcelWriter(filename, engine='openpyxl', mode='a' if perm_num > 1 else 'w') as writer:
            server_df.to_excel(writer, sheet_name=sheet_name, index=False)

def generate_task_params():
    task_info = []
    NUM_TASKS = parameters.taskno
    TASK_SIZE_RANGE = parameters.TASK_SIZE_RANGE
    
    a, b=parameters.Low_demand, parameters.High_demand
    # Mean and standard deviation
    mu = (a + b) / 2  # Mean at the midpoint of the interval
    sigma = (b - a) / 6  # Standard deviation chosen to fit the interval

    # Define the truncation limits in terms of the standard normal
    lower, upper = (a - mu) / sigma, (b - mu) / sigma


    for i in range(NUM_TASKS):
        task_size = np.random.randint(*TASK_SIZE_RANGE)
        
        # Generate samples
        computation_demand =  truncnorm.rvs(lower, upper, loc=mu, scale=sigma)
        task_info.append((i + 1, task_size, computation_demand))

    task_df = pd.DataFrame(task_info, columns=['Task_ID', 'Task_Size', 'Computation_Demand'])
    task_df.to_excel('task_parameters.xlsx', index=False)


def main():
    edge_frequencies, cloud_frequencies = generate_processing_frequencies()
    generate_server_info_per_permutation(edge_frequencies, cloud_frequencies, 'homogeneous', 'homogeneous_server_info.xlsx')
    generate_server_info_per_permutation(edge_frequencies, cloud_frequencies, 'heterogeneous', 'heterogeneous_server_info.xlsx')
    generate_task_params()
    print("Parameters defined in Excel files!")

if __name__ == "__main__":
    main()

