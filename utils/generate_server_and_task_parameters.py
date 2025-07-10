#generate_server_and_task_parameters.py


import pandas as pd
import random
from utils.configuration import ConfigLoader
import numpy as np

from scipy.stats import truncnorm

config = ConfigLoader()

NUM_EDGE_SERVERS = config.get('NUM_EDGE_SERVERS')
NUM_CLOUD_SERVERS = config.get('NUM_CLOUD_SERVERS')
STATE_PERMUTATIONS = config.get('STATE_PERMUTATIONS', [])  # Should be a list in config.yaml
STATES = config.get('STATES', {})  # Should be a dict in config.yaml
FAILURE_RATES = config.get('FAILURE_RATES', {})  # Should be a dict in config.yaml


def generate_processing_frequencies():
    edge_frequencies = [round(random.uniform(*config.get('EDGE_PROCESSING_FREQ_RANGE')), 2) for _ in range(NUM_EDGE_SERVERS)]
    cloud_frequencies = [round(random.uniform(*config.get('CLOUD_PROCESSING_FREQ_RANGE')), 2) for _ in range(NUM_CLOUD_SERVERS)]
    return edge_frequencies, cloud_frequencies

def generate_server_info_per_permutation(edge_frequencies, cloud_frequencies, scenario_type, filename):
    failure_rates = FAILURE_RATES
    for perm_num, permutation in enumerate(STATE_PERMUTATIONS, start=1):
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
                state_type = STATES[state][0].lower() if state in STATES else 'low'
                failure_model = "Permanent" if (state in STATES and random.random() < STATES[state][1]) else "Transient"
                if scenario_type == "homogeneous":
                    failure_rate_interval = failure_rates.get('edge', {}).get('homogeneous', {}).get(state_type, (0, 1))
                else:
                    failure_rate_interval = failure_rates.get('edge', {}).get('heterogeneous', {}).get(state_type, (0, 1))
                failure_rate = round(random.uniform(float(failure_rate_interval[0]), float(failure_rate_interval[1])), 6)
                row.extend([failure_rate, failure_model])
            server_info.append(row)

        for i in range(1, NUM_CLOUD_SERVERS + 1):
            server_id = str(i + NUM_EDGE_SERVERS)
            server_type = "Cloud"
            processing_frequency = cloud_frequencies[i-1]
            row = [server_id, server_type, processing_frequency]
            for state in permutation:
                state_type = STATES[state][0].lower() if state in STATES else 'low'
                failure_model = "Permanent" if (state in STATES and random.random() < STATES[state][1]) else "Transient"
                if scenario_type == "homogeneous":
                    failure_rate_interval = failure_rates.get('cloud', {}).get('homogeneous', {}).get(state_type, (0, 1))
                else:
                    failure_rate_interval = failure_rates.get('cloud', {}).get('heterogeneous', {}).get(state_type, (0, 1))
                failure_rate = round(random.uniform(float(failure_rate_interval[0]), float(failure_rate_interval[1])), 6)
                row.extend([failure_rate, failure_model])
            server_info.append(row)

        server_df = pd.DataFrame(server_info, columns=columns)
        sheet_name = f'{scenario_type.capitalize()}_Permutation_{perm_num}'
        with pd.ExcelWriter(filename, engine='openpyxl', mode='a' if perm_num > 1 else 'w') as writer:
            server_df.to_excel(writer, sheet_name=sheet_name, index=False)

def generate_task_params():
    task_info = []
    NUM_TASKS = config.get('taskno')
    TASK_SIZE_RANGE = config.get('TASK_SIZE_RANGE')
    a, b = config.get('Low_demand'), config.get('High_demand')
    mu = (a + b) / 2
    sigma = (b - a) / 6
    lower, upper = (a - mu) / sigma, (b - mu) / sigma
    for i in range(NUM_TASKS):
        task_size = np.random.randint(*TASK_SIZE_RANGE)
        computation_demand = truncnorm.rvs(lower, upper, loc=mu, scale=sigma)
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

