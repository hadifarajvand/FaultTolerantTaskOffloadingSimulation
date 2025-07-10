

# mainLoop.py


from env.server import Server
from env.task import Task
from env.EnvState import EnvironmentState
from utils.params import params

import simpy
import torch
import numpy as np
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import math
# import wandb  # Removed wandb import since we now use CSV logging
import csv


class MainLoop:
    def __init__(self,dm,buffer,total_episodes,maxtaskno,num_states,num_actions):
         
        self.dm=dm
        self.buffer=buffer
        self.num_states=num_states
        self.num_actions=num_actions
        self.total_episodes = total_episodes 
        self.rewardsAll=[]
        self.ep_reward_list = []
        self.ep_delay_list = []
        self.avg_reward_list = []

        #self.actor_loss = []  # List to track actor losses
        #self.critic_loss = []  # List to track critic losses
        self.this_episode=0
        self.G_state=[]
        self.G_action=[]
        self.index_of_actions = self.generate_combinations()
        self.episodic_reward=0
        self.episodic_delay=0
        
        self.tempbuffer={} # temporary key-value buffer tuple[i]=(s,a,r,s')
        self.taskCounter=1
        self.pendingList=[]
        self.maxTask=maxtaskno
        
        self.env = None
        self.env_state = None
        self.log_data = []
        self.task_Assignments_info = []
        self.SCENARIO_TYPE=params.SCENARIO_TYPE
        self.Permutation_Number=params.Permutation_Number
        self.agent_type = params.AGENT_TYPE
        self.log_dir = os.path.join('logs', self.agent_type)
        os.makedirs(self.log_dir, exist_ok=True)
        self.csv_log_file = os.path.join(self.log_dir, 'training_log.csv')
        with open(self.csv_log_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['episode', 'episodic_reward', 'avg_reward', 'episodic_delay', 'avg_delay'])
        
    def EP(self):
        
        while self.this_episode<self.total_episodes:

            #current_std_dev = params.std_dev * (params.decay_rate ** self.this_episode)
            # Ensure it doesn't go below the minimum value
            #params.std_dev = max(params.min_std_dev, current_std_dev)


            self.this_episode=self.this_episode+1
            self.episodic_reward = 0
            self.episodic_delay=0
            self.tempbuffer={}
            self.taskCounter=1
            self.pendingList=[]
            # Main simulation setup
            self.env = simpy.Environment()
            self.env_state = EnvironmentState()
            self.env_state.reset()
            #add servers and init environment
            self.setServers()
            self.env.process(self.Iteration()) # execute as a process
            # Run the simulation
            self.env.run()
            
            #input("Press Enter to continue to the next episode...")  

            
            
        # Plot the rewards after all episodes are completed
        # self.plot_rewards()  # Comment out static plot
        # self.plot_losses()   # Comment out static plot
        # wandb.finish() # Removed wandb.finish()

    def Iteration(self):
        
        while self.taskCounter<=self.maxTask:   # there are more tasks
            
            
            yield self.env.timeout(np.random.poisson(1/params.TASK_ARRIVAL_RATE))  # Using numpy.random.poisson for inter-arrival times
            
            task = Task(self.env, self.env_state, self.taskCounter) # Generate a task object
            
            self.env_state.add_task(task)  # to save task in a dictionary          
            
            #create the next state vector
            self.G_state = self.convert_state_to_normalized_array(self.env_state.get_state(), task) 
            
            torch_state = torch.FloatTensor(self.G_state).unsqueeze(0)

            #add this state to the last pending tuple in tempBuffer
            if self.taskCounter>1:
                tempx=list(self.tempbuffer[self.taskCounter-1])
                tempx[3]=self.G_state
                self.tempbuffer[self.taskCounter-1]=tuple(tempx)
                self.add_train()

            self.G_action = self.dm.policy(torch_state)
            if isinstance(self.G_action, torch.Tensor):
                self.G_action = self.G_action.detach().cpu().numpy().tolist()
            
            self.G_action=self.dm.addNoise(self.G_action,self.this_episode,self.total_episodes)
            X, Y, Z = self.extract_parameters() # (primary, backup, z) according to self.G_action
            
            #record the current state and action
            tempx=[self.G_state,self.G_action,1,[]] # (s,a,r,s')
            self.tempbuffer[self.taskCounter]=tuple(tempx)
                                 
            #start a simpy process for task 
            self.env.process(task.execute_task(X,Y,Z)) 
            #add to the pending list
            self.pendingList.append(self.taskCounter)
            
            # next task counter
            self.taskCounter=self.taskCounter+1
            
            # Check if we need to update server parameters
            #if self.taskCounter == self.maxTask // 4:
                #self.update_server_states(failureRateIndex=5, failureModelIndex=6)  #  second Failure_Rate_ and Failure_Model_
            #elif self.taskCounter == self.maxTask // 2:
                #self.update_server_states(failureRateIndex=7, failureModelIndex=8)  #  third Failure_Rate_ and Failure_Model_
            #elif self.taskCounter == 3 * self.maxTask // 4:
                #self.update_server_states(failureRateIndex=9, failureModelIndex=10)  #  fourth Failure_Rate_ and Failure_Model_

           


        tempx=list(self.tempbuffer[self.taskCounter-1])
        tempx[3]=self.G_state
        self.tempbuffer[self.taskCounter-1]=tuple(tempx)

        
        while len(self.pendingList)>0:
            
            # minimum computation demand of tasks
            yeild_time=self.env_state.get_min_computation_demand() # I defined a function to give me a minimum computation demand of tasks
            #print ("yield for some time")
            yield self.env.timeout(yeild_time)  # Wait for pending tasks  min( task size)
            #print("pendingList is:")
            #print(self.pendingList)
            self.add_train()

        self.ep_reward_list.append(self.episodic_reward)
        self.ep_delay_list.append(self.episodic_delay)
        avg_reward = np.mean(self.ep_reward_list[-40:])
        avg_delay=np.mean(self.ep_delay_list[-40:])
        self.log_data.append((self.this_episode, avg_reward, self.episodic_reward,avg_delay))

        # Log to CSV
        with open(self.csv_log_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([self.this_episode, self.episodic_reward, avg_reward, self.episodic_delay, avg_delay])

        print("Proposed Approach: Episode * {} * Avg Reward is ==> {}".format(self.this_episode, avg_reward),"This episode:",self.episodic_reward)
        #print("Episode * {} * Total Reward is ==> {}".format(self.this_episode, self.episodic_reward))
        self.avg_reward_list.append(avg_reward)
        
    def calcReward(self, taskID):
            
            task = self.env_state.get_task_by_id(taskID)
            
            
            z = task.z
            primaryStat = task.primaryStat
            backupStat = task.backupStat
            primaryFinished = task.primaryFinished
            primaryStarted = task.primaryStarted
            backupFinished = task.backupFinished
            backupStarted = task.backupStarted

            flag="s" # default value for succedeed task
            delay=None

            if z == 0:
                if primaryStat == 'success' and backupStat is None and primaryFinished is not None:
                    delay = primaryFinished - primaryStarted
                elif primaryStat == 'failure' and backupStat == 'success' and backupFinished is not None:
                    delay = backupFinished - primaryStarted
                elif primaryStat == 'failure' and backupStat == 'failure':
                    delay = backupFinished - primaryStarted   
                    flag="f" # if the task failed
                else: # primaryStat == 'failure' and backupStat is None /primaryStat is None and backupStat is None/other unready
                    flag="n" # if we dont know yet(it will be set after)
                    
            
            else:  # z == 1 (parallel mode)
                if primaryStat == 'success' and backupStat == 'success' and primaryFinished is not None and backupFinished is not None :
                    delay = min(primaryFinished, backupFinished) - primaryStarted
                elif primaryStat == 'success' and backupStat == 'failure' and primaryFinished is not None:
                    delay = primaryFinished - primaryStarted
                elif primaryStat == 'failure' and backupStat == 'success' and backupFinished is not None:
                    delay = backupFinished - backupStarted
                elif primaryStat == 'failure' and backupStat == 'failure':
                    delay = max(backupFinished - backupStarted,primaryFinished - primaryStarted)
                    flag="f" # if the task failed
                    
                elif primaryStat == 'success' and backupStat is None and primaryFinished is not None:
                    delay = primaryFinished - primaryStarted
                elif primaryStat is None and backupStat == 'success'  and backupFinished is not None:
                    delay = backupFinished - backupStarted
                else: # primaryStat == 'failure' and backupStat is None/primaryStat is None and backupStat == 'failure'/primaryStat is None and backupStat is None /other unready
                    flag="n" # if we dont know yet(it will be set after)
                    
            
            

            # Handle delay cases
            if flag =="f":
                                
                # Penalize failure with a weighted negative reward
                failure_penalty_weight = 3.0  # Example weight for failure penalty
                reward = -failure_penalty_weight * delay
                # Ensure a minimum penalty for failures to drive learning
                if reward > -3:
                    reward = -3  # Minimum penalty for any failure to ensure learning

            elif flag == "s":
                # Reward for successful task with lower delay being more rewarding
                success_reward_weight = 1.0  # Example weight for success reward
                reward = success_reward_weight * (math.log(1 - (1 / math.exp(math.sqrt(delay)))) / math.log(0.995))

            else:
                reward = None
                             
           
            return reward,delay
    
    def add_train(self):
        if self.buffer.buffer_counter>0:
            self.buffer.learn()
            '''            
            self.actor_loss.append(self.dm.actor_loss)
            self.critic_loss.append(self.dm.critic_loss)
            '''
            self.dm.update_target(self.dm.target_actor, self.dm.actor_model)
            self.dm.update_target(self.dm.target_critic, self.dm.critic_model)
        
        #compute pending rewards
        removeList=[]
        tempx=[]
        for taskid in self.pendingList:
            reward,delay = self.calcReward(taskid)
            #if reward !=1: # the reward is ready 
            if reward != None: # the reward is ready     
                self.episodic_reward += reward
                self.episodic_delay += delay # for delay
                self.rewardsAll.append(reward)
                #add completed tuples to the buffer
                tempx=list(self.tempbuffer[taskid])
                tempx[2]=reward
                self.tempbuffer[taskid]=tuple(tempx) 
                #print(self.tempbuffer[taskid])
                self.buffer.record(self.tempbuffer[taskid])
                #
                #print("training....")
                self.buffer.learn()
                self.dm.update_target(self.dm.target_actor, self.dm.actor_model)
                self.dm.update_target(self.dm.target_critic, self.dm.critic_model)
                #
                removeList.append(taskid)
        
        for t in removeList:
            self.pendingList.remove(t)
            task=self.env_state.get_task_by_id(t)
            self.task_Assignments_info.append((self.this_episode, task.id, task.primaryNode.server_id,task.primaryStarted, task.primaryFinished,task.primaryStat, task.backupNode.server_id,task.backupStarted, task.backupFinished,task.backupStat, task.z)) ## for trace
            self.env_state.remove_task(t)
 
    def dumpResults(self,fname1):
        if os.path.exists(fname1):
            os.remove(fname1)
        with open(fname1, 'w') as f:
            json.dump(self.avg_reward_list, f)
        ####
    
    def setServers(self):
        # Choose the appropriate file based on the scenario type
        excel_file = 'homogeneous_server_info.xlsx' if self.SCENARIO_TYPE == 'homogeneous' else 'heterogeneous_server_info.xlsx'

        # Read the specific sheet based on permutation number
        sheet_name = f'{self.SCENARIO_TYPE.capitalize()}_Permutation_{self.Permutation_Number}'
        server_info_df = pd.read_excel(excel_file, sheet_name=sheet_name)

        # Iterate over the DataFrame from the second row to create server objects
        for index, row in server_info_df.iterrows():
            server_id = row['Server_ID']
            server_type = row['Server_Type'] 
            processing_frequency = row['Processing_Frequency'] 
            # Extract Failure Rate and Failure Model from the 4th and 5th columns
            failure_rate = row.iloc[3]  #  4th column contains first Failure Rate
            failure_model = row.iloc[4]  #  5th column contains first Failure Model
            
            # Create a Server object
            server = Server(self.env, server_type, server_id, processing_frequency, failure_rate, failure_model)
            
            # Add server to the environment state and initialize state
            self.env_state.add_server_and_init_environment(server)



    def extract_parameters(self):
        if not self.G_action:
            raise ValueError("G_action array is empty")

        # Find the index of the maximum value in self.G_action
        max_index = self.G_action.index(max(self.G_action))

        # Retrieve the parameters from the precomputed list
        primary_server_id, backup_server_id, z_parameter = self.index_of_actions[max_index]

        # Retrieve the server objects using their IDs
        primary_server = self.env_state.get_server_by_id(primary_server_id)
        backup_server = self.env_state.get_server_by_id(backup_server_id)

        return primary_server, backup_server, z_parameter


    def convert_state_to_normalized_array(self, server_states, task):
        """Convert the list of server states to a concatenated array."""
        # Convert the server_states list directly to a numpy array
        #server_states_array = np.array(server_states)
        
        # Add the task profile to the state array using np.append
        #concatenated_array = np.append(server_states_array, [task.task_size, task.computation_demand])
        concatenated_array = np.append(server_states, [task.task_size, task.computation_demand])

        max_vals = np.max(concatenated_array, axis=0)
        min_vals = np.min(concatenated_array, axis=0)
        normalized_arr = (concatenated_array - min_vals) / (max_vals - min_vals + 1e-8)  # adding small value to avoid division by zero
        return normalized_arr
    
    def plot_rewards(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        plot_path = os.path.join(current_dir, 'rewards_plot.png')
        
        plt.plot(range(1, self.total_episodes + 1), self.ep_reward_list, label='Episodic Reward')
        plt.plot(range(1, self.total_episodes + 1), self.avg_reward_list, label='Average Reward', linestyle='--')
        plt.title('Episodic and Average Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend()
        plt.savefig(plot_path)  # Save the plot as an image in the current directory
        plt.close()
    
    
    '''
    def plot_losses(self):

        current_dir = os.path.dirname(os.path.abspath(__file__))
        plot_path = os.path.join(current_dir, 'loss_plot.png')
        plt.figure(figsize=(12, 5))

        # Plot Actor Loss
        plt.subplot(1, 2, 1)
        plt.plot(self.actor_loss, label='Actor Loss')
        plt.title('Actor Loss')
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.legend()

        # Plot Critic Loss
        plt.subplot(1, 2, 2)
        plt.plot(self.critic_loss, label='Critic Loss')
        plt.title('Critic Loss')
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        plt.savefig(plot_path)  # Save the plot as an image in the current directory
        plt.close()

    '''
    def update_server_states(self, failureRateIndex, failureModelIndex):
        # Map scenario type to the corresponding file name
        file_name = 'homogeneous_server_info.xlsx' if self.SCENARIO_TYPE == 'homogeneous' else 'heterogeneous_server_info.xlsx'
        
        # Construct the sheet name based on the permutation number
        sheet_name = f'{self.SCENARIO_TYPE.capitalize()}_Permutation_{self.Permutation_Number}'
        
        # Read the server information from the Excel sheet
        server_info_df = pd.read_excel(file_name, sheet_name=sheet_name)
        
        for index, row in server_info_df.iterrows():
            server_id = row['Server_ID']  
            
            # Extract Failure Rate and Failure Model from the index columns
            failure_rate = row.iloc[failureRateIndex]  # index+1 column contains Failure Rate
            failure_model = row.iloc[failureModelIndex]  # index+1 column contains Failure Model
           
            server = self.env_state.get_server_by_id(server_id)
            server.update_failure_params(failure_rate, failure_model)
    @staticmethod
    def generate_combinations():
        numberOfServers=params.serverNo
        index_of_actions = []

        # Generate combinations for z=0
        for i in range(1, numberOfServers+1):
            for j in range(1, numberOfServers+1):
                index_of_actions.append((i, j, 0))

        # Generate combinations for z=1
        for i in range(1, numberOfServers+1):
            for j in range(i + 1, numberOfServers+1):
                index_of_actions.append((i, j, 1))

        return index_of_actions    
    
