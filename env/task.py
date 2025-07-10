

# task.py

from utils.params import params
import pandas as pd
import math
import random


class Task:

    def __init__(self, env, state, id, params_file='task_parameters.xlsx'):
        self.env = env
        self.env_state = state
        self.id = id
        
        # Other attributes
        self.primaryNode = None
        self.backupNode = None
        self.z = None

        self.primaryStarted = None
        self.primaryFinished = None
        self.primaryStat = None
        
        #self.primary_next_failure = None
        self.primary_service_time = None

        self.backupStarted = None
        self.backupFinished = None
        self.backupStat = None         

        # Read task parameters from Excel file
        task_info_df = pd.read_excel(params_file)

        # Find parameters for current task ID
        task_row = task_info_df.loc[task_info_df['Task_ID'] == self.id]

        # Set task parameters
        self.task_size = task_row['Task_Size'].values[0]
        self.computation_demand = task_row['Computation_Demand'].values[0]
        
        #calculate task parameters
        #self.inpDelay = None  
        #self.outDelay = None  
        self.teta = None  

    def execute_task(self, X, Y, Z):

        self.primaryNode=X
        self.backupNode=Y
        self.z = Z
        
        if self.z == 0:
            self.primaryStarted = self.env.now
            yield self.env.process(self.primary())
            # teta
            if self.primaryStat == "failure":

                yield self.env.timeout(max(self.teta - (self.primaryFinished - self.primaryStarted), 0))
                self.backupStarted = self.env.now
                self.env.process(self.backup())
            
        else: ## z==1
            self.primaryStarted = self.backupStarted = self.env.now
            self.env.process(self.primary())
            self.env.process(self.backup())
    # Other methods...
    
    def primary(self):
        
        inpDelay , outDelay = self.calc_input_output_delay(self.primaryNode)
              
        yield self.env.timeout(inpDelay)

        
        Q_time= self.env.now
        with self.primaryNode.queue.request(priority=1) as req:
            yield req  # Queueing time in server
            Q_time= self.env.now - Q_time
            self.env_state.assign_task_to_server(self.primaryNode.server_id, self, "primary")# assign a task to a server as primary run
            # Calculate service time on primaryNode
            self.primary_service_time = self.computation_demand / self.primaryNode.processing_frequency
            #print("service_time", service_time , "for task",self.id,"in server " , self.primaryNode.server_id )
            failure_rate_adjusted=self.set_failure_rate(self.primaryNode)
            # Simulate execution either success or failed
            yield self.env.timeout(self.primary_service_time)
            
        # Generate the next failure probability            
        fault_prob= 1-math.exp(-failure_rate_adjusted * self.primary_service_time)
        r=random.uniform(0, 1)
        if(r<fault_prob):
            self.primaryStat = "failure"
            
        else:
            yield self.env.timeout(outDelay)
            self.primaryStat = "success"

        self.primaryFinished = self.env.now
        
        #print(f"Task {self.id} {'succeeded' if self.primaryStat == 'success' else 'failed'} on primary server {self.primaryNode.server_id}")
        self.env_state.complete_task(self.primaryNode.server_id, self, 'primary', self.primary_service_time)
        
        self.teta= 1.5 * (self.primary_service_time + inpDelay + outDelay + Q_time) 

    def backup(self):

        inpDelay , outDelay = self.calc_input_output_delay(self.backupNode)

        # Use PriorityRequest if backupNode is the same as primaryNode
        if self.backupNode == self.primaryNode: # Retry sterategy
            # no inpDelay
            with self.backupNode.queue.request(priority=0) as req: #high priority
                yield req  
                self.env_state.assign_task_to_server(self.backupNode.server_id, self, "backup") 
                backup_service_time = self.primary_service_time # as primary
                failure_rate_adjusted=self.set_failure_rate(self.backupNode)
                yield self.env.timeout(backup_service_time)

        else: # recovery block or first result strategy
            yield self.env.timeout(inpDelay)
            with self.backupNode.queue.request(priority=1) as req:
                yield req 
                self.env_state.assign_task_to_server(self.backupNode.server_id, self, "backup") 
                backup_service_time = self.computation_demand / self.backupNode.processing_frequency # may differ from primary according to frequency of backup server
                failure_rate_adjusted=self.set_failure_rate(self.backupNode)
                yield self.env.timeout(backup_service_time)

            
        
        
        fault_prob= 1-math.exp(-failure_rate_adjusted * backup_service_time)
        r=random.uniform(0, 1)
        if(r<fault_prob):
            self.backupStat = "failure"
        else:
            yield self.env.timeout(outDelay)
            self.backupStat = "success"
         
        self.backupFinished = self.env.now
        
        #print(f"Task {self.id} {'succeeded' if self.backupStat == 'success' else 'failed'} on backup server {self.backupNode.server_id}")
        self.env_state.complete_task(self.backupNode.server_id, self, "backup", backup_service_time)

    def calc_input_output_delay(self, server_object):
        if server_object.server_type == "Edge":
            # Calculate input delay for Edge
            
            inpDelay = 0
        else:
            # Calculate input delay for Cloud
            inpDelay = self.task_size / params.rsu_to_cloud_profile['bandwidth'] 

        # Output delay is the same as input delay
        outDelay = inpDelay   
        return inpDelay, outDelay
    
    
    def set_failure_rate(self, server_object):
            
            if (server_object.server_type=="Edge"):
                failure_rate_adjusted = server_object.failure_rate + params.alpha_edge[0] * len(server_object.queue.queue)
                if failure_rate_adjusted>params.alpha_edge[1]:
                    failure_rate_adjusted=params.alpha_edge[1]

            else:
                failure_rate_adjusted = server_object.failure_rate + params.alpha_cloud[0] * len(server_object.queue.queue)
                if failure_rate_adjusted>params.alpha_cloud[1]:
                    failure_rate_adjusted=params.alpha_cloud[1]

            return failure_rate_adjusted
            