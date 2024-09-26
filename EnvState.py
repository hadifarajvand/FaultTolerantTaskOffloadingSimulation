
#EnvState.py
import numpy as np
class EnvironmentState:
    def __init__(self):
        self.servers = {}  # Dictionary to store server objects {server_id: server_object, 'failure_rate': failure_rate, 'load': load}
        self.tasks = {}  # Dictionary to store generated task objects {task_id: task_object}
        
        self.num_completed_tasks = 0  # Number of completed tasks at all servers
   
    def add_server_and_init_environment(self, server_object):
        """Add a server object to the environment state."""
        server_id = server_object.server_id  # Extract the server ID from the server object
        #print(f"Adding server with ID {server_id}")
        self.servers[server_id] = {
            'server_object': server_object,
            'tasks_assigned': [],  # List of task objects assigned to this server
            'primary_failure_count': 1000000 * server_object.failure_rate,  # Initialize failure time for the server if it is selected as primary
            'backup_failure_count': 1000000 * server_object.failure_rate,  # Initialize failure time for the server if it is selected as backup
            'primary_executed_time': 1000000,  # Initialize executed tasks time for the server if it is selected as primary
            'backup_executed_time': 1000000,  # Initialize executed tasks time for the server if it is selected as backup
            'load': 0  # Initialize load for the server (sum of computation demands of tasks assigned to it)
        }

    def print_servers(self):
        """Print information about all servers in the environment."""
        if not self.servers:
            print("No servers available.")
            return
        
        for server_id, server_info in self.servers.items():
            server_object = server_info['server_object']
            failure_rate = server_info.get('failure_rate', 0)
            load = server_info.get('load', 0)
            
            print(f"Server ID: {server_id}")
            print(f"Failure Rate: {failure_rate}")
            print(f"Load: {load}")
            # Print additional information as needed

    def assign_task_to_server(self, server_id, task, selection):
        """Assign a task object to a server based on the selection (primary or backup)."""
              
        self.servers[server_id]['tasks_assigned'].append({'task': task, 'selection': selection})

        # Update 'load'
        self.servers[server_id]['load'] += task.computation_demand
        
    def complete_task(self, server_id, task, selection, execute_time):
        """Set parameters about completed task in the environment state."""
        tasks_assigned = self.servers[server_id]['tasks_assigned']
        for assigned_task in tasks_assigned:
            if assigned_task['task'] == task and assigned_task['selection'] == selection:
                # Update 'load'
                self.servers[server_id]['load'] -= task.computation_demand

                if selection == "primary" :
                    # Update 'primary_executed_tasks'
                    self.servers[server_id]['primary_executed_time'] += execute_time
                    if task.primaryStat == "failure":
                        # Update 'primary_failure_count'
                        self.servers[server_id]['primary_failure_count'] += execute_time
                    

                elif selection == "backup": 
                    # Update 'backup_executed_tasks'
                    self.servers[server_id]['backup_executed_time'] += execute_time
                    if task.backupStat == "failure":
                        # Update 'backup_failure_count'
                        self.servers[server_id]['backup_failure_count'] += execute_time

                
                self.num_completed_tasks += 1
                break
   
    def get_server_by_id(self, server_id):
        """Get a server object by its ID."""
        server_info = self.servers.get(server_id)
        if server_info:
            return server_info['server_object']
        else:
            return None
    
    def add_task(self, task_object):
        """Add a task object to the environment state."""
        task_id = task_object.id  # Extract the task ID from the task object
        self.tasks[task_id] = task_object

    def remove_task(self, task_id):
        """Remove a task object from the environment state."""
        if task_id in self.tasks:
            del self.tasks[task_id]
        else:
            print(f"Task with ID {task_id} not found in the task dictionary.")

    def get_task_by_id(self, task_id):
        """Get a task object by its ID."""
        return self.tasks.get(task_id)
   
    def get_min_computation_demand(self):
        """Get the minimum computation demand among all tasks."""
        if not self.tasks:
            print("No tasks available.")
            return None
        
        min_demand = float('inf')  # Initialize min_demand with positive infinity
        
        for task_id, task_obj in self.tasks.items():
            if task_obj.computation_demand < min_demand:
                min_demand = task_obj.computation_demand
        
        return min_demand
        
    def reset(self):
        """Reset the environment state."""
        self.servers = {}
        self.tasks= {}
        self.num_completed_tasks = 0

    '''
    def get_state(self):
        server_states = []
        for server_id, server_info in self.servers.items():
            server_object = server_info['server_object']

            # Calculate Primary failure rate
            primary_failure_rate = server_info['primary_failure_count'] / server_info['primary_executed_time']
            
            # Calculate Backup failure rate
            backup_failure_rate = server_info['backup_failure_count'] / server_info['backup_executed_time']
            
            frequency = server_object.processing_frequency
            
            load = server_info['load']

            # Append each piece of information to the server_states list
            server_states.extend([
                #server_id,
                primary_failure_rate,
                backup_failure_rate,
                frequency,
                load
            ])

        return server_states
    '''

    
    def get_state(self):
        primary_failure_rate=[]
        backup_failure_rate=[]
        frequency=[]
        load=[]

        server_states = []


        for server_id, server_info in self.servers.items():
            
            server_object = server_info['server_object']
            primary_failure_rate.append(server_info['primary_failure_count'] / server_info['primary_executed_time'])
            backup_failure_rate.append(server_info['backup_failure_count'] / server_info['backup_executed_time'])
            frequency.append(server_object.processing_frequency)
            load.append(server_info['load'])

            

        # convert to numpy array
        primary_failure_rate=np.array(primary_failure_rate)
        backup_failure_rate=np.array(backup_failure_rate)
        frequency=np.array(frequency)
        load=np.array(load)

        # concate them
        server_states=np.concatenate([primary_failure_rate,backup_failure_rate,frequency,load])

        return server_states
    
    def print_state(self):
        for server_state in self.get_state():
            print(f"Server ID: {server_state['server_id']}")
            print(f"Frequency: {server_state['frequency']}")
            print(f"Load: {server_state['load']}")
            print(f"primary_failure_rate: {server_state['primary_failure_rate']}")
            print(f"backup_failure_rate: {server_state['backup_failure_rate']}")