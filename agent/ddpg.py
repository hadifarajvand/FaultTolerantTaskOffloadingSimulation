
#ddpg.py
import tensorflow as tf
from keras import layers
import numpy as np

tf.config.functions_run_eagerly()
#tf.config.run_functions_eagerly(True)

class ddpgModel:
    """
    Deep Deterministic Policy Gradient (DDPG) model for continuous control.
    Contains actor and critic networks, target networks, and noise process.
    """
    def __init__(self, num_states, num_actions, std_dev, critic_lr, actor_lr, gamma, tau, activationFunction):
        """
        Initialize the DDPG model with actor/critic networks and optimizers.
        Args:
            num_states (int): Number of state features.
            num_actions (int): Number of action dimensions.
            std_dev (float): Standard deviation for exploration noise.
            critic_lr (float): Learning rate for critic.
            actor_lr (float): Learning rate for actor.
            gamma (float): Discount factor.
            tau (float): Soft update rate for target networks.
            activationFunction (str): Activation function for actor output.
        """
        self.activationFunction=activationFunction # string: tanh , softmax
        self.num_states=num_states
        self.num_actions=num_actions
        self.std_dev=std_dev
        self.critic_lr=critic_lr
        self.actor_lr=actor_lr
        self.gamma=gamma
        self.tau=tau
                
        #self.ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))
        self.ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1), theta=0.2)


        self.actor_model = self.get_actor()
        self.critic_model = self.get_critic()
        self.target_actor = self.get_actor()
        self.target_critic = self.get_critic()

        self.target_actor.set_weights(self.actor_model.get_weights())
        self.target_critic.set_weights(self.critic_model.get_weights())

        self.critic_optimizer = tf.keras.optimizers.Adam(self.critic_lr)
        self.actor_optimizer = tf.keras.optimizers.Adam(self.actor_lr)
        # To record training losses
        #self.actor_loss = []
        #self.critic_loss = []

    # This update target parameters slowly
    # Based on rate `tau`, which is much less than one.
    @tf.function
    def update_target(self,target_weights, weights):
        
        for (a, b) in zip(target_weights, weights):
            a.assign(b * self.tau + a * (1 - self.tau))

    def get_actor(self):
        # Initialize weights between -3e-3 and 3-e3
        last_init = tf.keras.initializers.RandomUniform(minval=-0.003, maxval=0.003)

        inputs = layers.Input(shape=(self.num_states,))#34
        
        out = layers.LayerNormalization()(inputs)
        out = layers.Dense(300, activation="relu")(out) # our: 400  retry:300      no rec:200
        out = layers.Dense(200, activation="relu")(out) #our:300    retry : 200    no rec:150
        outputs = layers.Dense(self.num_actions, activation=self.activationFunction, kernel_initializer=last_init)(out)#92

        # Our upper bound is 2.0 for Pendulum.
        #outputs = outputs * upper_bound
        model = tf.keras.Model(inputs, outputs)
        return model


    def get_critic(self):#evaluates the action taken by the actor by estimating the value of state-action pairs.

        state_input = layers.Input(shape=(self.num_states))
        state_out = layers.Dense(300, activation="relu")(state_input) #our"400  retry:300      no rec:200
        state_out = layers.Dense(200, activation="relu")(state_out)#our:300     retry:200      no rec:150

        action_input = layers.Input(shape=(self.num_actions))
        action_out = layers.Dense(200, activation="relu")(action_input) #our:300    retry:200      no rec:150

        concat = layers.Concatenate()([state_out, action_out])
        out = layers.LayerNormalization()(concat)

        out = layers.Dense(200, activation="relu")(out) #our:300    retry:200      no rec:150
  
        outputs = layers.Dense(1)(out)

        # Outputs single value for give state-action
        model = tf.keras.Model([state_input, action_input], outputs)

        return model

      
    def addNoise(self,sampled_actions,thisEpNo,totalEpNo):
        noise = self.ou_noise()
        noisy_actions=[]
        
        for sa in sampled_actions:
            sa=sa+noise[0]
            sa=np.clip(sa, -1, 1)
            noisy_actions.append(sa)
        return  noisy_actions
    
    
    def policy(self,state):
        
        
        sampled_actions = tf.squeeze(self.actor_model(state))
        #sampled_actions=sampled_actions.eval(session=tf.compat.v1.Session())
        return  sampled_actions
    
   
    def model_summary(self):
            def get_layers_summary(model):
                layers_info = []
                for layer in model.layers:
                    if isinstance(layer, layers.Dense):
                        #layers_info.append(f"Layer: {layer.name}, Neurons: {layer.units}, Activation: {layer.activation.__name__}")

                        layers_info.append(f"Neurons: {layer.units}")
                return layers_info

            actor_layers_summary = get_layers_summary(self.actor_model)
            critic_layers_summary = get_layers_summary(self.critic_model)

            return f"Actor Model Hidden Layers:\n" + "\n".join(actor_layers_summary) + "\n\n" + \
                f"Critic Model Hidden Layers:\n" + "\n".join(critic_layers_summary)


class OUActionNoise:
    """
    Ornstein-Uhlenbeck process for temporally correlated exploration noise.
    """
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        """Initialize the OU noise process."""
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        """Generate a new noise sample."""
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        """Reset the noise process to initial state."""
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


class Buffer:
    """
    Experience replay buffer for DDPG agent.
    Stores (state, action, reward, next_state) tuples and samples minibatches for training.
    """
    def __init__(self, ddpgObj, buffer_capacity=100000, batch_size=64):
        """Initialize the buffer with given capacity and batch size."""
        self.ddpgObj=ddpgObj
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, self.ddpgObj.num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, self.ddpgObj.num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, self.ddpgObj.num_states))

    def record(self, obs_tuple):
        """Store a new experience tuple in the buffer."""
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function
    def update(
        self, state_batch, action_batch, reward_batch, next_state_batch,
    ):
        """Update actor and critic networks using a minibatch of experiences."""
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            target_actions = self.ddpgObj.target_actor(next_state_batch, training=True)
            y = reward_batch + self.ddpgObj.gamma * self.ddpgObj.target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = self.ddpgObj.critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, self.ddpgObj.critic_model.trainable_variables)
        self.ddpgObj.critic_optimizer.apply_gradients(
            zip(critic_grad, self.ddpgObj.critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = self.ddpgObj.actor_model(state_batch, training=True)
            critic_value = self.ddpgObj.critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self.ddpgObj.actor_model.trainable_variables)
        self.ddpgObj.actor_optimizer.apply_gradients(
            zip(actor_grad, self.ddpgObj.actor_model.trainable_variables)
        )
        # Record the losses
        #self.ddpgObj.actor_loss.append(actor_loss.numpy())
        #self.ddpgObj.critic_loss.append(critic_loss.numpy())



    # We compute the loss and update parameters
    def learn(self):
        """Sample a minibatch and perform a learning step."""
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        self.update(state_batch, action_batch, reward_batch, next_state_batch)
