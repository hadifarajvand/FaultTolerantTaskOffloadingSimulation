import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the training log
log_path = os.path.join(os.path.dirname(__file__), '..', 'training_log.csv')
df = pd.read_csv(log_path)

# Plot episodic and average rewards
plt.figure(figsize=(10, 6))
plt.plot(df['episode'], df['episodic_reward'], label='Episodic Reward', alpha=0.7)
plt.plot(df['episode'], df['avg_reward'], label='Average Reward (window=40)', linestyle='--')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Episodic and Average Rewards Over Episodes')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'reward_plot.png'))
plt.close()

# Plot episodic and average delays
plt.figure(figsize=(10, 6))
plt.plot(df['episode'], df['episodic_delay'], label='Episodic Delay', alpha=0.7)
plt.plot(df['episode'], df['avg_delay'], label='Average Delay (window=40)', linestyle='--')
plt.xlabel('Episode')
plt.ylabel('Delay')
plt.title('Episodic and Average Delays Over Episodes')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'delay_plot.png'))
plt.close()

print('Plots saved as reward_plot.png and delay_plot.png in the analysis directory.') 