import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import wandb

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

torch.autograd.set_detect_anomaly(True)

class Agent(nn.Module):
    def __init__(self, env):
        super(Agent, self).__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(env.observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(env.observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, env.action_space.n), std=0.01),
        )
    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), self.critic(x)
    
def update(agent, optimizer, states, actions, log_probs, returns, advantages, step):
    
    b_inds = np.arange(step)
    num_minibatches = 4
    minibatch_size = step // num_minibatches
    for _ in range(update_epochs):
        np.random.shuffle(b_inds)
        for start in range(0, step, minibatch_size):
            end = start + minibatch_size
            idx = b_inds[start:end]
            state = states[idx]
            action = actions[idx]
            log_prob = log_probs[idx]
            advantage = advantages[idx].detach()
            cumulative_rewards = returns[idx]

            _, action_log_probs, predicted_vals = agent.get_action_and_value(state,action)

            ratio = torch.exp(action_log_probs.detach() - log_prob.detach())

            loss1 = advantage * ratio
            loss2 = advantage * torch.clamp(ratio,1-epsilon_clip,1+epsilon_clip)
            pg_loss = -torch.min(loss1,loss2).mean()

            v_loss = 0.5 * ((predicted_vals - cumulative_rewards) ** 2).mean()
            loss = pg_loss - v_loss * v_loss_coef

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def ppo(env, episode_rewards):

    # Initialize the Actor and Critic networks
    agent = Agent(env)

    # Initialize the optimizer for the networks
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)
    
    episode = 0
    num_steps = 200
    while episode < num_episodes:
        observation, info = env.reset()
        states = torch.zeros((num_steps, 1) + env.observation_space.shape)
        actions = torch.zeros((num_steps, 1) + env.action_space.shape)
        rewards = torch.zeros((num_steps, 1))
        log_probs = torch.zeros((num_steps, 1))

        terminated, truncated = False, False
        total_reward = 0
        step = 0
        # Training loop
        while not terminated and not truncated and step < num_steps:
            # Reset the environment
            state_tensor = torch.tensor(observation, dtype=torch.float)
            
            action, log_prob, _ = agent.get_action_and_value(state_tensor)

            # Perform actions and collect data for a minibatch
            next_obs, reward, terminated, truncated, info = env.step(action.item())

            states[step] = state_tensor
            actions[step] = action
            rewards[step] = torch.tensor(reward)
            log_probs[step] = log_prob

            total_reward += reward
            observation = next_obs
            step += 1
        if track:
            log_episode_reward(total_reward,episode_rewards)
        print(f"episode {episode} total_reward {total_reward:+0.2f}")

        # Compute the returns and advantages
        returns = torch.zeros((step, 1))
        advantages = torch.zeros((step, 1))

        discounted_reward = 0
        for i in range(step - 1, -1, -1):
            discounted_reward = rewards[i] * gamma + discounted_reward
            returns[i] = discounted_reward
  
        for i in range(step):
            value_estimates = agent.critic(states[i])
            advantages[i] = returns[i] - value_estimates
        
        # Update the networks using PPO
        update(agent, optimizer, states, actions, log_probs, returns, advantages, step)
        episode += 1

    env.close()

def log_episode_reward(reward,episode_rewards):
    episode_rewards.append(reward)
    wandb.log({"Episode Reward": reward})
if __name__ == "__main__":
    env = gym.make("LunarLander-v2")#,render_mode="human")

    env = gym.wrappers.RecordEpisodeStatistics(env)
    track = True
    if track:
        wandb.init(
            project="PPO",
            # sync_tensorboard=True,
            name="ppo",
            monitor_gym=True,
            save_code=True,
        )
    episode_rewards = []
    
    num_episodes = 64
    gamma = 0.99
    epsilon_clip = 0.2
    update_epochs = 4
    v_loss_coef = 0.5
    learning_rate = 2.5e-4
    
    
    ppo(env, episode_rewards)

    # Create a line plot of the episode rewards over time
    wandb.log({"Episode Rewards": wandb.plot.line(
        series=episode_rewards,
        xlabel="Time Step",
        ylabel="Reward",
        title="Episode Rewards",
    )})
