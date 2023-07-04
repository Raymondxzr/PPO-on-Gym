import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

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
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(env.observation_space.shape).prod(), 64)),
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
        return action, probs.log_prob(action)
    # , probs.entropy(), self.critic(x)

def update(agent, optimizer, states, actions, log_probs, returns, advantages, epsilon_clip, policy_epochs):
    """
    Update the actor and critic networks using the collected data.

    Args:
        actor_net (nn.Module): Actor network.
        critic_net (nn.Module): Critic network.
        actor_optimizer (torch.optim.Optimizer): Optimizer for the actor network.
        critic_optimizer (torch.optim.Optimizer): Optimizer for the critic network.
        states (list): List of collected states.
        actions (list): List of collected actions.
        log_probs (list): List of log probabilities of selected actions.
        returns (list): List of computed returns.
        advantages (list): List of computed advantages.
        epsilon_clip (float): Clipping parameter for PPO.
        policy_epochs (int): Number of policy optimization epochs.
        value_epochs (int): Number of value function optimization epochs.

    Returns:
        None
    """

    # Convert lists to tensors if necessary
    # Perform updates for a certain number of policy epochs
    b_inds = np.arange(len(states))
    minibatch_size = len(states) // 4
    for _ in range(policy_epochs):
        for start in range(0, len(states), minibatch_size):
            end = start + minibatch_size
            idx = b_inds[start:end]
        # for i in range(len(states)):
            state = states[idx]
            # print(states[i])
            action = actions[idx]
            log_prob = log_probs[idx]
            # Compute new action probabilities and log probabilities based on the current states
            advantage = advantages[idx]
            # print(advantage)
            cumulative_rewards = returns[idx]
            # state = states[i].clone()
            # state = state.clone()
            _, action_log_probabilities, predicted_val = agent.get_action_and_value(state,action)
            # action_probs = torch.softmax(action_logits, dim=-1)
            # action_distribution = Categorical(logits=action_logits)
            # action_probabilities = action_distribution.probs
            # action_log_probabilities = action_distribution.log_prob(action)

            # Compute the ratio of new and old action probabilities
            ratio = (action_log_probabilities - log_prob).exp()
            # Compute the surrogate objective for the policy update
            loss1 = advantage * ratio
            loss2 = advantage * torch.clamp(ratio,1-epsilon_clip,1+epsilon_clip)
            pg_loss = -torch.min(loss1,loss2).mean()

            # Value loss
            # Compute the value loss using the mean squared error between predicted values and returns
            v_loss = 0.5 * ((predicted_val - cumulative_rewards) ** 2).mean()
            loss = pg_loss - v_loss * 0.5

            optimizer.zero_grad()
            optimizer.step()

            loss.backward(retain_graph=True)
        # Compute the clipped surrogate objective

        # Compute the policy loss as the minimum of the clipped and unclipped surrogate objectives

        # Update the actor network using the policy loss

        # Perform updates for a certain number of value epochs
    # for _ in range(value_epochs):
    #     for i in range(len(states)):
    #         state = states[i]
    #         action = actions[i]
    #         log_prob = log_probs[i]
    #         # Compute value predictions for the current states
    #         predicted_val = agent.critic(state)
    #         cumulative_rewards = returns[state]
    #         # Compute the value loss using the mean squared error between predicted values and returns
    #         value_loss = nn.MSELoss()(predicted_val,cumulative_rewards)
    #         # Update the critic network using the value loss
    #         optimizer.zero_grad()
    #         value_loss.backward()
    #         optimizer.step()



def ppo(env, num_episodes, gamma, epsilon_clip, policy_epochs, learning_rate,episode_rewards):
    """
    Proximal Policy Optimization (PPO) algorithm implementation for the LunarLander environment.

    Args:
        env_name (str): Name of the Gym environment.
        num_episodes (int): Number of episodes to train.
        batch_size (int): Number of samples to collect before updating the networks.
        gamma (float): Discount factor for computing returns.
        epsilon_clip (float): Clipping parameter for PPO.
        policy_epochs (int): Number of policy optimization epochs.
        value_epochs (int): Number of value function optimization epochs.
        learning_rate (float): Learning rate for the optimizer.

    Returns:
        None
    """

    # Create the LunarLander environment    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Initialize the Actor and Critic networks
    agent = Agent(env)

    # Initialize the optimizer for the networks
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)
    
    episode = 0

    while episode < num_episodes:
        observation, info = env.reset()
        states = []
        rewards = []
        # infos = []
        log_probs = []
        actions = []
        terminated, truncated = False, False
        total_reward = 0
        steps = 0
        # Training loop
        while not terminated and not truncated:
            # Reset the environment
            state_tensor = torch.tensor(observation, dtype=torch.float)
            
            action, log_prob = agent.get_action_and_value(state_tensor)
            # if step == 0:
            #     states.append(state_tensor)
            #     actions.append(torch.tensor(action))
            # Perform actions and collect data for a minibatch
            next_obs, reward, terminated, truncated, info = env.step(action.item())
            # state_tensor = torch.tensor(next_obs, dtype=torch.float)
            states.append(state_tensor)
            actions.append(action)
            rewards.append(torch.tensor(reward))
            # infos.append(torch.tensor(info))
            log_probs.append(log_prob)
            total_reward += reward
            # Update the current state with the next state
            observation = next_obs
            # if steps % 20 == 0 or terminated or truncated:
            #     print("observations:", " ".join([f"{x:+0.2f}" for x in next_obs]))
            #     print(f"step {steps} total_reward {total_reward:+0.2f}")
            # steps += 1
        log_episode_reward(total_reward,episode_rewards)
        print(f"episode {episode} total_reward {total_reward:+0.2f}")
        # Compute the returns and advantages
        returns = [torch.tensor(0)] * len(states)
        discounted_reward = 0
        advantages = [torch.tensor(0)] * len(states)
        for i in range(len(rewards)-1,-1,-1):
            discounted_reward = rewards[i] * gamma + discounted_reward
            returns[i] = discounted_reward

        for i in range(len(states)):
            value_estimates = agent.critic(states[i])
            advantages[i] = returns[i] - value_estimates
        
        update(agent, optimizer, states, actions, log_probs, returns, advantages, epsilon_clip, policy_epochs)
        episode+=1
    # Update the networks using PPO

    env.close()

# if __name__ == "__main__":
#     env = gym.make("LunarLander-v2")
                #    ,render_mode="human")
    # enable_wind=True,
                #    wind_power=10, turbulence_power=1)
    # ppo(env, num_episodes, batch_size, gamma, epsilon_clip, policy_epochs, value_epochs, learning_rate):
def log_episode_reward(reward,episode_rewards):
    episode_rewards.append(reward)
    wandb.log({"Episode Reward": reward})
if __name__ == "__main__":
    env = gym.make("LunarLander-v2")

    env = gym.wrappers.RecordEpisodeStatistics(env)
    track = True
    if track:
        import wandb
        wandb.init(
            project="PPO",
            sync_tensorboard=True,
            name="ppo",
            monitor_gym=True,
            save_code=True,
        )

    # Initialize an empty list to store the episode rewards
    episode_rewards = []



    ppo(env, 20, 0.99, 0.2, 4, 2.5e-4, episode_rewards)

    # Create a line plot of the episode rewards over time
    wandb.log({"Episode Rewards": wandb.plot.line(
        series=episode_rewards,
        xlabel="Time Step",
        ylabel="Reward",
        title="Episode Rewards",
    )})
