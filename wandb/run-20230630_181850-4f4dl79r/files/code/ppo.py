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
        return action, probs.log_prob(action), self.critic(x)
    
def update(agent, optimizer, states, actions, log_probs, returns, advantages, epsilon_clip, policy_epochs,step):
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
    # b_states = states.detach().numpy()
    # b_actions = actions.detach().numpy()
    # b_log_probs = log_probs.detach().numpy()
    # b_advantages = advantages.detach().numpy()
    # b_returns = returns.detach().numpy()
    
    b_inds = np.arange(step)
    num_minibatches = 4
    minibatch_size = step // num_minibatches
    for _ in range(policy_epochs):
        np.random.shuffle(b_inds)
        # print("shuffled",b_inds)
        for start in range(0, step, minibatch_size):
            end = start + minibatch_size
            idx = b_inds[start:end]
            state = states[idx]
            # print("is state tensor?", type(states))
            # print("start, end, ministate",start,end,state)
            action = actions.long()[idx]
            # print("is action tensor?", type(actions))
            log_prob = log_probs[idx]
            # Compute new action probabilities and log probabilities based on the current states
            advantage = advantages[idx].detach()
            cumulative_rewards = returns[idx]

            # action = torch.tensor(action)
            _, action_log_probs, predicted_vals = agent.get_action_and_value(state,action)
            # action_probs = torch.softmax(action_logits, dim=-1)
            # action_distribution = Categorical(logits=action_logits)
            # action_probabilities = action_distribution.probs
            # action_log_probabilities = action_distribution.log_prob(action)

            # Compute the ratio of new and old action probabilities
            ratio = torch.exp(action_log_probs.detach() - log_prob.detach())
            # Compute the surrogate objective for the policy update
            # advantage = torch.tensor(advantage)
            # print(type(advantage),type(ratio))
            loss1 = advantage * ratio
            loss2 = advantage * torch.clamp(ratio,1-epsilon_clip,1+epsilon_clip)
            pg_loss = -torch.min(loss1,loss2).mean()

            # cumulative_rewards = torch.tensor(cumulative_rewards)
            # Value loss
            # Compute the value loss using the mean squared error between predicted values and returns
            v_loss = 0.5 * ((predicted_vals - cumulative_rewards) ** 2).mean()
            loss = pg_loss - v_loss * 0.5

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


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
    num_steps = 125
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
            if step % 20 == 0 or terminated or truncated:
                print("observations:", " ".join([f"{x:+0.2f}" for x in next_obs]))
                print(f"step {step} total_reward {total_reward:+0.2f}")
            # Update the current state with the next state
            observation = next_obs
            step += 1
        print("step:",step)
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
        update(agent, optimizer, states, actions, log_probs, returns, advantages, epsilon_clip, policy_epochs, step)
        episode += 1

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
    env = gym.make("LunarLander-v2",render_mode="human")

    env = gym.wrappers.RecordEpisodeStatistics(env)
    track = True
    if track:
        wandb.init(
            project="PPO",
            sync_tensorboard=True,
            name="ppo",
            monitor_gym=True,
            save_code=True,
        )

    # Initialize an empty list to store the episode rewards
    episode_rewards = []



    ppo(env, 10, 0.9, 0.2, 4, 2.5e-4, episode_rewards)

    # Create a line plot of the episode rewards over time
    wandb.log({"Episode Rewards": wandb.plot.line(
        series=episode_rewards,
        xlabel="Time Step",
        ylabel="Reward",
        title="Episode Rewards",
    )})
