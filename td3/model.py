import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477

class ConvFeature(nn.Module):
    
    def __init__(self, state_dim=(1, 240, 320), num_filters=8, filter_size=5, max_pool=4):
        
        super().__init__()
        self.conv1 = nn.Conv2d(state_dim[0], num_filters, filter_size, stride=3)
        self.conv2 = nn.Conv2d(num_filters, num_filters, 3, stride=3)
        self.state_dim = state_dim
        self.num_filters = num_filters
        self.max_pool = max_pool
        
    def forward(self, state):
        
        a = F.relu(self.conv1(state))
        a = F.dropout(a, p=0.2)
        a = F.relu(self.conv2(a))
        a = F.dropout(a, p=0.2)
        
        return a
    
    @property
    def fc_size(self):
        
        #return self.num_filters * int(self.state_dim[1]/self.max_pool) * int(self.state_dim[2]/self.max_pool)
        return self.num_filters * 6 * 8
    

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.conv_feature = ConvFeature(state_dim)
        self.fc1 = nn.Linear(self.conv_feature.fc_size, 32)
        self.fc2 = nn.Linear(32, action_dim)
        
        self.max_action = torch.FloatTensor(max_action).to(device)
        
    def forward(self, state):
        conv_feature = self.conv_feature.forward(state)
        conv_feature = conv_feature.reshape(-1, self.conv_feature.fc_size)
        a = F.relu(self.fc1(conv_feature))
        return self.max_action * torch.tanh(self.fc2(a))
    
    
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        
        super(Critic, self).__init__()
        self.conv_feature_1 = ConvFeature(state_dim)
        self.fc1 = nn.Linear(self.conv_feature_1.fc_size + action_dim, 32)
        self.fc2 = nn.Linear(32, 1)

        # Q2 architecture
        self.conv_feature_2 = ConvFeature(state_dim)
        self.fc3 = nn.Linear(self.conv_feature_2.fc_size + action_dim, 32)
        self.fc4 = nn.Linear(32, 1)


    def forward(self, state, action):

        conv_feature = self.conv_feature_1.forward(state)
        conv_feature = conv_feature.reshape(-1, self.conv_feature_1.fc_size)
        q1 = torch.cat([conv_feature, action], 1)
        q1 = self.fc1(q1)
        q1 = self.fc2(q1)

        conv_feature = self.conv_feature_2.forward(state)
        conv_feature = conv_feature.reshape(-1, self.conv_feature_2.fc_size)
        q2 = torch.cat([conv_feature, action], 1)
        q2 = self.fc3(q2)
        q2 = self.fc4(q2)
        return q1, q2


    def Q1(self, state, action):
        conv_feature = self.conv_feature_1.forward(state)
        conv_feature = conv_feature.reshape(-1, self.conv_feature_1.fc_size)
        q1 = torch.cat([conv_feature, action], 1)
        q1 = self.fc1(q1)
        q1 = self.fc2(q1)
        return q1


class TD3(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2
    ):

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = torch.tensor(max_action).to(device)
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0


    def select_action(self, state):
        state = torch.FloatTensor(state).to(device).unsqueeze(0)
        return self.actor(state).cpu().data.numpy().flatten()
    
    def update_target_networks(self):
        
        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def train(self, replay_buffer, batch_size=100):
        self.total_it += 1

        # Sample replay buffer 
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        
        target_noise = torch.randn(action.shape[0], action.shape[1]) * self.policy_noise
        target_noise_clipped = torch.clip(target_noise, -self.noise_clip, self.noise_clip).to(device)
        
        with torch.no_grad():
            
            target_action = self.actor_target(next_state)
            target_action_noised = torch.max(torch.min(self.max_action, target_action + target_noise_clipped), 
                                             -self.max_action)
            #target_action_noised = target_action
            
            q1_next, q2_next = self.critic_target(next_state, target_action_noised)
            
            y = reward + self.discount * not_done * torch.min(q1_next, q2_next)
        
        pred_q_1, pred_q_2 = self.critic.forward(state, action)
        
        critic_loss = torch.nn.functional.mse_loss(pred_q_1, y) + torch.nn.functional.mse_loss(pred_q_2, y)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        actor_loss = None # not always updated
        
        if self.total_it % self.policy_freq == 0:
            
            policy_act = self.actor(state)
            
            # why q1-here, just randomly rather than either of q1 or q2?
            actor_loss = -1 * self.critic.Q1(state, policy_act).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            
            self.actor_optimizer.step()
            
        self.update_target_networks()
        
        return actor_loss, critic_loss
    
    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
   
    
if __name__ == '__main__':
    #a = Actor((1, 30, 2), 2, np.array([1, 1])).to(device)
    #c = Critic((1, 30, 40), 2).to(device)
    #obs = torch.ones([1, 30 * 8, 40 * 8], device=device)
    td3 = TD3((4, 240, 320), 2, np.array([1, 1]))
    #act = a(obs)
    #val = c(obs, act)
    act = td3.select_action(np.ones([4, 60, 80]))
    from utils import ReplayBuffer
    #buf = ReplayBuffer((1, 30, 40), 2, 10)
    #buf.add(obs.cpu().numpy(), act, obs.cpu().numpy(), 1, 0)
    