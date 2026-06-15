# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn
import torch.nn.functional as F
            
class ActorCritic(nn.Module):
    is_recurrent = False
    def __init__(self,  
                        num_actions,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        encoder_hidden_dims=[512, 256, 128],
                        activation='elu',
                        init_noise_std=1.5,
                        num_props=12,
                        num_rays=31,
                        his_len=10,
                        **kwargs):
        if kwargs:
            print("ActorCriticNew.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCritic, self).__init__()

        activation = get_activation(activation)

  
        self.his_len = his_len
        self.num_rays = num_rays
        self.num_props = num_props
        self.num_obs_one_step = num_rays + num_props + 2  # 2 for target x,y
        self.num_obs_hist = self.num_obs_one_step * his_len
        self.num_actions = num_actions
        self.num_latent = 16

        mlp_input_dim_a = self.num_obs_one_step + self.num_latent
        mlp_input_dim_c = self.num_obs_one_step + self.num_latent
        mlp_input_dim_e = self.num_obs_hist

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)


        # Encoder
        encoder_layers = []
        encoder_layers.append(nn.Linear(mlp_input_dim_e, encoder_hidden_dims[0]))
        encoder_layers.append(activation)
        for l in range(len(encoder_hidden_dims)):
            if l == len(encoder_hidden_dims) - 1:
                encoder_layers.append(nn.Linear(encoder_hidden_dims[l], self.num_latent))
            else:
                encoder_layers.append(nn.Linear(encoder_hidden_dims[l], encoder_hidden_dims[l + 1]))
                encoder_layers.append(activation)
        self.encoder = nn.Sequential(*encoder_layers)


        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")
        print(f"Encoder MLP: {self.encoder}")

        self.rays_max = 2.0
        self.rays_min = 0.0
        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)
    
    def extract(self, observations):
        # obs_buf = torch.cat((
        #             self.prop_buf, # 12
        #             self.delay_rays.clip(min=0.1, max=5.0), # num_rays 31
        #             self.delay_goal, # 2
        #             ), dim=-1)
        obs_hist = observations
        obs_buf = observations[:, -self.num_obs_one_step:]
        props = obs_buf[:, :self.num_props]
        rays = obs_buf[:, self.num_props : self.num_props + self.num_rays]
        goals = obs_buf[:, -2:]

        return obs_buf, obs_hist, props, rays, goals
                    
    def update_distribution(self, observations):
        obs_buf, obs_hist, props, rays, goals = self.extract(observations)
        latent = self.encoder(obs_hist)
        observations = torch.cat(
                            (obs_buf, latent), dim=-1)  # detach? 
        mean = self.actor(observations)
        self.distribution = Normal(mean, mean*0. + self.std)
        self.mean = mean
        
    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        actions = self.distribution.sample()
        return actions
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, im_heights=None, im_rays=None):
        obs_buf, obs_hist, props, rays, goals = self.extract(observations)
        latent = self.encoder(obs_hist)
        observations = torch.cat(
                            (obs_buf, latent), dim=-1)  # detach? 
        mean = self.actor(observations)

        return mean

    def evaluate(self, observations, **kwargs):
        obs_buf, obs_hist, props, rays, goals = self.extract(observations)
        latent = self.encoder(obs_hist)
        observations = torch.cat(
                            (obs_buf, latent), dim=-1)  # detach? 
        value = self.critic(observations)
        return value
    
    
def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
