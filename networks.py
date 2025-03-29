import torch
import torch.nn.functional as fnn
import einops as eop
from torch import nn
from torch.distributions import Normal, Independent


class EncoderModel(nn.Module):
    def __init__(self, params):
        '''
        parameters: 整个训练的参数

        网络结构比较简单，只是一个 CNN 网络，输入维度为 3（RGB 图像），输出维度为 256。
        输入3 * 64 * 64 ，输出 256 * 2 * 2。
        '''
        super(EncoderModel, self).__init__()
        self.params = params
        self.encoder_net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2, padding='valid'),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding='valid'),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding='valid'),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding='valid'),
            nn.ReLU(),
        )

    def forward(self, obs):
        encoded_obs = self.encoder_net(obs)
        # 这行代码使用了 [einops](https://einops.rocks/) 库中的 `rearrange` 函数，将原本形状为 `[batch, channel, height, width]` 的张量展开（flatten）成形状为 `[batch, channel * height * width]` 的张量。这通常用于将卷积层的输出扁平化，以便后续通过全连接层（或其他处理）使用。
        encoded_obs = eop.rearrange(encoded_obs, 'b c h w -> b (c h w)')
        return encoded_obs


class RepresentationModel(nn.Module):
    def __init__(self, params):
        '''
        parameters: 整个训练的参数

        网络结构比较简单，只是一个全连接网络，输入维度为 h_dim + feat_dim，输出维度为 2 * z_dim。
        输出的是一个正态分布的均值和标准差，表示当前时刻的潜在状态。
        '''
        super(RepresentationModel, self).__init__()
        self.params = params
        self.repr_net = FeedForwardNet(
            input_dim=self.params['h_dim']+self.params['feat_dim'],
            output_dim=2*self.params['z_dim'],
            hidden_dim=self.params['h_dim'],
            n_layers=self.params['n_ff_layers']
        )

    def forward(self, h_state, encoded_obs):
        concat_input = torch.concat([h_state, encoded_obs], dim=1)
        mu, pre_std = torch.chunk(self.repr_net(concat_input), chunks=2, dim=1)
        # 对 pre_std 进行 softplus 操作并加上一个最小标准差，以确保标准差为正。
        std = fnn.softplus(pre_std + 0.55) + self.params['min_std']
        dist_posterior = Independent(Normal(loc=mu, scale=std), reinterpreted_batch_ndims=1)
        return dist_posterior


class RecurrentModel(nn.Module):
    def __init__(self, params, action_dim):
        '''
        params: 整个训练的参数
        action_dim: 动作的维度

        网络结构比较简单，只是一个 GRU 网络，输入维度为 z_dim + action_dim，输出维度为 h_dim。
        '''
        super(RecurrentModel, self).__init__()
        self.params = params
        # GRU 网络的输入维度为 z_dim + action_dim
        self.gru_net = nn.GRUCell(input_size=self.params['z_dim']+action_dim, hidden_size=self.params['h_dim'])

    def forward(self, h_state, z_state, action):
        gru_input = torch.concat([z_state, action], dim=1)
        next_h_state = self.gru_net(gru_input, h_state)
        return next_h_state


class TransitionModel(nn.Module):
    def __init__(self, params):
        '''
        parameters: 整个训练的参数

        网络结构比较简单，只是一个全连接网络，输入维度为 h_dim，输出维度为 2 * z_dim。
        输出的是一个正态分布的均值和标准差，表示当前时刻的潜在状态。
        '''
        super(TransitionModel, self).__init__()
        self.params = params
        self.transition_net = FeedForwardNet(
            input_dim=self.params['h_dim'],
            output_dim=2*self.params['z_dim'],
            hidden_dim=self.params['h_dim'],
            n_layers=self.params['n_ff_layers']
        )

    def forward(self, h_state):
        mu, pre_std = torch.chunk(self.transition_net(h_state), chunks=2, dim=1)
        # 对 pre_std 进行 softplus 操作并加上一个最小标准差，以确保标准差为正。
        std = fnn.softplus(pre_std + 0.55) + self.params['min_std']
        dist_prior = Independent(Normal(loc=mu, scale=std), reinterpreted_batch_ndims=1)
        return dist_prior


class DecoderModel(nn.Module):
    def __init__(self, params):
        '''
        params : 整个训练的参数

        网络有一个全连接层，将 h_state 和 z_state 连接起来，然后通过一系列的反卷积层将其解码为与环境观测相对应的重构输出。
        也就是输出为 3 * 64 * 64 的大小的分布
        '''
        super(DecoderModel, self).__init__()
        self.params = params
        self.fc_net = nn.Linear(in_features=self.params['h_dim']+self.params['z_dim'], out_features=1024)
        self.decoder_net = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, stride=2, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, stride=2, kernel_size=5),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, stride=2, kernel_size=6),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=3, stride=2, kernel_size=6)
        )

    def forward(self, h_state, z_state):
        concat_input = torch.concat([h_state, z_state], dim=-1)
        fc_output = self.fc_net(concat_input)
        reshaped_input = fc_output.view(-1, 256, 2, 2)
        mu_obs = self.decoder_net(reshaped_input)
        dist_obs = Independent(Normal(loc=mu_obs, scale=1.0), reinterpreted_batch_ndims=3)
        return dist_obs


class RewardModel(nn.Module):
    def __init__(self, params):
        '''
        params: 整个训练的参数

        网络结构比较简单，只是一个全连接网络，输入维度为 h_dim + z_dim，输出维度为 1。应该是直接预测奖励值。
        '''
        super(RewardModel, self).__init__()
        self.params = params
        self.reward_net = FeedForwardNet(
            input_dim=params['h_dim']+params['z_dim'],
            output_dim=1,
            hidden_dim=params['h_dim'],
            n_layers=self.params['n_ff_layers']
        )

    def forward(self, h_state, z_state):
        concat_input = torch.concat([h_state, z_state], dim=-1)
        mu_reward = self.reward_net(concat_input)
        dist_reward = Independent(Normal(loc=mu_reward, scale=1.0), reinterpreted_batch_ndims=1)
        return dist_reward


class FeedForwardNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers):
        '''
        前向传播网络

        param input_dim: 输入维度
        param output_dim: 输出维度
        param hidden_dim: 隐藏层维度
        param n_layers: 隐藏层的层数
        
        '''

        super(FeedForwardNet, self).__init__()
        self.to_hidden = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_dim),
            nn.ReLU()
        )
        self.hidden = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
                nn.ReLU()
            ) for _ in range(n_layers-1)
        ])
        self.from_hidden = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=output_dim)
        )

    def forward(self, x):
        to_hidden = self.to_hidden(x)
        hidden = self.hidden(to_hidden)
        return self.from_hidden(hidden)
