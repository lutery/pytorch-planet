# References
#       PlaNet Paper: https://arxiv.org/pdf/1811.04551
#
#       Implementation:
#           1. Danijar's repo: [https://github.com/danijar/planet]
#           1. Jaesik's repo: [https://github.com/jsikyoon/dreamer-torch]
#           2. Kaito's repo: [https://github.com/cross32768/PlaNet_PyTorch]

import torch
from torch import nn
from torch.distributions import kl_divergence
from networks import EncoderModel, RepresentationModel, RecurrentModel, TransitionModel, DecoderModel, RewardModel
from utils import get_device, get_dtype


class Planet(nn.Module):
    def __init__(self, params, action_dim):
        '''
        parameters: 整个训练的参数
        action_dim: 动作的维度
        '''

        super(Planet, self).__init__()
        self.params = params
        self.d_type = get_dtype(self.params['fp_precision'])
        self.device = get_device(self.params['device'])
        self.action_dim = action_dim
        '''
        这些模型在 PlaNet 框架中各有分工，简要说明如下：

        1. **EncoderModel**: 负责将观察（像素图像）编码为紧凑的特征向量，减少原始输入的维度。  
        2. **RepresentationModel**: 使用编码后的特征向量与先前的隐藏状态来生成后验分布 (posterior)，表示当前时刻的潜在状态。  
        3. **TransitionModel**: 生成先验分布 (prior)，描述在未观测到当前帧情况下，对潜在状态的预测。  
        4. **DecoderModel**: 将隐藏状态与潜在状态解码回与环境观测相对应的重构输出。  
        5. **RewardModel**: 从隐藏状态和潜在状态预测当前时刻的奖励，以便在学习及规划过程中使用。  
        6. **RecurrentModel**: 按序列更新并维护隐藏状态，结合潜在状态和动作来捕捉时间上的依赖关系。

        todo 描述这几个网络之间的数据流通情况
        '''
        self.rnn_model = RecurrentModel(params=self.params, action_dim=self.action_dim)
        self.obs_encoder = EncoderModel(params=self.params)
        self.repr_model = RepresentationModel(params=self.params)
        self.transition_model = TransitionModel(params=self.params)
        self.decoder_model = DecoderModel(params=self.params)
        self.reward_model = RewardModel(params=self.params)

    def __repr__(self):
        return 'PlaNet'

    def get_init_h_state(self, batch_size):
        '''
        创建一个初始的确定性隐藏状态
        '''
        return torch.zeros((batch_size, self.params['h_dim']), dtype=self.d_type, device=self.device)

    def forward(self, sampled_episodes):
        '''
        param sampled_episodes: 采样的 episode 数据
        sampled_episodes['obs']: 采样的观察数据chunk_length， batch_size， visual_resolution， visual_resolution， channels）
        sampled_episodes['action']: 采样的动作数据（chunk_length， batch_size， action_dim）
        sampled_episodes['reward']: 采样的奖励数据  chunk_length， batch_size， 1）
        '''
        # 创建一个字典来存储预测的分布
        # prior: 先验分布
        # posterior: 后验分布
        # recon_obs: 重构观察
        # reward: 奖励分布
        # 这里的分布是一个分布对象，包含均值和方差等信息 todo 
        dist_predicted = {'prior': list(), 'posterior': list(), 'recon_obs': list(), 'reward': list()}
        # 获取一个初始的确定性隐藏状态，shape 为（batch_size， h_dim）
        h_state = self.get_init_h_state(batch_size=self.params['batch_size'])
        for time_stamp in range(self.params['chunk_length']):
            # 遍历每一个序列的时间步
            input_obs = sampled_episodes['obs'][time_stamp] # 获取当前时间步的观察数据
            # 创建一个噪声观察数据，噪声的大小与像素位数有关
            # 这里的噪声是一个高斯噪声，均值为0，方差为1
            # 将噪声进行缩放后，使其范围满足和观察空间一致，因为高斯噪声本身就分布在0附近的左右，所以只需要使用正数缩放即可
            # 这行代码通过生成与 input_obs 同形状的标准正态分布噪声，并用 (1/2^(pixel_bit)) 缩放后加到 input_obs 上，从而为输入观测值注入噪声。这样做可以模拟低位深像素的量化效应或者作为数据正则化的一种手段。
            # 提高代码鲁棒性
            noisy_input_obs = (1/pow(2, self.params['pixel_bit']))*torch.randn_like(input_obs) + input_obs
            # 获取当前时间步的动作数据
            action = sampled_episodes['action'][time_stamp]

            # 对当前时间步的观察数据进行编码，得到一个编码后的观察数据
            # shape = （batch_size， h_dim）
            encoded_obs = self.obs_encoder(noisy_input_obs)
            # 根据上一个隐藏状态生（上个确定性隐藏状态（上个后验状态、上一个动作））成一个先验分布，shape = （batch_size， 2 * z_dim）
            z_prior = self.transition_model(h_state)
            # 根据编码后的观察数据和上一个隐藏状态生成一个后验分布（上个确定性隐藏状态（上个后验状态、上一个动作）， 本次编码后的观察），shape = （batch_size， 2 * z_dim）
            z_posterior = self.repr_model(h_state, encoded_obs)
            
            # 对后验分布进行采样，得到一个潜在状态
            # shape = （batch_size， z_dim）
            z_state = z_posterior.rsample()

            # 对潜在状态进行解码，得到一个重构观察数据，输入为潜在状态和上一个隐藏状态
            # shape = （batch_size， visual_resolution， visual_resolution， channels）
            dist_recon_obs = self.decoder_model(h_state, z_state)
            # 根据潜在状态和上一个隐藏状态生成一个奖励分布，输入为潜在状态和上一个隐藏状态
            # shape = （batch_size， 1）
            dist_reward = self.reward_model(h_state, z_state)
            # 根据潜在状态和动作数据更确定性新隐藏状态，输入为潜在状态和动作数据
            h_state = self.rnn_model(h_state, z_state, action)

            dist_predicted['prior'].append(z_prior)
            dist_predicted['posterior'].append(z_posterior)
            dist_predicted['recon_obs'].append(dist_recon_obs)
            dist_predicted['reward'].append(dist_reward)
        # 返回一个字典，包含了每个时间步先验分布，后验分布，重构观察数据和奖励分布
        return dist_predicted

    def compute_loss(self, target, dist_predicted):
        '''
        param target: 目标数据,也是采样的数据
        param dist_predicted: 预测的分布

        return net_loss: 总损失, 其他损失返回标量，用于记录
        '''
        # 创建一个大小和观察一致的分布
        sampled_reconstructed_obs = torch.stack([dist_recon_obs.rsample() for dist_recon_obs in dist_predicted['recon_obs']])
        # 创建一个大小和奖励一致的分布
        sampled_reward = torch.stack([dist_reward.rsample() for dist_reward in dist_predicted['reward']])
        # Individual loss terms
        # 重构观察损失，也就是重构观察数据和目标观察数据之间的均方误差
        recon_loss = ((target['obs'] - sampled_reconstructed_obs) ** 2).mean(dim=0).mean(dim=0).sum()
        # 预测中的先验分布和后验分布之间的 KL 散度损失，要保证其相近
        kl_loss = torch.stack(
            [kl_divergence(p=dist_posterior, q=dist_prior) for dist_prior, dist_posterior in
             zip(dist_predicted['prior'], dist_predicted['posterior'])]
        )
        # 对 KL 散度损失进行平均
        # 定义了 KL 散度的下限，通常被称为“free nats”技巧。在计算 KL 散度损失时，将每个时间步的 KL 值与 free_nats 比较，确保即使模型预测的 KL 散度非常低，也不会过分惩罚。这样可以防止模型过度拟合噪声，提高训练的稳定性
        kl_loss = (torch.maximum(kl_loss, torch.tensor([self.params['free_nats']])[0])).mean()
        # 奖励预测损失，计算目标奖励数据和预测奖励数据之间的均方误差
        reward_prediction_loss = ((target['reward'] - sampled_reward) ** 2).mean()
        # Net loss term 计算总损失
        net_loss = recon_loss + kl_loss + reward_prediction_loss
        return net_loss, (recon_loss.item(), kl_loss.item(), reward_prediction_loss.item())

