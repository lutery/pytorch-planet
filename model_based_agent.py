# References
#        Replay Buffer: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

import copy
from collections import namedtuple
import cv2
import gymnasium as gym
from torch import nn
from torch.optim import Adam
from torch.distributions import Normal, Independent
from torch.utils.tensorboard import SummaryWriter
from dm_control import suite
from dm_control.suite.wrappers import pixels
from gymnasium.wrappers import TimeLimit, PixelObservationWrapper, TransformObservation
from utils import *
from planet import Planet

'''
Transition 是一个使用 Python 的 namedtuple 定义的数据结构，用来存储一次环境交互的数据。它包含三个字段：

- observation：状态或观测值
- action：采取的动作
- reward：获得的奖励

在基于 Planet 的实现中，Transition 用来保存每一步环境交互的记录，方便后续对学习过程进行采样和训练。
'''
Transition = namedtuple('Transition',
                        ('observation', 'action', 'reward'))


class ModelBasedLearner:
    def __init__(self, params):
        self.params = params
        # 浮点数据精度16位、32位还是8位
        self.d_type = get_dtype(self.params['fp_precision'])
        self.device = get_device(self.params['device'])
        exp_tag, self.env, self.action_dim = self.get_env()
        self.replay_buffer = ReplayBuffer(params=self.params)
        # 也是一个世界模型
        self.world_model = Planet(params=self.params, action_dim=self.action_dim).to(self.d_type).to(self.device)
        print(f'Initialized {self.world_model} ({count_parameters(self.world_model)}) as the world-model')
        # 直接对整个世界模型进行优化，和之前的dreamer不一样，dreamer是对每个模型进行分别优化
        self.optimizer = Adam(params=self.world_model.parameters(), lr=self.params['lr'], eps=self.params['adam_epsilon'])
        self.logger = SummaryWriter(comment=exp_tag)

    def get_env(self):
        '''
        创建环境

        return 环境的标识、环境、动作维度
        '''
        if self.params['api_name'] == 'gym':
            return self.get_gym_env()
        elif self.params['api_name'] == 'dmc':
            return self.get_dm_control_env()
        else:
            raise NotImplementedError(f'{self.params["api_name"]} is not implemented')
        
    def get_gym_env(self):
        '''
        return 环境的标识、环境、动作维度
        '''

        exp_tag = '_' + self.params['env_name'].lower() + '_'
        # 创建原始环境
        env = gym.make(self.params['env_name'], render_mode='rgb_array')
        # 设置动作重复的次数，有点像跳帧
        env = ActionRepeat(env, n_repeat=self.params['action_repeat'])
        # 设置最大的episode步数，防止无限循环
        env = TimeLimit(env, max_episode_steps=self.params['max_episode_step'])
        # 将观察值转换为RGB像素？todo 查看加入了这个之后的效果
        # PixelObservationWrapper 将像素图像作为环境的主要观测值（observation），并将其存储在字典中（通常键为 'pixels'），从而统一了观测值的接口。
        env = PixelObservationWrapper(env)
        # 是 gymnasium.wrappers 提供的一个环境包装器，其作用是对环境的观测值（observation）进行自定义的变换或处理。它允许用户通过传入一个自定义的函数，对环境返回的观测值进行动态修改
        env = TransformObservation(env, lambda obs: self.process_gym_observation(obs['pixels']))
        env.reset(seed=self.params['rng_seed'])
        action_dim = env.action_space.shape[0]
        print(f'Initialized {self.params["env_name"]} as environment')
        return exp_tag, env, action_dim

    def get_dm_control_env(self):
        exp_tag = '_' + self.params['domain_name'] + '_' + self.params['task_name'] + '_'
        env = suite.load(
            domain_name=self.params['domain_name'],
            task_name=self.params['task_name'],
            task_kwargs={'random': self.params['rng_seed']}
        )
        env = pixels.Wrapper(env=env, render_kwargs={
            'height': self.params['observation_resolution'],
            'width': self.params['observation_resolution'],
            'camera_id': 0
        })
        env = GymWrapper(env)
        env = ActionRepeatDM(env, n_repeat=self.params['action_repeat'])
        env = TransformObservationDM(env, obs_transformation=self.process_dm_observation)
        env.reset()
        action_dim = env.action_space.shape[0]
        print(f'Initialized {self.params["domain_name"]}-{self.params["task_name"]} as environment')
        return exp_tag, env, action_dim

    def process_gym_observation(self, raw_obs):
        '''
        主要实现：缩放、量化、归一化、转换为tensor

        量化再归一化主要是为了将原始像素值分成更少的离散级别，然后映射到 [-0.5, 0.5]。如果直接将 8 位整数 (0–255) 线性缩放到 [-0.5, 0.5]，会跳过量化步骤，观测值仍会保留浮点表示；通过先量化再归一化，可以强制像素只有较少的离散级别，从而更好地模拟低精度输入或起到正则化的效果。
        '''
        bits = self.params['pixel_bit'] # 这里貌似是对像素进行量化，不采用8bit存储，todo
        visual_resolution = self.params['observation_resolution'] # 图像分辨率，即缩放图像大小
        # 先对观察进行缩放，shape：（visual_resolution, visual_resolution， chanels）
        resized_obs = cv2.resize(raw_obs, dsize=(visual_resolution, visual_resolution), interpolation=cv2.INTER_AREA)
        bins = 2 ** bits
        # 将图像转换为浮点数
        norm_ob = np.float16(resized_obs)
        if bits < 8:
            # 对图像进行量化
            norm_ob = np.floor(norm_ob / 2 ** (8 - bits))
        # 量化后进行归一化到[-0.5, 0.5]之间
        norm_ob = (norm_ob / bins) - 0.5
        # 转换为tensor
        processed_obs = torch.tensor(norm_ob, dtype=self.d_type)
        # 我感觉这里是将通道放在前面，也就是将（visual_resolution, visual_resolution， chanels）转换为（chanels， visual_resolution, visual_resolution）
        processed_obs = processed_obs.transpose(0, 2)
        return processed_obs

    def process_dm_observation(self, raw_obs):
        bits = self.params['pixel_bit']
        bins = 2 ** bits
        norm_ob = np.float16(raw_obs)
        if bits < 8:
            norm_ob = np.floor(norm_ob / 2 ** (8 - bits))
        norm_ob = (norm_ob / bins) - 0.5
        processed_obs = torch.tensor(norm_ob, dtype=self.d_type)
        processed_obs = processed_obs.transpose(0, 2)
        return processed_obs

    def collect_seed_episodes(self):
        '''
        随机动作预热缓冲区
        '''
        print('\n')
        while len(self.replay_buffer.memory) < self.params['n_seed_episodes']:
            # 打印当前收集的episode数量
            print(f'\rCollecting seed episodes ({1+len(self.replay_buffer.memory)}/{self.params["n_seed_episodes"]}) ... ', end='')
            prev_obs, _ = self.env.reset()
            episode_transitions = list()
            while True:
                action = self.env.action_space.sample()
                obs, reward, terminated, truncated, info = self.env.step(action)
                # 这里存储的动作和观察处于同一个时间t
                episode_transitions.append(
                    Transition(
                        observation=prev_obs,
                        action=torch.tensor(action, dtype=self.d_type),
                        reward=torch.tensor([reward], dtype=self.d_type)
                    )
                )
                prev_obs = copy.deepcopy(obs)
                if terminated or truncated:
                    break
            self.replay_buffer.push(episode_transitions)
        print(f'\rCollected {self.params["n_seed_episodes"]} episodes as initial seed data!')

    def collect_episode(self):
        '''
        收集新的episode
        '''
        print('\rCollecting a new episode with CEM-based planning ...', end='')
        self.world_model.eval()
        # 重置环境
        prev_obs, _ = self.env.reset()
        # 获取一个初始的确定性隐藏状态
        h_state = self.world_model.get_init_h_state(batch_size=1)
        episode_transitions = list()
        # 收集一次完整游戏周期的数据
        while True:
            # Inject observation noise
            # 这里的噪声是一个高斯噪声，均值为0，方差为1
            # 将噪声进行缩放后，使其范围满足和观察空间一致，因为高斯噪声本身就分布在0附近的左右，所以只需要使用正数缩放即可
            # 这行代码通过生成与 input_obs 同形状的标准正态分布噪声，并用 (1/2^(pixel_bit)) 缩放后加到 input_obs 上，从而为输入观测值注入噪声。这样做可以模拟低位深像素的量化效应或者作为数据正则化的一种手段。
            # 提高代码鲁棒性
            noisy_prev_obs = (1/pow(2, self.params['pixel_bit']))*torch.randn_like(prev_obs) + prev_obs
            # Get posterior states using observation 获取编码后的观察
            encoded_obs = self.world_model.obs_encoder(noisy_prev_obs.unsqueeze(dim=0).to(self.device))
            # 根据上一个隐藏状态和编码后的观察生成一个后验分布
            posterior_z = self.world_model.repr_model(h_state, encoded_obs)
            # 对后验分布进行采样，得到一个潜在状态
            # shape = （batch_size， z_dim）
            z_state = posterior_z.sample()
            # Get best action by planning in latent space through open-loop prediction
            with torch.no_grad():
                # 预测动作
                action = self.plan_action_with_cem(h_state, z_state)
            exploration_noise = Normal(loc=0.0, scale=self.params['action_epsilon']).sample(sample_shape=torch.Size(action.shape)).to(self.d_type).to(self.device)
            noisy_action = action + exploration_noise # 对动作加入噪声，这里可以看出其针对连续动作空间，对离散动作空间不支持
            # 执行动作
            obs, reward, terminated, truncated, info = self.env.step(noisy_action.to('cpu').numpy())
            # Get next latent state 预测下一个确定性隐藏状态
            h_state = self.world_model.rnn_model(h_state, z_state, noisy_action.unsqueeze(dim=0))
            # Save environment transition 将本次的动作收集起来
            episode_transitions.append(
                Transition(
                    observation=prev_obs,
                    action=noisy_action.to('cpu'),
                    reward=torch.tensor([reward], dtype=self.d_type)
                )
            )
            # 深拷贝，防止影响
            prev_obs = copy.deepcopy(obs)
            if terminated or truncated:
                break
        print('\rCollected a new episode!' + 50*' ')
        return episode_transitions

    def learn_with_planet(self):
        global_step = 0
        # 这里训练指定的步数吗？
        for learning_step in range(self.params['n_steps']):
            print(f'\n\nLearning step: {1+learning_step}/{self.params["n_steps"]}\n')
            # 切换训练模式
            self.world_model.train()
            # 进行 collect_interval 次的训练
            for update_step in range(self.params['collect_interval']):
                print(f'\rFitting world model : ({update_step+1}/{self.params["collect_interval"]})', end='')
                # 先进行采样
                sampled_episodes = self.replay_buffer.sample(self.params['batch_size'])
                # 对采样的每个时间步进行预测
                dist_predicted = self.world_model(sampled_episodes=sampled_episodes)
                # 计算损失
                loss, (recon_loss, kl_loss, reward_loss) = self.world_model.compute_loss(target=sampled_episodes, dist_predicted=dist_predicted)
                loss.backward()
                # 梯度求导
                nn.utils.clip_grad_value_(self.world_model.parameters(), clip_value=self.params['max_grad_norm'])
                self.optimizer.step()
                self.optimizer.zero_grad()
                # 记录损失
                self.logger.add_scalar('Reward/max_from_train_batch', sampled_episodes['reward'].max(), global_step)
                self.logger.add_scalar('TrainLoss/obs_recon', recon_loss, global_step)
                self.logger.add_scalar('TrainLoss/kl_div', kl_loss, global_step)
                self.logger.add_scalar('TrainLoss/reward_prediction', reward_loss, global_step)
                global_step += 1
            print('\rUpdated world model!' + 50*' ')
            # 每隔一定步数进行评估以及收集新的episode todo
            with torch.no_grad():
                # 进行数据采样，采样的同时会对预测的奖励、动作、观察加入噪声
                self.replay_buffer.push(self.collect_episode())
                self.evaluate_learning(step=learning_step+1)
                self.evaluate_video_prediction(step=learning_step+1)

    def evaluate_learning(self, step):
        '''
        评估学习网络
        '''
        print('\rEvaluating learning progress ...', end='')
        self.world_model.eval() # 进入评估模式
        prev_obs, _ = self.env.reset() # 重置游戏环境
        h_state = self.world_model.get_init_h_state(batch_size=1) # 获取新的起始确定性隐藏状态
        # observed_frames保存游戏观察
        # reconstructed_frames保存预测的游戏观察
        observed_frames, reconstructed_frames = list(), list()
        ep_reward = 0
        while True:
            observed_frames.append(prev_obs)
            # Get posterior states using observation
            # 对观察进行编码
            encoded_obs = self.world_model.obs_encoder(prev_obs.unsqueeze(dim=0).to(self.device))
            # 得到随机后验分布
            posterior_z = self.world_model.repr_model(h_state, encoded_obs)
            # 根据后验状态获取潜在状态
            z_state = posterior_z.sample()
            # Get best action by planning in latent space through open-loop prediction
            # 预测最好的动作
            action = self.plan_action_with_cem(h_state, z_state)
            obs, reward, terminated, truncated, info = self.env.step(action.to('cpu').numpy())
            ep_reward += reward
            # Reconstruct observation
            # 根据确定性隐藏状态和潜在状态对观察进行解码，重构prev_obs
            recon_obs = self.world_model.decoder_model(h_state, z_state).mean
            reconstructed_frames.append(recon_obs.squeeze())
            # Get next latent state 预测下一个确定性隐藏状态
            h_state = self.world_model.rnn_model(h_state, z_state, action.unsqueeze(dim=0))
            prev_obs = copy.deepcopy(obs)
            if terminated or truncated:
                break
        # 记录游戏观察以及重构的游戏观察
        observed_frames = torch.stack(observed_frames).unsqueeze(dim=0) + 0.5
        reconstructed_frames = torch.clip(torch.stack(reconstructed_frames).unsqueeze(dim=0) + 0.5, min=0.0, max=1.0)
        self.logger.add_scalar('Reward/test_episodes', ep_reward, step)
        if step % self.params['eval_gif_freq'] == 0:
            self.logger.add_video(f'ObservedTestEpisode/{step}', observed_frames.transpose(3, 4))
            self.logger.add_video(f'ReconstructedTestEpisode/{step}', reconstructed_frames.transpose(3, 4))
            print('\rLearning progress evaluation complete! Saved the episode!')
        else:
            print('\rLearning progress evaluation is complete!')

    def evaluate_video_prediction(self, step):
        '''
        进行模型预测观察的一次评估，应该是全部都是由模型对未来进行走势进行评估
        '''
        if step % self.params['vp_eval_freq'] == 0:
            print('\rEvaluating video prediction ability ...', end='')
            n_context_frames = 5 # 预测的帧数，这里仅预测5帧
            n_predicted_frames = 50
            self.world_model.eval()
            # 获取初始化观察
            prev_obs, _ = self.env.reset()
            # 初始化确定性隐藏状态
            h_state = self.world_model.get_init_h_state(batch_size=1)

            # observed_frames存储真实观察
            observed_frames, predicted_frames = list(), list()
            observed_frames.append(prev_obs)
            # Feed context
            for _ in range(n_context_frames):
                # 对环境进行编码
                encoded_obs = self.world_model.obs_encoder(prev_obs.unsqueeze(dim=0).to(self.device))
                # 生成后验状态分布
                posterior_z = self.world_model.repr_model(h_state, encoded_obs)
                # 采样后验分布，也就是随机性隐藏状态
                z_state = posterior_z.sample()
                # Get best action by planning in latent space through open-loop prediction
                # 获取预测的未来最好动作
                action = self.plan_action_with_cem(h_state, z_state)
                # 执行动作
                obs, reward, terminated, truncated, info = self.env.step(action.to('cpu').numpy())
                observed_frames.append(obs)
                # Reconstruct observation
                # 对观察进行重构预测，并存储在predicted_frames
                recon_obs = self.world_model.decoder_model(h_state, z_state).mean
                predicted_frames.append(recon_obs.squeeze())
                # 得到下一个的确定性状态
                h_state = self.world_model.rnn_model(h_state, z_state, action.unsqueeze(dim=0))
                prev_obs = copy.deepcopy(obs)

                # Generate prediction 预测未来的5帧的观察变化
                for _ in range(n_predicted_frames):
                    # 获取先验状态分布
                    prior_z = self.world_model.transition_model(h_state)
                    # 获取随机性隐藏状态
                    z_state = prior_z.sample()
                    # 预测未来最好的动作
                    action = self.plan_action_with_cem(h_state, z_state)
                    # 执行动作
                    obs, reward, terminated, truncated, info = self.env.step(action.to('cpu').numpy())
                    # 存储o真实obs
                    observed_frames.append(obs)
                    # Reconstruct observation
                    # 存储预测的obs，这里的预测obs就和真实的obs没有关系了，存粹的预测了
                    recon_obs = self.world_model.decoder_model(h_state, z_state).mean
                    predicted_frames.append(recon_obs.squeeze())
                    # 获取下一个确定性状态
                    h_state = self.world_model.rnn_model(h_state, z_state, action.unsqueeze(dim=0))
            
            # 将观察重构到0～1，。合并obs帧
            observed_frames = 0.5 + torch.stack(observed_frames[:-1]).to(self.device)
            # 将预测的观察重构到0～1，合并obs帧
            predicted_frames = torch.clip(0.5 + torch.stack(predicted_frames), min=0.0, max=1.0)
            # 将真实帧和预测帧对比合并起来
            overlay_frames = torch.clip(0.5*(1 - observed_frames) + 0.5*predicted_frames, min=0.0, max=1.0)

            # 将以上三个观察帧合并起来并记录到tensorboard中
            combined_frame = torch.cat([observed_frames, predicted_frames, overlay_frames], dim=3).unsqueeze(dim=0)
            self.logger.add_video(f'VideoPrediction/after_training_step_{step}', combined_frame.transpose(3, 4))
            print('\rVideo prediction evaluation is complete! Saved the episode!')

    def plan_action_with_cem(self, init_h_state, init_z_state):
        '''
        param init_h_state: 上一个隐藏状态
        param init_z_state: 本次潜在状态，也就是后验分布的潜在状态

        return: 根据不断的迭代得到认为的最好的动作
        '''
        # todo `planning_horizon` 参数指定了在规划过程中，评估候选动作序列时所预见的未来时间步数。在基于 Planet 算法的实现中，这个值控制了模型在每次规划时向前预测多少步，以便选择出最优的动作计划。
        action_dist = Independent(Normal(loc=torch.zeros(self.params['planning_horizon'], self.action_dim), scale=1.0), reinterpreted_batch_ndims=2)
        # todo plan_optimization_iter参数指定了在规划过程中，通过多次迭代优化候选动作序列，以便搜索出最优计划的迭代次数。也就是说，规划器会在10次迭代中不断改进候选计划
        for _ in range(self.params['plan_optimization_iter']):
            reward_buffer = list()
            # n_plans 数指定了在每次优化迭代中将采样多少条候选动作序列（计划）。也就是说，在每次规划过程中，会随机采样 1000 个动作序列，然后通过后续的评估机制选择出表现最佳的候选计划，以便在环境中执行
            # 也就是说随机生成n_plans个状态分布进行选择评估？
            h_state = torch.clone(init_h_state).repeat(self.params['n_plans'], 1)
            z_state = torch.clone(init_z_state).repeat(self.params['n_plans'], 1)
            # 生成候选动作，shape = todo
            candidate_plan = torch.clip_(
                action_dist.sample(sample_shape=torch.Size([self.params['n_plans']])).to(self.d_type).to(self.device),
                min=self.params['min_action'], max=self.params['max_action'])
            # 对每一个时间步的动作进行评估，这个时间步包含对未来观察的预测吧
            for time_step in range(self.params['planning_horizon']):
                batched_ts_action = candidate_plan[:, time_step, :]
                # Use learnt dynamics to get next hidden state 根据动作得到下一次的确定性隐藏状态
                h_state = self.world_model.rnn_model(h_state, z_state, batched_ts_action)
                # Get latent variables from transition model (prior) 得到下一个时间先验状态
                prior_z = self.world_model.transition_model(h_state)
                z_state = prior_z.sample() # 先验状态模拟后验状态
                # 利用确定性状态+模拟本地后验状态预测奖励分布
                predicted_reward = self.world_model.reward_model(h_state, z_state)
                # 对奖励进行裁剪，因为可能存在跳帧，则认为跳帧的每次动作的到的奖励都一样，那么跳帧期间的奖励=单帧的奖励*次数
                sampled_reward = torch.clip(predicted_reward.mean,
                                            min=self.params['min_reward'], max=(1+self.params['action_repeat'])*self.params['max_reward'])
                # 保存奖励缓冲区
                reward_buffer.append(sampled_reward)
            # 展平所有时间步的奖励
            plan_reward = torch.stack(reward_buffer).squeeze().sum(dim=0)
            # 根据奖励的大小，选择topK个奖励最高的动作分布，得到新的动作分布和均值
            # 再创建新的动作分布，重新进行动作采样重新进行评估
            chosen_actions = candidate_plan[torch.topk(plan_reward, k=self.params['top_k']).indices]
            action_mu, action_std = chosen_actions.mean(dim=0), chosen_actions.std(dim=0)
            action_dist = Independent(Normal(loc=action_mu, scale=action_std+1e-6), reinterpreted_batch_ndims=2)
        optimized_next_action = action_dist.mean[0]
        return optimized_next_action


class ReplayBuffer:
    '''
    
    应该是重放缓冲区
    '''

    def __init__(self, params):
        self.params = params
        self.d_type = get_dtype(self.params['fp_precision'])
        self.device = get_device(self.params['device'])
        self.memory = list() # 用list存储，这里存储的维度是一个游戏周期所有的步数

    def __len__(self):
        return len(self.memory)

    def push(self, episode):
        if len(episode) >= self.params['chunk_length']:
            self.memory.append(episode)

    def sample(self, n):
        '''
        这里传入的batch_size，采样batch_size个样本数据
        '''
        # 随机选择batch_size个episode
        sampled_indices = np.random.choice(len(self.memory), n, replace=True)
        # chunked_episodes看起来也是一个连续的片段
        chunked_episodes = list()
        for ep_idx in sampled_indices:
            # 随机选择一个起始位置
            start_idx = np.random.randint(low=0, high=len(self.memory[ep_idx])-self.params['chunk_length'])
            # 将选择连续的chunk_length个数据添加到chunked_episodes中
            chunked_episodes.append(self.memory[ep_idx][start_idx:start_idx+self.params['chunk_length']])
        # 此时chunked_episodes是一个list，里面存储的是每个episode的连续片段
        serialized_episodes = self.serialize_episode(chunked_episodes)
        return serialized_episodes

    def serialize_episode(self, list_episodes):
        '''
        return 返回一个字典，里面存储的是每个episode的observation、action、reward
        '''

        batched_ep_obs, batched_ep_action, batched_ep_reward = list(), list(), list()
        # 拆分每个episode的observation、action、reward到独立的缓存list中，list的每个元素是一个连续的obs、action、erward tensor
        for episode in list_episodes:
            ep_obs = torch.stack([transition.observation for transition in episode])
            ep_action = torch.stack([transition.action for transition in episode])
            ep_reward = torch.stack([transition.reward for transition in episode])
            batched_ep_obs.append(ep_obs)
            batched_ep_action.append(ep_action)
            batched_ep_reward.append(ep_reward)
        # 最后将所有的list转换为tensor
        # batched_ep_obs shape 是（batch_size， chunk_length， visual_resolution， visual_resolution， channels）
        # batched_ep_action shape 是（batch_size， chunk_length， action_dim）
        # batched_ep_reward shape 是（batch_size， chunk_length， 1）
        batched_ep_obs = torch.stack(batched_ep_obs).to(self.d_type).to(self.device)
        batched_ep_action = torch.stack(batched_ep_action).to(self.d_type).to(self.device)
        batched_ep_reward = torch.stack(batched_ep_reward).to(self.d_type).to(self.device)
        # batched_ep_obs shape 是（chunk_length， batch_size， visual_resolution， visual_resolution， channels）
        # batched_ep_action shape 是（chunk_length， batch_size， action_dim）
        # batched_ep_reward shape 是（chunk_length， batch_size， 1）
        # 这里的转换是为了将时间步放在前面，方便后续的训练
        return {'obs': batched_ep_obs.transpose(0, 1), 'action': batched_ep_action.transpose(0, 1), 'reward': batched_ep_reward.transpose(0, 1)}
