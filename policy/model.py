import copy, pickle
from typing import Dict, List
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from math import exp

from torch.utils.data import DataLoader

from utils.batch import stack_batch_multi, unstack_batch_multi, stack_batch, DictLoader

from .transformer import Encoder

INFINITY = 1e9

from utils.time import time_count


class MAPPO(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self._config = config.copy()

        self._state_dim = int(
            config["resolution"] * config["resolution"] * 4
            + self._config["sample_rate"] * self._config["step_time"] * 2
            + 1
        )

        self._vision_dim = int(config["resolution"] * config["resolution"] * 4)
        self._audio_dim = int(
            self._config["sample_rate"] * self._config["step_time"] * 2
        )

        self._action_dim = 4
        self._hid_dim_p = self._config["hid_dim_p"]
        self._hid_dim_v = self._config["hid_dim_v"]
        self._hid_dim_l = self._config["hid_dim_l"]

        self._build_model()
        self._init_model()

        self._eps = 1e-8
        self._step = 0
        self._optim = torch.optim.Adam(
            self.parameters(), lr=self._config["lr"], betas=(0.9, config["adam_beta2"])
        )

    def _build_model(self):
        # Networks
        # 这两个用foundation model的学习网络，vision使用sam
        # audio使用vggish
        self._vision_net = nn.Sequential(
            nn.Linear(self._vision_dim, self._hid_dim_l * 3),
            nn.ReLU(),
            nn.Linear(self._hid_dim_l * 3, self._hid_dim_l),
        ).cuda()
        self._audio_net = nn.Sequential(
            nn.Linear(self._audio_dim + 1, self._hid_dim_l * 2),
            nn.ReLU(),
            nn.Linear(self._hid_dim_l * 2, self._hid_dim_l),
        ).cuda()
        
        
        self._cross_attn1 = Encoder(self._hid_dim_l, self._hid_dim_l, 64, 4).cuda()
        self._cross_attn2 = Encoder(self._hid_dim_l, self._hid_dim_l, 64, 4).cuda()

        self._policy_net = nn.Sequential(
            nn.Linear(self._hid_dim_l, self._hid_dim_p),
            nn.ReLU(),
            nn.Linear(self._hid_dim_p, self._action_dim),
        ).cuda()

        self._value_net = nn.Sequential(
            nn.Linear(self._hid_dim_l, self._hid_dim_v),
            nn.ReLU(),
            nn.Linear(self._hid_dim_v, 1),
        ).cuda()

        self._lstm = nn.LSTM(self._hid_dim_l, self._hid_dim_l, batch_first=True).cuda()

    def _init_model(self):
        """Initialize model"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.constant_(m.bias.data, 0.0)

    @time_count
    def forward(self, input_d):
        """Build model that is capable to do forward process"""
        output_d = dict()

        N, T = input_d["audio"].shape[0], input_d["audio"].shape[1]

        input_d["camera"] = input_d["camera"].reshape(N, T, -1)
        input_d["audio"] = input_d["audio"].reshape(N, T, -1)
        input_d["step"] = input_d["step"].reshape(N, T, -1)

        ht = input_d["lstm_h"][:, 0, :].reshape(1, N, -1).detach().contiguous()
        ct = input_d["lstm_c"][:, 0, :].reshape(1, N, -1).detach().contiguous()
        v_s = input_d["camera"]
        a_s = torch.cat((input_d["audio"], input_d["step"]), dim=-1)

        v_s = self._vision_net(v_s)
        a_s = self._audio_net(a_s)

        x, _ = self._cross_attn1(v_s, a_s)
        x, _ = self._cross_attn2(x, x)

        lstm_out, (hn, cn) = self._lstm(x, (ht, ct))

        pi_logits = self._policy_net(lstm_out)
        v_value = self._value_net(lstm_out)

        output_d["lstm_h"] = hn.permute(1, 0, 2)
        output_d["lstm_c"] = cn.permute(1, 0, 2)

        output_d["rl_pred"] = torch.distributions.Categorical(logits=pi_logits).sample()
        output_d["rl_logits"] = pi_logits
        output_d["rl_value"] = v_value

        return output_d

    def build_loss(self, output_d, input_d):
        """Build PPO loss function and V-value loss function"""
        pi_logits = output_d["rl_logits"]  # (B, T+1, C)
        v_value = output_d["rl_value"].squeeze(-1)  # (B, T+1)

        mu_logits = input_d["rl_logits"]  # (B, T+1, C)
        action = input_d["rl_pred"].squeeze(-1)  # (B, T+1)
        mask = input_d["mask"]  # (B, T+1)
        old_v_value = input_d["rl_value"]  # (B, T+1)
        reward = input_d["reward"]  # (B, T+1)
        gamma = self._config["gamma"]

        """
        policy gradient
        """
        pi_dist = torch.distributions.Categorical(logits=pi_logits)
        mu_dist = torch.distributions.Categorical(logits=mu_logits)

        log_pi = pi_dist.log_prob(action)  # (B, T+1)
        log_mu = mu_dist.log_prob(action)  # (B, T+1)

        # (B, T)
        gae_advantage, gae_value = self._gae(
            reward, v_value, mask, gamma, self._config["gae_lambda"]
        )
        gae_adv_mean, gae_adv_std = gae_advantage.mean(), gae_advantage.std()
        if self._config["advantage_normalization"]:
            gae_advantage = (gae_advantage - gae_adv_mean) / (gae_adv_std + self._eps)

        ratio = (log_pi - log_mu).exp()[:, :-1] * mask[:, :-1]  # remove bootstrap
        surr1 = ratio * gae_advantage
        surr2 = (
            ratio.clamp(1.0 - self._config["ppo_eps"], 1.0 + self._config["ppo_eps"])
            * gae_advantage
        )

        pi_loss = -torch.min(surr1, surr2).mean()

        """
        value evaluation
        """
        if self._config["vf_clip"]:
            v_clip = old_v_value + (v_value - old_v_value).clamp(
                -self._config["vf_clip"], self._config["vf_clip"]
            )
            vf1 = (gae_value - v_value).pow(2)[:, :-1] * mask[:, :-1]
            vf2 = (gae_value - v_clip).pow(2)[:, :-1] * mask[:, :-1]
            vf_loss = torch.max(vf1, vf2).mean()
        else:
            vf1 = (gae_value - v_value).pow(2)[:, :-1] * mask[:, :-1]
            vf_loss = vf1.mean()

        explained_variance = self.explained_variance(v_value, gae_value)

        """
        entropy loss
        """
        ent = pi_dist.entropy()[:, :-1] * mask[:, :-1]
        ent_loss = ent.mean()

        pi_coef = self._config["pi_coef"]
        vf_coef = self._config["vf_coef"]
        # let ent_coef decay, uncertain max step
        ent_coef = self._config["ent_coef"] * (0.8 ** (self._step // 1000))
        # ent_coef = self._config["ent_coef"]

        total_loss = pi_coef * pi_loss + vf_coef * vf_loss - ent_coef * ent_loss
        loss_dict = {
            "pi_loss": pi_loss,
            "pi_coef": pi_coef,
            "vf_loss": vf_loss,
            "vf_coef": vf_coef,
            "ent_loss": ent_loss,
            "ent_coef": ent_coef,
            "explained_variance": explained_variance,
        }
        return total_loss, loss_dict
    def build_iql_loss(self, output_d, input_d):
        # 输出
        pi_logits = output_d["rl_logits"]  # (B, T+1, C)
        v_value = output_d["rl_value"].squeeze(-1)  # (B, T+1)

        # 输入
        action = input_d["rl_pred"].squeeze(-1)  # (B, T+1)
        mask = input_d["mask"]  # (B, T+1)
        reward = input_d["reward"]  # (B, T+1)
        old_v_value = input_d["rl_value"]  # (B, T+1)
        gamma = self._config["gamma"]
        alpha = self._config["alpha"]

        """
        1. 值函数损失计算
        """
        # TD目标值 (B, T)
        with torch.no_grad():
            # 不包括最后一时间步（bootstrap 时不要考虑 T+1 的值）
            td_target = reward + gamma * v_value[:, 1:] * mask[:, 1:]
            td_target = td_target.detach()

        # 值函数损失
        value_loss = nn.MSELoss()(v_value[:, :-1] * mask[:, :-1], td_target)

        """
        2. 策略损失计算
        """
        # 策略分布
        pi_dist = torch.distributions.Categorical(logits=pi_logits)
        log_pi = pi_dist.log_prob(action)  # (B, T+1)

        # 优势计算
        with torch.no_grad():
            advantage = v_value[:, :-1] - td_target  # (B, T)
            weights = torch.exp(alpha * advantage).clamp(max=10.0)  # 限制权重范围

        # 策略损失
        policy_loss = -(weights * log_pi[:, :-1] * mask[:, :-1]).mean()

        """
        3. 熵正则项
        """
        entropy_loss = pi_dist.entropy()[:, :-1] * mask[:, :-1]
        entropy_loss = entropy_loss.mean()

        """
        总损失
        """
        vf_coef = self._config["vf_coef"]
        ent_coef = self._config["ent_coef"]
        total_loss = value_loss + policy_loss - ent_coef * entropy_loss

        loss_dict = {
            "value_loss": value_loss,
            "policy_loss": policy_loss,
            "entropy_loss": entropy_loss,
            "weights_mean": weights.mean(),
        }

        return total_loss, loss_dict


    @staticmethod
    def explained_variance(u, v):
        return (((u - u.mean()) / u.std() * (v - v.mean()) / v.std()).mean()) ** 2

    @staticmethod
    def _gae(reward, value, mask, gamma, lam):
        value = value * mask
        delta = reward[:, :-1] + gamma * value[:, 1:] - value[:, :-1]
        gae_advantage = torch.zeros_like(value)
        gae_value = torch.zeros_like(value)
        for t in reversed(range(value.shape[1] - 1)):
            gae_advantage[:, t] = delta[:, t] + gamma * lam * gae_advantage[:, t + 1]
            gae_value[:, t] = value[:, t] + gae_advantage[:, t]
        return gae_advantage[:, :-1].detach(), gae_value.detach()

    @time_count
    def learn(self, input_d: Dict):
        """
        Train the model with data
        data: Dict, key(str) -> value(array).
        Each value has shape (B, T, *),
            B is batch-size,
            T is sequence length,
            * is the (arbitrary) shape of the key
        """
        self._step += 1

        for k, v in input_d.items():
            if type(v) == np.ndarray:
                input_d[k] = torch.from_numpy(v).cuda()
            else:
                input_d[k] = v.cuda()

        """
        forward process
        """
        output_d = self.forward(input_d)

        """
        calculating loss
        """
        total_loss, loss_dict = self.build_loss(output_d, input_d)

        """
        back propagation
        """
        self._optim.zero_grad()
        total_loss.backward()
        assert self._config["max_grad_norm"] > 0  # clip large gradient
        grad_norm = nn.utils.clip_grad_norm_(
            self.parameters(), max_norm=self._config["max_grad_norm"]
        )
        # learning rate schedule
        rate = self._get_rate(self._step, self._config["lr"], self._config["warmup"])
        for p in self._optim.param_groups:
            p["lr"] = rate
        self._optim.step()

        return {
            "total_loss": total_loss.data.cpu().numpy(),
            "vf_loss": loss_dict["vf_loss"].data.cpu().numpy(),
            "pi_loss": loss_dict["pi_loss"].data.cpu().numpy(),
            "explained_variance": loss_dict["explained_variance"].data.cpu().numpy(),
            "ent_loss": loss_dict["ent_loss"].data.cpu().numpy(),
        }

    @staticmethod
    def _get_rate(step, lr, warmup):
        lr_ = lr
        if warmup is not None and warmup > 0:
            if step < warmup:
                lr_ = lr * exp(-(step - warmup) * 0.0001)
        return lr_

    def inf(self, input_d: list[Dict]) -> Dict:
        """
        Forward process of model, input list of dict, output list of dict
        input: {key: (B, *)} -> {key: (B, 1, *)}
        output: input -> {key: (B, 1, *)} -> {key: (B, *)}
        """
        input_d = {
            k: torch.from_numpy(v[:, None]).type(torch.float32).cuda()
            for k, v in input_d.items()
        }  # add seq-length dimension
        output_d = self.forward(input_d)
        output_d = {
            k: v[:, 0].detach().cpu().numpy() for k, v in output_d.items()
        }  # remove seq-length dimension
        return output_d

    def save(self, path):
        """save model"""
        torch.save(self.state_dict(), path)

    def serializing(self):
        state_dict = self.state_dict()
        s = pickle.dumps(state_dict)
        return s

    def deserializing(self, s):
        state_dict = pickle.loads(s)
        self.load_state_dict(state_dict)


class BaseModel:
    def __init__(self, config: Dict):
        self._config = config
        self._model = MAPPO(copy.deepcopy(config))

    def _build_model(self):
        self._model._build_model()

    def _build_loss(self):
        self._model._build_loss()

    def _init_model(self):
        self._model._init_model()

    def learn(self, data: Dict):
        """BE CAREFUL !!!"""
        loader = DictLoader(data, self._config["batch_size"])
        d_list = list()
        bs_list = list()
        for data, bs in loader:
            d = self._model.learn(data)
            d_list.append(d)
            bs_list.append(bs)

        res_d = dict()
        for k in d_list[0].keys():
            res_d[k] = sum(
                [d_list[i][k] * bs for i in range(len(bs_list)) for bs in bs_list]
            ) / sum(bs_list)

        return res_d

    def save(self, path):
        """Save model"""
        self._model.save(path)

    def load_model(self, path):
        """Load model"""
        self._model.load_state_dict(torch.load(path))

    def inf(self, input_d_list: List) -> List:
        """BE CAREFUL !!!"""
        n_agents = len(input_d_list[0])

        input_d = stack_batch_multi(input_d_list)  # batch; form a dict

        rl_output = self._model.inf(input_d)

        output_list = unstack_batch_multi(rl_output, n_agents)  # split batch

        return output_list

    def load_model(self, s):
        self._model.deserializing(s)

    def dump_model(self):
        return self._model.serializing()
