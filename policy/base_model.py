import copy, pickle
from typing import Dict, List
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from math import exp

from torch.utils.data import DataLoader

INFINITY = 1e9


class MAPPO(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self._config = config.copy()

        self._state_dim = config["resolution"] * config["resolution"] * 3 
        self._action_dim = 3
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
        self._pi_net_fc1 = nn.Linear(self._hid_dim_l, self._hid_dim_p).cuda()
        self._pi_net_fc2 = nn.Linear(self._hid_dim_p, self._action_dim).cuda()

        self._v_net_fc1 = nn.Linear(self._hid_dim_l, self._hid_dim_v).cuda()
        self._v_net_fc2 = nn.Linear(self._hid_dim_v, 1).cuda()

        self._lstm = nn.LSTM(self._state_dim, self._hid_dim_l, batch_first=True).cuda()

    def _init_model(self):
        """Initialize model"""
        nn.init.trunc_normal_(self._pi_net_fc1.weight, std=0.02)
        nn.init.trunc_normal_(self._pi_net_fc2.weight, std=0.02)
        nn.init.trunc_normal_(self._v_net_fc1.weight, std=0.02)
        nn.init.trunc_normal_(self._v_net_fc2.weight, std=0.02)
        nn.init.constant_(self._pi_net_fc1.bias, 0.0)
        nn.init.constant_(self._pi_net_fc2.bias, 0.0)
        nn.init.constant_(self._v_net_fc1.bias, 0.0)
        nn.init.constant_(self._v_net_fc2.bias, 0.0)

        nn.init.trunc_normal_(self._lstm.weight_ih_l0, std=0.02)
        nn.init.trunc_normal_(self._lstm.weight_hh_l0, std=0.02)
        nn.init.constant_(self._lstm.bias_ih_l0, 0.0)
        nn.init.constant_(self._lstm.bias_hh_l0, 0.0)

    def forward(self, input_d):
        """Build model that is capable to do forward process"""
        output_d = dict()

        # input_d = {
        #     "camera": np.array([o["camera"] for o in obs]),
        #     "audio": np.array([o["audio"] for o in obs]),
        #     "mask": np.array(mask),
        #     "lstm_h": a["lstm_h"], # (num_agents, hid_dim_l)
        #     "lstm_c": a["lstm_c"], # (num_agents, hid_dim_l)
        # }

        (N, T, A) = input_d["camera"].shape[0], input_d["camera"].shape[1], input_d["camera"].shape[2]

        input_d["camera"] = input_d["camera"].reshape(N, T, -1)
        input_d["audio"] = input_d["audio"].reshape(N, T, -1)
        input_d["mask"] = input_d["mask"].reshape(N, T, -1)

        # TODO check sanity
        ht = input_d["lstm_h"][:, 0, :].reshape(1, N, -1).detach().contiguous()
        ct = input_d["lstm_c"][:, 0, :].reshape(1, N, -1).detach().contiguous()

        s = torch.cat((input_d["camera"], input_d["audio"]), dim=3)

        lstm_out, (hn, cn) = self._lstm(s, (ht, ct))
        pi_logits = self._pi_net_fc2(F.relu(self._pi_net_fc1(lstm_out)))
        # Mask out invalid actions
        pi_logits -= INFINITY * input_d["mask"]

        v_value = self._v_net_fc2(F.relu(self._v_net_fc1(lstm_out)))

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
        gamma = input_d["gamma"]   # (B, T+1)

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

    @staticmethod
    def explained_variance(u, v):
        return (((u - u.mean()) / u.std() * (v - v.mean()) / v.std()).mean()) ** 2

    @staticmethod
    def _gae(reward, value, mask, gamma, lam):
        value = value * mask
        delta = reward[:, :-1] + gamma[:, :-1] * value[:, 1:] - value[:, :-1]
        gae_advantage = torch.zeros_like(value)
        gae_value = torch.zeros_like(value)
        for t in reversed(range(value.shape[1] - 1)):
            gae_advantage[:, t] = delta[:, t] + gamma[:, :-1] * lam * gae_advantage[:, t + 1]
            gae_value[:, t] = value[:, t] + gae_advantage[:, t]
        return gae_advantage[:, :-1].detach(), gae_value.detach()

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


    def inf(self, input_d: Dict) -> Dict:
        """
        Forward process of model, input dict, output dict
        input: {key: (B, *)} -> {key: (B, 1, *)}
        output: input -> {key: (B, 1, *)} -> {key: (B, *)}
        """

        input_d = {
            k: torch.from_numpy(v[:, None]).cuda() for k, v in input_d.items()
        }  # add seq-length dimension
        output_d = self.forward(input_d)
        output_d = {
            k: v[:, 0].detach().cpu().numpy() for k, v in output_d.items()
        }  # remove seq-length dimension
        return output_d

    def save(self, path):
        """save model"""
        torch.save(self.state_dict(), path)
