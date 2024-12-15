import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# 数据集类
class OfflineDataset(Dataset):
    def __init__(self, data):
        """
        data: List of tuples (state, action, reward, next_state, done)
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Q网络和V网络
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)

# 核心 IQL 训练逻辑
class IQL:
    def __init__(self, state_dim, action_dim, tau=0.7, gamma=0.99, lr=3e-4):
        self.tau = tau
        self.gamma = gamma

        # 初始化 Q 网络和 V 网络
        self.q_net = MLP(state_dim + action_dim, 1)
        self.v_net = MLP(state_dim, 1)

        # 优化器
        self.q_optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.v_optimizer = optim.Adam(self.v_net.parameters(), lr=lr)

    def train(self, dataloader, epochs=50):
        for epoch in range(epochs):
            epoch_loss_q, epoch_loss_v = 0.0, 0.0
            for state, action, reward, next_state, done in dataloader:
                # 数据转换为张量
                state = state.float()
                action = action.float()
                reward = reward.float().unsqueeze(1)
                next_state = next_state.float()
                done = done.float().unsqueeze(1)

                # --- 更新 V 网络 ---
                with torch.no_grad():
                    q_values = self.q_net(torch.cat([state, action], dim=1))
                v_values = self.v_net(state)
                v_loss = ((q_values - v_values).clamp(min=0) ** 2).mean()
                self.v_optimizer.zero_grad()
                v_loss.backward()
                self.v_optimizer.step()

                # --- 更新 Q 网络 ---
                with torch.no_grad():
                    next_v_values = self.v_net(next_state)
                    target_q = reward + self.gamma * next_v_values * (1 - done)
                q_values = self.q_net(torch.cat([state, action], dim=1))
                q_loss = ((q_values - target_q) ** 2).mean()
                self.q_optimizer.zero_grad()
                q_loss.backward()
                self.q_optimizer.step()

                # 累计损失
                epoch_loss_q += q_loss.item()
                epoch_loss_v += v_loss.item()

            # 打印损失
            print(f"Epoch {epoch+1}/{epochs} | Q Loss: {epoch_loss_q:.4f} | V Loss: {epoch_loss_v:.4f}")

# 生成离线数据示例
def generate_synthetic_data(num_samples=1000, state_dim=4, action_dim=2):
    data = []
    for _ in range(num_samples):
        state = torch.rand(state_dim)
        action = torch.rand(action_dim)
        reward = torch.rand(1).item()
        next_state = torch.rand(state_dim)
        done = torch.randint(0, 2, (1,)).item()
        data.append((state, action, reward, next_state, done))
    return data

# 主函数
if __name__ == "__main__":
    # 参数
    state_dim = 4
    action_dim = 2
    batch_size = 64
    num_samples = 1000

    # 数据集和数据加载器
    data = generate_synthetic_data(num_samples, state_dim, action_dim)
    dataset = OfflineDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 初始化 IQL
    iql = IQL(state_dim, action_dim)

    # 训练
    iql.train(dataloader, epochs=50)
