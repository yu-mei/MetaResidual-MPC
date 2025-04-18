import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from collections import defaultdict
from torch.func import functional_call
import matplotlib.pyplot as plt
import random

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# -------------------------
# Load Meta-Dataset
# -------------------------
csv_path = os.path.join(os.path.dirname(__file__), '../meta_dataset_mpc/cartpole_meta_residual_mpc.csv')
df = pd.read_csv(csv_path)

# -------------------------
# Group data by task_id
# -------------------------
tasks = defaultdict(list)
for _, row in df.iterrows():
    tasks[row['task_id']].append((
        row['x'], row['x_dot'], row['theta'], row['theta_dot'], row['u'],
        row['res_x_ddot'], row['res_theta_ddot']
    ))

# Convert to tensors
task_data = {}
for tid, samples in tasks.items():
    data = torch.tensor(samples, dtype=torch.float32)
    inputs = data[:, :5]  # x, x_dot, theta, theta_dot, u
    targets = data[:, 5:]  # res_x_ddot, res_theta_ddot
    task_data[tid] = (inputs, targets)

print(f"✅ Loaded data for {len(task_data)} different tasks (task_id)")

# -------------------------
# MLP Model Definition
# -------------------------
class MLP(nn.Module):
    def __init__(self, input_dim=5, output_dim=2, hidden_dim=64, num_layers=2):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# -------------------------
# MAML Training Loop
# -------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    meta_lr = 1e-4
    inner_lr = 1e-3
    meta_batch_size = len(task_data)
    inner_steps = 1
    epochs = 10000
    K = 20  # Support/query set size

    input_dim = 5
    output_dim = 2
    hidden_dim = 64
    num_layers = 3

    model = MLP(input_dim, output_dim, hidden_dim, num_layers).to(device)
    meta_optimizer = torch.optim.Adam(model.parameters(), lr=meta_lr)
    train_losses = []

    print(f"🔧 Meta-learning config: {hidden_dim} hidden units, {num_layers} layers")
    print(f"🧠 Tasks: {len(task_data)}, Meta batch size: {meta_batch_size}")

    for epoch in range(epochs):
        meta_optimizer.zero_grad()
        meta_loss = 0.0
        n_tasks_used = 0

        task_ids = np.random.choice(list(task_data.keys()), meta_batch_size, replace=False)

        for tid in task_ids:
            x, y = task_data[tid]

            if len(x) < 2 * K:
                continue

            perm = torch.randperm(x.size(0))
            x, y = x[perm], y[perm]

            x_support, y_support = x[:K].to(device), y[:K].to(device)
            x_query, y_query = x[K:K+K].to(device), y[K:K+K].to(device)

            adapted_params = {name: param.clone() for name, param in model.named_parameters()}

            for _ in range(inner_steps):
                support_pred = functional_call(model, adapted_params, (x_support,))
                loss = F.mse_loss(support_pred, y_support)
                grads = torch.autograd.grad(loss, adapted_params.values(), create_graph=True)
                adapted_params = {
                    name: param - inner_lr * grad
                    for (name, param), grad in zip(adapted_params.items(), grads)
                }

            query_pred = functional_call(model, adapted_params, (x_query,))
            task_loss = F.mse_loss(query_pred, y_query)
            meta_loss += task_loss
            n_tasks_used += 1

        if n_tasks_used == 0:
            continue

        meta_loss = meta_loss / n_tasks_used
        meta_loss.backward()
        meta_optimizer.step()
        train_losses.append(meta_loss.item())

        if epoch % 1000 == 0 or epoch == epochs - 1:
            print(f"[Epoch {epoch+1:05d}] Meta Loss: {meta_loss.item():.6f}")

    # -------------------------
    # Save Model
    # -------------------------
    save_dir = os.path.dirname(__file__)
    save_path = os.path.join(save_dir, f"maml_cartpole_meta_init_{num_layers}_{hidden_dim}.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': input_dim,
        'output_dim': output_dim,
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,
    }, save_path)
    print(f"✅ Model saved to {save_path}")

    # -------------------------
    # Plot Training Loss
    # -------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Meta Loss (MSE)")
    plt.title("MAML Meta-Training Loss (CartPole Residuals)")
    plt.grid(True)
    plt.yscale("log")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
