import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from collections import defaultdict
from torch.func import functional_call
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

# -------------------------
# Load Van der Pol Meta-Dataset
# -------------------------
csv_path = os.path.join(os.path.dirname(__file__), '../dataset/vdp_meta_nominal_residual.csv')
df = pd.read_csv(csv_path)

tasks = defaultdict(list)
for _, row in df.iterrows():
    tasks[row['mu']].append((row['x1'], row['x2'], row['residual']))

task_data = {}
for mu, samples in tasks.items():
    data = torch.tensor(samples, dtype=torch.float32)
    inputs = data[:, :2]
    targets = data[:, 2].unsqueeze(1)
    task_data[mu] = (inputs, targets)

# -------------------------
# MLP Model Definition
# -------------------------
class MLP(nn.Module):
    def __init__(self, input_dim=2, output_dim=1, hidden_dim=256, num_layers=3):
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

    # Hyperparameters
    meta_lr = 1e-3
    inner_lr = 1e-2
    meta_batch_size = 10
    inner_steps = 1
    epochs = 20000
    K = 50

    input_dim = 2
    output_dim = 1
    hidden_dim = 64
    num_layers = 2

    model = MLP(input_dim, output_dim, hidden_dim, num_layers).to(device)
    meta_optimizer = torch.optim.Adam(model.parameters(), lr=meta_lr)
    train_losses = []

    for epoch in range(epochs):
        meta_optimizer.zero_grad()
        meta_loss = 0.0

        task_mus = np.random.choice(list(task_data.keys()), meta_batch_size, replace=False)

        for mu in task_mus:
            x, y = task_data[mu]
            perm = torch.randperm(x.size(0))
            x, y = x[perm], y[perm]

            x_support, y_support = x[:K].to(device), y[:K].to(device)
            x_query, y_query = x[K:K+K].to(device), y[K:K+K].to(device)

            # Clone model parameters
            adapted_params = {name: param for name, param in model.named_parameters()}

            # Inner loop adaptation
            for _ in range(inner_steps):
                support_pred = functional_call(model, adapted_params, (x_support,))
                loss = F.mse_loss(support_pred, y_support)
                grads = torch.autograd.grad(loss, adapted_params.values(), create_graph=True)
                adapted_params = {
                    name: param - inner_lr * grad
                    for (name, param), grad in zip(adapted_params.items(), grads)
                }

            # Outer loop (meta-update) loss
            query_pred = functional_call(model, adapted_params, (x_query,))
            task_loss = F.mse_loss(query_pred, y_query)
            meta_loss += task_loss

        meta_loss = meta_loss / meta_batch_size
        meta_loss.backward()
        meta_optimizer.step()

        train_losses.append(meta_loss.item())

        if epoch % 1000 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch + 1}/{epochs}, Meta Loss: {meta_loss.item():.6f}")

    # -------------------------
    # Save Results
    # -------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, f"maml_vdp_meta_init_{num_layers}_{hidden_dim}.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': input_dim,
        'output_dim': output_dim,
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,
    }, save_path)
    print(f"Model saved to {save_path}")

    # -------------------------
    # Plot Training Loss
    # -------------------------
    plt.plot(train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Meta Loss (MSE)")
    plt.title("MAML Meta-Training Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"MetaLearning/maml_meta_loss_{num_layers}_{hidden_dim}.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
