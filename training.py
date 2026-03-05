# train.py
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from datasets import sample_distribution
from sampling import euler_sampling
from toy_network import ToyModel

device = "cuda" if torch.cuda.is_available() else "cpu"

# ======================
# CONFIG
# ======================
dist_A = "moons"
dist_B = "circles"
batch_size = 512
num_steps = 20000
lr = 2e-5   # <-- LR più stabile
val_interval = 500

# ======================
# MODEL
# ======================
model = ToyModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

writer = SummaryWriter("runs/rectified_flow_2d")

best_val_loss = float("inf")

os.makedirs("checkpoints", exist_ok=True)

# ======================
# LOSS FUNCTION
# ======================
def compute_loss():

    x0 = sample_distribution(dist_A, batch_size).to(device)
    x1 = sample_distribution(dist_B, batch_size).to(device)

    # ===== NORMALIZZAZIONE STABILE =====
    mean = torch.cat([x0, x1], dim=0).mean(0, keepdim=True)
    std = torch.cat([x0, x1], dim=0).std(0, keepdim=True) + 1e-6

    x0 = (x0 - mean) / std
    x1 = (x1 - mean) / std
    # ===================================

    t = torch.rand(batch_size, device=device)
    t = t * 0.8 + 0.1  # t ∈ [0.1, 0.9] -> meno varianza
    xt = (1 - t).unsqueeze(1) * x0 + t.unsqueeze(1) * x1
    # Calcolo target velocity
    target_velocity = x1 - x0
    # Normalizzazione stabile
    target_velocity = target_velocity / (target_velocity.std(dim=0, keepdim=True) + 1e-6)

    pred_velocity = model(xt, t)

    loss = ((pred_velocity - target_velocity) ** 2).mean()

    return loss


# ======================
# TRAIN LOOP
# ======================
for step in range(num_steps):

    model.train()
    loss = compute_loss()

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # <-- stabilità
    optimizer.step()

    writer.add_scalar("Loss/train", loss.item(), step)

    if step % val_interval == 0:

        model.eval()
        with torch.no_grad():
            val_loss = compute_loss()

        writer.add_scalar("Loss/val", val_loss.item(), step)

        print(f"Step {step} | Train {loss.item():.6f} | Val {val_loss.item():.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = f"checkpoints/best_step_{step}_valloss_{val_loss:.4f}.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "step": step,
                "val_loss": val_loss
            }, save_path)
            print(f"Saved best model at step {step}")

writer.close()