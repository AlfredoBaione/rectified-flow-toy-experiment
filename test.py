# sample -> statistical snapshot + final trajectories + interactive slider on TensorBoard
import torch
import matplotlib.pyplot as plt
from io import BytesIO
from torch.utils.tensorboard import SummaryWriter
from datasets import sample_distribution
from toy_network import ToyModel
from matplotlib.widgets import Slider

device = "cuda" if torch.cuda.is_available() else "cpu"

# ======================
# CONFIG
# ======================
dist_A = "moons"
dist_B = "circles"
n_points = 200
steps = 50
writer = SummaryWriter("runs/rectified_flow_2d_snapshots_and_trajectories")

# ======================
# FUNCTION TO CONVERT FIGURE TO IMAGE
# ======================
def plot_to_image(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    import torchvision
    import PIL.Image as Image
    image = Image.open(buf)
    image = torchvision.transforms.ToTensor()(image)
    buf.close()
    return image

# ======================
# LOADING THE TRAINED MODEL
# ======================
model = ToyModel().to(device)
checkpoint = torch.load("checkpoints/best_step_17000_valloss_0.5399.pt", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# ======================
# INITIAL AND TARGET SAMPLE
# ======================
xA = sample_distribution(dist_A, n_points).to(device)
xB = sample_distribution(dist_B, n_points).to(device)

mean = torch.cat([xA, xB], dim=0).mean(0, keepdim=True)
std = torch.cat([xA, xB], dim=0).std(0, keepdim=True) + 1e-6
xA = (xA - mean) / std
xB = (xB - mean) / std

x = xA.clone()
dt = 1.0 / steps
trajectories = [x.clone().cpu()]

# ======================
# SAMPLING STEP-BY-STEP
# ======================
with torch.no_grad():
    for i in range(steps):
        t = torch.ones(n_points, device=device) * ((i + 0.5) / steps)
        v = model(x, t)
        x = x + v * dt
        trajectories.append(x.clone().cpu())

# ======================
# STATISTICAL SNAPSHOT  STEP 10,20,30,40,50 (unchanged)
# ======================
snapshot_steps = [9,19,29,39,49]
for idx in snapshot_steps:
    fig, ax = plt.subplots(figsize=(6,6))
    xt = trajectories[idx]
    ax.scatter(xB[:,0], xB[:,1], s=40, color='green', alpha=0.8, label="Target")
    ax.scatter(xt[:,0], xt[:,1], s=50, color='red', alpha=0.95, label=f"Transformed - step {idx+1}")
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_title(f"Rectified Flow – Snapshot Step {idx+1}")
    ax.legend()
    ax.grid(True, alpha=0.2)
    writer.add_image(f"RectifiedFlow/Snapshot_Step_{idx+1}", plot_to_image(fig), global_step=idx+1)
    plt.close(fig)

# ======================
# COMPLETE TRAJECTORIES WITH INTERACTIVE SLIDER
# ======================
for step_idx, traj in enumerate(trajectories):
    fig, ax = plt.subplots(figsize=(6,6))
    # Target green
    ax.scatter(xB[:,0], xB[:,1], s=40, color='green', alpha=0.8, label="Target")
    # Trajectories black
    for p in range(n_points):
        xs = [t[p,0] for t in trajectories[:step_idx+1]]
        ys = [t[p,1] for t in trajectories[:step_idx+1]]
        ax.plot(xs, ys, color='black', alpha=0.3)
    # Transformed points at the current step (red)
    ax.scatter(traj[:,0], traj[:,1], s=50, color='red', alpha=0.95, label=f"Transformed - step {step_idx}")
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_title("Rectified Flow – Trajectories (Interactive Step)")
    ax.grid(True, alpha=0.2)
    ax.legend()
    # Saving step-by-step to simulate the interactive bar on TensorBoard
    writer.add_image("RectifiedFlow/Trajectories", plot_to_image(fig), global_step=step_idx)
    plt.close(fig)

writer.close()
print("Sampling completed. Statistical snapshot and trajectories with slider saved on TensorBoard!")
