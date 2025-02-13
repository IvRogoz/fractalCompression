import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os


# Import time module
import time

os.makedirs("checkpoints", exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, is_first=False, omega_0=30):
        super().__init__()
        self.in_features = in_features
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        self.init_weights()
        
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                bound = np.sqrt(6 / self.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)
                
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

class Siren(nn.Module):
    def __init__(self, in_features=2, hidden_features=256, hidden_layers=3, out_features=3, omega_0=30):
        super().__init__()
        layers = []
        layers.append(SineLayer(in_features, hidden_features, is_first=True, omega_0=omega_0))
        for _ in range(hidden_layers):
            layers.append(SineLayer(hidden_features, hidden_features, is_first=False, omega_0=omega_0))
        final_linear = nn.Linear(hidden_features, out_features)
        with torch.no_grad():
            bound = np.sqrt(6 / hidden_features) / omega_0
            final_linear.weight.uniform_(-bound, bound)
        layers.append(final_linear)
        self.net = nn.Sequential(*layers)
        
    def forward(self, coords):
        return self.net(coords)

target_image = Image.open("Lenna.jpg").convert("RGB")
target_np = np.array(target_image) / 255.0  # Normalize to [0,1]
H, W, _ = target_np.shape

# Prepare a coordinate grid in the range [-1, 1]
xs = np.linspace(-1, 1, W)
ys = np.linspace(-1, 1, H)
grid_x, grid_y = np.meshgrid(xs, ys)
coords = np.stack([grid_x, grid_y], axis=-1)  # shape: [H, W, 2]
coords = torch.tensor(coords, dtype=torch.float32).reshape(-1, 2).to(device)
pixels = torch.tensor(target_np, dtype=torch.float32).reshape(-1, 3).to(device)

model = Siren(in_features=2, hidden_features=256, hidden_layers=3, out_features=3, omega_0=30).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

# record start time
start = time.time()

num_epochs = 5000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = model(coords)
    loss = criterion(output, pixels)
    loss.backward()
    optimizer.step()
    
    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
        with torch.no_grad():
            output_img = model(coords).reshape(H, W, 3).cpu().numpy()
        output_img_clipped = np.clip(output_img, 0, 1)
        output_img_uint8 = (output_img_clipped * 255).astype(np.uint8)

        image_epoch = Image.fromarray(output_img_uint8)
        image_epoch.save(f"checkpoints/reconstructed_epoch_{epoch}.png")

end = time.time()
print("The time of execution of above program is :", (end-start), "s")

with torch.no_grad():
    output_img = model(coords).reshape(H, W, 3).cpu().numpy()

output_img_clipped = np.clip(output_img, 0, 1)
output_img_uint8 = (output_img_clipped * 255).astype(np.uint8)
recon_image = Image.fromarray(output_img_uint8)
recon_image.save("reconstructed_image.png")
print("Saved reconstructed_image.png.")

# TEST
high_res = 1024
xs_hr = np.linspace(-1, 1, high_res)
ys_hr = np.linspace(-1, 1, high_res)
grid_x_hr, grid_y_hr = np.meshgrid(xs_hr, ys_hr)
coords_hr = np.stack([grid_x_hr, grid_y_hr], axis=-1)
coords_hr = torch.tensor(coords_hr, dtype=torch.float32).reshape(-1, 2).to(device)

with torch.no_grad():
    output_hr = model(coords_hr).reshape(high_res, high_res, 3).cpu().numpy()

output_hr_clipped = np.clip(output_hr, 0, 1)
output_hr_uint8 = (output_hr_clipped * 255).astype(np.uint8)
hr_image = Image.fromarray(output_hr_uint8)
hr_image.save("high_resolution_image.png")
print("Saved high_resolution_image.png.")

torch.save(model.state_dict(), "siren_model_state_dict.pth")
print("Saved model state dictionary to siren_model_state_dict.pth.")

torch.save(model, "siren_full_model.pth")
print("Saved FULL model state dictionary to siren_full_model.pth.")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(target_np)
axes[0].set_title("Original Image")
axes[0].axis('off')

# Reconstructed 
axes[1].imshow(output_img_clipped)
axes[1].set_title("Reconstructed Image")
axes[1].axis('off')

# High resolution 
axes[2].imshow(output_hr_clipped)
axes[2].set_title("High Resolution Image")
axes[2].axis('off')

plt.tight_layout()
plt.show()
