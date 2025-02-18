import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

os.makedirs("checkpoints", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

#  SIREN DEFINITIONS
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
                
    def forward(self, x):
        # print(f"[DEBUG: SineLayer] input shape: {x.shape}")
        out = torch.sin(self.omega_0 * self.linear(x))
        # print(f"[DEBUG: SineLayer] output shape: {out.shape}\n")
        return out

class Siren(nn.Module):
    def __init__(self, in_features=2, hidden_features=256, hidden_layers=3, out_features=3, omega_0=30):
        super().__init__()
        layers = []
        # First Sine layer
        layers.append(SineLayer(in_features, hidden_features, is_first=True, omega_0=omega_0))
        # Hidden Sine layers
        for _ in range(hidden_layers):
            layers.append(SineLayer(hidden_features, hidden_features, is_first=False, omega_0=omega_0))
        # Final linear
        final_linear = nn.Linear(hidden_features, out_features)
        with torch.no_grad():
            bound = np.sqrt(6 / hidden_features) / omega_0
            final_linear.weight.uniform_(-bound, bound)
        layers.append(final_linear)
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, coords):
        return self.net(coords)

# TRAINING FUNCTION
def train_siren(model, coords, pixels, epochs=1000, lr=1e-4, log_interval=200):
    model = model.to(device)
    coords = coords.to(device)
    pixels = pixels.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(coords)             # shape (N, 3)
        loss = criterion(out, pixels)   # shape (N, 3)
        loss.backward()
        optimizer.step()
        
        if epoch % log_interval == 0:
            print(f"[train_siren] Epoch {epoch}, Loss: {loss.item():.6f}")
    return model

# DISTILLATION
def distill_siren(student, teacher, coords, epochs=1000, lr=1e-4, log_interval=200):
    student = student.to(device)
    teacher = teacher.to(device)
    coords = coords.to(device)
    
    # Precompute teacher's outputs
    with torch.no_grad():
        teacher_out = teacher(coords)  # shape (N, 3)
    
    optimizer = optim.Adam(student.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        student_out = student(coords)  # shape (N, 3)
        loss = criterion(student_out, teacher_out)
        loss.backward()
        optimizer.step()
        
        if epoch % log_interval == 0:
            print(f"[distill_siren] Epoch {epoch}, Loss: {loss.item():.6f}")
    return student

# PRUNING + FINE-TUNING
def prune_siren(model, amount=0.5):
    parameters_to_prune = []
    for module in model.modules():
        if isinstance(module, nn.Linear):
            parameters_to_prune.append((module, 'weight'))
    
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount
    )
    # Remove re-param so pruned weights are actually set to zero
    for (module, _) in parameters_to_prune:
        prune.remove(module, 'weight')
    
    print(f"Pruned {amount*100:.0f}% of weights globally.")

def finetune_siren(model, coords, pixels, epochs=500, lr=1e-5):
    model = model.to(device)
    coords = coords.to(device)
    pixels = pixels.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(coords)
        loss = criterion(out, pixels)
        loss.backward()
        optimizer.step()
    print("Finished fine-tuning after pruning.")

# CONVERT TO FP16 (GPU)
def convert_to_fp16(model):
    model.half()
    # Convert existing parameters to half
    for module in model.modules():
        if isinstance(module, nn.Linear):
            module.weight.data = module.weight.data.half()
            if module.bias is not None:
                module.bias.data = module.bias.data.half()
    return model

# RECONSTRUCTION
def reconstruct_image(model, coords, H, W, is_fp16=False):
    model = model.to(device)
    if is_fp16:
        coords = coords.half()
    coords = coords.to(device)
    
    with torch.no_grad():
        out = model(coords)
    out_np = out.cpu().numpy().reshape(H, W, 3)
    
    # Convert to float32 for Matplotlib
    out_np_clipped = np.clip(out_np, 0, 1).astype(np.float32)
    return out_np_clipped

def show_side_by_side(imgA, imgB, titleA="Original", titleB="Reconstructed"):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].imshow(imgA)
    axs[0].set_title(titleA)
    axs[0].axis("off")
    
    axs[1].imshow(imgB)
    axs[1].set_title(titleB)
    axs[1].axis("off")
    plt.tight_layout()
    plt.show()

# FILE SIZE HELPERS
def print_file_size(filename):
    size_bytes = os.path.getsize(filename)
    size_kb = size_bytes / 1024
    print(f"{filename} size: {size_kb:.2f} KB")

# MAIN
if __name__ == "__main__":
    image_path = "Lena_small.jpg" 
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img, dtype=np.float32) / 255.0
    H, W, _ = img_np.shape
    print(f"[INFO] Loaded image: shape={img_np.shape} (H={H}, W={W}, C=3)")

    # Flatten image => shape (H*W, 3)
    pixels = torch.tensor(img_np).reshape(-1, 3)
    print(f"[INFO] Flattened pixels shape: {pixels.shape}")

    # Flatten coordinates => shape (H*W, 2)
    xs = torch.linspace(-1, 1, W)
    ys = torch.linspace(-1, 1, H)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    coords = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)
    print(f"[INFO] Flattened coords shape: {coords.shape}")

    # --- TEACHER (bigger) ---
    teacher = Siren(in_features=2, hidden_features=256, hidden_layers=4, out_features=3, omega_0=130)
    print("\n--- Training TEACHER (bigger) ---")
    teacher = train_siren(teacher, coords, pixels, epochs=5000, lr=1e-4, log_interval=200)

    # Save TEACHER
    torch.save(teacher, "teacher_full_model.pth")
    torch.save(teacher.state_dict(), "teacher_state_dict.pth")

    # --- STUDENT (smaller) ---
    student = Siren(in_features=2, hidden_features=128, hidden_layers=2, out_features=3, omega_0=130)
    print("\n--- Distilling from TEACHER to STUDENT ---")
    student = distill_siren(student, teacher, coords, epochs=5000, lr=1e-4, log_interval=200)

    # --- PRUNE + FINE-TUNE ---
    print("\n--- Pruning STUDENT by 50% & Fine-Tuning ---")
    prune_siren(student, amount=0.5)
    finetune_siren(student, coords, pixels, epochs=5000, lr=1e-5)

    # Save STUDENT FP32
    torch.save(student, "student_fp32_full_model.pth")
    torch.save(student.state_dict(), "student_fp32_state_dict.pth")

    # --- Convert STUDENT to FP16 (GPU) ---
    print("\n--- Converting STUDENT to FP16 ---")
    student = student.to(device)
    student = convert_to_fp16(student)

    # Save STUDENT FP16
    torch.save(student, "student_fp16_full_model.pth")
    torch.save(student.state_dict(), "student_fp16_state_dict.pth")

    # Compare Reconstructions at original resolution
    recon_teacher = reconstruct_image(teacher, coords, H, W, is_fp16=False)
    recon_student_fp16 = reconstruct_image(student, coords, H, W, is_fp16=True)

    show_side_by_side(img_np, recon_teacher,
                      titleA="Original",
                      titleB="Teacher (FP32) Reconstruction")
    show_side_by_side(img_np, recon_student_fp16,
                      titleA="Original",
                      titleB="Student (Pruned+FP16) Reconstruction")

    # Compare at higher resolution
    high_res = 1024
    xs_hr = torch.linspace(-1, 1, high_res)
    ys_hr = torch.linspace(-1, 1, high_res)
    grid_y_hr, grid_x_hr = torch.meshgrid(ys_hr, xs_hr, indexing='ij')
    coords_hr = torch.stack([grid_x_hr, grid_y_hr], dim=-1).reshape(-1, 2)

    hr_teacher = reconstruct_image(teacher, coords_hr, high_res, high_res, is_fp16=False)
    hr_student_fp16 = reconstruct_image(student, coords_hr, high_res, high_res, is_fp16=True)

    show_side_by_side(hr_teacher, hr_student_fp16,
                      titleA="Teacher High-Res",
                      titleB="Student (Pruned+FP16) High-Res")

    # 10) Print File Sizes
    print("\n--- Model File Sizes ---")
    def print_all_sizes():
        for fname in [
            "teacher_full_model.pth",
            "teacher_state_dict.pth",
            "student_fp32_full_model.pth",
            "student_fp32_state_dict.pth",
            "student_fp16_full_model.pth",
            "student_fp16_state_dict.pth"
        ]:
            if os.path.exists(fname):
                size_bytes = os.path.getsize(fname)
                size_kb = size_bytes / 1024
                print(f"{fname}: {size_kb:.2f} KB")
            else:
                print(f"{fname} not found.")

    print_all_sizes()

    print("\nDONE. Teacher -> Student Distillation, Pruning, FP16, no shape mismatch, saved models, compared file sizes.")
