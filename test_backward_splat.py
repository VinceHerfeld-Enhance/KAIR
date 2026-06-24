import torch
from vsr.utils.splatting import backward_gridsample

B=2
H = 128
W = 128
C = 3
out_nc = 3
sf = 4
device = "cuda"

hz, wz = H * sf, W * sf

grid = torch.randn(B, H, W,2, requires_grad=True).to(device)
y = torch.randn(B, C+1, H, W, requires_grad=True).to(device)
z = torch.zeros((B, out_nc + 1, hz, wz), device=device).requires_grad_(True)



out = backward_gridsample.apply(y, z, grid, False, "bilinear")
loss = torch.abs(z- out).mean()
loss.backward()

print(grid.grad)  # should not be None
