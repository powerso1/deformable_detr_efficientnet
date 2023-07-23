import torch

W_ = 4
H_ = 3
device = 'cpu'

ref_x, ref_y = torch.meshgrid(torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device),
                              torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device))


print(ref_x)
print(ref_y)
ref = torch.stack((ref_x, ref_y), -1)
print(ref.shape)
