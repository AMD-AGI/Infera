import torch
print(torch.__version__, flush=True)
x = torch.empty(1024, device='cuda:0')
print('allocated', flush=True)
x.fill_(1)
torch.cuda.synchronize()
print('filled', x[0].item(), flush=True)
