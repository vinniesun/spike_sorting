import torch

class SwapAdjacent:
    def __init__(self, p=1.0):
        self.p = p

    def __call__(self, x):
        if torch.rand(1).item() < self.p:
            return x

        mask = torch.where(x != 0)[0]

        x_clone = x.clone()
        mask_idx = torch.randperm(mask.shape[0])[0].item()
        idx = mask[mask_idx].item()

        if torch.rand(1).item() > 0.5:
            if idx >= x.shape[0]-1:
                x_clone[[idx, idx-1]] = x_clone[[idx-1, idx]]
            else:
                x_clone[[idx, idx+1]] = x_clone[[idx+1, idx]]
        else:
            if idx == 0:
                x_clone[[idx, idx+1]] = x_clone[[idx+1, idx]]
            else:
                x_clone[[idx, idx-1]] = x_clone[[idx-1, idx]]

        return x_clone
    