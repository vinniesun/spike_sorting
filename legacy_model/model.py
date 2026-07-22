import torch
import torch.nn as nn

import snntorch as snn
from model.resonator import Resonator

class SNNResonatorModel(nn.Module):
    def __init__(self, num_classes):
        # The model we are recreating have the following architecture:
        # 10×10×2−6c2−24c3−10o
        # Where the input image is 10x10x2 (2 because of the polarity)

        super().__init__()

        self.conv1 = nn.Conv2d(2, 6, kernel_size=2)     # Output Dim is Batch_size*6*9*9
        self.res1 = Resonator(reset_mechanism="subtract")
        self.conv2 = nn.Conv2d(6, 24, kernel_size=3)    # Output Dim is Batch_size*24*7*7
        self.res2 = Resonator(reset_mechanism="subtract")
        self.fc = nn.Linear(24*7*7, num_classes)                 # After flattening, the input dimension becomes 1176 (24*7*7)
        self.res3 = Resonator(reset_mechanism="subtract")

    def init_resonators(self, input_, dts_):
        self.forward(input_, dts_)

    def forward(self, x, dts):
        # Expected Input Dimension is Num_Steps*Batch_size*channels*height*Width
        self.res1.reset_mem()
        self.res2.reset_mem()
        self.res3.reset_mem()

        spk3_hist, x3_hist, y3_hist = [], [], []
        for step in range(x.shape[1]):  # Loop through number of steps
            dt = dts[:, step]
            cur1 = self.conv1(x[:, step, ...])
            spk1, x1, y1 = self.res1(cur1, dt)
            cur2 = self.conv2(spk1)
            spk2, x2, y2 = self.res2(cur2, dt)
            spk2 = spk2.view(x.shape[0], -1)
            cur3 = self.fc(spk2)
            spk3, x3, y3 = self.res3(cur3, dt)

            spk3_hist.append(spk3)
            x3_hist.append(x3)
            y3_hist.append(y3)

        return torch.stack(spk3_hist, dim=0), torch.stack(x3_hist, dim=0), torch.stack(y3_hist, dim=0)
    
class SNNLIFModel(nn.Module):
    def __init__(self, num_classes, beta):
        # The model we are recreating have the following architecture:
        # 10×10×2−6c2−24c3−10o
        # Where the input image is 10x10x2 (2 because of the polarity)

        super().__init__()

        self.conv1 = nn.Conv2d(2, 6, kernel_size=2)     # Output Dim is Batch_size*6*9*9
        self.lif1 = snn.Leaky(beta=beta, init_hidden=False)
        self.conv2 = nn.Conv2d(6, 24, kernel_size=3)    # Output Dim is Batch_size*24*7*7
        self.lif2 = snn.Leaky(beta=beta, init_hidden=False)
        self.fc = nn.Linear(24*7*7, num_classes)                 # After flattening, the input dimension becomes 1176 (24*7*7)
        self.lif3 = snn.Leaky(beta=beta, init_hidden=False)

    def forward(self, x):
        # Expected Input Dimension is Num_Steps*Batch_size*channels*height*Width
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        spk3_hist, mem3_hist = [], []
        for step in range(x.shape[1]):  # Loop through number of steps
            cur1 = self.conv1(x[:, step, ...])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.conv2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2 = spk2.view(x.shape[0], -1)
            cur3 = self.fc(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)

            spk3_hist.append(spk3)
            mem3_hist.append(mem3)

        return torch.stack(spk3_hist, dim=0), torch.stack(mem3_hist, dim=0)

# if __name__ == "__main__":
#     dummy_input = torch.ones(256, 25, 2, 10, 10)
#     # dummy_input = dummy_input.permute(1, 0, 2, 3, 4)

#     net = SNNModel(num_classes=10, beta=0.95)

#     spk_out, mem_out = net(dummy_input)

#     print(f"spk_out dim is: {spk_out.shape}")
#     print(f"mem_out dim is: {mem_out.shape}")