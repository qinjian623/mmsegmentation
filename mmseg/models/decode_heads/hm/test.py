import torch

from mmseg.models import FPN

if __name__ == '__main__':
    neck = FPN(in_channels=[512, 512, 512, 1024], out_channels=256, num_outs=1, start_level=1, end_level=2)
    r = neck([torch.rand(1, 512, 32, 32), torch.rand(1, 512, 16, 16), torch.rand(1, 512, 8, 8), torch.rand(1, 1024, 4, 4)])
    for i in r:
        print(i.shape)
