import torch

from mmseg.models import build_segmentor
from mmseg.models.backbones.haomo import DLA34

if __name__ == '__main__':
    m = DLA34(pretrained=False, sync_bn=False)
    print(m._bb.base_layer[0].weight.max())
    checkpoint = torch.load("/home/jian/output/latest.pth", map_location=torch.device('cpu'))

    remap_dict = {}
    for k, v in checkpoint['state_dict'].items():
        if k.startswith("backbone."):
            remap_dict[k.replace("backbone.", "")] = v
    m.load_state_dict(remap_dict)
    print(m._bb.base_layer[0].weight.max())
    torch.save(m._bb.state_dict(), "/home/jian/output/dla34.bb.pth")
    # print(m._bb)
