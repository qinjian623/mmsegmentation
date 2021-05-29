import torch
import torchvision as tv
import torch.nn as nn
from torchvision.models._utils import IntermediateLayerGetter

from ..builder import BACKBONES

# We don't need MMSeg resnets

@BACKBONES.register_module()
class TVResNet(nn.Module):
    def __init__(self, arch="r18", pretrained=True, sync_bn=False, **kwargs):
        super(TVResNet, self).__init__(**kwargs)
        if arch=="r18":
            self._bb = tv.models.resnet18(pretrained=pretrained)
        elif arch=="r34":
            self._bb = tv.models.resnet34(pretrained=pretrained)
        elif arch=="r50":
            self._bb = tv.models.resnet50(pretrained=pretrained)
        else:
            raise NotImplementedError("Only arch in [r18, r34, r50] now.")
        if sync_bn:
            self._bb = nn.SyncBatchNorm.convert_sync_batchnorm(self._bb)
        self._bbody = IntermediateLayerGetter(self._bb,
                                              {'layer1': 's4', 'layer2': 's8', 'layer3': 's16', 'layer4': 's32'})

    def init_weights(self, pretrained=True):
        pass
        # if pretrained:
        #     state_dict = load_state_dict_from_url(model_urls[arch],
        #                                           progress=progress)
        #     self.load_state_dict(state_dict)

    def forward(self, x):
        ret = self._bbody(x)
        return list(ret.values())



if __name__ == '__main__':
    bb = TVResNet()
    d = torch.rand(4, 3, 224, 224)
    for i in bb(d):
        print(i.shape)