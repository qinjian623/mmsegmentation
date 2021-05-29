import torch
import torchvision as tv
import torch.nn as nn
from torchvision.models._utils import IntermediateLayerGetter
import timm

from ..builder import BACKBONES

# We don't need MMSeg resnets


arch_tab = {
    'r18': tv.models.resnet18,
    'r34': tv.models.resnet34,
    'r50': tv.models.resnet50,
    'r101': tv.models.resnet101,
    'r152': tv.models.resnet152,
    'rnext50': tv.models.resnext50_32x4d,
    'rnext101': tv.models.resnext101_32x8d,
}
@BACKBONES.register_module()
class TVResNet(nn.Module):
    def __init__(self, arch="r18", pretrained=True, sync_bn=False, **kwargs):
        super(TVResNet, self).__init__(**kwargs)
        if arch in arch_tab:
            self._bb = arch_tab[arch](pretrained=pretrained)
        else:
            raise NotImplementedError("This arch {} not supported yet. All archs : {}".format(arch, arch_tab.keys()))
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


@BACKBONES.register_module()
class DLA34(nn.Module):
    def __init__(self, arch="r18", pretrained=True, sync_bn=False, **kwargs):
        super(DLA34, self).__init__(**kwargs)
        self._bb = timm.create_model('dla34', pretrained=pretrained)
        if sync_bn:
            self._bb = nn.SyncBatchNorm.convert_sync_batchnorm(self._bb)

        self._bbody = IntermediateLayerGetter(self._bb,
                                              {'level2': 's4', 'level3': 's8', 'level4': 's16', 'level5': 's32'})

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
    print(bb)
    for i in bb(d):
        print(i.shape)