import torch.nn as nn
from mmcv.runner import force_fp32
from mmseg.models.necks.fpn import FPN

from ...builder import HEADS
from ...decode_heads.decode_head import BaseDecodeHead


#
# class RPNNeck(nn.Module):
#     def __init__(self, lvls=[512, 1024], channels=256, conv_cfg=None, norm_cfg=None, act_cfg=None):
#         super(RPNNeck, self).__init__()
#         self.lvl_convs = []
#         for i, inc in enumerate(lvls):
#             self.lvl_convs.append(ConvModule(
#                 inc,
#                 channels,
#                 kernel_size=3,
#                 padding=1,
#                 dilation=1,
#                 conv_cfg=conv_cfg,
#                 norm_cfg=norm_cfg,
#                 act_cfg=act_cfg))
#         self.upsample = nn.Upsample(scale_factor=(2, 2))  # 3 times
#
#         # Conv after residual plus
#         self.convs = []
#         for i, inc in enumerate(lvls):
#             self.convs.append(ConvModule(
#                 channels,
#                 channels,
#                 kernel_size=3,
#                 padding=1,
#                 dilation=1,
#                 conv_cfg=conv_cfg,
#                 norm_cfg=norm_cfg,
#                 act_cfg=act_cfg))
#
#     def forward(self, x):
#         eq_ch_xs = [lv_conv(lv_x) for lv_x, lv_conv in zip(x, self.lvl_convs)]
#         output = None
#         for eq_ch_x, conv in zip(eq_ch_xs[::-1], self.convs[::-1]):
#             if output is None:
#                 output = eq_ch_x
#             else:
#                 output += eq_ch_x
#             output = conv(output)
#             output = self.upsample(output)
#         return output
#

@HEADS.register_module
class AFHead(BaseDecodeHead):
    def __init__(self, feat_num=32, heads={"hm": 1, "vaf": 2, "haf": 1}, fpn_out_level=0, **kwargs):
        super().__init__(**kwargs)
        # TODO Make sure all cfgs works well.
        self._neck = FPN(self.in_channels, out_channels=feat_num, num_outs=1, start_level=fpn_out_level,
                         end_level=fpn_out_level + 1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg,
                         act_cfg=self.act_cfg)
        self._heads = heads
        for head, nc in self._heads.items():
            fc = nn.Conv2d(feat_num, nc, kernel_size=3, stride=1, padding=1, bias=True)
            self.__setattr__(head, fc)

        # Called by upper level...
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self._transform_inputs(x)
        # [b x [512, 1024] x h x w]
        x = self._neck(x)  # FPN
        assert len(x) == 1
        x = x[0]
        z = {}
        for head in self._heads:
            z[head] = self.__getattr__(head)(x)
        return z

    def forward_train(self, inputs, img_metas, train_cfg, **kwargs):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.

            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        out = self.forward(inputs)
        losses = self.losses(out, **kwargs)
        return losses

    @force_fp32(apply_to=('out',))
    def losses(self, out, **kwargs):
        """Compute LaneAF loss."""
        loss = dict()
        loss_seg, loss_vaf, loss_haf = self.loss_decode(out, **kwargs)
        loss['seg'] = loss_seg
        loss['vaf'] = loss_vaf
        loss['haf'] = loss_haf
        return loss_seg + loss_vaf + loss_haf
