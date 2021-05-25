import os.path as osp

from mmseg.datasets import CustomDataset
from mmseg.datasets import DATASETS


@DATASETS.register_module()
class HMDataset(CustomDataset):
    """DRIVE dataset.

    In segmentation map annotation for HM. The
    ``img_suffix`` is fixed to '.png' and ``seg_map_suffix`` is fixed to
    '_label.png'.
    """
    CLASSES = (
        "Nill",
        "Building.common",
        "Building.tunnel",
        "Marks.lane",
        "MO.living",
        "MO.no_id",
        "MO.pedestrian",
        "MO.temp_static",
        "MO.vehicle.ground",
        "Nature.vegetation",
        "Separator.road_boundary",
        "Separator.std",
        "Separator.wall",
        "Space.vehicle",
        "Traffic.facility.affiliated",
        "Traffic.light",
        "Traffic.sign",
    )
    PALETTE = None

    def __init__(self, **kwargs):
        super(HMDataset, self).__init__(
            img_suffix='.json.png',
            seg_map_suffix='_label.png',
            reduce_zero_label=True,
            **kwargs)
        assert osp.exists(self.img_dir)
