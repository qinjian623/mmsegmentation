import os.path as osp

from mmseg.datasets import CustomDataset
from mmseg.datasets import DATASETS


@DATASETS.register_module()
class HMDataset(CustomDataset):
    """HM Seg dataset.

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

    PALETTE = [
        [6, 235, 68],
        [203, 213, 104],
        [2, 2, 169],
        [247, 129, 7],
        [236, 184, 69],
        [239, 86, 208],
        [31, 170, 7],
        [24, 166, 169],
        [25, 39, 42],
        [252, 73, 124],
        [52, 31, 161],
        [156, 24, 38],
        [17, 213, 171],
        [85, 219, 203],
        # "sse-eraser": [75, 195, 52],
        [65, 100, 8],
        [237, 40, 140],
        [169, 83, 76],
    ]

    def __init__(self, **kwargs):
        super(HMDataset, self).__init__(
            img_suffix='.json.png',
            seg_map_suffix='_label.png',
            reduce_zero_label=True,
            **kwargs)
        assert osp.exists(self.img_dir)
