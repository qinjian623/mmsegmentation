import os

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from mmseg.datasets import DATASETS
from mmseg.datasets.hm.utils import transforms as tf
from mmseg.datasets.hm.utils.affinity_fields import generateAFs
from scipy.interpolate import CubicSpline
from torch.utils.data import Dataset


def coord_op_to_ip(x, y, scale):
    # (208*scale, 72*scale) --> (208*scale, 72*scale+14=590) --> (1664, 590) --> (1640, 590)
    if x is not None:
        x = int(scale * x)
        x = x * 1640. / 1664.
    if y is not None:
        y = int(scale * y + 14)
    return x, y


def coord_ip_to_op(x, y, scale):
    # (1640, 590) --> (1664, 590) --> (1664, 590-14=576) --> (1664/scale, 576/scale)
    if x is not None:
        x = x * 1664. / 1640.
        x = int(x / scale)
    if y is not None:
        y = int((y - 14) / scale)
    return x, y


def get_lanes_culane(seg_out, samp_factor):
    # fit cubic spline to each lane
    h_samples = range(589, 240, -10)
    cs = []
    lane_ids = np.unique(seg_out[seg_out > 0])
    for idx, t_id in enumerate(lane_ids):
        xs, ys = [], []
        for y_op in range(seg_out.shape[0]):
            x_op = np.where(seg_out[y_op, :] == t_id)[0]
            if x_op.size > 0:
                x_op = np.mean(x_op)
                x_ip, y_ip = coord_op_to_ip(x_op, y_op, samp_factor)
                xs.append(x_ip)
                ys.append(y_ip)
        if len(xs) >= 10:
            cs.append(CubicSpline(ys, xs, extrapolate=False))
        else:
            cs.append(None)
    # get x-coordinates from fitted spline
    lanes = []
    for idx, t_id in enumerate(lane_ids):
        lane = []
        if cs[idx] is not None:
            y_out = np.array(h_samples)
            x_out = cs[idx](y_out)
            for _x, _y in zip(x_out, y_out):
                if np.isnan(_x):
                    continue
                else:
                    lane += [_x, _y]
        else:
            print("Lane completely missed!")
        if len(lane) <= 16:
            continue
        lanes.append(lane)
    return lanes


@DATASETS.register_module()
class CULane(Dataset):
    CLASSES = None
    PALETTE = None

    def __init__(self, path, image_set='train', test_mode=False,
                 random_transforms=False):
        super(CULane, self).__init__()
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)
        assert image_set in ('train', 'val', 'test'), "image_set is not valid!"
        self.input_size = (256, 512)  # original image res: (590, 1640) -> (590-14, 1640+24)/2
        self.output_scale = 0.25
        self.samp_factor = 2. / self.output_scale
        self.data_dir_path = path
        self.image_set = image_set
        self.random_transforms = random_transforms
        # normalization transform for input images
        self.mean = [0.485, 0.456, 0.406]  # [103.939, 116.779, 123.68]
        self.std = [0.229, 0.224, 0.225]  # [1, 1, 1]
        self.ignore_label = 255
        if self.random_transforms:
            self.transforms = transforms.Compose([
                tf.GroupRandomScale(size=(0.5, 0.6), interpolation=(cv2.INTER_LINEAR, cv2.INTER_NEAREST)),
                tf.GroupRandomCropRatio(size=(self.input_size[1], self.input_size[0])),
                tf.GroupRandomHorizontalFlip(),
                tf.GroupRandomRotation(degree=(-1, 1), interpolation=(cv2.INTER_LINEAR, cv2.INTER_NEAREST),
                                       padding=(self.mean, (self.ignore_label,))),
                tf.GroupNormalize(mean=(self.mean, (0,)), std=(self.std, (1,))),
            ])
        else:
            self.transforms = transforms.Compose([
                tf.GroupRandomScale(size=(0.5, 0.5), interpolation=(cv2.INTER_LINEAR, cv2.INTER_NEAREST)),
                tf.GroupNormalize(mean=(self.mean, (0,)), std=(self.std, (1,))),
            ])
        print("Creating Index...")
        self.create_index()
        print("Creating Index DONE")

    def create_index(self):
        self.img_list = []
        self.seg_list = []

        listfile = os.path.join(self.data_dir_path, "list", "{}.txt".format(self.image_set))
        if not os.path.exists(listfile):
            raise FileNotFoundError("List file doesn't exist. Label has to be generated! ...")

        with open(listfile) as f:
            for line in f:
                l = line.strip()
                if self.image_set == 'test':
                    self.img_list.append(os.path.join(self.data_dir_path,
                                                      l[1:]))  # l[1:]  get rid of the first '/' so as for os.path.join
                else:
                    self.img_list.append(os.path.join(self.data_dir_path,
                                                      l[0:]))
                if self.image_set == 'test':
                    self.seg_list.append(os.path.join(self.data_dir_path, 'laneseg_label_w16_test', l[0:-3] + 'png'))
                else:
                    self.seg_list.append(os.path.join(self.data_dir_path, 'laneseg_label_w16', l[0:-3] + 'png'))

    def __getitem__(self, idx):
        img = cv2.imread(self.img_list[idx]).astype(np.float32) / 255.  # (H, W, 3)
        if self.image_set == "test":
            seg = np.zeros(img.shape[:2])
        else:
            seg = cv2.imread(self.seg_list[idx], cv2.IMREAD_UNCHANGED)  # (H, W)
        seg = np.tile(seg[..., np.newaxis], (1, 1, 3))  # (H, W, 3)
        seg = cv2.resize(seg, (1024, 512), interpolation=cv2.INTER_NEAREST)
        img = cv2.resize(img, (1024, 512), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img, seg = self.transforms((img, seg))
        seg = cv2.resize(seg, None, fx=self.output_scale, fy=self.output_scale, interpolation=cv2.INTER_NEAREST)

        mask = seg[:, :, 0].copy()
        mask[seg[:, :, 0] >= 1] = 1
        mask[seg[:, :, 0] == self.ignore_label] = self.ignore_label

        # create AFs
        seg_wo_ignore = seg[:, :, 0].copy()
        seg_wo_ignore[seg_wo_ignore == self.ignore_label] = 0
        vaf, haf = generateAFs(seg_wo_ignore.astype(np.long), viz=False)
        af = np.concatenate((vaf, haf[:, :, 0:1]), axis=2)

        # convert all outputs to torch tensors
        img = torch.from_numpy(img).permute(2, 0, 1).contiguous().float()
        seg = torch.from_numpy(seg[:, :, 0]).contiguous().long().unsqueeze(0)
        mask = torch.from_numpy(mask).contiguous().float().unsqueeze(0)
        af = torch.from_numpy(af).permute(2, 0, 1).contiguous().float()

        return {'img': img, 'seg': seg, 'mask': mask, 'af': af, 'img_metas': []}

    def __len__(self):
        return len(self.img_list)

    def evaluate(self,
                 results,
                 metric='mIoU',
                 logger=None,
                 efficient_test=False,
                 **kwargs):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. 'mIoU' and
                'mDice' are supported.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.

        Returns:
            dict[str, float]: Default metrics.
        """

        if isinstance(metric, str):
            metric = [metric]
        allowed_metrics = ['mIoU', 'mDice']
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError('metric {} is not supported'.format(metric))
        eval_results = {}
        gt_seg_maps = self.get_gt_seg_maps(efficient_test)
        if self.CLASSES is None:
            num_classes = len(
                reduce(np.union1d, [np.unique(_) for _ in gt_seg_maps]))
        else:
            num_classes = len(self.CLASSES)
        ret_metrics = eval_metrics(
            results,
            gt_seg_maps,
            num_classes,
            self.ignore_index,
            metric,
            label_map=self.label_map,
            reduce_zero_label=self.reduce_zero_label)
        class_table_data = [['Class'] + [m[1:] for m in metric] + ['Acc']]
        if self.CLASSES is None:
            class_names = tuple(range(num_classes))
        else:
            class_names = self.CLASSES
        ret_metrics_round = [
            np.round(ret_metric * 100, 2) for ret_metric in ret_metrics
        ]
        for i in range(num_classes):
            class_table_data.append([class_names[i]] +
                                    [m[i] for m in ret_metrics_round[2:]] +
                                    [ret_metrics_round[1][i]])
        summary_table_data = [['Scope'] +
                              ['m' + head
                               for head in class_table_data[0][1:]] + ['aAcc']]
        ret_metrics_mean = [
            np.round(np.nanmean(ret_metric) * 100, 2)
            for ret_metric in ret_metrics
        ]
        summary_table_data.append(['global'] + ret_metrics_mean[2:] +
                                  [ret_metrics_mean[1]] +
                                  [ret_metrics_mean[0]])
        print_log('per class results:', logger)
        table = AsciiTable(class_table_data)
        print_log('\n' + table.table, logger=logger)
        print_log('Summary:', logger)
        table = AsciiTable(summary_table_data)
        print_log('\n' + table.table, logger=logger)

        for i in range(1, len(summary_table_data[0])):
            eval_results[summary_table_data[0]
            [i]] = summary_table_data[1][i] / 100.0
        for idx, sub_metric in enumerate(class_table_data[0][1:], 1):
            for item in class_table_data[1:]:
                eval_results[str(sub_metric) + '.' +
                             str(item[0])] = item[idx] / 100.0

        if mmcv.is_list_of(results, str):
            for file_name in results:
                os.remove(file_name)
        return eval_results
