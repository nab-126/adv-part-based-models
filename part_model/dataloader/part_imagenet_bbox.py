import json
import os
import random

# from DINO.datasets.coco import ConvertCocoPolysToMask, dataset_hook_register
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as data
import torchvision
from PIL import Image

from DINO.datasets.coco import ConvertCocoPolysToMask
from DINO.util.box_ops import box_cxcywh_to_xyxy, box_iou
from part_model.utils import get_seg_type, np_temp_seed
from part_model.utils.colors import COLORMAP
from part_model.utils.eval_sampler import DistributedEvalSampler
from part_model.utils.image import get_seg_type

# from .segmentation_transforms import (
#     CenterCrop,
#     Compose,
#     RandomHorizontalFlip,
#     RandomResizedCrop,
#     Resize,
#     ToTensor,
# )

CLASSES = {
    "Quadruped": 4,
    "Biped": 5,
    "Fish": 4,
    "Bird": 5,
    "Snake": 2,
    "Reptile": 4,
    "Car": 3,
    "Bicycle": 4,
    "Boat": 2,
    "Aeroplane": 5,
    "Bottle": 2,
}


######################################################
# some hookers for training


class label2compat:
    def __init__(self) -> None:
        self.category_map_str = {
            "1": 1,
            "2": 2,
            "3": 3,
            "4": 4,
            "5": 5,
            "6": 6,
            "7": 7,
            "8": 8,
            "9": 9,
            "10": 10,
            "11": 11,
            "13": 12,
            "14": 13,
            "15": 14,
            "16": 15,
            "17": 16,
            "18": 17,
            "19": 18,
            "20": 19,
            "21": 20,
            "22": 21,
            "23": 22,
            "24": 23,
            "25": 24,
            "27": 25,
            "28": 26,
            "31": 27,
            "32": 28,
            "33": 29,
            "34": 30,
            "35": 31,
            "36": 32,
            "37": 33,
            "38": 34,
            "39": 35,
            "40": 36,
            "41": 37,
            "42": 38,
            "43": 39,
            "44": 40,
            "46": 41,
            "47": 42,
            "48": 43,
            "49": 44,
            "50": 45,
            "51": 46,
            "52": 47,
            "53": 48,
            "54": 49,
            "55": 50,
            "56": 51,
            "57": 52,
            "58": 53,
            "59": 54,
            "60": 55,
            "61": 56,
            "62": 57,
            "63": 58,
            "64": 59,
            "65": 60,
            "67": 61,
            "70": 62,
            "72": 63,
            "73": 64,
            "74": 65,
            "75": 66,
            "76": 67,
            "77": 68,
            "78": 69,
            "79": 70,
            "80": 71,
            "81": 72,
            "82": 73,
            "84": 74,
            "85": 75,
            "86": 76,
            "87": 77,
            "88": 78,
            "89": 79,
            "90": 80,
        }
        self.category_map = {
            int(k): v for k, v in self.category_map_str.items()
        }

    def __call__(self, target, img=None):
        labels = target["labels"]
        res = torch.zeros(labels.shape, dtype=labels.dtype)
        for idx, item in enumerate(labels):
            res[idx] = self.category_map[item.item()] - 1
        target["label_compat"] = res
        if img is not None:
            return target, img
        else:
            return target


class label_compat2onehot:
    def __init__(self, num_class=80, num_output_objs=1):
        self.num_class = num_class
        self.num_output_objs = num_output_objs
        if num_output_objs != 1:
            raise DeprecationWarning(
                "num_output_objs!=1, which is only used for comparison"
            )

    def __call__(self, target, img=None):
        labels = target["label_compat"]
        place_dict = {k: 0 for k in range(self.num_class)}
        if self.num_output_objs == 1:
            res = torch.zeros(self.num_class)
            for i in labels:
                itm = i.item()
                res[itm] = 1.0
        else:
            # compat with baseline
            res = torch.zeros(self.num_class, self.num_output_objs)
            for i in labels:
                itm = i.item()
                res[itm][place_dict[itm]] = 1.0
                place_dict[itm] += 1
        target["label_compat_onehot"] = res
        if img is not None:
            return target, img
        else:
            return target


class box_label_catter:
    def __init__(self):
        pass

    def __call__(self, target, img=None):
        labels = target["label_compat"]
        boxes = target["boxes"]
        box_label = torch.cat((boxes, labels.unsqueeze(-1)), 1)
        target["box_label"] = box_label
        if img is not None:
            return target, img
        return target


def label2onehot(label, num_classes):
    """
    label: Tensor(K)
    """
    res = torch.zeros(num_classes)
    for i in label:
        itm = int(i.item())
        res[itm] = 1.0
    return res


class RandomSelectBoxlabels:
    def __init__(
        self,
        num_classes,
        leave_one_out=False,
        blank_prob=0.8,
        prob_first_item=0.0,
        prob_random_item=0.0,
        prob_last_item=0.8,
        prob_stop_sign=0.2,
    ) -> None:
        self.num_classes = num_classes
        self.leave_one_out = leave_one_out
        self.blank_prob = blank_prob

        self.set_state(
            prob_first_item, prob_random_item, prob_last_item, prob_stop_sign
        )

    def get_state(self):
        return [
            self.prob_first_item,
            self.prob_random_item,
            self.prob_last_item,
            self.prob_stop_sign,
        ]

    def set_state(
        self, prob_first_item, prob_random_item, prob_last_item, prob_stop_sign
    ):
        sum_prob = (
            prob_first_item + prob_random_item + prob_last_item + prob_stop_sign
        )
        assert sum_prob - 1 < 1e-6, (
            f"Sum up all prob = {sum_prob}. prob_first_item:{prob_first_item}"
            + f"prob_random_item:{prob_random_item}, prob_last_item:{prob_last_item}"
            + f"prob_stop_sign:{prob_stop_sign}"
        )

        self.prob_first_item = prob_first_item
        self.prob_random_item = prob_random_item
        self.prob_last_item = prob_last_item
        self.prob_stop_sign = prob_stop_sign

    def sample_for_pred_first_item(self, box_label: torch.FloatTensor):
        box_label_known = torch.Tensor(0, 5)
        box_label_unknown = box_label
        return box_label_known, box_label_unknown

    def sample_for_pred_random_item(self, box_label: torch.FloatTensor):
        n_select = int(random.random() * box_label.shape[0])
        box_label = box_label[torch.randperm(box_label.shape[0])]
        box_label_known = box_label[:n_select]
        box_label_unknown = box_label[n_select:]
        return box_label_known, box_label_unknown

    def sample_for_pred_last_item(self, box_label: torch.FloatTensor):
        box_label_perm = box_label[torch.randperm(box_label.shape[0])]
        known_label_list = []
        box_label_known = []
        box_label_unknown = []
        for item in box_label_perm:
            label_i = item[4].item()
            if label_i in known_label_list:
                box_label_known.append(item)
            else:
                # first item
                box_label_unknown.append(item)
                known_label_list.append(label_i)
        box_label_known = (
            torch.stack(box_label_known)
            if len(box_label_known) > 0
            else torch.Tensor(0, 5)
        )
        box_label_unknown = (
            torch.stack(box_label_unknown)
            if len(box_label_unknown) > 0
            else torch.Tensor(0, 5)
        )
        return box_label_known, box_label_unknown

    def sample_for_pred_stop_sign(self, box_label: torch.FloatTensor):
        box_label_unknown = torch.Tensor(0, 5)
        box_label_known = box_label
        return box_label_known, box_label_unknown

    def __call__(self, target, img=None):
        box_label = target["box_label"]  # K, 5

        dice_number = random.random()

        if dice_number < self.prob_first_item:
            (
                box_label_known,
                box_label_unknown,
            ) = self.sample_for_pred_first_item(box_label)
        elif dice_number < self.prob_first_item + self.prob_random_item:
            (
                box_label_known,
                box_label_unknown,
            ) = self.sample_for_pred_random_item(box_label)
        elif (
            dice_number
            < self.prob_first_item + self.prob_random_item + self.prob_last_item
        ):
            box_label_known, box_label_unknown = self.sample_for_pred_last_item(
                box_label
            )
        else:
            box_label_known, box_label_unknown = self.sample_for_pred_stop_sign(
                box_label
            )

        target["label_onehot_known"] = label2onehot(
            box_label_known[:, -1], self.num_classes
        )
        target["label_onehot_unknown"] = label2onehot(
            box_label_unknown[:, -1], self.num_classes
        )
        target["box_label_known"] = box_label_known
        target["box_label_unknown"] = box_label_unknown

        return target, img


class RandomDrop:
    def __init__(self, p=0.2) -> None:
        self.p = p

    def __call__(self, target, img=None):
        known_box = target["box_label_known"]
        num_known_box = known_box.size(0)
        idxs = torch.rand(num_known_box)
        # indices = torch.randperm(num_known_box)[:int((1-self).p*num_known_box + 0.5 + random.random())]
        target["box_label_known"] = known_box[idxs > self.p]
        return target, img


class BboxPertuber:
    def __init__(self, max_ratio=0.02, generate_samples=1000) -> None:
        self.max_ratio = max_ratio
        self.generate_samples = generate_samples
        self.samples = self.generate_pertube_samples()
        self.idx = 0

    def generate_pertube_samples(self):
        import torch

        samples = (
            (torch.rand(self.generate_samples, 5) - 0.5) * 2 * self.max_ratio
        )
        return samples

    def __call__(self, target, img):
        known_box = target["box_label_known"]  # Tensor(K,5), K known bbox
        K = known_box.shape[0]
        known_box_pertube = torch.zeros(K, 6)  # 4:bbox, 1:prob, 1:label
        if K == 0:
            pass
        else:
            if self.idx + K > self.generate_samples:
                self.idx = 0
            delta = self.samples[self.idx : self.idx + K, :]
            known_box_pertube[:, :4] = known_box[:, :4] + delta[:, :4]
            iou = (
                torch.diag(
                    box_iou(
                        box_cxcywh_to_xyxy(known_box[:, :4]),
                        box_cxcywh_to_xyxy(known_box_pertube[:, :4]),
                    )[0]
                )
            ) * (1 + delta[:, -1])
            known_box_pertube[:, 4].copy_(iou)
            known_box_pertube[:, -1].copy_(known_box[:, -1])

        target["box_label_known_pertube"] = known_box_pertube
        return target, img


class RandomCutout:
    def __init__(self, factor=0.5) -> None:
        self.factor = factor

    def __call__(self, target, img=None):
        unknown_box = target["box_label_unknown"]  # Ku, 5
        known_box = target["box_label_known_pertube"]  # Kk, 6
        Ku = unknown_box.size(0)

        known_box_add = torch.zeros(Ku, 6)  # Ku, 6
        known_box_add[:, :5] = unknown_box
        known_box_add[:, 5].uniform_(0.5, 1)

        known_box_add[:, :2] += (
            known_box_add[:, 2:4] * (torch.rand(Ku, 2) - 0.5) / 2
        )
        known_box_add[:, 2:4] /= 2

        target["box_label_known_pertube"] = torch.cat(
            (known_box, known_box_add)
        )
        return target, img


class RandomSelectBoxes:
    def __init__(self, num_class=80) -> None:
        Warning("This is such a slow function and will be deprecated soon!!!")
        self.num_class = num_class

    def __call__(self, target, img=None):
        boxes = target["boxes"]
        labels = target["label_compat"]

        # transform to list of tensors
        boxs_list = [[] for i in range(self.num_class)]
        for idx, item in enumerate(boxes):
            label = labels[idx].item()
            boxs_list[label].append(item)
        boxs_list_tensor = [
            torch.stack(i) if len(i) > 0 else torch.Tensor(0, 4)
            for i in boxs_list
        ]

        # random selection
        box_known = []
        box_unknown = []
        for idx, item in enumerate(boxs_list_tensor):
            ncnt = item.shape[0]
            nselect = int(
                random.random() * ncnt
            )  # close in both sides, much faster than random.randint
            # import ipdb; ipdb.set_trace()
            item = item[torch.randperm(ncnt)]
            # random.shuffle(item)
            box_known.append(item[:nselect])
            box_unknown.append(item[nselect:])
        # import ipdb; ipdb.set_trace()
        # box_known_tensor = [torch.stack(i) if len(i) > 0 else torch.Tensor(0,4) for i in box_known]
        # box_unknown_tensor = [torch.stack(i) if len(i) > 0 else torch.Tensor(0,4) for i in box_unknown]
        # print('box_unknown_tensor:', box_unknown_tensor)
        target["known_box"] = box_known
        target["unknown_box"] = box_unknown
        return target, img


# class BoxCatter():
#     def __init__(self) -> None:
#         pass

#     def __call__(self, target, img):
#         """
#         known_box_cat:
#             - Tensor(k*5),
#                 * Tensor[:, :4]: bbox,
#                 * Tensor[:, -1]: label
#         """
#         known_box = target['known_box']
#         boxes_list = []
#         for idx, boxes in enumerate(known_box):
#             nbox = boxes.shape[0]
#             boxes_idx = torch.cat([boxes, torch.Tensor([idx] * nbox).unsqueeze(1)], 1)
#             boxes_list.append(boxes_idx)
#         known_box_cat = torch.cat(boxes_list, 0)
#         target['known_box_cat'] = known_box_cat
#         return target, img


class MaskCrop:
    def __init__(self) -> None:
        pass

    def __call__(self, target, img):
        known_box = target["known_box"]
        h, w = img.shape[1:]  # h,w
        # imgsize = target['orig_size'] # h,w
        # import ipdb; ipdb.set_trace()
        scale = torch.Tensor([w, h, w, h])

        # _cnt = 0
        for boxes in known_box:
            if boxes.shape[0] == 0:
                continue
            box_xyxy = box_cxcywh_to_xyxy(boxes) * scale
            for box in box_xyxy:
                x1, y1, x2, y2 = [int(i) for i in box.tolist()]
                img[:, y1:y2, x1:x2] = 0
                # _cnt += 1
        # print("_cnt:", _cnt)
        return target, img


dataset_hook_register = {
    "label2compat": label2compat,
    "label_compat2onehot": label_compat2onehot,
    "box_label_catter": box_label_catter,
    "RandomSelectBoxlabels": RandomSelectBoxlabels,
    "RandomSelectBoxes": RandomSelectBoxes,
    "MaskCrop": MaskCrop,
    "BboxPertuber": BboxPertuber,
}

##################################################################################


def get_aux_target_hacks_list(image_set, args):
    if args.modelname in ["q2bs_mask", "q2bs"]:
        aux_target_hacks_list = [
            label2compat(),
            label_compat2onehot(),
            RandomSelectBoxes(num_class=args.num_classes),
        ]
        if args.masked_data and image_set == "train":
            # aux_target_hacks_list.append()
            aux_target_hacks_list.append(MaskCrop())
    elif args.modelname in [
        "q2bm_v2",
        "q2bs_ce",
        "q2op",
        "q2ofocal",
        "q2opclip",
        "q2ocqonly",
    ]:
        aux_target_hacks_list = [
            label2compat(),
            label_compat2onehot(),
            box_label_catter(),
            RandomSelectBoxlabels(
                num_classes=args.num_classes,
                prob_first_item=args.prob_first_item,
                prob_random_item=args.prob_random_item,
                prob_last_item=args.prob_last_item,
                prob_stop_sign=args.prob_stop_sign,
            ),
            BboxPertuber(max_ratio=0.02, generate_samples=1000),
        ]
    elif args.modelname in ["q2omask", "q2osa"]:
        if args.coco_aug:
            aux_target_hacks_list = [
                label2compat(),
                label_compat2onehot(),
                box_label_catter(),
                RandomSelectBoxlabels(
                    num_classes=args.num_classes,
                    prob_first_item=args.prob_first_item,
                    prob_random_item=args.prob_random_item,
                    prob_last_item=args.prob_last_item,
                    prob_stop_sign=args.prob_stop_sign,
                ),
                RandomDrop(p=0.2),
                BboxPertuber(max_ratio=0.02, generate_samples=1000),
                RandomCutout(factor=0.5),
            ]
        else:
            aux_target_hacks_list = [
                label2compat(),
                label_compat2onehot(),
                box_label_catter(),
                RandomSelectBoxlabels(
                    num_classes=args.num_classes,
                    prob_first_item=args.prob_first_item,
                    prob_random_item=args.prob_random_item,
                    prob_last_item=args.prob_last_item,
                    prob_stop_sign=args.prob_stop_sign,
                ),
                BboxPertuber(max_ratio=0.02, generate_samples=1000),
            ]
    else:
        aux_target_hacks_list = None

    return aux_target_hacks_list


class PartImageNetBBOXDataset(torchvision.datasets.CocoDetection):
    def __init__(
        self,
        img_folder,
        class_label_file,
        ann_file,
        transforms,
        aux_target_hacks=None,
    ):
        super(PartImageNetBBOXDataset, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(False)
        self.aux_target_hacks = aux_target_hacks

        with open(class_label_file, "r") as openfile:
            self.image_to_label = json.load(openfile)

        self.classes = list(sorted(set(self.image_to_label.values())))
        self.num_classes = len(self.classes)

    def change_hack_attr(self, hackclassname, attrkv_dict):
        target_class = dataset_hook_register[hackclassname]
        for item in self.aux_target_hacks:
            if isinstance(item, target_class):
                for k, v in attrkv_dict.items():
                    setattr(item, k, v)

    def get_hack(self, hackclassname):
        target_class = dataset_hook_register[hackclassname]
        for item in self.aux_target_hacks:
            if isinstance(item, target_class):
                return item

    def __getitem__(self, idx):
        """
        Output:
            - target: dict of multiple items
                - boxes: Tensor[num_box, 4]. \
                    Init type: x0,y0,x1,y1. unnormalized data.
                    Final type: cx,cy,w,h. normalized data. 
        """
        try:
            img, target = super(PartImageNetBBOXDataset, self).__getitem__(idx)
        except:
            print("Error idx: {}".format(idx))
            idx += 1
            img, target = super(PartImageNetBBOXDataset, self).__getitem__(idx)

        image_id = self.ids[idx]
        target = {"image_id": image_id, "annotations": target}
        img, target = self.prepare(img, target)

        if self._transforms is not None:
            img, target = self._transforms(img, target)
        # convert to needed format
        if self.aux_target_hacks is not None:
            for hack_runner in self.aux_target_hacks:
                target, img = hack_runner(target, img=img)

        class_label = self.image_to_label[str(idx)]
        return img, class_label, target


# from DINO.datasets.coco import CocoDetection

# class PartImageNetBBOXDataset(CocoDetection):
#     def __init__(
#         self,
#         img_folder,
#         class_label_file,
#         ann_file,
#         transforms,
#         aux_target_hacks=None
#     ):
#         super(PartImageNetBBOXDataset, self).__init__(img_folder, ann_file, transforms, False, aux_target_hacks)

#         with open(class_label_file, 'r') as openfile:
#             self.image_to_label = json.load(openfile)

#         self.classes = list(sorted(set(self.image_to_label.values())))
#         self.num_classes = len(self.classes)

#     def __getitem__(self, index):
#         img, target = super().__getitem__(index)
#         # print(type(img))
#         # qq
#         class_label = self.image_to_label[str(index)]
#         return img, target, class_label


def get_loader_sampler_bbox(args, transforms, split):
    is_train = split == "train"

    # TODO: add as arg
    root = Path(args.bbox_label_dir)
    # root = Path('/data/shared/PartImageNet/PartBoxSegmentations')
    # root = Path('/data1/chawins/PartImageNet/PartBoxSegmentations')
    # root = Path('/global/scratch/users/nabeel126/PartImageNet/PartBoxSegmentations')

    if not args.sample:
        PATHS = {
            "train": (
                root / "train",
                root / "image_labels" / "train.json",
                root / "annotations" / "train.json",
            ),
            "val": (
                root / "val",
                root / "image_labels" / "val.json",
                root / "annotations" / "val.json",
            ),
            "test": (
                root / "test",
                root / "image_labels" / "test.json",
                root / "annotations" / "test.json",
            ),
        }
    else:
        PATHS = {
            "train": (
                root / "train",
                root / "image_labels" / "train_sample.json",
                root / "annotations" / "train_sample.json",
            ),
            "val": (
                root / "val",
                root / "image_labels" / "val_sample.json",
                root / "annotations" / "val_sample.json",
            ),
            "test": (
                root / "test",
                root / "image_labels" / "test_sample.json",
                root / "annotations" / "test_sample.json",
            ),
        }

    img_folder, class_label_file, ann_file = PATHS[split]

    # part_imagenet_dataset = PartImageNetBBOXDataset(
    #     img_folder,
    #     class_label_file,
    #     ann_file,
    #     transforms,
    #     aux_target_hacks=None
    # )

    # add some hooks to datasets
    aux_target_hacks_list = get_aux_target_hacks_list(split, args)
    part_imagenet_dataset = PartImageNetBBOXDataset(
        img_folder,
        class_label_file,
        ann_file,
        transforms,
        aux_target_hacks=aux_target_hacks_list,
    )

    sampler = None
    if args.distributed:
        shuffle = None
        if is_train:
            sampler = torch.utils.data.distributed.DistributedSampler(
                part_imagenet_dataset,
                shuffle=True,
                seed=args.seed,
                drop_last=False,
            )
        else:
            # Use distributed sampler for validation but not testing
            sampler = DistributedEvalSampler(
                part_imagenet_dataset, shuffle=False, seed=args.seed
            )
    else:
        # shuffle = is_train
        shuffle = True

    from DINO.util.misc import collate_fn

    batch_size = args.batch_size
    loader = torch.utils.data.DataLoader(
        part_imagenet_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
        collate_fn=collate_fn,  # turns images into NestedTensors
    )

    # from DINO.util.misc import collate_fn
    # batch_size = args.batch_size
    # loader = torch.utils.data.DataLoader(
    #     part_imagenet_dataset,
    #     batch_size=batch_size,
    #     shuffle=shuffle,
    #     num_workers=args.workers,
    #     pin_memory=True,
    #     sampler=sampler,
    #     drop_last=is_train
    # )

    # TODO: can we make this cleaner?
    # PART_IMAGENET["part_to_class"] = part_imagenet_dataset.part_to_class
    PART_IMAGENET_BBOX["num_classes"] = part_imagenet_dataset.num_classes
    # PART_IMAGENET["num_seg_labels"] = part_imagenet_dataset.num_seg_labels

    setattr(args, "num_classes", part_imagenet_dataset.num_classes)
    # pto = part_imagenet_dataset.part_to_object
    # if seg_type == "part":
    #     seg_labels = len(pto)
    # elif seg_type == "fg":
    #     seg_labels = 2
    # else:
    #     seg_labels = pto.max().item() + 1
    # setattr(args, "seg_labels", seg_labels)

    return loader, sampler


import DINO.datasets.transforms as T


def make_coco_transforms(
    image_set, fix_size=False, strong_aug=False, args=None
):
    normalize = T.Compose(
        [
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # config the params for data aug
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    max_size = 1333
    scales2_resize = [400, 500, 600]
    scales2_crop = [384, 600]

    # update args from config files
    scales = getattr(args, "data_aug_scales", scales)
    max_size = getattr(args, "data_aug_max_size", max_size)
    scales2_resize = getattr(args, "data_aug_scales2_resize", scales2_resize)
    scales2_crop = getattr(args, "data_aug_scales2_crop", scales2_crop)

    # resize them
    data_aug_scale_overlap = getattr(args, "data_aug_scale_overlap", None)
    if data_aug_scale_overlap is not None and data_aug_scale_overlap > 0:
        data_aug_scale_overlap = float(data_aug_scale_overlap)
        scales = [int(i * data_aug_scale_overlap) for i in scales]
        max_size = int(max_size * data_aug_scale_overlap)
        scales2_resize = [
            int(i * data_aug_scale_overlap) for i in scales2_resize
        ]
        scales2_crop = [int(i * data_aug_scale_overlap) for i in scales2_crop]
    # else:
    #     scales = getattr(args, 'data_aug_scales', scales)
    #     max_size = getattr(args, 'data_aug_max_size', max_size)
    #     scales2_resize = getattr(args, 'data_aug_scales2_resize', scales2_resize)
    #     scales2_crop = getattr(args, 'data_aug_scales2_crop', scales2_crop)

    datadict_for_print = {
        "scales": scales,
        "max_size": max_size,
        "scales2_resize": scales2_resize,
        "scales2_crop": scales2_crop,
    }
    print("data_aug_params:", json.dumps(datadict_for_print, indent=2))

    if image_set == "train":
        if fix_size:
            return T.Compose(
                [
                    T.RandomHorizontalFlip(),
                    T.RandomResize([(max_size, max(scales))]),
                    # T.RandomResize([(512, 512)]),
                    normalize,
                ]
            )

        if strong_aug:
            import DINO.datasets.sltransform as SLT

            return T.Compose(
                [
                    T.RandomHorizontalFlip(),
                    T.RandomSelect(
                        T.RandomResize(scales, max_size=max_size),
                        T.Compose(
                            [
                                T.RandomResize(scales2_resize),
                                T.RandomSizeCrop(*scales2_crop),
                                T.RandomResize(scales, max_size=max_size),
                            ]
                        ),
                    ),
                    SLT.RandomSelectMulti(
                        [
                            SLT.RandomCrop(),
                            # SLT.Rotate(10),
                            SLT.LightingNoise(),
                            SLT.AdjustBrightness(2),
                            SLT.AdjustContrast(2),
                        ]
                    ),
                    # # for debug only
                    # SLT.RandomCrop(),
                    # SLT.LightingNoise(),
                    # SLT.AdjustBrightness(2),
                    # SLT.AdjustContrast(2),
                    # SLT.Rotate(10),
                    normalize,
                ]
            )

        return T.Compose(
            [
                T.RandomHorizontalFlip(),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=max_size),
                    T.Compose(
                        [
                            T.RandomResize(scales2_resize),
                            T.RandomSizeCrop(*scales2_crop),
                            T.RandomResize(scales, max_size=max_size),
                        ]
                    ),
                ),
                normalize,
            ]
        )

    if image_set in ["val", "eval_debug", "train_reg", "test"]:

        if os.environ.get("GFLOPS_DEBUG_SHILONG", False) == "INFO":
            print("Under debug mode for flops calculation only!!!!!!!!!!!!!!!!")
            return T.Compose(
                [
                    T.ResizeDebug((1280, 800)),
                    normalize,
                ]
            )

        return T.Compose(
            [
                T.RandomResize([max(scales)], max_size=max_size),
                normalize,
            ]
        )

    raise ValueError(f"unknown {image_set}")


def load_part_imagenet_bbox(args):
    try:
        strong_aug = args.strong_aug
    except:
        strong_aug = False

    train_transforms = make_coco_transforms(
        "train", fix_size=args.fix_size, strong_aug=strong_aug, args=args
    )
    train_loader, train_sampler = get_loader_sampler_bbox(
        args, train_transforms, "train"
    )

    val_transforms = make_coco_transforms(
        "val", fix_size=args.fix_size, strong_aug=False, args=args
    )
    val_loader, _ = get_loader_sampler_bbox(args, val_transforms, "val")
    test_loader, _ = get_loader_sampler_bbox(args, val_transforms, "test")

    return train_loader, train_sampler, val_loader, test_loader


PART_IMAGENET_BBOX = {
    "normalize": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    },
    "loader": load_part_imagenet_bbox,
    "input_dim": (3, 224, 224),
    "colormap": COLORMAP,
}
