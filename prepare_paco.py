"""
Code is adapted from https://github.com/micco00x/py-pascalpart

Usage examples:
python prepare_pascal_part_v3.py --data-dir ~/data/pascal_part/ --name name
"""
import argparse
from collections import defaultdict, OrderedDict
import copy
import cv2
import json
import glob
import os
import random

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import PIL
from PIL import Image
import pycocotools.mask as mask_util
from tqdm import tqdm

OBJ_CLASSES = ['trash_can','handbag','ball','basket','bicycle','book','bottle','bowl','can','car_(automobile)','carton','cellular_telephone','chair','cup','dog','drill','drum_(musical_instrument)','glass_(drink_container)','guitar','hat','helmet','jar','knife','laptop_computer','mug','pan_(for_cooking)','plate','remote_control','scissors','shoe','slipper_(footwear)','stool','table','towel','wallet','watch','wrench','belt','bench','blender','box','broom','bucket','calculator','clock','crate','earphone','fan','hammer','kettle','ladder','lamp','microwave_oven','mirror','mouse_(computer_equipment)','napkin','newspaper','pen','pencil','pillow','pipe','pliers','plastic_bag','scarf','screwdriver','soap','sponge','spoon','sweater','tape_(sticky_cloth_or_paper)','telephone','television_set','tissue_paper','tray','vase']
assert len(OBJ_CLASSES) == 75

PART_CLASSES = ['antenna','apron','arm','back','back_cover','backstay','bar','barrel','base','base_panel','basket','bezel','blade','body','border','bottom','bowl','bracket','bridge','brush','brush_cap','buckle','bulb','bumper','button','cable','camera','canopy','cap','capsule','case','clip','closure','colied_tube','control_panel','cover','cuff','cup','decoration','dial','door_handle','down_tube','drawer','drawing','ear','ear_pads','embroidery','end_tip','eraser','eye','eyelet','face','face_shield','fan_box','fender','ferrule','finger_hole','fingerboard','finial','flap','food_cup','foot','footrest','fork','frame','fringes','gear','grille','grip','hand','handle','handlebar','head','head_tube','headband','headlight','headstock','heel','hem','hole','hood','housing','inner_body','inner_side','inner_wall','insole','jaw','joint','key','keyboard','label','lace','lead','left_button','leg','lid','light','lining','logo','loop','lower_bristles','lug','mirror','motor','mouth','neck','neckband','nose','nozzle','nozzle_stem','outer_side','outsole','page','pedal','pedestal_column','pediment','pickguard','pipe','pom_pom','prong','pull_tab','punt','push_pull_cap','quarter','rail','right_button','rim','ring','rod','roll','roof','rough_surface','runningboard','saddle','screen','screw','scroll_wheel','seal_ring','seat','seat_stay','seat_tube','shade','shade_cap','shade_inner_side','shaft','shank','shelf','shoulder','side','side_button','sign','sipper','skirt','sleeve','slider','spindle','splashboard','spout','steeringwheel','stem','step','sticker','stile','strap','stretcher','string','switch','swivel','table_top','tail','taillight','tank','tapering_top','teeth','terry_bar','text','throat','time_display','tip','toe_box','tongue','top','top_cap','top_tube','touchpad','trunk','turnsignal','turntable','vamp','vapour_cover','visor','welt','wheel','window','windowpane','windshield','wiper','wire','yoke','zip']
assert len(PART_CLASSES) == 200

# https://en.wikipedia.org/wiki/YUV#SDTV_with_BT.601
_M_RGB2YUV = [[0.299, 0.587, 0.114], [-0.14713, -0.28886, 0.436], [0.615, -0.51499, -0.10001]]
_M_YUV2RGB = [[1.0, 0.0, 1.13983], [1.0, -0.39465, -0.58060], [1.0, 2.03211, 0.0]]

# https://www.exiv2.org/tags.html
_EXIF_ORIENT = 274  # exif 'Orientation' tag

cell_size = 256

def _get_box_from_bin_mask(bin_mask):
    box_mask = np.zeros_like(bin_mask)
    if bin_mask.sum() == 0:
        return box_mask
    y, x = np.where(bin_mask)
    ymin, ymax = y.min(), y.max()
    xmin, xmax = x.min(), x.max()
    box_mask[ymin : ymax + 1, xmin : xmax + 1] = 1
    return box_mask

def get_obj_and_part_anns(annotations):
    """
    Returns a map between an object annotation ID and 
    (object annotation, list of part annotations) pair.
    """
    obj_ann_id_to_anns = {ann["id"]: (ann, []) for ann in annotations if ann["id"] == ann["obj_ann_id"]}
    for ann in annotations:
        if ann["id"] != ann["obj_ann_id"]:
            obj_ann_id_to_anns[ann["obj_ann_id"]][1].append(ann)
    return obj_ann_id_to_anns
    
def expand_bounding_box(box, factor, im_height, im_width):
    """
    Expands a bounding box by the specified factor.
    Args:
        box:        (4, ) NumPy array with bounding box in (left, top, width, height)
                    format
        factor:     Expansion factor (e.g., 1.5)
        im_height:  Image height
        im_width:   Image width

    Returns:
        expanded_box: (4, ) NumPy array with expanded bounding box
    """
    # Compute the center of the bounding box
    center_x = box[0] + box[2] / 2
    center_y = box[1] + box[3] / 2

    # Compute the new width and height of the bounding box
    new_width = box[2] * factor
    new_height = box[3] * factor

    # Compute the new left and top coordinates of the bounding box
    new_left = center_x - new_width / 2
    new_top = center_y - new_height / 2

    # Make sure the new bounding box does not exceed the image dimensions
    new_right = min(new_left + new_width, im_width)
    new_bottom = min(new_top + new_height, im_height)

    new_left = max(new_left, 0)
    new_top = max(new_top, 0)

    # Return the expanded bounding box
    return np.array([new_left, new_top, new_right - new_left, new_bottom - new_top])

def crop_and_resize(img, obj_bbox, part_bbox, crop_size):
    x, y, w, h = obj_bbox
    crop_img = img[int(y) : int(y + h), int(x) : int(x + w)]
    
    x_part, y_part, w_part, h_part = part_bbox
    x_part -= x
    y_part -= y
    max_side = max(w, h)
    resize_factor = crop_size / max_side
    new_w, new_h = round(resize_factor * w), round(resize_factor * h)
    
    crop_img = cv2.resize(crop_img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    
    # update the part_bbox coordinates relative to the cropped image
    x_part_cropped, y_part_cropped = round(x_part * resize_factor), round(y_part * resize_factor)
    w_part_cropped, h_part_cropped = round(w_part * resize_factor), round(h_part * resize_factor)
    part_bbox_cropped = (x_part_cropped, y_part_cropped, w_part_cropped, h_part_cropped)
    
    return crop_img, part_bbox_cropped

def _apply_exif_orientation(image):
    """
    Applies the exif orientation correctly.

    This code exists per the bug:
    https://github.com/python-pillow/Pillow/issues/3973
    with the function `ImageOps.exif_transpose`. The Pillow source raises errors with
    various methods, especially `tobytes`

    Function based on:
    https://github.com/wkentaro/labelme/blob/v4.5.4/labelme/utils/image.py#L59
    https://github.com/python-pillow/Pillow/blob/7.1.2/src/PIL/ImageOps.py#L527

    Args:
        image (PIL.Image): a PIL image

    Returns:
        (PIL.Image): the PIL image with exif orientation applied, if applicable
    """
    if not hasattr(image, "getexif"):
        return image

    try:
        exif = image.getexif()
    except Exception:  # https://github.com/facebookresearch/detectron2/issues/1885
        exif = None

    if exif is None:
        return image

    orientation = exif.get(_EXIF_ORIENT)

    method = {
        2: Image.FLIP_LEFT_RIGHT,
        3: Image.ROTATE_180,
        4: Image.FLIP_TOP_BOTTOM,
        5: Image.TRANSPOSE,
        6: Image.ROTATE_270,
        7: Image.TRANSVERSE,
        8: Image.ROTATE_90,
    }.get(orientation)

    if method is not None:
        return image.transpose(method)
    return image

def convert_PIL_to_numpy(image, format):
    """
    Convert PIL image to numpy array of target format.

    Args:
        image (PIL.Image): a PIL image
        format (str): the format of output image

    Returns:
        (np.ndarray): also see `read_image`
    """
    if format is not None:
        # PIL only supports RGB, so convert to RGB and flip channels over below
        conversion_format = format
        if format in ["BGR", "YUV-BT.601"]:
            conversion_format = "RGB"
        image = image.convert(conversion_format)
    image = np.asarray(image)
    # PIL squeezes out the channel dimension for "L", so make it HWC
    if format == "L":
        image = np.expand_dims(image, -1)

    # handle formats not supported by PIL
    elif format == "BGR":
        # flip channels if needed
        image = image[:, :, ::-1]
    elif format == "YUV-BT.601":
        image = image / 255.0
        image = np.dot(image, np.array(_M_RGB2YUV).T)

    return image


def read_image(file_name, format=None):
    """
    Read an image into the given format.
    Will apply rotation and flipping if the image has such exif information.

    Args:
        file_name (str): image file path
        format (str): one of the supported image modes in PIL, or "BGR" or "YUV-BT.601".

    Returns:
        image (np.ndarray):
            an HWC image in the given format, which is 0-255, uint8 for
            supported image modes in PIL or "BGR"; float (0-1 for Y) for YUV-BT.601.
    """
    image = Image.open(file_name)
    image = _apply_exif_orientation(image)
    return convert_PIL_to_numpy(image, format)
    
def plot_metaclass_counts(object_class_to_count, plot_name):
    train_counts = [object_class_to_count["train"][class_name] for class_name in OBJ_CLASSES]
    val_counts = [object_class_to_count["val"][class_name] for class_name in OBJ_CLASSES]
    test_counts = [object_class_to_count["test"][class_name] for class_name in OBJ_CLASSES]

    plt.figure(figsize=(20, 16))
    plt.bar(OBJ_CLASSES, train_counts, label='train')
    plt.bar(OBJ_CLASSES, val_counts, bottom=train_counts, label='val')
    plt.bar(OBJ_CLASSES, test_counts, bottom=[sum(x) for x in zip(train_counts, val_counts)], label='test')

    # add space below chart for labels
    plt.subplots_adjust(bottom=0.2)
    plt.xlabel('Class Names')
    plt.ylabel('Counts')
    plt.xticks(rotation=90)
    plt.legend()
    plt.grid()
    plt.savefig(plot_name)

def plot_image_and_bbox(img, bbox, filename):
    fig, ax = plt.subplots()
    ax.imshow(img)
    rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.savefig(filename)

def annToRLE(ann, h, w):
    """
    Convert annotation which can be polygons, uncompressed RLE to RLE.
    :return: binary mask (numpy 2D array)
    """
    segm = ann['segmentation']
    if type(segm) == list:
        if len(segm) == 0:
            return None
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = mask_util.frPyObjects(segm, h, w)
        rle = mask_util.merge(rles)
    elif type(segm['counts']) == list:
        # uncompressed RLE
        rle = mask_util.frPyObjects(segm, h, w)
    else:
        # rle
        rle = ann['segmentation']
    return rle

# Load annotations from the annotation folder of PASCAL-Part dataset:
if __name__ == "__main__":
    # Parse arguments from command line:
    parser = argparse.ArgumentParser(
        description="Prepare PACO dataset for classification tasks"
    )
    parser.add_argument(
        "--data-dir", default="~/data/", type=str, help="Path to dataset"
    )
    parser.add_argument(
        "--name", default="temp", type=str, help="Name the new part dataset"
    )
    parser.add_argument("--seed", default=0, type=int, help="Random seed")
    parser.add_argument("--use-box-seg", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--bbox-expand-factor", default=1.0, type=float, help="Factor by which to expand bounding boxes")
    parser.add_argument("--ignore-objects-without-parts", action="store_true", help="Ignore objects without parts")
    parser.add_argument('--split_ratios', type=str, default='0.8,0.1,0.1',
                        help='comma-separated split ratios for train, validation and test sets')
    parser.add_argument("--sample", action="store_true", help="Use a subset of PACO - split ratios do not have to sum to 1")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    image_root_dir = os.path.join(args.data_dir)

    # Extract the split ratios from the string
    train_split, val_split, test_split = [float(split_ratio) for split_ratio in args.split_ratios.split(',')]

    # define new train/val/test splits
    if args.sample:
        assert train_split + val_split + test_split < 1
    else:
        assert train_split + val_split + test_split - 1 < 1e-5
    

    obj_id_to_obj_class = {}
    img_id_to_obj_id = {}

    # if args.debug:
    object_class_to_count = {}
    for partition in ["train", "val", "test"]:
        object_class_to_count[partition] = OrderedDict()
        for obj_class in OBJ_CLASSES:
            object_class_to_count[partition][obj_class] = 0
    
    # collect all data
    obj_cat_id_to_name = {}
    part_cat_id_to_name = {}
    obj_cat_id_to_part_cat_id = defaultdict(set)

    num_objs = 0
    num_objs_without_parts = 0
    small_object_count = 0

    seen_obj_ids = set()
    seen_part_ids = set()

    # for dataset_name in ["lvis"]:
    for dataset_name in ["lvis", "ego4d"]:
        for partition in ["train", "val", "test"]:
            if dataset_name == "ego4d" and partition == "test":
                # skip since no segmentation masks
                continue

            dataset_file_name = os.path.join(args.data_dir, f"paco_{dataset_name}_v1_{partition}.json")
            with open(dataset_file_name) as f:
                dataset = json.load(f)

            # create dictionary mapping image_id to image dimensions
            image_id_to_image_dims = {}
            for ann in dataset['images']:
                image_id_to_image_dims[ann['id']] = [ann['height'], ann['width']] # height, width

            obj_ann_id_to_anns = get_obj_and_part_anns(dataset["annotations"])

            cat_id_to_name = {d["id"]: d["name"] for d in dataset["categories"]}

            for obj_id in obj_ann_id_to_anns:
                obj_ann, part_anns = obj_ann_id_to_anns[obj_id]

                if args.ignore_objects_without_parts and len(part_anns) == 0:
                    continue

                image_id = obj_ann["image_id"]
                obj_cat_id = obj_ann["category_id"]
                bbox = obj_ann["bbox"]

                # if object is less than 1% of the total image, skip object
                image_height, image_width = image_id_to_image_dims[image_id]
                if (bbox[2] * bbox[3]) / (image_height * image_width) < 0.01:
                    small_object_count += 1
                    continue
                
                obj_class = cat_id_to_name[obj_cat_id]

                seen_obj_ids.add(obj_cat_id)
                for part_ann in part_anns:
                    part_cat_id = part_ann["category_id"]
                    obj_cat_id_to_part_cat_id[obj_cat_id].add(part_cat_id)
                    seen_part_ids.add(part_cat_id)
                
                obj_id_to_obj_class[(dataset_name, partition, obj_id)] = [obj_class, None]

                if (dataset_name, partition, image_id) not in img_id_to_obj_id:
                    img_id_to_obj_id[(dataset_name, partition, image_id)] = []    
                img_id_to_obj_id[(dataset_name, partition, image_id)].append(obj_id)

                num_objs += 1
                if len(part_anns) == 0:
                    num_objs_without_parts += 1
                object_class_to_count[partition][obj_class] += 1

            print(f"[INFO] Dataset {dataset_name}-{partition} num objects: {len(obj_ann_id_to_anns)} \n")


    for dataset_name in ["lvis", "ego4d"]:
        for partition in ["train", "val", "test"]:
            dataset_file_name = os.path.join(args.data_dir, f"paco_{dataset_name}_v1_{partition}.json")
            if os.path.exists(dataset_file_name):
                with open(dataset_file_name) as f:
                    dataset = json.load(f)

                if not obj_cat_id_to_name or not part_cat_id_to_name:
                    
                    for d in dataset["categories"]:
                        if d["supercategory"] == 'OBJECT':
                            if d['id'] in seen_obj_ids:
                                obj_cat_id_to_name[d["id"]] = d["name"]
                        elif d["supercategory"] == 'PART':
                            if d['id'] in seen_part_ids:
                                part_cat_id_to_name[d["id"]] = d["name"]
                break

    print('[INFO] Total Number of objects', num_objs)
    print('[INFO] Total Number small objects', small_object_count)
    print('[INFO] Total Number of objects without part annotations', num_objs_without_parts)
    print('[INFO] Proportion of objects without part annotations', num_objs_without_parts/num_objs)

    plot_metaclass_counts(object_class_to_count, plot_name="test_chart.png")

    if args.debug:
        for obj_cat_id in obj_cat_id_to_part_cat_id:
            print('obj name: ', obj_cat_id_to_name[obj_cat_id])
            for part_cat_id in obj_cat_id_to_part_cat_id[obj_cat_id]:
                print('part name:', part_cat_id_to_name[part_cat_id])
            print()

    # repartition data into previously define splits
    all_obj_ids = list(obj_id_to_obj_class.keys())
    num_samples = len(all_obj_ids)
    np.random.shuffle(all_obj_ids)
    num_train, num_val, num_test = int(train_split * num_samples), int(val_split * num_samples), int(test_split * num_samples)
    val_idx = all_obj_ids[:num_val]
    test_idx = all_obj_ids[num_val : num_val + num_test]
    train_idx = all_obj_ids[num_val + num_test: num_val + num_test + num_train]

    print('num train', num_train)
    print('num val', num_val)
    print('num test', num_test)
        
    # reset count dict
    object_class_to_count = {}
    for partition in ["train", "val", "test"]:
        object_class_to_count[partition] = OrderedDict()
        for obj_class in OBJ_CLASSES:
            object_class_to_count[partition][obj_class] = 0

    # populate count dict for new data split
    for new_partition, indices in zip(["train", "val", "test"], [train_idx, val_idx, test_idx]):
        for dataset_name, partition, obj_id in indices:
            obj_class, _ = obj_id_to_obj_class[(dataset_name, partition, obj_id)]
            obj_id_to_obj_class[(dataset_name, partition, obj_id)] = [obj_class, new_partition] # add new partition 
            object_class_to_count[new_partition][obj_class] += 1

    # if args.debug:
    plot_metaclass_counts(object_class_to_count, plot_name="test_chart_reshuffled.png")
    
    # map new object id to object name
    assert len(obj_cat_id_to_name) <= 75
    categories_ann = []
    for ci, (obj_cat_id, name) in enumerate(obj_cat_id_to_name.items()):
        ann = {"id": ci, "name": name}
        categories_ann.append(ann)
    print('Number of Object in Dataset', len(obj_cat_id_to_name))

    assert len(part_cat_id_to_name) <= 456
    # map new part id to object name
    part_cat_id_to_seg_mask_id = {}
    part_categories_ann = []
    ann = {"id": 0, "name": "background"} # 0 is background
    part_categories_ann.append(ann)
    for ci, (part_cat_id, name) in enumerate(part_cat_id_to_name.items()):
        part_cat_id_to_seg_mask_id[part_cat_id] = ci + 1 # add 1 for background
        ann = {"id": ci + 1, "name": name} # add 1 for background
        # print(ann)
        part_categories_ann.append(ann)
    print('Number of Parts in Dataset', len(part_categories_ann))
    UNLABELED_ID = -1

    # TODO: save to json file instead?
    part_to_object = [0] * len(part_categories_ann) 
    for obj_cat_id in obj_cat_id_to_part_cat_id:
        obj_name = obj_cat_id_to_name[obj_cat_id]
        for part_cat_id in obj_cat_id_to_part_cat_id[obj_cat_id]:
            part_to_object[part_cat_id_to_seg_mask_id[part_cat_id]] = OBJ_CLASSES.index(obj_name)
       
    new_ann_data = {}
    for partition in ["train", "test", "val"]:
        new_ann_data[partition] = {}
        new_ann_data[partition]['images'] = []
        new_ann_data[partition]['annotations'] = []
        new_ann_data[partition]['categories'] = categories_ann
        new_ann_data[partition]['part_categories'] = part_categories_ann

    # create folders to save images and segmentation masks
    for partition in ["train", "val", "test"]:
        data_path = os.path.join(args.data_dir, "PartSegmentations", args.name, partition, "images")
        seg_masks_path = os.path.join(args.data_dir, "PartSegmentations", args.name, partition, "seg_masks")
        os.makedirs(data_path, exist_ok=True)
        os.makedirs(seg_masks_path, exist_ok=True)
    
    # keep track of obj and part count in new data repartition
    partition_to_obj_count = {}
    partition_to_part_count = {}
    for partition in ["train", "val", "test"]:
        partition_to_obj_count[partition] = 0
        partition_to_part_count[partition] = 0

    exit_loop = False
    
    print("[INFO] Saving images and segmentation masks")
    for partition in ["train", "val", "test"]:
        for dataset_name in ["lvis", "ego4d"]:
            if dataset_name == "ego4d" and partition == "test":
                # skip since no segmentation masks
                continue
        
            dataset_file_name = os.path.join(args.data_dir, f"paco_{dataset_name}_v1_{partition}.json")

            # Load dataset.
            with open(dataset_file_name) as f:
                dataset = json.load(f)

            # Extract maps from dataset.
            cat_id_to_name = {d["id"]: d["name"] for d in dataset["categories"]}
            image_id_to_image_file_name = {d["id"]: os.path.join(image_root_dir, dataset_name, d["file_name"]) for d in dataset["images"]}
            obj_ann_id_to_anns = get_obj_and_part_anns(dataset["annotations"])
                
            for obj_ann_id in tqdm(obj_ann_id_to_anns):
                ann, part_anns = obj_ann_id_to_anns[obj_ann_id]

                image_id = ann["image_id"]
                obj_id = ann["id"]
                obj_cat_id = ann["category_id"]
                orig_bbox = ann["bbox"]

                # get obj class and new assigned partition
                if (dataset_name, partition, obj_id) not in obj_id_to_obj_class:
                    # skipping small objects
                    continue
                obj_class, new_partition = obj_id_to_obj_class[(dataset_name, partition, obj_id)]

                if not new_partition:
                    continue

                if args.sample:
                    if new_partition == 'train' and partition_to_obj_count['train'] > num_train:
                        continue
                    elif new_partition == 'val' and partition_to_obj_count['val'] > num_val:
                        continue
                    elif new_partition == 'test' and partition_to_obj_count['test'] > num_test:
                        continue

                # Read the image.
                im_fn = image_id_to_image_file_name[image_id]
                img = read_image(im_fn, format="RGB")
                obj_name = cat_id_to_name[obj_cat_id]
            
                # generate segmentation for whole image, even if there are multiple objects
                orig_image_height, orig_image_width, _ = img.shape

                image_seg_mask = np.zeros((orig_image_height, orig_image_width), dtype=np.int8)
                # get all objects from image
                img_obj_ids = img_id_to_obj_id[(dataset_name, partition, image_id)]
                for img_obj_id in img_obj_ids:
                    image_obj_ann, image_part_anns = obj_ann_id_to_anns[img_obj_id]
                    # no part specific part masks
                    if len(image_part_anns) == 0:
                        if args.ignore_objects_without_parts:
                            continue
                        rle = annToRLE(image_obj_ann, h=orig_image_height, w=orig_image_width)
                        if rle:
                            cat_id = UNLABELED_ID
                            # cat_id = -1 
                            part_mask = mask_util.decode(rle)
                            image_seg_mask = part_mask * cat_id + (1 - part_mask) * image_seg_mask
                    else:
                        for image_part_ann in image_part_anns:
                            # part_id = partition_to_part_count[new_partition]
                            image_obj_cat_id = image_obj_ann["category_id"]
                            rle = image_part_ann["segmentation"]
                            cat_id = image_part_ann["category_id"]
                            assert cat_id in obj_cat_id_to_part_cat_id[image_obj_cat_id]
                            # cat_id = obj_cat_id_to_seg_mask_id[image_obj_cat_id]
                            cat_id = part_cat_id_to_seg_mask_id[cat_id]
                            part_mask = mask_util.decode(rle)
                            image_seg_mask = part_mask * cat_id + (1 - part_mask) * image_seg_mask

                image_seg_mask = image_seg_mask.astype(np.int16)
                if args.debug:
                    # Plot the image
                    Image.fromarray(img).save("test_mask.png")
                    # Plot the segmentation mask
                    plt.imsave('test_mask.png', image_seg_mask, cmap='viridis')
                
                new_obj_id = partition_to_obj_count[new_partition]
                
                # Square the box and expand it by a factor of args.bbox_expand_factor.
                bbox = expand_bounding_box(orig_bbox, args.bbox_expand_factor, orig_image_height, orig_image_width)
                
                assert bbox[0] >= 0
                assert bbox[1] >= 0
                assert bbox[0] + bbox[2] <= orig_image_width
                assert bbox[1] + bbox[3] <= orig_image_height
                if bbox[2] < 1 or bbox[3] < 1:
                    print("Mask too small, skipping...")
                    continue
                
                # Crop and resize to fixed max side both the image and the mask.
                img_crop, _ = crop_and_resize(img, bbox, bbox, cell_size)
                
                # Turn annotation to mask
                img_height, img_width, _ = img_crop.shape
                seg_mask = np.zeros((img_height, img_width), dtype=np.int8)
                
                if len(part_anns) == 0:
                    # handle cases where there are no part annotations
                    part_id = partition_to_part_count[new_partition]                    
                    cat_id = ann["category_id"]
                    obj_name = cat_id_to_name[cat_id]
                    cat_id = UNLABELED_ID # no part specific part masks
                    seg_mask, part_bbox = crop_and_resize(image_seg_mask, bbox, orig_bbox, cell_size)
                    area = part_bbox[2] * part_bbox[3]

                    part_ann = {
                        "id" : part_id,
                        "image_id" : new_obj_id,
                        "category_id": cat_id,
                        "area" : area,
                        "bbox" : part_bbox, # [x,y,width,height],
                        "iscrowd": 0,
                    }
                    new_ann_data[new_partition]['annotations'].append(part_ann)
                    partition_to_part_count[new_partition] += 1

                else:
                    for part_ann in part_anns:
                        part_id = partition_to_part_count[new_partition]
                        cat_id = part_ann["category_id"]
                        assert cat_id in obj_cat_id_to_part_cat_id[obj_cat_id]
                        part_name = cat_id_to_name[cat_id] 
                        # print('part_name', part_name)
                        cat_id = part_cat_id_to_seg_mask_id[cat_id]
                        part_bbox = part_ann["bbox"]
                        seg_mask, part_bbox = crop_and_resize(image_seg_mask, bbox, part_bbox, cell_size)
                        area = part_bbox[2] * part_bbox[3]
    
                        part_ann = {
                            "id" : part_id,
                            "image_id" : new_obj_id,
                            "category_id": cat_id,
                            "area" : area,
                            "bbox" : part_bbox, # [x,y,width,height],
                            "iscrowd": part_ann["iscrowd"],
                        }

                        new_ann_data[new_partition]['annotations'].append(part_ann)
                        partition_to_part_count[new_partition] += 1

                        if args.debug:
                            plot_image_and_bbox(img_crop, part_bbox, "test_mask.png")
                
                assert seg_mask.max() <= len(part_categories_ann)
                assert seg_mask.min() >= -1
                
                # save cropped obj from image
                data_path = os.path.join(args.data_dir, "PartSegmentations", args.name, new_partition, "images")
                image_filename = "{:07d}.png".format(new_obj_id)
                image_path = os.path.join(data_path, image_filename)
                Image.fromarray(img_crop).save(image_path)

                # save segmentation mask for obj
                seg_masks_path = os.path.join(args.data_dir, "PartSegmentations", args.name, new_partition, "seg_masks")
                mask_filename = "{:07d}.tif".format(new_obj_id)
                seg_mask_path = os.path.join(seg_masks_path, mask_filename)
                    
                Image.fromarray(seg_mask).save(seg_mask_path)

                image_ann = {
                    "id" : new_obj_id,
                    "supercategory" : obj_class,
                    "width" : img_width,
                    "height" : img_height,
                    "file_name" : image_filename,
                    "seg_filename" : mask_filename,
                }
                new_ann_data[new_partition]['images'].append(image_ann)
                partition_to_obj_count[new_partition] += 1

                if args.sample and partition_to_obj_count['train'] > num_train and partition_to_obj_count['val'] > num_val and partition_to_obj_count['test'] > num_test:
                    exit_loop = True
                    break
            if exit_loop:
                break
        if exit_loop:
                break
        
    # save annotations in COCO format            
    ann_file_path = os.path.join(
        args.data_dir, "PartSegmentations", args.name
    )
    for partition in ["train", "val", "test"]:
        ann_filename = os.path.join(ann_file_path, f"{partition}.json")
        with open(ann_filename, 'w') as f:
            json.dump(new_ann_data[partition], f)

        # print statistics
        print(f"[INFO] Partition: {partition}")
        print(f"[INFO] Num. Images: {partition_to_obj_count[partition]}")
        print(f"[INFO] Num. Annotations: {partition_to_part_count[partition]}")
        print()