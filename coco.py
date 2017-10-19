"""
Mask R-CNN
Configurations and data loading code for MS COCO.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import os
import numpy as np

# Download and install the Python COCO tools from https://github.com/waleedka/coco
# That's a fork from the original https://github.com/pdollar/coco with a bug
# fix for Python 3.
# I submitted a pull request https://github.com/cocodataset/cocoapi/pull/50
# If the PR is merged then use the original repo.
# Note: Edit PythonAPI/Makefile and replace "python" with "python3".
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

from config import Config
import common
import model as modellib


ROOT_DIR = os.getcwd()
MODEL_DIR = ROOT_DIR  # todo:


############################################################
#  Configurations
############################################################

class CocoConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "coco"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 80  # COCO has 80 classes


############################################################
#  Dataset
############################################################

class CocoDataset(common.Dataset):
    def load_coco(self, dataset_dir, subset, class_ids=None, class_map=None):
        """Load a subset of the COCO dataset.
        dataset_dir: The root directory of the COCO dataset.
        subset: What to load (train, val, minival, val35k)
        class_ids: If provided, only loads images that have the given classes.
        class_map: TODO
        """
        # Path
        image_dir = os.path.join(dataset_dir, "train2014" if subset == "train"
                                 else "val2014")

        # COCO object
        json_path_dict = {
            "train": "annotations/instances_train2014.json",
            "val": "annotations/instances_val2014.json",
            "minival": "annotations/instances_minival2014.json",
            "val35k": "annotations/instances_valminusminival2014.json",
        }
        coco = COCO(os.path.join(dataset_dir, json_path_dict[subset]))

        # Load all classes or a subset?
        if not class_ids:
            # All classes
            class_ids = sorted(coco.getCatIds())

        # All images or a subset?
        if class_ids:
            image_ids = []
            for id in class_ids:
                # todo: does this cause duplicate image ids?
                image_ids.extend(list(coco.getImgIds(catIds=[id])))
            # Remove duplicates
            image_ids = list(set(image_ids))
        else:
            # All images
            image_ids = list(coco.imgs.keys())

        # Add classes
        for i in class_ids:
            self.add_class("coco", i, coco.loadCats(i)[0]["name"])

        # Add images
        for i in image_ids:
            self.add_image("coco", image_id=i,
                           path=os.path.join(image_dir, coco.imgs[i]['file_name']),
                           width=coco.imgs[i]["width"],
                           height=coco.imgs[i]["height"],
                           annotations=coco.loadAnns(coco.getAnnIds(imgIds=[i], iscrowd=False)))

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "coco":
            return super(self.__class__).load_mask(image_id)

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for i, annotation in enumerate(annotations):
            class_id = self.map_source_class_id("coco.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"], image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(self.__class__).load_mask(image_id)

    def image_reference(self, image_id):
        """Return a link to the image in the COCO Website."""
        info = self.image_info[image_id]
        if info["source"] == "coco":
            return "http://cocodataset.org/#explore?id={}".format(info["id"])
        else:
            super(self.__class__).image_reference(self, image_id)


    #-- TODO: The following two functions are copied from pycocotools with small changes.

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m


############################################################
#  COCO Evaluation
############################################################

def build_coco_results(dataset, image_ids, rois, class_ids, scores, masks):
    """Arrange resutls to match COCO specs in http://cocodataset.org/#format
    dataset: A Dataset class
    todo:
    """
    assert False, "masks are transposed"
    # If no results, return an empty list
    if rois is None:
        return []

    results = []
    for image_ix in range(len(image_ids)):
        image_id = image_ids[image_ix]
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score = scores[i]

            # TODO: Compute bbox in original image dimensions
            bbox = np.around(rois[i], 1)

            result = {
                "image_id": image_id,
                "category_id": dataset.class_info[class_id]["map"][0][1],
                "bbox": [bbox[1], bbox[0], bbox[3]-bbox[1], bbox[2]-bbox[0]],
                "score": score,
                "segmentation": maskUtils.encode(np.asfortranarray(masks[i]))
            }
            results.append(result)
    return results


############################################################
#  COCO Evaluation on Many Images
############################################################

# # Pick COCO images from the validation set
# image_ids = [id for id in dataset.image_ids if dataset.image_info[id]["ds"] == "coco"]

# # Limit to a subset
# limit = 150
# image_ids = image_ids[:limit]

# # Get corresponding COCO image IDs.
# coco_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

# t_prediction = 0
# t_start = time.time()

# results = []
# for i in range(len(image_ids)):
#     image_id = image_ids[i]

#     image, shape, window, mask, mask_classes = \
#         dataset.load_image_and_mask(image_id, min_dim=config.IMAGE_MIN_DIM, 
#                                         max_dim=config.IMAGE_MAX_DIM,
#                                         padding=config.IMAGE_PADDING)
#     normalized_image = common.mold_image(image[np.newaxis,...], config)
#     image_meta = common.compose_image_meta(image_id, shape, window, 
#                                     np.zeros([dataset.num_classes], dtype=np.int32))[np.newaxis,...]

#     # Run object detection. One image at a time.
#     t_pred_start = time.time()
#     detections, mrcnn_class, mrcnn_bbox, mrcnn_mask, rois, \
#     rpn_class, rpn_bbox = model.predict([normalized_image, image_meta], verbose=0)
#     t_prediction += (time.time() - t_pred_start)

#     # Process detections
#     final_rois, final_class_ids, final_scores, final_masks =\
#         process_image_detections(detections[0], mrcnn_mask[0], shape, window, config)

#     image_results = build_coco_results(dataset, coco_image_ids[i:i+1], 
#                                        final_rois, final_class_ids, final_scores, final_masks)
#     results.extend(image_results)


# # Load results. This also modifies the 'results' list with additional attributes.
# coco_results = coco.loadRes(results)

# # Evaluate
# cocoEval = COCOeval(coco, coco_results, "segm")
# cocoEval.params.imgIds  = coco_image_ids
# # Limit evaluation to classes included in the training set
# cocoEval.params.catIds = list(filter(None, [dataset.class_id_to_external(id, "coco") 
#                                             for id in dataset.ds_class_ids['coco']]))
# cocoEval.evaluate()
# cocoEval.accumulate()
# cocoEval.summarize()

# print("Prediction time: {}. Average {}/image".format(t_prediction, t_prediction/len(image_ids)))
# print("Total time: ", time.time() - t_start)



############################################################
#  Compare Bounding Boxes
############################################################

# Compare our bounding boxes, which are computed from the 
# instance masks, to the COCO provided bounding boxes. The 
# test shows that 1~2% of values have a difference >= 1px, 
# and about 0.1% have a difference greater than 5px. This 
# difference is neglegable for most practical purposes. 
# It might affect the COCO AP score a little bit, but this 
# is not intended for a competition.

# # Pick random images to test on
# image_ids = np.random.choice(dataset_train.image_ids, 1000)
# diff = []
# for image_id in image_ids:
#     # Skip non-COCO images
#     info = dataset_train.image_info[image_id]
#     if info['source'] != 'coco' or not info['annotations']:
#         continue
#     mask, class_ids = dataset_train.load_mask(image_id)
#     bbox = common.extract_bboxes(mask)
#     coco_class_ids = [dataset_train.get_source_class_id(c, "coco") for c in class_ids]
#     coco_bbox = np.array([
#         [a['bbox'][1], a['bbox'][0], a['bbox'][1]+a['bbox'][3], a['bbox'][0]+a['bbox'][2]]
#         for a in dataset_train.image_info[image_id]['annotations'] 
#         if a['category_id'] in coco_class_ids
#         ])
#     # Skip on instance count mismatch. Happens when we skip
#     # an instance for being too small, for example.
#     if bbox.shape[0] < coco_bbox.shape[0]:
#         continue
#     # Mean diff between coco bbox and our bbox
#     diff.append(coco_bbox - bbox)
# diff = np.concatenate(diff, axis=0)
# print("images: {}   objects: {}\n".format(len(image_ids), diff.shape[0]))
# print("Stats for differences in y1, x1, y2, x2:\n")
# print("mean: ", np.mean(diff, axis=0))
# print("std: ", np.std(diff, axis=0))
# print("min: ", np.min(diff, axis=0))
# print("max: ", np.max(diff, axis=0))
# print("% diff >= 1px: ", np.sum(np.abs(diff) >= 1, axis=0) / diff.shape[0])
# print("% diff >= 5px: ", np.sum(np.abs(diff) >= 5, axis=0) / diff.shape[0])
# print("% diff >= 10px: ", np.sum(np.abs(diff) >= 10, axis=0) / diff.shape[0])



############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on MS COCO.')
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/coco/",
                        help='Directory of the MS-COCO dataset')
    args = parser.parse_args()
    print("Dataset directory: ", args.dataset)

    # Configurations
    config = CocoConfig()
    config.print()

    # Training dataset
    dataset_train = CocoDataset()
    dataset_train.load_coco(args.dataset, "train")  # todo: load val35k
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CocoDataset()
    dataset_val.load_coco(args.dataset, "minival")
    dataset_val.prepare()

    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=config)

    # Load Weights Trained on MS COCO
    # todo: arg to control initial weights
    TEST_MODEL = os.path.join(MODEL_DIR, "mask_rcnn_coco.h5")
    model.load_weights(TEST_MODEL, by_name=True)

    # Start training
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=40, layers='heads')
