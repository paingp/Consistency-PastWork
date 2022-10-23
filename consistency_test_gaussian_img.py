# -*- coding: utf-8 -*-
"""consistency-test-gaussian-img.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1uoZCpUwmfUW91uuD52T6339_mjA6h9Vj
"""

import cv2
import custom_dataset
import numpy as np
import torch
import torchvision

from postprocessing_utils import compare_pred_w_gt_boxes_only
from tqdm.notebook import tqdm

MOT15_TRAIN_PATH = os.path.join('../A/', 'MOT15/train/')
GT_SUBPATH = 'gt/'
IMG_SUBPATH = 'img1/'
OUTPUT_DIR = 'results/'

# Uses CUDA-ready GPU 0 if CUDA is available, otherwise, stick with the CPU
target_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Automatically downloads the model from the Internet, set for 91 unique classes (from COCO dataset)
frcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, num_classes=91).to(target_device)
frcnn = frcnn.eval()

mot_dataset = custom_dataset.MOT2015(MOT15_TRAIN_PATH, transform=torchvision.transforms.Compose([custom_dataset.ApplySkimageGaussian(), custom_dataset.ToTensor()]))
frcnn_dataloader = torch.utils.data.DataLoader(mot_dataset, batch_size=1, shuffle=False, num_workers=0)

def get_consistency(sample_1:dict, gt_matches_1:set, sample_2:dict, gt_matches_2:set):
    overlapping_matches = gt_matches_1.intersection(gt_matches_2)
    sample_1_gt_bbox_ids = set([int(bbox_id) for bbox_id in sample_1["bbox_ids"]])
    sample_2_gt_bbox_ids = set([int(bbox_id) for bbox_id in sample_2["bbox_ids"]])
    overlapping_gt_ids = sample_1_gt_bbox_ids.intersection(sample_2_gt_bbox_ids)
    unique_matches_in_sample_1 = gt_matches_1 - overlapping_matches
    stuff_sample_2_should_have_caught = unique_matches_in_sample_1.intersection(overlapping_gt_ids)
    unique_matches_in_sample_2 = gt_matches_2 - overlapping_matches
    stuff_sample_1_should_have_caught = unique_matches_in_sample_2.intersection(overlapping_gt_ids)

    if len(overlapping_gt_ids) != 0:
        consistency = (len(overlapping_gt_ids) - len(stuff_sample_2_should_have_caught) - len(stuff_sample_1_should_have_caught))/len(overlapping_gt_ids)
    else:
        consistency = 1.0
        
    return stuff_sample_1_should_have_caught, stuff_sample_2_should_have_caught, consistency

current_folder_name = ""
total_misses = {}
total_matches = {}
total_consistency = {}
prev_matches = None
prev_sample = None
total_imgs_per_folder = {}
for i, sample in tqdm(enumerate(frcnn_dataloader)):
    predictions = frcnn(sample["img"].to(target_device))[0] # Retrieve the single element in the prediction list
    pred_boxes = predictions["boxes"].to("cpu")
    pred_scores = predictions["scores"].to("cpu")
    gt_boxes = sample["bboxes"]
    gt_ids = sample["bbox_ids"]

    gt_boxes_tensor = torch.Tensor(gt_boxes)
    gt_ids_tensor = torch.Tensor(gt_ids)

    if len(gt_boxes) == 3:
        gt_boxes_tensor = gt_boxes_tensor.squeeze(0)
        gt_ids_tensor = gt_ids_tensor.squeeze(0)

    matches = compare_pred_w_gt_boxes_only(pred_boxes, pred_scores, gt_boxes_tensor, gt_ids_tensor)
    if current_folder_name == sample["folder_name"][0]:
        
        stuff_sample_1_should_have_caught, stuff_sample_2_should_have_caught, consistency = get_consistency(sample, matches, prev_sample, prev_matches)

        missed_matches = stuff_sample_1_should_have_caught.union(stuff_sample_2_should_have_caught)
        total_misses[current_folder_name] += len(missed_matches)
        total_matches[current_folder_name] += len(matches) 
        total_consistency[current_folder_name] += consistency
        total_imgs_per_folder[current_folder_name] += 1

        if len(missed_matches) > 0:
            cv2_img = sample["img"].squeeze().numpy().transpose((1, 2, 0)) * 255
            cv2_img = cv2.cvtColor(cv2_img.astype(np.uint8), cv2.COLOR_RGB2BGR)

            color = (0, 0, 255)
            thickness = 10
        
            for miss in missed_matches: # All the images missed
                for bbox_index, bbox_id in enumerate(sample["bbox_ids"]):
                    if bbox_id == miss:
                        cv2_img = cv2.rectangle(cv2_img, tuple(sample["bboxes"][bbox_index][0:2]), tuple(sample["bboxes"][bbox_index][2:4]), color, thickness)
                        
                        try:
                            os.mkdir(os.path.join(OUTPUT_DIR, current_folder_name))
                        except:
                            pass
                        cv2.imwrite(os.path.join(OUTPUT_DIR, current_folder_name, sample["img_name"][0]), cv2_img)
                        break
    else:
        current_folder_name = sample["folder_name"][0]
        total_misses[current_folder_name] = 0
        total_matches[current_folder_name] = 0
        total_consistency[current_folder_name] = 0
        total_imgs_per_folder[current_folder_name] = 0
        prev_matches = None
        prev_sample = None
        print("MISSES", total_misses)
        print("MATCHES", total_matches)
        print("Consistency", total_consistency)
        print("Per", total_imgs_per_folder)
        print("\n")
    prev_matches = matches
    prev_sample = sample

for folder_name in total_consistency:
    folder = os.path.join(MOT15_TRAIN_PATH, folder_name, IMG_SUBPATH)
    total_consistency[folder_name] /= len(os.listdir(folder))

for key in total_consistency:
    print("{0}: {1:.2f} \t\t ACCURACY: {2}".format(key, 100*(1-total_consistency[key]), total_matches[key]))
