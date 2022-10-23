import cv2
import os
import torch
import torchvision
import numpy as np
from skimage import io, filters, exposure

class ApplyHorizontalFlip(object):
    def __call__(self,sample):
        folder_name = sample["folder_name"]
        img_name = sample["img_name"]
        img = sample["img"]
        gt_bboxes = sample["bboxes"]
        gt_bbox_ids = sample["bbox_ids"]
        filtered_img = np.copy(img[:, ::-1])
        img_width = filtered_img.shape[1] # HxWxC
        img_midpoint = img_width//2
        for i, bbox in enumerate(gt_bboxes): # Flip all bounding boxes
            new_bbox = bbox
            new_bbox[0] = 2*img_midpoint - bbox[0] # x1
            new_bbox[2] = 2*img_midpoint - bbox[2] # x2
            gt_bboxes[i] = new_bbox
        return {"folder_name": folder_name, "img_name": img_name, "img": filtered_img, "bboxes": gt_bboxes, "bbox_ids": gt_bbox_ids}

class ApplyWebpCompress(object):
    def __call__(self,sample):
        folder_name = sample["folder_name"]
        img_name = sample["img_name"]
        img = sample["img"]
        gt_bboxes = sample["bboxes"]
        gt_bbox_ids = sample["bbox_ids"]
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 40]
        _, img_cv2 = cv2.imencode(".jpg", img, encode_param)
        filtered_img=cv2.imdecode(img_cv2, 1)
        return {"folder_name": folder_name, "img_name": img_name, "img": filtered_img, "bboxes": gt_bboxes, "bbox_ids": gt_bbox_ids}

class ApplySkimageUnsharpMask(object):
    def __call__(self,sample):
        folder_name = sample["folder_name"]
        img_name = sample["img_name"]
        img = sample["img"]
        gt_bboxes = sample["bboxes"]
        gt_bbox_ids = sample["bbox_ids"]
        filtered_img = filters.unsharp_mask(img, multichannel=True, preserve_range=True)
        return {"folder_name": folder_name, "img_name": img_name, "img": filtered_img, "bboxes": gt_bboxes, "bbox_ids": gt_bbox_ids}

class ApplySkimageGaussian(object):
    def __call__(self,sample):
        folder_name = sample["folder_name"]
        img_name = sample["img_name"]
        img = sample["img"]
        gt_bboxes = sample["bboxes"]
        gt_bbox_ids = sample["bbox_ids"]
        filtered_img = filters.gaussian(img, preserve_range=True)
        return {"folder_name": folder_name, "img_name": img_name, "img": filtered_img, "bboxes": gt_bboxes, "bbox_ids": gt_bbox_ids}

class ApplySkimageGammaCorrection(object):
    def __call__(self,sample):
        folder_name = sample["folder_name"]
        img_name = sample["img_name"]
        img = sample["img"]
        gt_bboxes = sample["bboxes"]
        gt_bbox_ids = sample["bbox_ids"]
        filtered_img = exposure.adjust_gamma(img, 0.7)
        return {"folder_name": folder_name, "img_name": img_name, "img": filtered_img, "bboxes": gt_bboxes, "bbox_ids": gt_bbox_ids}

class ToTensor(object):
    def __call__(self, sample):
        """
        Returns the sample, converted from a [0, 255] numpy array to a [0, 1] float tensor.
        """

        folder_name = sample["folder_name"]
        img_name = sample["img_name"]
        img = sample["img"].astype("uint8")
        gt_bboxes = sample["bboxes"]
        gt_bbox_ids = sample["bbox_ids"]
        return {"folder_name": folder_name, "img_name": img_name, "img": torchvision.transforms.ToTensor()(img), "bboxes": gt_bboxes, "bbox_ids": gt_bbox_ids}

class MOT2015(torch.utils.data.Dataset):
    def __init__(self, root_dir:str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.folder_names = sorted(os.listdir(root_dir))
        self.folder_samples = []
        self.samples = []

        for folder_name in self.folder_names:
            folder_imgs = []
            folder_img_names = sorted([img_file for img_file in os.listdir(os.path.join(self.root_dir, folder_name, "img1/"))])
            
            for img_name in folder_img_names:
                sample = {"folder_name": folder_name, "img_name": img_name, "bboxes": [], "bbox_ids": []}
                folder_imgs.append(sample)

            with open(os.path.join(self.root_dir, folder_name, "gt/gt.txt")) as gt:
                gt_lines = gt.readlines()
                
                for gt_line in gt_lines:
                    gt_line = gt_line.split(',')
                    frame_num = int(float(gt_line[0]))-1
                    obj_id = int(float(gt_line[1]))
                    bbox_xmin = int(float(gt_line[2]))
                    bbox_ymin = int(float(gt_line[3]))
                    bbox_xmax = bbox_xmin + int(float(gt_line[4]))
                    bbox_ymax = bbox_ymin + int(float(gt_line[5]))
                    folder_imgs[frame_num]["bboxes"].append([bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax])
                    folder_imgs[frame_num]["bbox_ids"].append(obj_id)
            
            self.folder_samples.append(folder_imgs)
        
        for folder_imgs in self.folder_samples:
            self.samples += folder_imgs

    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, i):
        folder_name = self.samples[i]["folder_name"]
        img_name = self.samples[i]["img_name"]
        img = io.imread(os.path.join(self.root_dir, folder_name, "img1/", img_name))
        
        gt_bboxes = self.samples[i]["bboxes"]
        gt_bbox_ids = self.samples[i]["bbox_ids"]
        sample = {"folder_name": folder_name, "img_name": img_name, "img": img, "bboxes": gt_bboxes, "bbox_ids": gt_bbox_ids}

        if self.transform is not None:
            sample = self.transform(sample)
        return sample
