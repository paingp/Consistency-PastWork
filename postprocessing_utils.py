import torch
import torchvision


def get_consistency_boxes_only(gt_1: set, matches_1: set, gt_2: set, matches_2: set) -> tuple:
    """Given the ground truth and the correct predictions (matches) for 2 similar images, calculates consistency

    Args:
        gt_1 (set): All bounding box IDs in the first image's ground truth
        matches_1 (set): All bounding box IDs correctly predicted (matched) in the first image
        gt_2 (set):  All bounding box IDs in the second image's ground truth
        matches_2 (set): All bounding box IDs correctly predicted (matched) in the second image

    Returns:
        tuple: a set of IDs missed in image 1 that were matched in image 2, a set of IDs missed in image 2 that were matched in image 1, calculated consistency (float)
    """
    matches_overlapping = matches_1.intersection(matches_2)
    gt_overlapping = gt_1.intersection(gt_2)
    unique_matches_1 = matches_1 - matches_overlapping
    unique_matches_2 = matches_2 - matches_overlapping
    missed_matches_in_1 = unique_matches_2.intersection(gt_overlapping)
    missed_matches_in_2 = unique_matches_1.intersection(gt_overlapping)

    if len(gt_overlapping) != 0:
        consistency = (len(gt_overlapping) - len(missed_matches_in_1) - len(missed_matches_in_2)) / len(gt_overlapping)
    else:
        consistency = 1.0
        
    return missed_matches_in_1, missed_matches_in_2, consistency

def compare_pred_w_gt_boxes_only(pred_boxes: torch.Tensor, pred_confidence_scores: torch.Tensor, gt_boxes: torch.Tensor, gt_ids: torch.Tensor, confidence_thresh: float = 0.7, iou_thresh: float = 0.5) -> set:
    """Given N bounding box predictions and M ground truth, returns a set containing the IDs of the ground truth that got correctly predicted.

    A ground truth ID is determined as "correctly predicted" iff:
    - the prediction box is above the IoU threshold

    Uses Non-maximum suppression to avoid multi-matching. Screens out predictions below the given confidence level.

    Args:
        pred_boxes (torch.Tensor): An Nx4 tensor for N predicted bounding boxes as (x1, y1, x2, y2)
        pred_confidence_scores (torch.Tensor): An N-element tensor of confidence scores for each predicted bounding box
        gt_boxes (torch.Tensor): An Mx4 tensor for M ground truth bounding boxes as (x1, y1, x2, y2)
        gt_ids (torch.Tensor): An M-element tensor storing the IDs of each ground truth bounding box
        confidence_thresh (float, optional): Confidence score threshold below which a predicted bounding box is discarded. Defaults to 0.5.
        iou_thresh (float, optional): IoU threshold above which two bounding boxes are considered matching. Also used for NMS. Defaults to 0.5.

    Returns:
        set: A list of all ground truth IDs that are matched correctly.
    """
    pred_boxes = pred_boxes[pred_confidence_scores >= confidence_thresh]  # This creates a new Tensor
    # Again, new Tensor
    pred_confidence_scores = pred_confidence_scores[pred_confidence_scores >= confidence_thresh]

    idxes_to_keep_after_nms = torchvision.ops.nms(pred_boxes, pred_confidence_scores, iou_thresh)
    if len(idxes_to_keep_after_nms) == 0 or len(gt_boxes) == 0:
        return set(), 0
    pred_boxes = pred_boxes[idxes_to_keep_after_nms]
    pred_confidence_scores = pred_confidence_scores[idxes_to_keep_after_nms]

    # Each pred_box gets its own row in this matrix, where each element in the row is the IOU with the corresponding box in gt_boxes
    mat_of_ious = torchvision.ops.box_iou(pred_boxes, gt_boxes)

    gt_ids_matched = set()  # Guarantees that nothing will be reported as matched twice
    num_preds = mat_of_ious.shape[0] # num rows
    # For each prediction (aka each row), find the best match (aka highest IoU, argmax) 
    for row in mat_of_ious:
        if torch.count_nonzero(row) > 0:  
            best_match_gt_idx = torch.argmax(row)
        else: # if the entire row is zero, that means the IoUs all were zero, skip
            continue

        if row[best_match_gt_idx] > iou_thresh:
            gt_ids_matched.add(int(gt_ids[best_match_gt_idx]))

    return gt_ids_matched, num_preds


def compare_pred_w_gt(pred_boxes: torch.Tensor, pred_confidence_scores: torch.Tensor, pred_classes: torch.Tensor, gt_boxes: torch.Tensor, gt_labels: torch.Tensor, gt_ids: torch.Tensor, confidence_thresh: float = 0.5, iou_thresh: float = 0.5) -> list:
    """Given N bounding box predictions and M ground truth, returns a set containing the IDs of the ground truth that got correctly predicted.

    A ground truth ID is determined as "correctly predicted" iff:
    - the prediction box is above the IoU threshold
    - the predicted class is correct

    Uses Non-maximum suppression to avoid multi-matching. Screens out predictions below the given confidence level.

    Args:
        pred_boxes (torch.Tensor): An Nx4 tensor for N predicted bounding boxes as (x1, y1, x2, y2)
        pred_confidence_scores (torch.Tensor): An N-element tensor of confidence scores for each predicted bounding box
        pred_classes (torch.Tensor): An N-element tensor of corresponding classes for each predicted bounding box
        gt_boxes (torch.Tensor): An Mx4 tensor for M ground truth bounding boxes as (x1, y1, x2, y2)
        gt_labels (torch.Tensor): An M-element tensor of corresponding labels for each ground truth bounding box
        gt_ids (torch.Tensor): An M-element tensor storing the IDs of each ground truth bounding box
        confidence_thresh (float, optional): Confidence score threshold below which a predicted bounding box is discarded. Defaults to 0.5.
        iou_thresh (float, optional): IoU threshold above which two bounding boxes are considered matching. Also used for NMS. Defaults to 0.5.

    Returns:
        list: A list of all ground truth IDs that are matched correctly.
    """
    pred_boxes = pred_boxes[pred_confidence_scores >= confidence_thresh]  # This creates a new Tensor
    pred_classes = pred_classes[pred_confidence_scores >= confidence_thresh]
    # Again, new Tensor
    pred_confidence_scores = pred_confidence_scores[pred_confidence_scores >= confidence_thresh]

    idxes_to_keep_after_nms = torchvision.ops.nms(pred_boxes, pred_confidence_scores, iou_thresh)
    pred_boxes = pred_boxes[idxes_to_keep_after_nms]
    pred_classes = pred_classes[idxes_to_keep_after_nms]
    pred_confidence_scores = pred_confidence_scores[idxes_to_keep_after_nms]

    # Each pred_box gets its own row in this matrix, where each element in the row is the IOU with the corresponding box in gt_boxes
    mat_of_ious = torchvision.ops.box_iou(pred_boxes, gt_boxes)

    gt_ids_matched = set()  # Guarantees that nothing will be reported as matched twice

    # For each prediction (aka each row), find the best match (aka highest IoU, argmax) 
    for i, row in enumerate(mat_of_ious):
        predicted_class = pred_classes[i]
        if torch.count_nonzero(row) > 0:  
            best_match_gt_idx = torch.argmax(row)
        else: # if the entire row is zero, that means the IoUs all were zero, skip
            continue

        if row[best_match_gt_idx] > iou_thresh and gt_labels[best_match_gt_idx] == predicted_class:
            gt_ids_matched.add(int(gt_ids[best_match_gt_idx]))

    return gt_ids_matched


if __name__ == "__main__":
    pred_boxes = [(0, 0, 50, 50), (100, 100, 150, 150), (200, 200, 250, 250)]
    # Confidence scores of 0.5 for the first box, 0.4 for the second...
    pred_scores = [0.5, 0.4, 0.7]
    pred_classes = [1, 1, 1]  # Predicted the class was '1' each time.
    gt_boxes = [(0, 0, 50, 50), (100, 100, 150, 150), (400, 400, 450, 450), (200, 200, 250, 250)]  # 4 ground truth boxes
    gt_labels = [1, 1, 3, 1]  # each box has labels
    gt_ids = [1, 2, 3, 4]  # each box also has object trackign ID

    print(compare_pred_w_gt(torch.Tensor(pred_boxes), torch.Tensor(pred_scores), torch.Tensor(pred_classes), torch.Tensor(gt_boxes), torch.Tensor(gt_labels), torch.Tensor(gt_ids)))
    print(compare_pred_w_gt_boxes_only(torch.Tensor(pred_boxes), torch.Tensor(pred_scores), torch.Tensor(gt_boxes),  torch.Tensor(gt_ids)))