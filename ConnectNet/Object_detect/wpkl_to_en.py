import os
import pandas as pd
from pycocotools.coco import COCO
import argparse

from ensemble_boxes import *

parser = argparse.ArgumentParser(description='')
parser.add_argument('--pkl', required=False, help='output file')
parser.add_argument('--csv', required=True, help='submission file')

args = parser.parse_args()

prediction_strings = []
file_names = []
coco = COCO('../../input/data/test.json')

plk1=  "./work_dirs/detectors_r50_x1_trash_k1/epoch_48.pkl"
plk2=  "./work_dirs/detectors_r50_x1_trash_k1/epoch_60.pkl"
output1 = pd.read_pickle(plk1)
output2 = pd.read_pickle(plk2)


imag_ids = coco.getImgIds()

img_size = 512.

weights = [2, 1] # ensemble weights for model 1 and model 2
iou_thr = 0.6
skip_box_thr = 0.0001

for idx in range(len(output1)):
    
    boxes_list1, boxes_list2, boxes_list = [], [], []
    scores_list1, scores_list2, scores_list = [], [], []
    labels_list1, labels_list2, labels_list = [], [], []
    
    # model 1
    for label, boxes_in_label in enumerate(output1[idx]):
        for box_and_score in boxes_in_label:
            scores_list1.append(box_and_score[4])
            boxes_list1.append(box_and_score[:4] / img_size)
            labels_list1.append(label)
    
    # model 2
    for label, boxes_in_label in enumerate(output2[idx]):
        for box_and_score in boxes_in_label:
            scores_list2.append(box_and_score[4])
            boxes_list2.append(box_and_score[:4] / img_size)
            labels_list2.append(label)
    
    boxes_list = [boxes_list1, boxes_list2]
    scores_list = [scores_list1, scores_list2]
    labels_list = [labels_list1, labels_list2]
    
    boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    
    prediction_string = ''
    image_info = coco.loadImgs(coco.getImgIds(imgIds=idx))[0]
    
    for i, box in enumerate(boxes):
        prediction_string += str(int(labels[i])) + ' ' + str(scores[i])[:11] + ' ' + str(box[0]*img_size)[:9] + ' '  + str(box[1]*img_size)[:9] + ' '  + str(box[2]*img_size)[:9] + ' '  + str(box[3]*img_size)[:9] + ' '
    prediction_strings.append(prediction_string)
    file_names.append(image_info['file_name'])


submission = pd.DataFrame()
submission['PredictionString'] = prediction_strings
submission['image_id'] = file_names
submission.to_csv(args.csv, index=None)
print(submission.head())