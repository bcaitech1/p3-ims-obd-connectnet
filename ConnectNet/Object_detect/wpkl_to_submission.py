import os
import pandas as pd
from pycocotools.coco import COCO
import argparse

from ensemble_boxes import *

parser = argparse.ArgumentParser(description='')
parser.add_argument('--pkl', required=True, help='output file')
parser.add_argument('--csv', required=True, help='submission file')

args = parser.parse_args()

prediction_strings = []
file_names = []
coco = COCO('../../input/data/test.json')

output = pd.read_pickle(args.pkl)
imag_ids = coco.getImgIds()

img_size = 512.

iou_thr = 0.6
skip_box_thr = 0.0001

for idx, img in enumerate(output):
    
    boxes_list = []
    scores_list = []
    labels_list = []
    
    for label, boxes_in_label in enumerate(img):
        for box_and_score in boxes_in_label:
            scores_list.append(box_and_score[4])
            boxes_list.append(box_and_score[:4] / img_size)
            labels_list.append(label)
            
    boxes, scores, labels = weighted_boxes_fusion([boxes_list], [scores_list], [labels_list], weights=None, iou_thr=iou_thr, skip_box_thr=0.0)
    
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