import os
import pandas as pd
from pycocotools.coco import COCO
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--pkl', required=True, help='output file')
parser.add_argument('--csv', required=True, help='submission file')

args = parser.parse_args()

prediction_strings = []
file_names = []
coco = COCO('../../input/data/test.json')

output = pd.read_pickle(args.pkl)
imag_ids = coco.getImgIds()

for i, out in enumerate(output):
    prediction_string = ''
    image_info = coco.loadImgs(coco.getImgIds(imgIds=i))[0]
    for j in range(11):
        for o in out[0][j]:
            prediction_string += str(j) + ' ' + str(o[4]) + ' ' + str(o[0]) + ' ' + str(o[1]) + ' ' + str(
                o[2]) + ' ' + str(o[3]) + ' '

    prediction_strings.append(prediction_string)
    file_names.append(image_info['file_name'])

submission = pd.DataFrame()
submission['PredictionString'] = prediction_strings
submission['image_id'] = file_names
submission.to_csv(args.csv, index=None)
print(submission.head())