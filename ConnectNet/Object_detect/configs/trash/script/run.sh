#!/bin/sh

python tools/train.py configs/trash/detectors/detectors_r50_x1_trash.py 

python  tools/train.py configs/trash/detectors/detectors_r50_x1_trash_basic.py 