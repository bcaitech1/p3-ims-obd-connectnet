 python tools/train.py configs/trash/faster_rcnn/faster_rcnn_r50_fpn_1x_trash.py
 
 
 python tools/test.py configs/trash/detectors/detectors_r50_x1_trash_basic.py  work_dirs/detectors_r50_x1_trash_basic/epoch_48.pth --out work_dirs/detectors_r50_x1_trash_basic/epoch_48_s.pkl

 python pkl_to_submission.py --pkl work_dirs/detectors_r50_x1_trash_basic/epoch_48_s.pkl --csv output/de8.csv

 python pkl_to_submission.py --pkl work_dirs/htc_dectors_r50_x1_trash/epoch_12.pkl --csv output/de6.csv

 python tools/test.py configs/trash/detectors/htc2.py  work_dirs/detectors_r50_x1_trash_basic/epoch_12.pth --out work_dirs/htc2/epoch_12.pkl

python pkl_to_submission.py --pkl work_dirs/detectors_r50_x1_trash_basic/epoch_48_s.pkl --csv output/de8.csv


python tools/test.py configs/trash/detectors/detectors_r50_x1_trash_k1.py  work_dirs/detectors_r50_x1_trash_k1/epoch_60.pth --out work_dirs/detectors_r50_x1_trash_k1/epoch_60.pkl


python wpkl_to_submission.py --pkl work_dirs/detectors_r50_x1_trash_k1/epoch_18.pkl --csv output/tet.csv