import argparse
import shutil
import os


def custom_filename(x):
    # 146.batch_02_vt_0205.txt -> batch_02_vt/0205.txt
    x = x.split(".")[1]
    x = x.split("_")
    x_f = '_'.join(x[:-1])
    x = "/".join([x_f,x[-1]])
    x = x+".jpg"
    return x

def movefiles(opt):

    file_list = []
    file_list = os.listdir(opt.source) # source의 모든 파일 확인
    # dst_file_list = list(map(custom_filename,file_list))

    for i,name in enumerate(file_list):
        # shutil.copy(os.path.join(opt.source,name), os.path.join(opt.dst,dst_file_list[i]))
        shutil.copy(os.path.join(opt.source,name), os.path.join(opt.dst,name))



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='/opt/ml/p3-ims-obd-connectnet/junbae/Detection/yolov5/runs/detect/psudo_fold_2/labels', help='source')
    parser.add_argument('--dst', type=str, default='/opt/ml/input/data/labels/test', help='dst')  # file/folder, 0 for webcam
   
    # sorce = "/opt/ml/p3-ims-obd-connectnet/junbae/Detection/yolov5/runs/detect/fold2/labels"
    # dest = "/opt/ml/input/data/labels/test"
    opt = parser.parse_args()
    movefiles(opt = opt)