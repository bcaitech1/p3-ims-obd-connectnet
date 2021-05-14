# 폴더선택
# 폴더의 txt들 읽기
    # 하나씩 열어서 배열로 만들기
    # 파일 하나당 한줄 만들기

import os
import pandas as pd

def make_filename(x):
    file_names = x.split(".")
    return [int(file_names[0]),x]

def line_make(x):
    x = x.replace("\n","")
    x = x.split(" ")
    x[2] = float(x[2]) * 512.0 # x center
    x[3] = float(x[3]) * 512.0 # y center
    x[4] = float(x[4]) * 512.0 # w
    x[5] = float(x[5]) * 512.0 # h
    
    x[2] = x[2] - x[4]/2
    x[3] = x[3] - x[5]/2

    x[4]+=x[2]
    x[5]+=x[3]
    x[2] = str(x[2])
    x[3] = str(x[3])
    x[4] = str(x[4])
    x[5] = str(x[5])
    return ' '.join(x)



def make_submission(path_dir):
    
    prediction_strings = []
    file_names = []

    file_list = os.listdir(path_dir)
    #labels삭제
    file_list.remove("labels")
    file_list = list(map(make_filename,file_list))
    file_list.sort(key=lambda x:x[0])

    for i,file_name in file_list:
        file_name = file_name.replace(".jpg",".txt")
        #파일 존재 유무 확인
        if os.path.exists(os.path.join(path_dir,"labels",file_name)):
            f = open(os.path.join(path_dir,"labels",file_name), 'r')
            #predictions
            lines = f.readlines()
            lines = list(map(line_make,lines))
            lines = ' '.join(lines)
            prediction_strings.append(lines)
            f.close()
        else:
            prediction_strings.append("")
        # file_names
        file_name = file_name.split(".")[1]
        file_name = file_name.split("_")
        file_name_f = '_'.join(file_name[:-1])
        file_name = "/".join([file_name_f,file_name[-1]])
        file_name = file_name+".jpg"
        file_names.append(file_name)
        
    return prediction_strings,file_names
    


if __name__ == '__main__':
    det_dir = "/opt/ml/code/yolov5/runs/detect/exp"
    submission_name = 'yolov5_epoch57_batch32_best.csv'
    prediction_strings,file_names = make_submission(det_dir)
    

    print(prediction_strings)
    submission = pd.DataFrame()
    submission['PredictionString'] = prediction_strings
    submission['image_id'] = file_names
    submission.to_csv(os.path.join(det_dir,submission_name), index=None)
    print(submission.head())
