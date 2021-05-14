# json파일 읽기
# images
    # train
    # val
# labels
    # train
    # val

# train.json읽기
# images는 images/train/폴더로 복사
# labels는 labels/train/폴더에 생성
import shutil
import json
import os

def make_data(data_path, json_path,folder_name,is_test=False):

    # images폴더생성
        # folder_name폴더생성
    if not os.path.exists(os.path.join(data_path,"images")):
        os.makedirs(os.path.join(data_path,"images"))
    
    if not os.path.exists(os.path.join(data_path,"images",folder_name)):
        os.makedirs(os.path.join(data_path,"images",folder_name))

    # labels폴더생성
        # folder_name폴더생성
    if not is_test:
        if not os.path.exists(os.path.join(data_path,"labels")):
            os.makedirs(os.path.join(data_path,"labels"))
        
        if not os.path.exists(os.path.join(data_path,"labels",folder_name)):
            os.makedirs(os.path.join(data_path,"labels",folder_name))


    with open(json_path) as json_file:
        json_data = json.load(json_file)
        # print(json_data.keys()) #['images', 'annotations'] 이미지, 어노테이션들
        save = {}
        for e in json_data["images"]:
            changename = e["file_name"].replace("/","_")
            # 파일 이동
            save[e['id']]= changename
            shutil.copy(os.path.join(data_path,e["file_name"]), os.path.join(data_path,"images",folder_name,changename))
            
        if not is_test:
            ann_save={}
            for e in json_data["annotations"]:
                if ann_save.get(e['image_id'])==None:
                    ann_save[e['image_id']] = []
                ann_save[e['image_id']].append([e['category_id'],*e["bbox"]])

                
            # print(ann_save)
            for k,v in ann_save.items():
                name = save[k]
                name = name.replace(".jpg",".txt")
                data = []
                for d in v:
                    cat_id = d[0]
                    d[1] = d[1] + d[3]/2
                    d[2] = d[2] + d[4]/2

                    d[1] /= 512.0
                    d[2] /= 512.0
                    d[3] /= 512.0
                    d[4] /= 512.0
                    d = list(map(str,d))
                    data.append(' '.join(d))
                # print("-------")
                write_data = '\n'.join(data)
                # # print(write_data)
                # # print(name)
                f = open(os.path.join(data_path,"labels",folder_name,name), 'w')
                f.write(write_data)
                f.close()

if __name__ == '__main__':
    data_path = "/opt/ml/input/data"
    json_path = "/opt/ml/input/data/train.json"
    folder_name = "train"
    make_data(json_path,folder_name)