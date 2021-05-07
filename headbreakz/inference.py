import json
import requests
import os
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse

def load_model(saved_model_path,model_name,device):        
    model_path = os.path.join(saved_model_path, model_name)
    model.load_state_dict(torch.load(saved_model_path, map_location=device))

    return model

@torch.no_grad()
def test(model, data_loader, device,transform ,test_dir,output_file):
    size = 256
    print('Start prediction.')
    model.eval()
    
    file_name_list = []
    preds_array = np.empty((0, size*size), dtype=np.long)
    
    with torch.no_grad():
        for step, (imgs, image_infos) in enumerate(test_loader):

            # inference (512 x 512)
            outs = model(torch.stack(imgs).to(device))
            oms = torch.argmax(outs, dim=1).detach().cpu().numpy()
            
            # resize (256 x 256)
            temp_mask = []
            for img, mask in zip(np.stack(imgs), oms):                
                transformed = transform(image=img, mask=mask)
                mask = transformed['mask']
                temp_mask.append(mask)

            oms = np.array(temp_mask)
            
            oms = oms.reshape([oms.shape[0], size*size]).astype(int)
            preds_array = np.vstack((preds_array, oms))
            
            file_name_list.append([i['file_name'] for i in image_infos])
    print("End prediction.")
    print("Save CSV file")
    file_names = [y for x in file_name_list for y in x]

    submission = pd.read_csv(test_dir +'sample_submission.csv', index_col=None)

    for file_name, string in zip(file_names, preds):
        submission = submission.append({"image_id" : file_name, "PredictionString" : ' '.join(str(e) for e in string.tolist())}, 
                                   ignore_index=True)
    submission.to_csv(test_dir + output_file , index=False)

    print('Finish save csv file')
    print('Submit Prediction result csv file')
    submit(user_key, os.path.join(test_dir, output_file),desc)
    


def submit(user_key='', file_path = '', desc=""):
    if not user_key:
        raise Exception("No UserKey" )
    url = urlparse('http://ec2-13-124-161-225.ap-northeast-2.compute.amazonaws.com:8000/api/v1/competition/28/presigned_url/?description=&hyperparameters={%22training%22:{},%22inference%22:{}}')
    qs = dict(parse_qsl(url.query))
    qs['description'] = desc
    parts = url._replace(query=urlencode(qs))
    url = urlunparse(parts)

    print(url)
    headers = {
        'Authorization': user_key
    }
    res = requests.get(url, headers=headers)
    print(res.text)
    data = json.loads(res.text)
    
    submit_url = data['url']
    body = {
        'key':'app/Competitions/000028/Users/{}/Submissions/{}/output.csv'.format(str(data['submission']['user']).zfill(8),str(data['submission']['local_id']).zfill(4)),
        'x-amz-algorithm':data['fields']['x-amz-algorithm'],
        'x-amz-credential':data['fields']['x-amz-credential'],
        'x-amz-date':data['fields']['x-amz-date'],
        'policy':data['fields']['policy'],
        'x-amz-signature':data['fields']['x-amz-signature']
    }
    requests.post(url=submit_url, data=body, files={'file': open(file_path, 'rb')})


test_dir = "./submission/" # 변경: output 파일 폴더 
user_key = "Bearer 15d0d31523f456f79bc39b22507b3c5a3a00fd0f" # 변경 :AI Stage F12 -> Network 탭 -> 새로고침 -> auth/ -> Headers -> Authorization : Bearer 값 복사
desc = "inceptionresnetv2_miou"   # 변경 : 파일에 대한 설명
output_file = "inceptionresnetv2.csv" #변경 : output 파일
