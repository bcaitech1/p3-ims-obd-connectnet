import json
import requests
import os
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse

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


############################################
test_dir = "../../submission/" # 변경: output 파일 폴더 
user_key = "Bearer ???" # 변경 :AI Stage F12 -> Network 탭 -> 새로고침 -> auth/ -> Headers -> Authorization : Bearer 값 복사
desc = "description 안녕하세요"   # 변경 : 파일에 대한 설명
output_file = "Baseline_FCN8s(pretrained).csv" #변경 : output 파일

submit(user_key, os.path.join(test_dir, output_file),desc)


############################################
# 실행 방법
# python ./submit.py