

## yolov5 데이터형식 맞춰주기

### yolo의 데이터형식
- yolo는 데이터 형식이 (x_center, y_center, w,h)로 되어있다.
- 기존에 알고있는 (x_min,y_min,w,h)와 다른 모습인것을 유의합시다.

### 파일 변환
- datachange.py파일 수행
  - json파일을 읽어서 아래 폴더구조에 맞게 파일을 복사해준다.

### yolov5의 폴더 구조
- yolov5에 custom 데이터를 이용하기위해서는 폴더 구조를 맞춰줘야 한다.
- 폴더구조
  - data
    - images
      - train - .jpg파일들이들어간다.
      - val
    - labels
      - train - .txt파일이 들어간다.
      - val
- 폴더가 images와 labels로 나뉜것을 볼 수 있다.
- images와 labels는 서로 같은 이름의 `파일명`을 가지며 서로 쌍이된다.
  - images/train/image1.jpg
  - labels/train/image1.txt
- labels폴더에는 쌍이되는 이미지의 `카테고리 아이디 ,bbox(x_center,y_center,w,h)`가 저장해야한다.
  - 들어가는 좌표는 normalize되어야한다.
  - ex) 이미지 크기512기준일때, x_center가 30일때 30/512가 들어가야한다.
  - ex) 0 0.1 0.2 0.4 0.2


### yolov5의 불러오는 방식
- yolov5는 `data/`폴더 내에 `.yaml`형식으로 `데이터의 위치`, `클래스 갯수`, `클래스 라벨`을 지정해줘야한다.
- ```yaml
    # 예시
    train: /input/data/images/train # train폴더 위치
    val: /input/data/images/val # validation폴더 위치

    nc: 11 # 클래스갯수

    names: ["UNKNOWN","General trash","Paper","Paper pack","Metal","Glass", "Plastic","Styrofoam","Plastic bag","Battery","Clothing"] # 클래스 라벨들
  ```


### pretrain 가져오기
- `bash weights/download_weights.sh`실행


### 학습진행
- train.py파일 실행
  - `python train.py --img-size 512 --batch-size 32 --epochs 100 --data bcai.yaml --weights yolov5x.pt`
  - 이미지크기 512, 배치 32, epoch 100
  - bcai.yaml(위에서 만든 yaml, data폴더에 넣으면 알아서 인식)
  - weight는 원하는 pretrain모델 가져온다.


### 학습결과 확인
- detect.py실행
  - `python detect.py --img-size 512 --source /opt/ml/input/data/images/test --weights runs/train/exp/weights/best.pt --save-txt --save-conf`
  - 이미지 사이즈 512
  - source는 test이미지 불러오기
  - weight는 학습한 모델 경로
  - save-txt는 학습결과를 txt로 출력할지
  - save-conf는 bbox내의 물체분류한 confidence로 같이 출력할지 여부

### submission생성
- `python mk_submission.py`

### psudo 적용
- 기존train(img,label)파일과 더불어 test(img만존재)에서 학습한 모델에서 label을 생성
- 기존train과 test(label생성)을 같이 학습을 적용
- test파일에서 psudo labeling적용
  - 일반 detect는 (cls, x_center,y_center,w,h)로 저장됩니다.
  - detect_psudo라는 파일로는 (cls, x_min,y_min,w,h)/이미지크기 로 저장을 해준다.
  - `python detect_psudo.py --img-size 512 --source /opt/ml/input/data/images/test --weights /opt/ml/p3-ims-obd-connectnet/junbae/Detection/yolov5/runs/train/fold2/weights/best.pt --save-txt --nosave --name psudo_fold_2`
- psudo label한 파일 이동
  - `python move_psudo_label.py --source /opt/ml/p3-ims-obd-connectnet/junbae/Detection/yolov5/runs/detect/psudo_fold_2/labels --dst /opt/ml/input/data/labels/test`
- data `yaml`도 psudo test를 추가
  - train부분에 `/input/data/images/test`추가
- 기존 학습한weight를 통해 재학습
  - `python train.py --img-size 512 --batch-size 32 --epochs 300 --data bcai_k2_psudo.yaml --weights /opt/ml/p3-ims-obd-connectnet/junbae/Detection/yolov5/runs/train/fold2/weights/best.pt --name fold2_psudo`



### 참고
- https://github.com/ultralytics/yolov5


