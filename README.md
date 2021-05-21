### 부스트캠프 - AI Tech

### P stage 3 - Semantic Segmentation & Object Detection

> `P stage 3 대회 진행 과정과 결과를 기록한 Team Git repo 입니다. 대회 특성상 수정 혹은 삭제된 부분이 존재 합니다`

---

#### ✔️ Segmentation Task
   * Rank: 17
   * LB: 0.6205

#### ✔️ Object Detection Task
   * Rank : 5
   * LB: 0.4789



### 📋 Table of content

[Team 소개](#Team)<br>
[Gound Rule](#rule)<br>

#### 1. [Segmentation](#seg)

1.1 [대회 전략](#strategy1)<br>
1.2 [Model](#model1)<br>
1.3 [Loss](#loss1)<br>
1.4 [Augmentation](#aug1)<br>
1.5 [실패한 부분](#fail1)<br>
1.6 [회고 & 과제](#try1)<br>

#### 2. [Object Detection](#obd)

2.1 [대회 전략](#strategy2)<br>
2.2 [Model](#model2)<br>
2.3 [Augmentation](#aug2)<br>
2.4 [사용한 기술](#skill2)<br>
2.5 [Ensemble](#ensemble2)<br>
2.6 [실패한 부분](#fail2)<br>


---

### 🌏Team - ConnectNet <a name = 'Team'></a>

* 김종호_T1034 [![Github Badge](https://img.shields.io/badge/-Github-161c22?style=flat&logo=github&link=https://github.com/philgineer/)](https://github.com/Headbreakz) [![Blog Badge](http://img.shields.io/badge/Blog-51a9fe?style=flat&link=https://philgineer.com/)](https://headbreakz.tistory.com/)
* 김현우_T1045 [![Github Badge](https://img.shields.io/badge/-Github-161c22?style=flat&logo=github&link=https://github.com/philgineer/)](https://github.com/LethalSun)
* 김현우_T1046 [![Github Badge](https://img.shields.io/badge/-Github-161c22?style=flat&logo=github&link=https://github.com/philgineer/)](https://github.com/akorea)
* 배철환_T1086 [![Github Badge](https://img.shields.io/badge/-Github-161c22?style=flat&logo=github&link=https://github.com/philgineer/)](https://github.com/bcc0830)
* 서준배_T1097 [![Github Badge](https://img.shields.io/badge/-Github-161c22?style=flat&logo=github&link=https://github.com/philgineer/)](https://github.com/deokisys)
* 윤준호_T1138 [![Github Badge](https://img.shields.io/badge/-Github-161c22?style=flat&logo=github&link=https://github.com/philgineer/)](https://github.com/philgineer) [![Blog Badge](http://img.shields.io/badge/Blog-51a9fe?style=flat&link=https://philgineer.com/)](https://philgineer.com/)



#### Ground rule <a name = 'rule'></a>

1. 공유 코드 작성 시
   * `작성자 표시`
   * `함수` 단위 작성
   * `merge` 시  .py 형식 사용
   
2. 모델 로그 기록
   * `Wandb` 사용

3. 회의 기록 작성

4. 자료 공유 및 연락 - Slack

5. Code review

   * Git에 code review 요청, 
   * Peer session 활용

6. 결과 제출 시

   * 실험 내용 결과 공유 및 기록 



---

### 🔍Segmentation <a name = 'seg'></a>

[전체 실험 내용](https://docs.google.com/spreadsheets/d/1JiopsJGh2aBIpnw7WPP2OvHHAEYdR9s0kT86OwruvAk/edit#gid=0)

#### 1. 대회 전략 <a name = 'strategy1'></a>

* Daily Mission 수행
* Augmentation & Loss 조합 실험
* Model 선정
* Skill  
  * TTA
  
  * SWA
  
  * Pseudo Labeling
  
  * Image 생성 - 김현우T1045 진행
  
    

#### 2. Model <a name = 'model1'></a>

1. efficientb3-noisy-student , FPN

- LB 점수 : 0.6248

- 모델 : decoder : FPN, backbone : efficientb3-noisy-student

- loss : Jaccard + SoftCE

- optimizer : AdamP (learning_rate = 0.0001), LookAhead 

- hyperparameters : Batch size 4, Epochs : 40

- augmentation

  - HorizontalFlip
  - ShiftScaleRotate
  - RandomBrightnessContrast
  - VerticalFlip
  - OneOf
    - A.RandomResizedCrop(512,512,scale = (0.5,0.8),p=0.8)
    - A.CropNonEmptyMaskIfExists(height=300, width=300, p=0.2),], p=0.5)
    - A.Resize(256, 256)
- SWA 



2. se_resnext101_32x4d, FPN

- LB 점수: 0.6228 (public)
- 모델 : decoder : FPN, backbone : se_resnext101_32x4d
- loss : Jacarrd
- optimizer : Adam (learning_rate = 0.00001)
- hyperparameters : Batch size 16, Epochs : 15
- augmentation
  - HorizontalFlip
  - VerticalFlip
  - ShiftScaleRotate
  - RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.2, p=0.5)
  - RandomResizedCrop(512,512,scale = (0.5,0.8))



3. efficient-b3 , FPN

- LB 점수: 0.5897 (public)
- 모델 : decoder : FPN, backbone : efficient-b3
- loss : Cross Entropy
- optimizer : AdamW (learning_rate = 0.00001)
- augmentation
  - HorizontalFlip
  - ShiftScaleRotate
  - RandomBrightnessContrast
  - RandomResizedCrop
  - OpticalDistortion
  - VerticalFlip
- pseudo hyperparameters : batch 8, epochs 20
- pseudo 학습: Fold로 나뉜 모델, 각각 psudo labeling 학습 진행



#### 3. Loss <a name = 'loss1'></a>

동일한 모델을 사용하여, Loss 값에 따른 Score 실험

- Decoder : deeplabV3+
- Backbone : efficientb3-noisy-student
- Optimizer : AdamW

![seg_chart](https://github.com/bcaitech1/p3-ims-obd-connectnet/blob/master/Team/headbreakz/Image/seg_chart.png?raw=true)



#### 4. Augmentation <a name = 'aug1'></a>

* 데이터에 적용 가능한 Augmentation [자세한 내용](https://github.com/bcaitech1/p3-ims-obd-connectnet/blob/akorea/akorea/segment/tips/argument.md)

![arg0](https://github.com/bcaitech1/p3-ims-obd-connectnet/blob/akorea/akorea/segment/images/arg0.png?raw=true)



* 사진을 어떻게 자를 것인가? [자세한 내용](https://github.com/bcaitech1/p3-ims-obd-connectnet/blob/akorea/akorea/segment/tips/crop.md)

![crop0](https://github.com/bcaitech1/p3-ims-obd-connectnet/blob/akorea/akorea/segment/images/crop0.png?raw=true)

```python 
Scale58 = RandomResizedCrop(512,512,scale = (0.5,0.8))
Scale68 =RandomResizedCrop(512,512,scale = (0.6,0.8))
Scale46 = RandomResizedCrop(512,512,scale = (0.4,0.6))
Scale24 = RandomResizedCrop(512,512,scale = (0.2,0.4))
```



#### 5. 실패한 부분 <a name = 'fail1'></a>

- Unet, Unet ++, Unet3+을 적용하려고 했으나 성능이 나오지 않았다.

- pseudo labeling

  - 학습 방법
    - label이 없는 데이터셋(test셋)에 대해 매 배치마다 생성 및 학습
    - 50batch마다 기존 train셋을 1epoch학습 진행

  ```markdown
  for each_test_image in test_loader:
  	model.eval()
  	output = output of model with each_test_image
  	oms = output label
  	model.train()
  	unlabled_loss = alpha_wight * CE(output,oms)
  	if batch % 50 == 0:
  		for each_train_image in train_loader:
  			train_output = output of model with each_train_image
        
  ```

  → 그러나,  CE(output, mos)는 사실상 같은 값이기 때문에 값이 0에 수렴하여 의미가 없다. 따라서, 위  pseudo code는 원래의  pseudo-labeling의 의도와는 다른 방식으로 작동한다. 그럼에도 불구하고 적용하지 않았을 때 보다 0.06이 상승하는 효과가 있었는데, 이는 단순히 내부 for문에 의해 train이 추가적으로 이루어진 결과에 기인한다고 생각한다.



#### 6. 회고 & 과제 <a name = 'try1'></a>

* Library 버전 통일하기

- EDA를 통해서 이미지 특성에 따라 실험하기 
- 시각화 코드도 한번 짜보기
- 학습 진행 시 ,다른 작업을 못할 때는 SOTA 모델 TTA나 CRF 같이 테크닉 탐색
- loss를 조합 시,  특정 모델에서만 좋았던 것일 수도 있으니 참고하기
- 최대한 작은 모델로 실험 하기 -> 기본적 조합 (loss, optim, augmentation, batch_size, lr, epoch)
- 기준 점수 (ex. 현재 single SOTA 점수의 +- 15%)를 충족하지 못하면 과감하게 드랍하기
- csv 파일로 soft voting, hard voting 앙상블 기능을 미리 구현
- 개인별 앙상블 미리 실험 
- 낮에는 작은 모델로 기능 테스트를 진행하고, 밤에는 성능이 좋은 모델로 실험을 진행함 -> 성능이 좋은 모델은 모두 다른 조합으로 진행함
- pseudo labeling 다른팀과 비교해서 검증 및 재사용
- 목표를 세분화해서 각각 데드라인을 지정
- 시간이 부족하면 validation score 측정 빼고 실험
- GPU 20 시간 돌리기!!!
- 항상 모델 pt 저장해서 필요할 때 사용하기
- 시각화 코드도 한번 짜보기



---

---



### 🔍Object Detection <a name = 'obd'></a>

[전체 실험 내용](https://docs.google.com/spreadsheets/d/1JiopsJGh2aBIpnw7WPP2OvHHAEYdR9s0kT86OwruvAk/edit#gid=346165051)

#### 1. 대회 전략 <a name = 'strategy2'></a>

* [Global wheat detection](https://www.kaggle.com/c/global-wheat-detection) 분석을  통해 사전 실험의 방향성을 수립

* 모델 설정
* augmentation & loss 조합 실험
* Multi Scale Train
* TTA
* WBF
* Pseudo Labeling  
* hyper parameter  튜닝

  

#### 2. Model <a name = 'model2'></a>

1. Cascade R-CNN 계열

   - ResNext101 / FPN / Cascade R-CNN

     - LB: 0.4781
     - optimizer : SGD (learning_rate = 0.02)
     - loss:  CrossEntropyLoss (Class loss),  SmoothL1Loss (Bbox loss)
     - hyperparameters : batch : 16, epochs : 50

   - ResNet50 / RFN + SAC / Cascade R-CNN (DetectoRS)

     - LB : 0.5121
     - optimizer : SGD (learning_rate = 0.01)
     - loss:  SoftCrossEntropyLoss (Class loss),  SmoothL1Loss (Bbox loss)
     - hyperparameters : batch : 4, epochs : 48 or 60
     - 5 fold cross-validation
     - TTA 

   - ResNext101 / RFN + SAC / Cascade R-CNN (DetectoRS)

     - LB: 0.5247
     - optimizer : SGD (learning_rate = 0.01)
     - loss:  SoftCrossEntropyLoss (Class loss),  SmoothL1Loss (Bbox loss)
     - hyperparameters : batch : 4, epochs : 48 or 60
     - TTA : vertical, horizontal flip, 512,  768 resize

     

2. YOLO 계열

   - DarkNet / SPP / YOLO v5
     - LB : 0.4916
     - loss : CrossEntropy (150 epoch models), Focal Loss (240 epoch models)
     - optimizer : SGD (learning_rate = 0.01)
     - hyperparameters : batch : 32, epochs : 150 or 240
     - TTA 
     - 원본 사이즈의 절반으로 Multi-scale train 진행 

       

3. Swin 계열

   - SwinTransformer / FPN / Mask R-CNN
     - LB: 0.5486
     - cls_loss: LabelSmooth + CE + Focal (각 box_head 별)
     - bbox_loss: SmoothL1Loss
     - optimizer: AdamW (learning_rate = 0.0001)
     - 재학습 
       - loss 변경 
       - augmentation 추가
     - TTA

       

#### 3. Augmentation <a name = 'aug2'></a>

- Mosaic

  - 이미지 4장을 각각 무작위로 잘라서 하나의 사진으로 만드는 augmentation
  - cutmix와 차이점 : cutmix는 자른 사진이 다른 사진을 가리는 구조, Mosaic은 4장의 랜덤하게 자른 사진을 하나로 합치는 구조

    ![mosaic](https://github.com/bcaitech1/p3-ims-obd-connectnet/blob/master/Team/headbreakz/Image/mosaic.png?raw=true)
    
    

- Mixup
- RandomRotate90
- HueSaturationValue
- CLAHE
- RandomBrightnessContrast
- RGBShift
- Blur
- MotionBlur
- GaussNoise
- ShiftScaleRotate
- Multi-scale



#### 4. 사용한 기술 <a name = 'skill2'></a>

1. Add Data
   * 외부 데이터 이용은 대회 규칙 상 금지
   * 마스크와 BBOX를 이용해서 원하는 오브젝트를 분리해서 다른 이미지에 붙일수 있을것이라고 생각.
   * 데이터상에서 Battery, Clothes, Metal, PaperPack, Glass 오브젝트가 부족한것으로 바악해서 해당 오브젝트를 기존 이미지에 추가하는 방식으로 데이터를 증강.
   * 부족하다고 판단된 오브젝트를 각 500개씩 증가
   * 데이터 추가 후 기본 베이스 라인 코드로 테스트 결과 0.05정도 점수 상승해서 폴드별로 데이터 추가
   * SWIN_T에서는 마스크 부분이 새로 생성된 이미지에 존재하지 않아 적용하지 못함.



2. WBF
   * 여러개의 bounding box를 각각의 확률을 가중평균으로 하여 하나의 bounding box로 나타내는 방식



3. Pseudo-Labeling

   - 기본모델 : Fold = 2 / size = 640 / EfficientDetD6

   1. 기본모델을 10 epoch 학습

      * 학습 데이터는 기본 train data + Pseudo-Labeling test data mixup

   2. 1에서 학습한 모델을 다시 6에폭동안 학습

      * 학습 데이터는 기본 train data + 1단계 모델 Pseudo-Labeling test data mixup

      

4. WBF hyperparameter

   * IOU threshold 와 Skip box threshold 조정
   * IOU threshold는 특정 임계값 이하로 내리면 오히려 성능이 하락
   * Skip box threshold는 낮은 값인 경우, test data visualization 시 지나치게 많은 박스가 존재하여 직관적으로는 납득하기 어려움
   *  최종값
     *  IOU threshold : 0.4, Skip box threshold : 0.01

     

#### 5. Ensemble <a name = 'ensemble2'></a>
![model_em](https://github.com/bcaitech1/p3-ims-obd-connectnet/blob/akorea/akorea/segment/images/model_em.png?raw=true)
- 총 26개 모델을 WBF와 threshold 최적화를 이용하여 앙상블
- stratified kfold방식으로 데이터셋을 5개(fold0,fold1,fold2,fold3,fold4)로 나뉘어 학습하여 앙상블
- 기준(0.5이상)을 넘긴 모델 앙상블 목록
  - YOLO v5
    - fold0, fold1, fold2, fold3, fold4
    - augment적용 fold0 ,fold1, fold2, fold3
    - fold4(img size 256)
  - Swin T
    - fold0, fold1, fold2, fold3
    - fold4(img size 768)
  - Cascade R-CNN
    - ResNet50
      - fold0, fold1, fold2, fold3
      - trainall data
    - ResNet101
      - fold0, fold1, fold2, fold3, fold4
      - train all data



#### 6. 실패한 부분 <a name = 'fail2'></a>

- pseudo labeling
  - 테스트 데이터셋을 inference 하면 csv 파일이 생성된다. LB 성능이 가장 좋은 결과 파일을 csv 파일을 기준으로 pseudo labeling 을 생성한다.
  - BBox 성능이 0.75 이상의 값만 읽어 COCO dataset  의 파일인 pseudo.json 파일을 생성한다.
  - pseudo.json 파일로 모델을 재학습시킨 모델의 성능을 올린다.






# Reference

### Paper

* [Zero-Shot Object Detection](https://arxiv.org/abs/1804.04340)


### GIt

* [Segmentation loss](https://github.com/JunMa11/SegLoss)
* [Segmentation Models](https://github.com/qubvel/segmentation_models.pytorch)
* [Lookahead optimizer](https://github.com/alphadl/lookahead.pytorch)
* [Weighted Boxes Fusion](https://github.com/ZFTurbo/Weighted-Boxes-Fusion)
* [Swin Transformer Object Detection](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection)


### Site & Kaggle

* [TACO Dataset](http://tacodataset.org/)
* [DeepLab_V3+](https://medium.com/hyunjulie/2%ED%8E%B8-%EB%91%90-%EC%A0%91%EA%B7%BC%EC%9D%98-%EC%A0%91%EC%A0%90-deeplab-v3-ef7316d4209d)
* [BCE loss](https://sosoeasy.tistory.com/337)
* [Loss Function 이해하기](https://keepdev.tistory.com/48)
* [MMDetection documentation](https://mmdetection.readthedocs.io/en/latest/index.html)
* [Kaggle Wheat Detection](https://www.kaggle.com/c/global-wheat-detection)



### 편의 기능

* [GIt 사용하기](https://github.com/bcaitech1/p3-ims-obd-connectnet/blob/master/headbreakz/git.md)
* [서버에서 제출하기 CODE](https://github.com/bcaitech1/p3-ims-obd-connectnet/blob/akorea/akorea/submit.py)
* [Wandb 간단 사용팁](https://github.com/bcaitech1/p3-ims-obd-connectnet/blob/akorea/akorea/tips/wandb.md)

