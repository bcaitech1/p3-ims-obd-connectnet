# 부스트캠프 Tech AI Stage 3
- object detect & segment
- 2021-04-27 ~ 2021-05-21

## 기본 데이터베이스
- 외부에 버려진 쓰레기 사진 이미지
    - 쓰레기 종류
        - "UNKNOWN","General trash","Paper","Paper pack","Metal","Glass", "Plastic","Styrofoam","Plastic bag","Battery","Clothing"  
    - 쓰레기 위치
        - bbox
        - segmentation

## segmentation
- 이미지로부터 쓰레기를 찾아서 sementic segmentation 및 분류 진행
- FPN+effieientnet_b3에 fold1에대해 psudo 학습 진행

## object detect
- 이미지로부터 쓰레기를 찾고(bbox) 쓰레기를 분류
- yolov5와 SwinT모델 학습 진행


