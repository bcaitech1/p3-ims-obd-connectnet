# 파일설명

- random seed는 21로 진행

## train.py
- 학습진행
- args는 따로 추가하지 않음
- 모델을추가(or models.py에 따로 구현)하여 진행
- 저장되는 파일이름을 따로 지정

## inference.py
- 학습한 모델을 불러서 submission.csv생성
- args는 추가하지않음
- 모델이름과, 저장파일이름 지정

## model.py
- 구현한 모델들

## loss.py
- loss모음

## submit.py
- config.py에 submitkey변수에 키를 넣어준다.
- 보낼 파일명, 설명을 적어서 실행하면 된다.