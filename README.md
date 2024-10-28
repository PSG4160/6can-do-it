# [AI_8기] 6조 머신러닝 & 딥러닝 팀과제

## 개발 환경

![https://noticon-static.tammolo.com/dgggcrkxq/image/upload/v1566791609/noticon/nen1y11gazeqhejw7nm1.png](https://noticon-static.tammolo.com/dgggcrkxq/image/upload/v1566791609/noticon/nen1y11gazeqhejw7nm1.png) ![https://noticon-static.tammolo.com/dgggcrkxq/image/upload/v1626170585/noticon/uqui2rrxtt26ngudnhdu.png](https://noticon-static.tammolo.com/dgggcrkxq/image/upload/v1626170585/noticon/uqui2rrxtt26ngudnhdu.png)



## 참여 인원

| ✭**박성규** | **김민철** | **이**시헌 | **박윤지** |
|:--------:|:-------:|:-------:|:-------:|

## 진행한 프로젝트 목록

### 타이타닉 생존자 예측

> 타이타닉 탑승객 데이터셋을 활용해 생존자를 예측하는 모델을 만드는 프로젝트

#### 목표

1.  데이터셋 불러오기

2.  feature 분석

3.  feature engineering

4.  모델 학습시키기

#### 결과

##### 모델 성능 비교

| 모델          | Accuracy | Precision (0) | Recall (0) | F1-Score (0) | Precision (1) | Recall (1) | F1-Score (1) |
| ----------- | -------- | ------------- | ---------- | ------------ | ------------- | ---------- | ------------ |
| **로지스틱 회귀** | 0.8045   | 0.82          | 0.86       | 0.84         | 0.78          | 0.73       | 0.76         |
| **결정 트리**   | 0.7709   | 0.83          | 0.76       | 0.80         | 0.70          | 0.78       | 0.74         |
| **XGBoost** | 0.8045   | 0.82          | 0.86       | 0.84         | 0.78          | 0.73       | 0.76         |

##### 요약

- **로지스틱 회귀**와 **XGBoost** 모델은 동일한 정확도(80.45%)를 기록하며, 생존자를 잘 예측했습니다.
- **결정 트리**는 상대적으로 낮은 정확도(77.09%)를 보였지만, 클래스 1(생존자)의 Recall이 가장 높았습니다.
- 전체적으로 **로지스틱 회귀**와 **XGBoost**가 유사한 성능을 보이며, 두 모델 모두 **결정 트리**보다 우수한 성능을 나타냈습니다.

### 영화 리뷰 감성 분석

> 영화 리뷰 데이터를 사용하여 긍정적/부정적 감정을 분류하는 모델을 만드는 프로젝트 

#### 목표

1. 데이터셋 불러오기

2. 데이터 전처리

3. feature 분석 (EDA)

4. 리뷰 예측 모델 학습시키기 (LSTM)

#### 추가 목표

- [x]  NLP 이용

- [x] 긍정 / 부정 리뷰의 워드 클라우드 그려보기


