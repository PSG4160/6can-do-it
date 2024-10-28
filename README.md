# [AI_8기] 6조 머신러닝 & 딥러닝 팀과제

## 개발 환경

![https://noticon-static.tammolo.com/dgggcrkxq/image/upload/v1566791609/noticon/nen1y11gazeqhejw7nm1.png](https://noticon-static.tammolo.com/dgggcrkxq/image/upload/v1566791609/noticon/nen1y11gazeqhejw7nm1.png) ![https://noticon-static.tammolo.com/dgggcrkxq/image/upload/v1626170585/noticon/uqui2rrxtt26ngudnhdu.png](https://noticon-static.tammolo.com/dgggcrkxq/image/upload/v1626170585/noticon/uqui2rrxtt26ngudnhdu.png)



## 참여 인원

| ✭**박성규** | **김민철** | **이**시헌 | **박윤지** |
|:--------:|:-------:|:-------:|:-------:|

## 진행한 프로젝트 목록

### ⛴️ 타이타닉 생존자 예측

> 타이타닉 탑승객 데이터셋을 활용해 생존자를 예측하는 모델을 만드는 프로젝트

#### 목표

1.  데이터셋 불러오기

2.  feature 분석

3.  feature engineering

4.  모델 학습시키기

####     타이타닉 생존자 예측 결과 모델 성능 비교

| **모델**      | **Accuracy** | <span style="color:red">**Precision (희생자)**</span> | <span style="color:blue">**Precision (생존자)**</span> | <span style="color:red">**Recall (희생자)**</span> | <span style="color:blue">**Recall (생존자)**</span> | <span style="color:red">**F1-Score (희생자)**</span> | <span style="color:blue">**F1-Score (생존자)**</span> |
| ----------- | ------------ | -------------------------------------------------- | --------------------------------------------------- | ----------------------------------------------- | ------------------------------------------------ | ------------------------------------------------- | -------------------------------------------------- |
| **로지스틱 회귀** | 0.8045       | <span style="color:red">0.82</span>                | <span style="color:blue">0.78</span>                | <span style="color:red">0.86</span>             | <span style="color:blue">0.73</span>             | <span style="color:red">0.84</span>               | <span style="color:blue">0.76</span>               |
| **결정 트리**   | 0.7709       | <span style="color:red">0.83</span>                | <span style="color:blue">0.70</span>                | <span style="color:red">0.76</span>             | <span style="color:blue">0.78</span>             | <span style="color:red">0.80</span>               | <span style="color:blue">0.74</span>               |
| **XGBoost** | 0.8045       | <span style="color:red">0.82</span>                | <span style="color:blue">0.78</span>                | <span style="color:red">0.86</span>             | <span style="color:blue">0.73</span>             | <span style="color:red">0.84</span>               | <span style="color:blue">0.76</span>               |

##### 요약

- **로지스틱 회귀**와 **XGBoost** 모델은 동일한 정확도(80.45%)로 높은 성능을 보임.
- **결정 트리**는 정확도는 상대적으로 낮지만, 생존자 클래스(1)의 Recall이 높아 생존자를 잘 예측.
- **로지스틱 회귀**와 **XGBoost** 모델이 결정 트리보다 전반적으로 우수한 성능을 보임.

### 🎬 영화 리뷰 감성 분석

> 영화 리뷰 데이터를 사용하여 긍정적/부정적 감정을 분류하는 모델을 만드는 프로젝트 

#### 목표

1. 데이터셋 불러오기

2. 데이터 전처리

3. feature 분석 (EDA)

4. 리뷰 예측 모델 학습시키기 (LSTM)

#### 추가 목표

- [x]  NLP 이용

- [x] 긍정 / 부정 리뷰의 워드 클라우드 그려보기


### 🔍과제 특이점
과제 특이점

#### 🌟 박성규



#### 🌟 김민철



#### 🌟 이시헌



#### 🌟 박윤지



