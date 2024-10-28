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

@

### 📊 Random Forest

#### 1. **Accuracy**:

- **Accuracy**: <span style="color:orange;">**77.1%**</span>  
  
  > 전체 테스트 데이터 중 모델이 올바르게 예측한 비율.

---

#### 2. **Classification Report**:

| Class      | Precision                              | Recall                                 | F1-Score                               | Support                              |
| ---------- | -------------------------------------- | -------------------------------------- | -------------------------------------- | ------------------------------------ |
| **0 (사망)** | <span style="color:blue;">0.83</span>  | <span style="color:blue;">0.76</span>  | <span style="color:blue;">0.80</span>  | <span style="color:blue;">105</span> |
| **1 (생존)** | <span style="color:green;">0.70</span> | <span style="color:green;">0.78</span> | <span style="color:green;">0.74</span> | <span style="color:green;">74</span> |

- **Precision**: 
  
  - 클래스 0: <span style="color:blue;">83%</span> (모델이 사망으로 예측한 것 중 실제 사망 비율)
  - 클래스 1: <span style="color:green;">70%</span> (모델이 생존으로 예측한 것 중 실제 생존 비율)

- **Recall**: 
  
  - 클래스 0: <span style="color:blue;">76%</span> (실제 사망자 중 모델이 올바르게 예측한 비율)
  - 클래스 1: <span style="color:green;">78%</span> (실제 생존자 중 모델이 올바르게 예측한 비율)

- **F1-Score**: 
  
  - 클래스 0: <span style="color:blue;">0.80</span>
  - 클래스 1: <span style="color:green;">0.74</span>

---

#### 3. **Averages**:

- **Macro Average**:
  
  - Precision: <span style="color:purple;">77%</span>
  - Recall: <span style="color:purple;">77%</span>
  - F1-Score: <span style="color:purple;">77%</span>

- **Weighted Average**:
  
  - Precision: <span style="color:orange;">78%</span>
  - Recall: <span style="color:orange;">77%</span>
  - F1-Score: <span style="color:orange;">77%</span>

---

### 📝 Conclusion:

- 전체적으로 <span style="color:orange;">**77.1%**</span>의 정확도를 보이며, 클래스 0(사망)에서 더 높은 성능을 나타냅니다. 클래스 1(생존)의 성능은 약간 낮지만 여전히 양호합니다. 
- 모델은 생존자 예측에서 신뢰성을 가지며, 사망자를 예측하는 데 더 높은 정밀도를 보여줍니다.

# XGBoost 모델 결과

- **모델 정확도**: `0.8045`  
  모델이 테스트 데이터에서 약 80.45%의 정확도로 생존자를 예측했습니다.

### Classification Report

          precision    recall  f1-score   support
    
       0       0.82      0.86      0.84       105
       1       0.78      0.73      0.76        74

accuracy                           0.80       179

- **Precision**: 양성으로 예측한 것 중 실제로 양성인 비율  
- **Recall**: 실제 양성 중에서 양성으로 올바르게 예측한 비율  
- **F1-Score**: Precision과 Recall의 조화 평균  
- **Support**: 각 클래스의 실제 샘플 수  

### 요약

- **클래스 0 (생존하지 않은 사람)**: 
  
  - Precision: 82%
  - Recall: 86%

- **클래스 1 (생존한 사람)**: 
  
  - Precision: 78%
  - Recall: 73%

전반적으로 모델의 성능은 양호하며, 생존자 예측에 효과적입니다.

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


