# 수정 테스트
# 데이터 불러오기
import seaborn as sns


titanic = sns.load_dataset('titanic')
print(titanic.head())

print(titanic.describe())
    #count : 결측값이 아닌 데이터 수
    #mean : 평균값
    #std : 분산 (퍼짐정도)
    #min : 최소값
    #25% : 하위 25%에 해당하는 값
    #50% : 중간값
    #75 % : 상위 25%에 해당하는 값
    #max : 최대값

# 결측값 확인
print(titanic.isnull().sum())

# Age(나이)의 결측치는 중앙값으로, Embarked(승선 항구)의 결측치는 최빈값으로 대체해주세요. 모두 대체한 후에, 대체 결과를 isnull() 함수와 sum()  함수를 이용해서 확인해주세요. 

titanic['age'] = titanic['age'].fillna(titanic['age'].median())
titanic['embarked'] = titanic['embarked'].fillna(titanic['embarked'].mode()[0])

print(titanic['age'].isnull().sum())
print(titanic['embarked'].isnull().sum())

# 수치형으로 인코딩

titanic['sex'] = titanic['sex'].map({'male': 0, 'female': 1})
titanic['alive'] = titanic['alive'].map({'no': 1, 'yes': 0})
titanic['embarked'] = titanic['embarked'].map({'C': 0, 'Q': 1, 'S': 2,})

print(titanic['sex'].head())
print(titanic['alive'].head())
print(titanic['embarked'].head())

# 새로운 feature 생성
# SibSip(타이타닉호에 동승한 자매 및 배우자의 수), Parch(타이타닉호에 동승한 부모 및 자식의 수)를 통해서 family_size(가족크기)를 생성해주세요. 새로운 feature를 `head` 함수를 이용해 확인해주세요.
titanic['family_size'] = titanic['sibsp'] + titanic['parch'] + 1

print(titanic['family_size'].head())

print(titanic.head())

# 4. **모델 학습시키기 (Logistic Regression, Random Forest, XGBoost)**
    
#    4-1. 모델 학습 준비 
    
#    이제 모델을 학습시키기 위한 데이터를 준비하겠습니다. 학습에 필요한 feature은 'survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', ‘family_size’ 입니다. feature과 target을 분리해주세요.  그 다음 데이터 스케일링을 진행해주세요.

titanic = titanic[['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'family_size']]
X = titanic.drop('survived', axis=1) # feature
y = titanic['survived'] # target

# Logistic Regression
import pandas as pd

from sklearn.model_selection import train_test_split # 훈련세트 만들기
from sklearn.linear_model import LogisticRegression # 로지스틱 회귀 라이브러리
from sklearn.preprocessing import StandardScaler # 스케일링 라이브러리
from sklearn.metrics import accuracy_score, classification_report # 모델 평가 라이브러리

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 데이터 스케일링
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 모델 생성 및 학습
model = LogisticRegression()
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 평가
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

# Random Forest
from sklearn.tree import DecisionTreeClassifier
# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 데이터 스케일링
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 모델 생성 및 학습
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 평가
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

# XGboost
import xgboost as xgb
from sklearn.metrics import mean_squared_error

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 데이터 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# XGBoost 모델 생성
xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

# 모델 학습
xgb_model.fit(X_train_scaled, y_train)

# 예측
y_pred_xgb = xgb_model.predict(X_test_scaled)

# 평가
mse_xgb = mean_squared_error(y_test, y_pred_xgb) # 잔차 제곱의 평균/ 작은 값일수록 실제 값과 가까움

print(f'XGBoost 모델의 MSE: {mse_xgb}')