....................................................................![](https://noticon-static.tammolo.com/dgggcrkxq/image/upload/v1603679366/noticon/dcvetqndre7gda3ttijy.gif)....................................................................

---

# [AI_8기] 6조 머신러닝 & 딥러닝 팀과제 

| **팀원** | ✭박성규                                                                                            | 김민철                                                                                              | 이시헌                                                                                            | 박윤지                                                                                             |
|:------:|:-----------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------:|
|        | ![박성규님](https://github.com/user-attachments/assets/40f97c52-c562-44b0-bef6-12289e149d27) | ![김민철님](https://github.com/user-attachments/assets/28b83bd5-13c2-4249-beab-64f7567e1816) | ![이시헌님](https://github.com/user-attachments/assets/7b91b2aa-c113-44ed-8f41-e8df1ef7d06d) | ![박윤지님](https://github.com/user-attachments/assets/8d5be377-1a58-4f88-9ee2-176d1e1d162e) |
| **역할** | 오류 제어 및 REPO 관리                                                                                 | 예측 모델 성능 향상                                                                                      | README.md 작성                                                                                   | GIT 충돌 관리 및 팀원 코드 리뷰                                                                            |

## 개발 환경

![://noticon-static.tammolo.com/dgggcrkxq/image/upload/v1566791609/noticon/nen1y11gazeqhejw7nmhttps1.png](https://noticon-static.tammolo.com/dgggcrkxq/image/upload/v1566791609/noticon/nen1y11gazeqhejw7nm1.png) ![https://noticon-static.tammolo.com/dgggcrkxq/image/upload/v1626170585/noticon/uqui2rrxtt26ngudnhdu.png](https://noticon-static.tammolo.com/dgggcrkxq/image/upload/v1626170585/noticon/uqui2rrxtt26ngudnhdu.png)![https://noticon-static.tammolo.com/dgggcrkxq/image/upload/v1632975248/noticon/sph4ujixspcnhzpw8zky.png](https://noticon-static.tammolo.com/dgggcrkxq/image/upload/v1632975248/noticon/sph4ujixspcnhzpw8zky.png)

## 진행한 프로젝트 목록

- ### [타이타닉 생존자 예측](#%EF%B8%8F-타이타닉-생존자-예측)

- ### [영화 리뷰 감성 분석](#-영화-리뷰-감성-분석)

### ⛴️ 타이타닉 생존자 예측

> 타이타닉 탑승객 데이터셋을 활용해 생존자를 예측하는 모델을 만드는 프로젝트

#### 목표

<details>
<summary>1. 데이터셋 불러오기</summary>

```python
import seaborn as sns

titanic = sns.load_dataset('titanic')
```

> titanic Dataset

<!-- dataset df -->

<div> 
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>survived</th>
      <th>pclass</th>
      <th>sex</th>
      <th>age</th>
      <th>sibsp</th>
      <th>parch</th>
      <th>fare</th>
      <th>embarked</th>
      <th>class</th>
      <th>who</th>
      <th>adult_male</th>
      <th>deck</th>
      <th>embark_town</th>
      <th>alive</th>
      <th>alone</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>S</td>
      <td>Third</td>
      <td>man</td>
      <td>True</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>no</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>C</td>
      <td>First</td>
      <td>woman</td>
      <td>False</td>
      <td>C</td>
      <td>Cherbourg</td>
      <td>yes</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>S</td>
      <td>Third</td>
      <td>woman</td>
      <td>False</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>yes</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>S</td>
      <td>First</td>
      <td>woman</td>
      <td>False</td>
      <td>C</td>
      <td>Southampton</td>
      <td>yes</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>S</td>
      <td>Third</td>
      <td>man</td>
      <td>True</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>no</td>
      <td>True</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>886</th>
      <td>0</td>
      <td>2</td>
      <td>male</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>13.0000</td>
      <td>S</td>
      <td>Second</td>
      <td>man</td>
      <td>True</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>no</td>
      <td>True</td>
    </tr>
    <tr>
      <th>887</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>19.0</td>
      <td>0</td>
      <td>0</td>
      <td>30.0000</td>
      <td>S</td>
      <td>First</td>
      <td>woman</td>
      <td>False</td>
      <td>B</td>
      <td>Southampton</td>
      <td>yes</td>
      <td>True</td>
    </tr>
    <tr>
      <th>888</th>
      <td>0</td>
      <td>3</td>
      <td>female</td>
      <td>NaN</td>
      <td>1</td>
      <td>2</td>
      <td>23.4500</td>
      <td>S</td>
      <td>Third</td>
      <td>woman</td>
      <td>False</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>no</td>
      <td>False</td>
    </tr>
    <tr>
      <th>889</th>
      <td>1</td>
      <td>1</td>
      <td>male</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>30.0000</td>
      <td>C</td>
      <td>First</td>
      <td>man</td>
      <td>True</td>
      <td>C</td>
      <td>Cherbourg</td>
      <td>yes</td>
      <td>True</td>
    </tr>
    <tr>
      <th>890</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>32.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.7500</td>
      <td>Q</td>
      <td>Third</td>
      <td>man</td>
      <td>True</td>
      <td>NaN</td>
      <td>Queenstown</td>
      <td>no</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
<p>891 rows × 15 columns</p>
</div>
</details>
<br>

<details>
<summary>2. feature 분석</summary>

> 데이터 프레임 첫 5행

```python
titanic.head()
```

<!-- head df -->

<div>
<table border="1" class="dataframe">
   <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>survived</th>
      <th>pclass</th>
      <th>sex</th>
      <th>age</th>
      <th>sibsp</th>
      <th>parch</th>
      <th>fare</th>
      <th>embarked</th>
      <th>class</th>
      <th>who</th>
      <th>adult_male</th>
      <th>deck</th>
      <th>embark_town</th>
      <th>alive</th>
      <th>alone</th>
    </tr>
   </thead>
   <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>S</td>
      <td>Third</td>
      <td>man</td>
      <td>True</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>no</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>C</td>
      <td>First</td>
      <td>woman</td>
      <td>False</td>
      <td>C</td>
      <td>Cherbourg</td>
      <td>yes</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>S</td>
      <td>Third</td>
      <td>woman</td>
      <td>False</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>yes</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>S</td>
      <td>First</td>
      <td>woman</td>
      <td>False</td>
      <td>C</td>
      <td>Southampton</td>
      <td>yes</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>S</td>
      <td>Third</td>
      <td>man</td>
      <td>True</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>no</td>
      <td>True</td>
    </tr>
   </tbody>
   </table>
   </div>

> 통계 확인 

```python
titanic.describe()
```

`타이타닉 데이터셋 주요 항목 (행 row)`

| <span style="color:blue">**항목**</span>       | <span style="color:blue">**설명**</span> |
| -------------------------------------------- | -------------------------------------- |
| <span style="color:blue">**survived**</span> | 승객 생존 여부 (0 = 사망, 1 = 생존)              |
| <span style="color:green">**pclass**</span>  | 객실 등급 (1 = 1등석, 2 = 2등석, 3 = 3등석)      |
| <span style="color:purple">**age**</span>    | 승객 나이                                  |
| <span style="color:orange">**sibsp**</span>  | 동반한 형제자매 및 배우자 수                       |
| <span style="color:orange">**parch**</span>  | 동반한 부모 및 자녀 수                          |
| <span style="color:teal">**fare**</span>     | 승객이 지불한 운임 금액                          |

`타이타닉 데이터셋 주요 통계 (열 Column)`

| <span style="color:blue">**지표**</span>    | <span style="color:blue">**설명**</span>  |
| ----------------------------------------- | --------------------------------------- |
| <span style="color:blue">**count**</span> | 데이터가 존재하는 항목의 개수 (결측치를 제외한 값의 개수)       |
| <span style="color:green">**mean**</span> | 값들의 평균                                  |
| <span style="color:purple">**std**</span> | 표준편차 (데이터가 평균으로부터 얼마나 퍼져 있는지를 나타냄)      |
| <span style="color:orange">**min**</span> | 데이터의 최소값                                |
| <span style="color:teal">**25%**</span>   | 하위 25%에 해당하는 값. 데이터의 25%가 이 값보다 작음      |
| <span style="color:orange">**50%**</span> | 중위값 (데이터의 중간 값). 데이터의 50%가 이 값보다 작거나 같음 |
| <span style="color:teal">**75%**</span>   | 상위 25%에 해당하는 값. 데이터의 75%가 이 값보다 작음      |
| <span style="color:red">**max**</span>    | 데이터의 최대값                                |

> 데이터셋 통계

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>survived</th>
      <th>pclass</th>
      <th>age</th>
      <th>sibsp</th>
      <th>parch</th>
      <th>fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>714.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.383838</td>
      <td>2.308642</td>
      <td>29.699118</td>
      <td>0.523008</td>
      <td>0.381594</td>
      <td>32.204208</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.486592</td>
      <td>0.836071</td>
      <td>14.526497</td>
      <td>1.102743</td>
      <td>0.806057</td>
      <td>49.693429</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.420000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>20.125000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.910400</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>28.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>14.454200</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>38.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>31.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>80.000000</td>
      <td>8.000000</td>
      <td>6.000000</td>
      <td>512.329200</td>
    </tr>
  </tbody>
</table>
</div>
<br>

</details>

<br>

<details>
<summary>
3. feature engineering
</summary>

> 결측치 처리

`결측치 갯수 확인`

```python
titanic.isnull().sum()
```

| <span style="color:blue">**항목**</span>            | <span style="color:blue">**결측치 수**</span> |
| ------------------------------------------------- | ----------------------------------------- |
| <span style="color:blue">**survived**</span>      | 0                                         |
| <span style="color:green">**pclass**</span>       | 0                                         |
| <span style="color:purple">**sex**</span>         | 0                                         |
| <span style="color:orange">**age**</span>         | 177                                       |
| <span style="color:teal">**sibsp**</span>         | 0                                         |
| <span style="color:blue">**parch**</span>         | 0                                         |
| <span style="color:green">**fare**</span>         | 0                                         |
| <span style="color:purple">**embarked**</span>    | 2                                         |
| <span style="color:orange">**class**</span>       | 0                                         |
| <span style="color:teal">**who**</span>           | 0                                         |
| <span style="color:blue">**adult_male**</span>    | 0                                         |
| <span style="color:green">**deck**</span>         | 688                                       |
| <span style="color:purple">**embark_town**</span> | 2                                         |
| <span style="color:orange">**alive**</span>       | 0                                         |
| <span style="color:teal">**alone**</span>         | 0                                         |

`결측치 값 대체`

```python
#Age(나이)의 결측치는 중앙값으로, Embarked(승선 항구)의 결측치는 최빈값으로 대체. 
titanic['age'].fillna(titanic['age'].median(), inplace=True)
titanic['embarked'].fillna(titanic['embarked'].mode()[0], inplace=True)

# 대체한 후에, 대체 결과를 isnull() 함수와 sum()  함수를 이용해서 확인
print(titanic['age'].isnull().sum())
print(titanic['embarked'].isnull().sum())
```

`수치형으로 인코딩`

```python
# Sex(성별)를 남자는 0, 여자는 1로 변환. 
# alive(생존여부)를 True는 1, False는 0으로 변환. 
# Embarked(승선 항구)는 ‘C’는 0으로, Q는 1으로, ‘S’는 2로 변환. 
# 모두 변환한 후에, 변환 결과를 head 함수를 이용해 확인. 


titanic['sex'] = titanic['sex'].map({'male': 0, 'female': 1})
titanic['alive'] = titanic['alive'].map({'no': 1, 'yes': 0})
titanic['embarked'] = titanic['embarked'].map({'C': 0, 'Q': 1, 'S': 2,})

print(titanic['sex'].head())
print(titanic['alive'].head())
print(titanic['embarked'].head())
```

`새로운 feature 생성`

```python
#Sibsp , Parch 를 통해 family_size 생성
#새로운 Feature를 head함수를 이용해 확인

titanic['family_size'] = titanic['sibsp'] + titanic['parch'] + 1

print(titanic['family_size'].head())
```

> 가족구성원 항목 추가된 데이터프레임

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>survived</th>
      <th>pclass</th>
      <th>sex</th>
      <th>age</th>
      <th>sibsp</th>
      <th>parch</th>
      <th>fare</th>
      <th>embarked</th>
      <th>class</th>
      <th>who</th>
      <th>adult_male</th>
      <th>deck</th>
      <th>embark_town</th>
      <th>alive</th>
      <th>alone</th>
      <th>family_size</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>2</td>
      <td>Third</td>
      <td>man</td>
      <td>True</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>1</td>
      <td>False</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>0</td>
      <td>First</td>
      <td>woman</td>
      <td>False</td>
      <td>C</td>
      <td>Cherbourg</td>
      <td>0</td>
      <td>False</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>2</td>
      <td>Third</td>
      <td>woman</td>
      <td>False</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>0</td>
      <td>True</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>2</td>
      <td>First</td>
      <td>woman</td>
      <td>False</td>
      <td>C</td>
      <td>Southampton</td>
      <td>0</td>
      <td>False</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>2</td>
      <td>Third</td>
      <td>man</td>
      <td>True</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>1</td>
      <td>True</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>886</th>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>13.0000</td>
      <td>2</td>
      <td>Second</td>
      <td>man</td>
      <td>True</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>1</td>
      <td>True</td>
      <td>1</td>
    </tr>
    <tr>
      <th>887</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>19.0</td>
      <td>0</td>
      <td>0</td>
      <td>30.0000</td>
      <td>2</td>
      <td>First</td>
      <td>woman</td>
      <td>False</td>
      <td>B</td>
      <td>Southampton</td>
      <td>0</td>
      <td>True</td>
      <td>1</td>
    </tr>
    <tr>
      <th>888</th>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>28.0</td>
      <td>1</td>
      <td>2</td>
      <td>23.4500</td>
      <td>2</td>
      <td>Third</td>
      <td>woman</td>
      <td>False</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>1</td>
      <td>False</td>
      <td>4</td>
    </tr>
    <tr>
      <th>889</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>30.0000</td>
      <td>0</td>
      <td>First</td>
      <td>man</td>
      <td>True</td>
      <td>C</td>
      <td>Cherbourg</td>
      <td>0</td>
      <td>True</td>
      <td>1</td>
    </tr>
    <tr>
      <th>890</th>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>32.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.7500</td>
      <td>1</td>
      <td>Third</td>
      <td>man</td>
      <td>True</td>
      <td>NaN</td>
      <td>Queenstown</td>
      <td>1</td>
      <td>True</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>891 rows × 16 columns</p>
</div>

</details>
<br>

<details>
<summary>
4. 모델 학습시키기 (Logistic Regression, Decision Tree, XGBoost)  
</summary>

`데이터 스케일링 진행`

```py
#feature와 target 분리

titanic = titanic[['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'family_size']]
X = titanic.drop('survived', axis=1) # feature
y = titanic['survived'] # target

# x는 승객의 생존 여부를 제외한 나머지 모든 열을 학습에 사용할 특징
# y는 승객이 생존했는지의 여부
# x로 y를 예측
```

> Logistic Regression

```py
# Logistic Regression

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

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
```

> 🔍 Logistic Regression 결과 요약

| **지표**                | **희생자 (0)**                      | **생존자 (1)**                      |
| --------------------- | -------------------------------- | -------------------------------- |
| **정밀도 (Precision)**   | 0.82 (모델이 예측한 '희생자' 중 실제 희생자 비율) | 0.78 (모델이 예측한 '생존자' 중 실제 생존자 비율) |
| **재현율 (Recall)**      | 0.86 (실제 희생자 중 정확히 예측한 비율)       | 0.73 (실제 생존자 중 정확히 예측한 비율)       |
| **F1-스코어 (F1-Score)** | 0.84 (정밀도와 재현율의 조화평균)            | 0.76 (정밀도와 재현율의 조화평균)            |
| **지원 (Support)**      | 105                              | 74                               |

| **평균 지표**          | **값**                                         |
| ------------------ | --------------------------------------------- |
| **정확도 (Accuracy)** | 0.80 (전체 데이터에서 정확히 예측한 비율)                    |
| **Macro 평균**       | Precision: 0.80, Recall: 0.79, F1-Score: 0.80 |
| **Weighted 평균**    | Precision: 0.80, Recall: 0.80, F1-Score: 0.80 |

**요약**: Logistic Regression 모델은 약 80%의 정확도를 보이며, 희생자를 예측하는 데 있어서 재현율이 높아(0.86) 희생자를 잘 예측. 생존자에 대한 재현율은 상대적으로 낮아(0.73) 생존자를 놓치는 경향이 약간 있음.

> Decision Tree

```py
#Decision Tree

from sklearn.tree import DecisionTreeClassifier  # Decision Tree 분류기
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
```

> 🔍 Decision Tree 모델 결과 요약

| **지표**                | **희생자 (0)**                      | **생존자 (1)**                      |
| --------------------- | -------------------------------- | -------------------------------- |
| **정밀도 (Precision)**   | 0.83 (모델이 예측한 '희생자' 중 실제 희생자 비율) | 0.70 (모델이 예측한 '생존자' 중 실제 생존자 비율) |
| **재현율 (Recall)**      | 0.76 (실제 희생자 중 정확히 예측한 비율)       | 0.78 (실제 생존자 중 정확히 예측한 비율)       |
| **F1-스코어 (F1-Score)** | 0.80 (정밀도와 재현율의 조화평균)            | 0.74 (정밀도와 재현율의 조화평균)            |
| **지원 (Support)**      | 105                              | 74                               |

| **평균 지표**          | **값**                                         |
| ------------------ | --------------------------------------------- |
| **정확도 (Accuracy)** | 0.77 (전체 데이터에서 정확히 예측한 비율)                    |
| **Macro 평균**       | Precision: 0.77, Recall: 0.77, F1-Score: 0.77 |
| **Weighted 평균**    | Precision: 0.78, Recall: 0.77, F1-Score: 0.77 |

**요약**: Decision Tree 모델은 약 77%의 정확도를 보이며, 희생자 예측에서 정밀도가 높아(0.83) 희생자를 잘 분류하는 경향이 있음. 생존자의 재현율이 다소 높아(0.78) 생존자를 놓치는 경우는 적으나, 정밀도가 희생자에 비해 낮아(0.70) 생존자 예측 정확도가 떨어질 수 있음.

> XGBoost

```py
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
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
print(f'XGBoost 모델의 MSE: {mse_xgb}')
```

> 🔍 XGBoost 모델 결과 요약

| **지표**                | **희생자 (0)**                      | **생존자 (1)**                      |
| --------------------- | -------------------------------- | -------------------------------- |
| **정밀도 (Precision)**   | 0.82 (모델이 예측한 '희생자' 중 실제 희생자 비율) | 0.78 (모델이 예측한 '생존자' 중 실제 생존자 비율) |
| **재현율 (Recall)**      | 0.86 (실제 희생자 중 정확히 예측한 비율)       | 0.73 (실제 생존자 중 정확히 예측한 비율)       |
| **F1-스코어 (F1-Score)** | 0.84 (정밀도와 재현율의 조화평균)            | 0.76 (정밀도와 재현율의 조화평균)            |
| **지원 (Support)**      | 105                              | 74                               |

| **평균 지표**          | **값**                                         |
| ------------------ | --------------------------------------------- |
| **정확도 (Accuracy)** | 0.80 (전체 데이터에서 정확히 예측한 비율)                    |
| **Macro 평균**       | Precision: 0.80, Recall: 0.79, F1-Score: 0.80 |
| **Weighted 평균**    | Precision: 0.80, Recall: 0.80, F1-Score: 0.80 |

**요약**: XGBoost 모델은 약 80%의 정확도를 보이며, 희생자 예측에서 높은 재현율(0.86)로 실제 희생자를 잘 식별하는 경향이 있음. 생존자 예측에서는 정밀도가 상대적으로 높아(0.78) 생존자를 더 정확하게 예측하며, 생존자 재현율은 0.73으로 다소 낮음. 전반적으로, XGBoost 모델은 희생자 식별에 강점을 보임.

</details>

<br>

<details>
<summary>
5. 모델별 시각화 자료 (추가)
</summary>

> 혼동 행렬 시각화

```py
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# 혼동 행렬 계산
cm = confusion_matrix(y_test, y_pred)

# 혼동 행렬 시각화
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Cancer', 'Cancer'], yticklabels=['Not Cancer', 'Cancer'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
```

![confusionmatrix](https://github.com/user-attachments/assets/d377b51b-89ca-49db-a371-e93f2cb9580b)

> 결정 트리 시각화

```py
from sklearn.tree import plot_tree

# 결정 트리 시각화
plt.figure(figsize=(12, 8))
plot_tree(model, filled=True, feature_names=X.columns, class_names=['Not Survived', 'Survived'])
plt.title('Decision Tree Visualization')
plt.show()
```

![plottree](https://github.com/user-attachments/assets/22f0a9a5-fdff-4971-b30a-bbc9adbcc70d)

> XGBoost 시각화

`예측 결과와 실제 값을 비교하는 산점도 시각화`

```py
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_xgb, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # 대각선 추가
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values (XGBoost)')
plt.show()
```

![xgboost1](https://github.com/user-attachments/assets/2de22727-acbe-4130-a8a6-00df409d19be)

`XGBOOST의 특성 중요도 시각화`

```py
plt.figure(figsize=(12, 8))
xgb.plot_importance(xgb_model, importance_type='weight', max_num_features=10)
plt.title('Feature Importance (XGBoost)')
plt.show()
```

![xgboost2](https://github.com/user-attachments/assets/6171d604-b6b4-4f84-97ff-90693c919e21)

</details>
<br>
<details>
<summary>
6. 모델 성능 비교 (추가)
</summary>

> 🐳 타이타닉 생존자 예측 결과 모델 성능 비교

| **모델**                  | **Accuracy** | <span style="color:red">**Precision (희생자)**</span> | <span style="color:blue">**Precision (생존자)**</span> | <span style="color:red">**Recall (희생자)**</span> | <span style="color:blue">**Recall (생존자)**</span> | <span style="color:red">**F1-Score (희생자)**</span> | <span style="color:blue">**F1-Score (생존자)**</span> |
| ----------------------- | ------------ | -------------------------------------------------- | --------------------------------------------------- | ----------------------------------------------- | ------------------------------------------------ | ------------------------------------------------- | -------------------------------------------------- |
| **Logistic Regression** | 0.8045       | <span style="color:red">0.82</span>                | <span style="color:blue">0.78</span>                | <span style="color:red">0.86</span>             | <span style="color:blue">0.73</span>             | <span style="color:red">0.84</span>               | <span style="color:blue">0.76</span>               |
| **Decision Tree**       | 0.7709       | <span style="color:red">0.83</span>                | <span style="color:blue">0.70</span>                | <span style="color:red">0.76</span>             | <span style="color:blue">0.78</span>             | <span style="color:red">0.80</span>               | <span style="color:blue">0.74</span>               |
| **XGBoost**             | 0.8045       | <span style="color:red">0.82</span>                | <span style="color:blue">0.78</span>                | <span style="color:red">0.86</span>             | <span style="color:blue">0.73</span>             | <span style="color:red">0.84</span>               | <span style="color:blue">0.76</span>               |

##### 요약

- **Logistic Regression**와 **XGBoost** 모델은 동일한 정확도(80.45%)로 높은 성능을 보임.
- **Decision Tree**는 정확도는 상대적으로 낮지만, 생존자 클래스(1)의 Recall이 높아 생존자를 잘 예측.
- **Logistic Regression**와 **XGBoost** 모델이 Decision Tree보다 전반적으로 우수한 성능을 보임.

</details>

### 🎬 영화 리뷰 감성 분석

> 영화 리뷰 데이터를 사용하여 긍정적/부정적 감정을 분류하는 모델을 만드는 프로젝트 

#### 목표

1. 데이터셋 불러오기

2. 데이터 전처리

3. feature 분석 (EDA)

4. 리뷰 예측 모델 학습시키기 (LSTM)

#### 추가 목표

- [x] NLP 이용

- [x] 긍정 / 부정 리뷰의 워드 클라우드 그려보기 

#### 
