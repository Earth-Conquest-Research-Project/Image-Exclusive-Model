# 캡스톤 그로스 1차 보고서

## 1. Team Info
### 1.1 과제명
**Image-Exclusive Model for Galaxy Merger Classification: Training on Simulations and Inference on Observations**

### 1.2 팀 정보
- 팀 번호: 07  
- 팀 이름: 우주정복 연구젝트 (SpaceConquest ResearchProject)

### 1.3 팀 구성원
- 이한나 (2271050)  
- 정소은 (2271053)  
- 정은채 (2271056)


## 2. Project-Summary

### 2.1 문제 정의
은하 병합은 단순히 두 은하의 물리적 결합을 넘어, 가스 밀도, 별 형성률, 중심 블랙홀의 성장 등 은하의 물리·화학적 진화 전반에 중대한 영향을 미친다. 따라서 병합 여부뿐 아니라 그 시점을 파악하는 것은 은하의 물리적 특성과 우주 진화 맥락을 해석하는 데 중요한 의미를 갖는다.  
기존 은하 병합 분류 연구는 이미지 기반 딥러닝 모델에 의존해 왔다. 그러나 뚜렷한 형태학적 흔적이 남는 **ongoing merger**와 달리, **pre-merger**와 **post-merger** 단계는 형태적 특징이 약하거나 일시적이어서 이미지 기반 판별이 본질적으로 어렵다.  
또한, 학습용 시뮬레이션 이미지를 생성·전처리하는 과정 자체가 막대한 연산 자원과 시간을 요구하며, 모델 학습과 추론 단계 또한 고성능 GPU와 긴 처리 시간이 필요하다. 이로 인해 이미지는 대규모 관측 카탈로그에 직접 적용하기에 실용성이 떨어지고 재현성·해석 가능성에도 제약이 따른다. 따라서 보다 연산 효율적이면서 과학적 해석이 용이한 대안적 접근이 요구된다.


### 2.2 기존 연구와의 비교
- **기존 연구**  
  - Pearson+2024: 시뮬레이션 이미지를 기반으로 galaxy merger stage 분류를 시도한 연구이다.  
  - non-merger, pre-merger, post-merger까지 분류하며, 정확도 **0.81±0.01** 수준에 도달했다.

- **우리 연구의 차별점**  
  - 효율성을 위해 이미지를 배제하고, 관측 가능한 은하의 **물리량만 사용**하여 모델을 학습한다.


### 2.3 제안 내용
이 연구는 galaxy merger가 은하 진화와 우주론에서 중요한 역할을 한다는 점에 주목하고, 특히 merger의 시점(pre-merger vs post-merger)이 은하의 물리적 특성에 다른 영향을 줄 수 있음을 강조한다.  
그러나 관측 이미지만으로는 현재 시점의 순간만 볼 수 있어 merger stage를 정확히 구분하기 어렵다. 기존에는 이미지 기반 머신러닝 모델로 merger stage를 예측하려는 시도가 있었지만, 이들은 고화질 이미지와 높은 연산 자원을 요구하는 한계가 있다.  
따라서 본 연구는 **이미지를 배제하고, 은하의 분광·측광 기반 물리 파라미터만을 활용해 merger stage를 분류하는 모델**을 제안하며, 이미지 기반 분석에 준하는 성능을 달성하는 것을 목표로 한다.


### 2.4 기대 효과 및 의의
- 이미지 없이도 merger stage 분류 성능을 확보하여 **계산 효율성 및 실제 관측 적용성**을 증대시킨다.  
- 물리량 기반 분류 결과가 이미지 기반 연구와 유사한 성능을 보일 경우, **경제적이고 실용적인 대안**으로 자리매김할 수 있다.  
- 향후 **멀티모달 모델(이미지+물리량)**을 위한 기초 연구 데이터셋 및 baseline 성능을 제공한다.


### 2.5 주요 기능
- **데이터 활용**  
  - Spectroscopic data: color_Ur, color_gr, color_gi, SFR_200Myr, AxisRatio, B_T, SurfaceBrightness, EffectiveRadius, AbsMag_U, AbsMag_B, AbsMag_V, AbsMag_g, AbsMag_r, AbsMag_i, StellarMass, Stellarmetallicity, StellarVelDispesion  
  - Photometric data: GiniCoefficient, Asymmetry, B/T, axisRatio  

- **모델 기능**  
  - 위 물리량 데이터를 사용하여 은하 병합 단계를 **non-merger, pre-merger, post-merger**로 분류한다.


## 3. Project-Design (과제 설계)

### 3.1 요구사항 정의

#### 목표 사용자
- 천문학자  
- 데이터 과학자  
- 시뮬레이션 및 관측 데이터를 다루는 연구자  

#### 문제점
- Ongoing merger 외에는 이미지 기반 분류가 어렵다.  
- 기존 이미지 기반 딥러닝 모델은 연산 자원 소모가 크고, 고해상도 이미지를 요구한다.  
- 이미지 데이터를 이용한 학습은 관측 데이터 한계로 인해 적용성이 떨어진다.  
- 결측치가 많은 물리량 데이터의 활용도가 낮다.  

#### 해결책
- 이미지를 완전히 제외한 **물리량 기반 모델 (Image-exclusive model)**을 개발한다.  
- Scaler를 통해 값들을 정규화하고 결측치를 **MICE**로 보정하여, 모델 학습에 적합한 형태의 데이터를 구축한다.  
- Classical ML 모델과 Deep Learning 모델 등 다양한 실험을 통해 **가장 높은 정확도를 가진 모델**을 선정한다.  
- **Illustris TNG 시뮬레이션 데이터** 기반으로 학습한 뒤 **SDSS 실제 관측 데이터**에 적용한다.  


#### 실험 상세 설명
1. **데이터 전처리**  
   - Illustris TNG 시뮬레이션 데이터를 활용하여 병합 단계를 레이블링한다.  
   - Scaler를 통해 Normalization을 적용한다.  
   - Imputer 라이브러리 내 **MICE**를 사용해 결측치를 보정한다.  

2. **머신러닝 알고리즘 비교**  
   - 아래 모델들의 non-merger/pre-merger/post-merger 단계 분류 성능을 비교하여 가장 좋은 성능의 모델을 선정한다.  
   - **Classical ML**  
     - Tree-based: Decision Tree, Random Forest  
     - Boosting-based: Gradient Boost, LightGBM, CatBoost, XGBoost  
     - Linear Model: Logistic Regression  
   - **Deep Learning**  
     - MLP  
     - FT-Transformer  
     - Tab Transformer  
   - **Ensemble Model**: TabM  

3. **관측 데이터 적용**  
   - TTA를 통해 시뮬레이션 데이터로 학습된 모델을 실제 데이터에 적응시킨다.  
   - 모델에 SDSS 은하 물리량 데이터를 넣어 merger stage를 추론한다.  

4. **결과 해석 및 의의**  
   - merger stage에 따른 물리량 변화 양상과 각 물리량 사이의 상관관계를 분석한다.  
   - image-exclude 모델의 활용 가능성과 한계를 평가한다.  


### 3.2 전체 시스템 구성
<img width="11525" height="6371" alt="Group 9" src="https://github.com/user-attachments/assets/41c26a24-8268-47a0-bb6a-22602e2fe84b" />
<img width="9384" height="6294" alt="Group 10" src="https://github.com/user-attachments/assets/3be1adac-e5f1-45cb-a71b-6bf649b2ff17" />
