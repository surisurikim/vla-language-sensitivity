# VLA Language Sensitivity Research

> VLA 모델은 instruction의 어떤 언어적 특성에 민감하며,
> 그 실패 모드는 언어 이해 단계인가, 아니면 action 생성 단계인가?

---

## 배경 및 동기

**VLA (Vision-Language-Action) 모델**이란?
카메라로 보는 시각 정보(Vision)와 사람이 주는 언어 명령(Language)을 함께 받아서, 로봇의 실제 움직임(Action)을 출력하는 모델입니다.

예를 들어 `"빨간 컵을 집어서 오른쪽 접시에 올려줘"` 같은 명령을 주면, 로봇 팔이 그에 맞게 움직이는 거죠.

**왜 언어가 중요한가?**
VLA 모델은 언어 명령을 잘 이해해야 올바른 행동을 할 수 있습니다.
그런데 같은 의미라도 표현 방식이 달라지면 로봇이 실패할 수 있습니다.
이 연구는 그 이유를 파헤칩니다.

---

## 핵심 연구 질문

1. **어떤 언어적 특성**이 VLA 모델의 성능에 영향을 주는가?
   - 공간 정보 (`"왼쪽"`, `"위에"`)
   - 색깔, 모양 등 시각적 속성
   - 명령 구조 (단순 vs 복잡)
   - 학습 데이터에 없는 표현 (OOD language)

2. **실패가 발생하면**, 그게 어느 단계의 문제인가?
   - 언어를 이해 못한 것인가? (language understanding 실패)
   - 언어는 이해했지만 행동으로 변환을 못한 것인가? (action generation 실패)

3. **Prompt Engineering**으로 얼마나 극복할 수 있는가?

---

## 사용할 모델 및 환경

### 모델
| 모델 | 설명 |
|---|---|
| **OpenVLA** | 가장 접근하기 쉬운 오픈소스 VLA 모델. HuggingFace에서 바로 내려받기 가능 |
| (추후 비교) Octo | 두 번째 옵션. 다른 구조로 비교 실험 가능 |

### 시뮬레이션 환경
| 환경 | 설명 |
|---|---|
| **SimplerEnv** | VLA 모델 평가 전용으로 만들어진 환경. 설치가 쉽고 OpenVLA와 잘 연동됨 |
| **LIBERO** | 언어 명령 다양성이 풍부한 환경. 130개 이상의 태스크 포함 |

> 처음에는 SimplerEnv로 시작해서 감을 익히고, 이후 LIBERO로 확장할 계획입니다.

---

## 실험 계획

> **중요**: 이 계획은 실험을 해보면서 계속 수정될 수 있습니다.
> 실험을 통해 문제점을 발견하고, 그에 따라 방향을 바꾸는 것이 목표입니다.

---

### Phase 0: 환경 세팅 및 베이스라인 확인

**목표**: VLA 모델을 시뮬레이터에서 돌려보고, 기본 동작을 확인한다.

**할 일**:
- [ ] SimplerEnv 설치 및 OpenVLA 연동
- [ ] 기본 태스크 몇 가지를 원래 instruction으로 실행
- [ ] 성공률 기록 (이게 앞으로의 비교 기준 = Baseline)

**확인할 것**:
- 어떤 태스크가 있는가?
- 기본 성공률은 어느 정도인가?
- 실패할 때 어떤 식으로 실패하는가?

---

### Phase 1: 언어 변형 실험 (Instruction Variation)

**목표**: 같은 의미의 명령을 다르게 표현했을 때 성공률이 어떻게 달라지는지 측정한다.

**실험 축 (변형 방향)**:

#### 1-A. Paraphrasing (같은 말, 다른 표현)
```
원본:   "pick up the cup"
변형 1: "grab the cup"
변형 2: "take the cup"
변형 3: "get the cup and lift it"
```
- 의미는 같은데 단어가 달라지면 어떻게 되는가?

#### 1-B. Spatial Grounding (공간 정보 추가)
```
원본:   "pick up the cup"
변형 1: "pick up the cup on the left"
변형 2: "pick up the cup near the plate"
변형 3: "pick up the red cup at the top"
```
- 공간 정보를 추가하면 더 잘하는가, 아니면 혼란스러워하는가?

#### 1-C. Instruction Decomposition (단계 분해)
```
원본:   "pick up the cup and place it on the plate"
변형 1: 먼저 "pick up the cup", 다음에 "place it on the plate" (2단계로 분리)
```
- 복잡한 명령을 쪼개면 성공률이 올라가는가?

#### 1-D. OOD Language (학습 데이터 밖의 표현)
```
원본:   "pick up the cup"
변형 1: "컵을 집어줘" (한국어)
변형 2: "Please kindly lift the cup" (격식체)
변형 3: "yo grab that cup" (비격식/슬랭)
```
- 학습 분포 밖의 표현에 모델이 어떻게 반응하는가?

#### 1-E. Ambiguity Injection (의도적 모호함)
```
원본:   "pick up the red cup"
변형 1: "pick up the cup" (색깔 정보 제거, 여러 컵이 있는 상황)
변형 2: "pick up the thing" (물체 이름 제거)
```
- 모호한 명령에서 모델은 어떻게 행동하는가? 랜덤인가, 패턴이 있는가?

**측정 지표**:
- Task success rate (태스크 성공률)
- 실패 유형 분류 (어디서 실패하는가?)

---

### Phase 2: 실패 모드 분석

**목표**: Phase 1에서 발견된 실패들이 언어 이해 문제인지, action 생성 문제인지 구분한다.

**방법론 (아이디어)**:
- 모델 내부의 언어 임베딩 값을 시각화 → 의미가 비슷한 instruction끼리 가깝게 모이는가?
- 실패 케이스를 영상으로 녹화하여 패턴 분류:
  - 아예 엉뚱한 물체를 잡으려 함 → language understanding 실패
  - 맞는 물체에 가지만 동작이 어색함 → action 생성 실패

> **Note**: 이 Phase는 Phase 1 결과를 보고 구체적인 방법을 결정할 예정입니다.

---

### Phase 3: Prompt Engineering으로 개선 시도

**목표**: Phase 2에서 파악한 실패 원인을 바탕으로, prompt를 개선하여 성공률을 높인다.

**시도할 방법**:

#### Chain-of-Thought Prompting
```
기존:  "pick up the cup"
개선:  "Look at the scene. There is a cup on the table.
        Pick up the cup by grasping it from the top."
```

#### Structured Prompting (구조화된 명령)
```
개선:  "Object: red cup. Action: pick up. Location: left side of table."
```

#### Context-Enriched Prompting (맥락 추가)
```
개선:  "You are a robot arm. pick up the red cup carefully."
```

**측정**: Phase 0 베이스라인 대비 성공률이 얼마나 올라가는가?

---

### Phase 4: 결과 정리 및 인사이트 도출

**목표**: 실험 결과를 정리하고, 어떤 언어적 특성이 VLA 성능에 중요한지 결론 내린다.

- 어떤 변형이 가장 큰 성능 변화를 만들었는가?
- Prompt Engineering의 한계는 어디인가?
- 이 결과가 action tokenizer 연구 (다른 레포)와 어떻게 연결되는가?

---

## 예상 도전 과제

| 도전 | 설명 |
|---|---|
| 재현성 | 시뮬레이터도 랜덤 요소가 있어 여러 번 실행해 평균을 내야 함 |
| 변수 통제 | instruction만 바꾸고 다른 건 그대로 유지해야 함 |
| 실패 분류 | 어떤 기준으로 실패를 분류할지 정의하기 어려울 수 있음 |

---

## 참고 자료

- [OpenVLA](https://github.com/openvla/openvla) - 주 모델
- [SimplerEnv](https://github.com/simpler-env/SimplerEnv) - 평가 환경
- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) - 언어 다양성 환경
- OpenVLA 논문: *OpenVLA: An Open-Source Vision-Language-Action Model* (2024)

---

## 실험 로그

> 실험을 진행하면서 여기에 결과를 기록합니다.

| 날짜 | 실험 | 결과 | 다음 단계 |
|---|---|---|---|
| - | - | - | - |
