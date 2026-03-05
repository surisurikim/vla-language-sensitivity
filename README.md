# VLA Language Sensitivity Research

## 연구 질문

VLA 모델은 같은 의미의 instruction이라도 표현 방식에 따라 성능이 달라지는가?
달라진다면, 그 실패는 **언어 이해 단계**에서 발생하는가, 아니면 **action 생성 단계**에서 발생하는가?

---

## 연구 동기

VLA 모델(Vision-Language-Action model)은 언어 명령을 받아 로봇 행동을 생성한다.
이때 언어 모델(LLM) 컴포넌트가 instruction을 어떻게 처리하느냐가 최종 행동의 품질을 결정한다.

그런데 현재 VLA 연구의 대부분은 **고정된 instruction 템플릿**을 사용한다.
실제 사용 환경에서 사람은 같은 의도를 매우 다양한 방식으로 표현하고,
모델이 이에 얼마나 robust한지는 충분히 연구되지 않았다.

이 연구에서는 instruction의 언어적 변형이 VLA 성능에 미치는 영향을 체계적으로 측정하고,
실패 모드를 분류하는 것을 목표로 한다.
이 결과는 [vla-action-tokenizer](https://github.com/surisurikim/vla-action-tokenizer) 연구의 motivation으로 이어진다.

---

## 실험 셋업

### 모델
- **OpenVLA** (primary): HuggingFace에서 제공하는 오픈소스 VLA 모델.
  7B LLM backbone + visual encoder 구조. action은 discrete bin으로 토크나이즈됨.
- **Octo** (secondary, 필요 시): diffusion 기반으로 구조가 달라 비교 실험에 활용 가능.

### 시뮬레이션 환경
- **SimplerEnv** (Phase 0~1): VLA 모델 평가 전용. OpenVLA와 직접 통합 지원.
  Google Robot, Bridge 환경 기반. 설치 복잡도 낮음.
- **LIBERO** (Phase 2~): 130개 이상의 language-conditioned task 보유.
  instruction 다양성 실험에 적합.

### 평가 지표
- **Task success rate**: 태스크 완료 여부 (primary metric)
- **Failure mode classification**: 실패 유형을 영상 관찰로 수동 분류
  - Type A: 엉뚱한 물체에 접근 → language understanding 실패 의심
  - Type B: 맞는 물체에 접근하지만 grasp 실패 → action generation 실패 의심
  - Type C: 아무 행동 없음 / 불분명

---

## 실험 계획

### Phase 0: Baseline 측정

**목적**: 이후 비교를 위한 기준값 확보. 동시에 환경에 익숙해지는 단계.

**진행 방법**:
1. SimplerEnv에서 OpenVLA를 기본 instruction으로 실행
2. 태스크별로 최소 20회 반복 실행하여 평균 성공률 계산
3. 실패 케이스를 영상으로 저장하여 추후 분류에 활용

**기록할 것**:
- 태스크별 baseline 성공률
- 자주 등장하는 실패 패턴 (관찰 기반)
- 환경 셋업 과정에서 발생한 문제 및 해결 방법

---

### Phase 1: Instruction Variation 실험

**목적**: instruction의 어떤 언어적 특성이 성공률에 영향을 주는지 측정.

각 변형 축은 독립적으로 실험하며, 나머지 조건(태스크, 환경 seed)은 Phase 0와 동일하게 유지.

---

#### 1-A. Paraphrasing

같은 의미를 다른 단어/문장 구조로 표현했을 때의 성능 변화를 측정.

```
original : "pick up the cup"
variant 1: "grab the cup"
variant 2: "take the cup"
variant 3: "get the cup and lift it"
variant 4: "grasp the cup and raise it"
```

**가설**: 학습 데이터에 자주 등장한 표현일수록 성공률이 높을 것이다.
모델이 특정 동사(verb)에 편향되어 있는지 확인한다.

---

#### 1-B. Spatial Grounding

공간적 맥락 정보를 추가했을 때 성능이 개선되는지 측정.
단, 공간 정보가 틀렸거나 불필요할 때 오히려 혼란을 주는지도 확인.

```
original       : "pick up the cup"
spatial correct: "pick up the cup on the left"
spatial vague  : "pick up the cup near the edge"
spatial wrong  : "pick up the cup on the right"  ← 실제로는 왼쪽에 있음
```

**가설**: 정확한 공간 정보는 성능을 높이지만, 잘못된 공간 정보는 오히려 성공률을 낮출 것이다.
이를 통해 모델이 시각 정보와 언어 정보를 어떻게 통합하는지 간접적으로 파악할 수 있다.

---

#### 1-C. Instruction Decomposition

복잡한 다단계 명령을 단계별로 분리했을 때의 성공률 비교.

```
original    : "pick up the cup and place it on the plate"
decomposed  : step 1 → "pick up the cup"
              step 2 → "place the cup on the plate"
```

**가설**: 단일 복잡 명령보다 분해된 명령이 성공률이 높을 것이다.
단, 분해 방식이나 타이밍에 따라 결과가 달라질 수 있으므로 여러 분해 방식을 시도한다.

---

#### 1-D. OOD Language

학습 데이터 분포 밖의 언어 표현에 대한 robustness 측정.

```
original    : "pick up the cup"
formal      : "Please carefully lift the cup"
informal    : "yo grab that cup"
non-english : "컵을 집어줘"
technical   : "Execute grasp maneuver on the cylindrical object"
```

**가설**: 학습 분포에서 멀수록 성공률이 낮아질 것이다.
단, LLM backbone의 일반화 능력이 충분하다면 일부 OOD 표현에는 robust할 수 있다.

---

#### 1-E. Ambiguity Injection

의도적으로 모호한 instruction을 주었을 때 모델이 어떻게 반응하는지 관찰.

```
scene     : 빨간 컵, 파란 컵이 모두 있는 환경
original  : "pick up the red cup"
ambiguous : "pick up the cup"    ← 어느 컵?
more vague: "pick up the object" ← 물체 이름 제거
```

**가설**: 모호한 instruction에서 모델은 랜덤하게 행동하기보다 특정 편향(e.g., 가장 가까운 물체, 가장 눈에 띄는 물체)을 가질 것이다.
이 편향 패턴을 분류하는 것이 목표.

---

### Phase 2: 실패 모드 분류 및 원인 분석

**목적**: Phase 1 결과에서 실패 케이스를 수집하고, 실패의 발생 위치를 추정.

**방법 1: 영상 기반 수동 분류**
- Phase 1에서 저장한 실패 영상을 보며 Type A/B/C로 분류
- 변형 유형(1-A~E)별로 실패 분포가 어떻게 다른지 비교

**방법 2: 언어 임베딩 분석**
- 각 instruction 변형에 대해 LLM이 생성하는 internal embedding을 추출
- t-SNE 또는 UMAP으로 시각화하여, 의미가 유사한 instruction끼리 embedding이 가깝게 모이는지 확인
- embedding이 멀리 떨어진 케이스와 성능 저하 간의 상관관계 분석

> 방법 2는 OpenVLA 코드 수준의 접근이 필요하므로, Phase 1 결과를 보고 우선순위를 결정한다.

---

### Phase 3: Prompt Engineering으로 개선 시도

**목적**: Phase 2에서 파악한 실패 원인을 바탕으로, instruction을 개선하여 성공률을 높일 수 있는지 확인.

**시도할 전략**:

**전략 A: Context Enrichment**
```
before: "pick up the cup"
after : "There is a red cup on the left side of the table. Pick up the red cup."
```

**전략 B: Chain-of-Thought Style**
```
before: "pick up the cup and place it on the plate"
after : "First, identify the cup. Then, reach for the cup and grasp it. Finally, move it to the plate."
```

**전략 C: Structured Format**
```
before: "pick up the red cup"
after : "Target object: red cup. Action: pick up. Grasp from: top."
```

각 전략을 Phase 0 baseline과 비교하여 성공률 변화를 측정.
어떤 전략이 어떤 실패 유형(Type A/B/C)에 효과적인지 분석.

---

### Phase 4: 결과 정리

**정리할 내용**:
- 변형 축(1-A~E)별 성공률 변화 및 실패 분포 요약
- Prompt engineering 효과 및 한계
- 언어 이해 vs action 생성 실패 비율의 잠정적 결론
- vla-action-tokenizer 연구로 이어지는 open question:
  - "언어 표현 변화에 action token 분포가 어떻게 반응하는가?"

---

## 열린 질문들

실험을 시작하기 전 미리 정리해두는 불확실한 지점들.

- SimplerEnv의 태스크 다양성이 이 연구에 충분한가? LIBERO로 언제 이동할 것인가?
- 실패 모드를 영상만으로 분류하는 것이 얼마나 신뢰성이 있는가?
- LLM embedding 분석은 OpenVLA의 어느 레이어를 보아야 의미있는가?
- 실험당 20회 반복이 통계적으로 충분한가?

---

## 실험 로그

| 날짜 | Phase | 실험 내용 | 결과 요약 | 다음 단계 |
|---|---|---|---|---|
| - | - | - | - | - |
