# Hypothesis 1: Background Context Dependency

## 실험 원리

### 입력 구성

DiT에 "헤어 없이 얼굴+배경만 보이는 이미지"를 배경으로 넣어주기 위해, hair 영역(m = 1)을 검정(0)으로 마스킹하여 background를 생성:

$$
\text{bg} = I_{tgt} \cdot (1 - m)
$$

sketch와 matte는 `ref_id` (braid_2534 등) 것으로 **완전 고정**.  
background만 다른 사람의 얼굴로 교체하여, DiT의 global attention이 얼굴 픽셀을 읽고 hair 생성에 반영하는지 검증.

### 추론 흐름 (denoising loop)

```
background ──VAE encode──▶ z_bg  (clean latent, 64×64)
sketch      ──VAE encode──▶ sketch_latent
matte       ──MatteCNN──▶  matte_feat
ctrl_cond = [sketch_latent + matte_feat,  matte_latent]  # [B,17,64,64]


z = randn(...)   # 순수 노이즈로 시작

for t in timesteps:
    residuals_cond    = ControlNet(z, ctrl_cond)
    blended           = matte_gated_blend(residuals_cond, matte_tokens)
    noise_pred_cond   = Transformer(z, blended)
    noise_pred_uncond = Transformer(z, zeros)          # CFG: 제로 residuals

    [CFG]  →  z = scheduler.step(noise_pred, t, z)
    [Compositor]  →  z = composite(z, z_bg, matte, sigma)
```

**CFG** (Classifier-Free Guidance):


**Compositor** — 매 스텝 배경 복원:

Flow Matching 공식으로 현재 σ 수준에 맞게 배경 latent를 noising:



Soft Matte: σ가 작아질수록(클린에 가까울수록) blur 반경 증가:

$
**Compositor의 역할**:  
- **hair 영역** (m̃ ≈ 1): DiT가 예측한 latent 유지 → 헤어 자유 생성  
- **non-hair 영역** (m̃ ≈ 0): z_bg를 현재 σ 수준으로 noising 후 덮어씌움 → 얼굴/배경을 원본 픽셀로 보존  
- **경계 soft blur**: σ → 0 일수록 m̃에 Gaussian blur 적용 → 경계 artifact 방지

### 출력 추출

$$
I_{hair} = I_{result} \cdot m
$$

변경되는 것은 `background`(=얼굴 픽셀)뿐이고, sketch/matte/seed/compositor 파라미터는 동일하므로  
결과 차이가 있다면 DiT global attention이 **배경(얼굴)을 hair 생성에 반영**하고 있다는 증거.


## 나노바나나 표정 실험

동일 인물의 표정(무표정/웃음/슬픔)만 바꿨을 때 헤어 생성 결과.
sketch + matte 고정, background = 나노바나나 생성 이미지.

### Full / Hair Only 정의

**Full**: 모델 inference의 전체 출력 이미지 (512×512).

$$
I_{full} = I_{result}
$$

- **헤어 영역** (matte = 1): DiT가 sketch + matte 조건으로 새로 생성한 헤어
- **비헤어 영역** (matte = 0): Compositor가 매 denoising step마다 background latent(z_bg)로 복원한 얼굴·배경 원본 픽셀

**Hair Only**: Full 결과에 오리지널 matte를 곱하여 헤어 영역만 추출.

$$
I_{hair} = I_{result} \cdot m
$$

비헤어 영역은 모두 검정(0)이 되며, 표정 변화에 따라 생성된 헤어가 실제로 달라지는지를 격리하여 관찰하기 위한 시각화.

### 생성 원리

```
background (나노바나나 이미지)
    ↓ VAE encode
z_bg  ← 비헤어 영역 복원의 기준 latent

sketch (고정) + matte (고정)
    ↓ ControlNet condition
DiT denoising (28 steps, CFG=7.5)
    ↓ 매 step: Compositor가 matte 기준으로
      헤어 영역 = DiT 예측 유지
      비헤어 영역 = z_bg 복원
    ↓
I_full  →  × matte  →  I_hair
```

변경되는 것은 background(나노바나나 표정)뿐이고 sketch·matte·seed는 동일하므로,
Hair Only 결과에 차이가 있다면 **DiT global attention이 얼굴 context를 헤어 생성에 반영**하고 있다는 증거.

### braid_2537

| 표정 | 원본 (나노바나나) | Full | Hair Only |
|:-:|:-:|:-:|:-:|
| 무표정 (baseline) | <img src="dataset/braid/img/test/braid_2537.png" width="200"> | <img src="results/hypothesis1/nanobana_2537/braid_2537_full_braid_2537.png" width="200"> | <img src="results/hypothesis1/nanobana_2537/braid_2537_bg_braid_2537.png" width="200"> |
| 웃는 얼굴 | <img src="dataset/nanobanana/braid_2537_smile.png" width="200"> | <img src="results/hypothesis1/nanobana_2537/braid_2537_full_braid_2537_smile.png" width="200"> | <img src="results/hypothesis1/nanobana_2537/braid_2537_bg_braid_2537_smile.png" width="200"> |
| 슬픈 얼굴 | <img src="dataset/nanobanana/braid_2537_sad.png" width="200"> | <img src="results/hypothesis1/nanobana_2537/braid_2537_full_braid_2537_sad.png" width="200"> | <img src="results/hypothesis1/nanobana_2537/braid_2537_bg_braid_2537_sad.png" width="200"> |

### braid_2562

| 표정 | 원본 | Full | Hair Only |
|:-:|:-:|:-:|:-:|
| 무표정 (baseline) | <img src="dataset/braid/img/test/braid_2562.png" width="200"> | <img src="results/hypothesis1/nanobana_2562/braid_2562_full_braid_2562.png" width="200"> | <img src="results/hypothesis1/nanobana_2562/braid_2562_bg_braid_2562.png" width="200"> |
| 웃는 얼굴 | <img src="dataset/nanobanana/braid_2562_smile.png" width="200"> | <img src="results/hypothesis1/nanobana_2562/braid_2562_full_braid_2562_smile.png" width="200"> | <img src="results/hypothesis1/nanobana_2562/braid_2562_bg_braid_2562_smile.png" width="200"> |
| 슬픈 얼굴 | <img src="dataset/nanobanana/braid_2562_sad.png" width="200"> | <img src="results/hypothesis1/nanobana_2562/braid_2562_full_braid_2562_sad.png" width="200"> | <img src="results/hypothesis1/nanobana_2562/braid_2562_bg_braid_2562_sad.png" width="200"> |

---

## 실험 한계 및 개선 방향

배경 변화가 헤어 생성에 미치는 영향을 분리하려면, **matte와 sketch는 반드시 고정**되어야 하며, 훈련 데이터의 **헤어 포즈(공간적 위치)도 거의 일정**하게 유지되어야 한다. matte와 sketch는 같은 헤어 영역을 공유하므로 하나가 바뀌면 다른 하나도 의미를 잃는다.

### 현재 실험의 문제저기

| 케이스 | 문제 | 원인 |
|:-:|:-:|:-:|
| **braid_2562** | "Hair Only" 결과에 배경이 비치고, 헤어 적용이 이상함 | 나노바나나 생성 시 헤어 포즈가 약간 바뀌어 hair 영역 자체가 변함 → 오리지널 matte와 생성된 헤어 위치가 불일치 |
| **braid_2534 (affine)** | affine transform이 sketch를 target matte에 얼추 맞추지만 완벽하지 않음 | 공간적 불일치가 잔존하여 오히려 해석 혼란 가중 |

### braid_2562 문제 상세

나노바나나가 생성한 이미지에서 헤어 포즈가 바뀌었기 때문에, 오리지널 matte를 그대로 적용하면:
- matte 영역(오리지널 헤어 위치)에 배경이 노출됨
- 생성된 헤어가 matte 바깥 영역에 걸쳐 나타남

→ "Hair Only" 시각화 자체가 의미 없는 결과가 됨. 이 케이스는 실험 조건을 충족하지 못한다.

### 올바른 실험 방향

**기준**: braid_2537 나노바나나 표정 실험이 이 조건을 가장 잘 충족함 (헤어 포즈가 유지된 상태에서 표정만 변경).

**braid_2534 개선안**:  
나노바나나에게 braid_2534의 헤어 영역을 semantic map 형태로 제공하고, **헤어 영역의 형태만 다르게** 생성하도록 요청. affine transform으로 sketch를 다른 인물 matte에 맞추는 방식은 중단.

**원칙**: 오리지널 matte + sketch semantic을 최대한 유지한 채로, 변수(배경 또는 표정)만 교체하여 실험할 것.

---

# Hypothesis 2: Background Context Dependency

sketch + matte를 각 인물 본인 것으로 고정하고, **순수 배경 영역만 3가지로 교체**했을 때 헤어 생성이 달라지는지 검증.

## 실험 원리

얼굴·헤어 영역은 원본 피험자 픽셀 그대로 유지하고, **순수 배경 영역**(얼굴·헤어 모두 제외)만 교체:

$$
m_{subj} = \text{clip}(m_{hair} + m_{face},\; 0,\; 1)
$$

$$
\text{bg} = I_{orig} \cdot m_{subj} \;+\; I_{new} \cdot (1 - m_{subj})
$$

- m_hair: 헤어 마스크  
- m_face: 얼굴 마스크 (face detector 또는 segmentation)  
- I_orig: 피험자 원본 이미지 (훈련 오리지널)  
- I_new: 배경 조건 이미지

모델은 이 background를 VAE로 인코딩해 z_bg를 만들고, compositor가 m_subj = 0 영역을 z_bg로 복원.  
DiT의 global attention이 순수 배경 픽셀까지 읽어 hair 생성에 반영하는지가 핵심.

> **대조군 설정 원칙**: 변수는 배경 픽셀 한 가지뿐이어야 한다.  
> sketch, matte, 얼굴 픽셀, seed, compositor 파라미터는 모두 동일하게 고정.  
> `complex_bg`에 다른 피험자(braid_2572 등)의 이미지를 그대로 사용하면 얼굴 픽셀까지 바뀌므로 **변수 통제 실패**. 반드시 훈련 오리지널(본인 이미지)에서 배경 영역만 교체할 것.

## 실험 설계

| 조건 | 순수 배경 영역 (1 - m_subj) | 얼굴·헤어 영역 |
|:-:|:-:|:-:|
| `white_bg` | 흰색 (255, 255, 255) | 원본 유지 |
| `texture_bg` | 8×8 체커보드 (회색) | 원본 유지 |
| `complex_bg` | 복잡한 실내/외 배경 씬 (인물 없음) | 원본 유지 |


## 결과에서 얼굴이 보이는 이유

### 큰 그림

모델은 이미지를 두 단계로 만든다.

1. **DiT가 노이즈에서 헤어를 생성**
2. **Compositor가 매 denoising 스텝마다 "헤어 아닌 영역"을 원본으로 복원**

얼굴이 보이는 이유는 **2번** 때문이다.

### Compositor의 동작 방식

Compositor는 매 스텝마다 다음 공식으로 합성한다:

$$
z_{out} = z_{pred} \cdot 	ilde{m}_{hair} \;+\; z_{bg}^{(\sigma)} \cdot (1 - 	ilde{m}_{hair})
$$

쉽게 풀면:

| 영역 | 처리 |
|---|---|
| **헤어 영역**  | DiT가 생성한 결과 사용 |
| **헤어 아닌 영역**  | `z_bg` (입력 배경)에서 그대로 복원 |

여기서 핵심: compositor가 쓰는 마스크는 **hair matte만** 이다.

```python
z = self.compositor(
    z, z_bg, mt_latent,   # ← hair matte만 (m_hair), 얼굴 포함 안 됨
    sigma, noise=bg_noise,
)
```

얼굴은 `m_hair = 0` 이므로 Compositor가 **28번의 denoising 스텝 내내 z_bg로 덮어씌운다.**

### z_bg 안에 얼굴이 들어있는 이유

hypothesis2의 입력 배경 이미지는 다음과 같이 만들었다:


- $m_{subj} = 	ext{clip}(m_{hair} + m_{face},\; 0,\; 1)$ → 얼굴·헤어 영역 보존
- 즉 `z_bg`를 VAE로 인코딩하면 **얼굴 픽셀이 z_bg 안에 그대로 담긴다**

### 전체 흐름 정리

```
입력 배경 이미지 (얼굴+헤어 원본 + 바뀐 배경)
        ↓ VAE 인코딩
       z_bg  ← 얼굴 정보 포함됨

denoising 28스텝 동안 매 스텝마다:
  헤어 영역   = DiT가 새로 생성
  얼굴+배경   = z_bg에서 복원  ← 얼굴이 여기서 나오는 것

        ↓
최종 출력 이미지에 얼굴이 보임
```

> **결론**: 얼굴이 보이는 건 DiT가 생성한 게 아니다.  
> Compositor가 28번 내내 "얼굴은 헤어 아니니까 z_bg로 덮어쓰기"를 반복한 결과다.  
> 이 실험에서 진짜 관심 대상은 **헤어 영역**이 배경 조건(white/texture/complex)에 따라 달라지는지이며, 얼굴은 그냥 따라오는 것이다.

---

## 결과

### braid_2534

| 배경 조건 | 입력 배경 | 생성 결과 |
|:-:|:-:|:-:|
| white_bg | <img src="results/hypothesis2/braid_2534/white_bg_input_bg.png" width="200"> | <img src="results/hypothesis2/braid_2534/white_bg.png" width="200"> |
| texture_bg | <img src="results/hypothesis2/braid_2534/texture_bg_input_bg.png" width="200"> | <img src="results/hypothesis2/braid_2534/texture_bg.png" width="200"> |
| complex_bg | <img src="results/hypothesis2/braid_2534/complex_bg_input_bg.png" width="200"> | <img src="results/hypothesis2/braid_2534/complex_bg.png" width="200"> |

### braid_2562

| 배경 조건 | 입력 배경 | 생성 결과 |
|:-:|:-:|:-:|
| white_bg | <img src="results/hypothesis2/braid_2562/white_bg_input_bg.png" width="200"> | <img src="results/hypothesis2/braid_2562/white_bg.png" width="200"> |
| texture_bg | <img src="results/hypothesis2/braid_2562/texture_bg_input_bg.png" width="200"> | <img src="results/hypothesis2/braid_2562/texture_bg.png" width="200"> |
| complex_bg | <img src="results/hypothesis2/braid_2562/complex_bg_input_bg.png" width="200"> | <img src="results/hypothesis2/braid_2562/complex_bg.png" width="200"> |

### braid_2574

| 배경 조건 | 입력 배경 | 생성 결과 |
|:-:|:-:|:-:|
| white_bg | <img src="results/hypothesis2/braid_2574/white_bg_input_bg.png" width="200"> | <img src="results/hypothesis2/braid_2574/white_bg.png" width="200"> |
| texture_bg | <img src="results/hypothesis2/braid_2574/texture_bg_input_bg.png" width="200"> | <img src="results/hypothesis2/braid_2574/texture_bg.png" width="200"> |
| complex_bg | <img src="results/hypothesis2/braid_2574/complex_bg_input_bg.png" width="200"> | <img src="results/hypothesis2/braid_2574/complex_bg.png" width="200"> |

### braid_2653

| 배경 조건 | 입력 배경 | 생성 결과 |
|:-:|:-:|:-:|
| white_bg | <img src="results/hypothesis2/braid_2653/white_bg_input_bg.png" width="200"> | <img src="results/hypothesis2/braid_2653/whiteㅊbg.png" width="200"> |
| texture_bg | <img src="results/hypothesis2/braid_2653/texture_bg_input_bg.png" width="200"> | <img src="results/hypothesis2/braid_2653/texture_bg.png" width="200"> |
| complex_bg | <img src="results/hypothesis2/braid_2653/complex_bg_input_bg.png" width="200"> | <img src="results/hypothesis2/braid_2653/complex_bg.png" width="200"> |
