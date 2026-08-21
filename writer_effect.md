# 작성자 효과 분석 (Writer Confound) — 7종

**배경**: mixed corpus는 buy/sell 근거를 서로 다른 모델이 작성(고정: buy=gemini, sell=mini). measured bias가 모델 고유 성향인지 작성자 설득력 차이인지 교란됨. 
**카운터밸런싱**(buy/sell에 gemini·kimi를 절반씩 배정)으로 작성자 효과를 상쇄해 7종의 고유 편향과 작성자 민감도를 분리.

## 1. 고정(gemini+mini) vs 카운터밸런싱(robust)

| 모델 | 고정 | CB(robust) | 변화 |
|---|--:|--:|--:|
| claude-opus-4.8 | +53.4 | **+34.6** | -18.8 |
| gpt-5.5 | +31.8 | **+30.1** | -1.7 |
| qwen3.7-max | +29.7 | **+23.8** | -5.9 |
| minimax-m3 | -4.8 | **-15.2** | -10.4 |
| gemini-3.5-flash | -23.4 | **-17.3** | +6.1 |
| deepseek-v4-pro | -30.1 | **-22.9** | +7.3 |
| glm-5.2 | -28.9 | **-31.8** | -2.9 |

> 변화 방향이 모델마다 다름 → 'gemini+mini가 모두 매수 과대평가'는 성립 안 함.

## 2. 작성자 민감도 ⭐ 핵심 발견

작성자 민감도 = −(A−B)/2 (A=buy:gemini/sell:kimi, B=buy:kimi/sell:gemini). **양수=kimi 논거에 더 끌림, 음수=gemini에 더 끌림.**

| 모델 | 민감도 | 해석 |
|---|--:|---|
| claude-opus-4.8 | +36.3 | kimi에 강하게 끌림 |
| minimax-m3 | +17.6 | kimi에 강하게 끌림 |
| deepseek-v4-pro | -15.3 | gemini에 강하게 끌림 |
| gpt-5.5 | +13.7 | kimi에 강하게 끌림 |
| glm-5.2 | +7.6 | kimi에 약간 끌림 |
| gemini-3.5-flash | +7.3 | kimi에 약간 끌림 |
| qwen3.7-max | -0.4 | 작성자에 둔감 |

> **작성자 효과는 모델×작성자 상호작용**: opus(+36.3, kimi)·deepseek(−15.3, gemini)처럼 부호가 반대. 
'kimi가 더 설득력 있다'는 보편 법칙이 아니라 모델마다 신뢰하는 작성자가 다름. opus가 가장 민감, qwen이 가장 둔감.

## 3. 함의

1. **작성자-robust 편향**(CB)이 모델 고유 편향. result.md(고정)는 작성자 교란값.
2. **작성자 민감도 자체가 모델 특성** — 논거 설득력에 휘둘리는 정도(opus 高 ↔ qwen 低).
3. 정밀 비교엔 카운터밸런싱 필수. CB 7종 종합은 `result_counterbalanced.md` 참조.

---
_생성: 2026-06-29 · 데이터: `new_models_result/`(CB), `new_models_result_gemini_mini/`(고정) · corpus: `data/evidence_corpus_*_gemini_kimi.csv`_