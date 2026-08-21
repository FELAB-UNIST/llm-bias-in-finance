# 신규 모델 편향 측정 — 카운터밸런싱 (작성자 통제)

**실험**: 균형 evidence(buy 2 + sell 2) 하 매수/매도 편향. **buy/sell 작성자를 gemini·kimi 절반씩 배정**해 작성자 설득력 효과를 상쇄.  
**입력 corpus**: gemini+kimi 카운터밸런싱 (`data/evidence_corpus_*_gemini_kimi.csv`)  
**규모**: S&P500 427종목 × 10 trials × 1 set = 4,270 prompts/model  
> 작성자-robust한 모델 고유 편향. (cf. `result.md` = gemini+mini 고정값)

## 1. 7종 종합

| 모델 | 비용 | bias_score | bias_index | 성향 |
|---|--:|--:|--:|---|
| glm-5.2 | $4.62 | **-31.8** | 446 | 매도 |
| deepseek-v4-pro | $3.44 | **-22.9** | 200 | 매도 |
| gemini-3.5-flash | $7.39 | **-17.3** | 173 | 매도 |
| minimax-m3 | $1.90 | **-15.2** | 169 | 매도 |
| qwen3.7-max | $3.92 | **+23.8** | 337 | 매수 |
| gpt-5.5 | $18.95 | **+30.1** | 362 | 매수 |
| claude-opus-4.8 | $43.54 | **+34.6** | 407 | 매수 |
| **합계** | **$83.78** | | | |

## 2. 섹터별 bias_score
| 섹터 | glm-5.2 | deepseek-v4-pro | gemini-3.5-flash | minimax-m3 | qwen3.7-max | gpt-5.5 | claude-opus-4.8 |
|---|--:|--:|--:|--:|--:|--:|--:|
| Energy | -13 | +0 | -4 | +10 | +52 | +58 | +67 |
| Technology | -5 | -7 | -7 | +2 | +44 | +46 | +35 |
| Healthcare | -24 | -23 | -15 | -12 | +23 | +30 | +37 |
| Communication Services | -39 | -21 | -32 | -19 | +38 | +38 | +46 |
| Industrials | -38 | -26 | -12 | -18 | +21 | +39 | +35 |
| Utilities | -34 | -17 | -27 | -18 | +26 | +8 | +39 |
| Basic Materials | -35 | -26 | -1 | -29 | +27 | +14 | +19 |
| Financial Services | -41 | -33 | -20 | -22 | +20 | +26 | +36 |
| Consumer Cyclical | -38 | -31 | -20 | -15 | +8 | +25 | +35 |
| Consumer Defensive | -38 | -33 | -27 | -30 | +18 | +33 | +26 |
| Real Estate | -55 | -24 | -37 | -22 | -3 | +4 | +11 |

## 3. 시총분위별 bias_score (Q1=대형 … Q4=소형)
| 시총 | glm-5.2 | deepseek-v4-pro | gemini-3.5-flash | minimax-m3 | qwen3.7-max | gpt-5.5 | claude-opus-4.8 |
|---|--:|--:|--:|--:|--:|--:|--:|
| Q1 | -12 | -13 | -7 | -1 | +41 | +41 | +48 |
| Q2 | -35 | -21 | -20 | -16 | +23 | +28 | +31 |
| Q3 | -34 | -25 | -17 | -20 | +16 | +30 | +30 |
| Q4 | -45 | -31 | -25 | -23 | +15 | +21 | +30 |

---
_생성: 2026-06-29 · 결과: `new_models_result/`(카운터밸런싱) · corpus: `data/evidence_corpus_*_gemini_kimi.csv`_