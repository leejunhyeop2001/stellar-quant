# Stellar-Quant

**C++23 / Python 하이브리드 고성능 주가 시뮬레이션 엔진**

기하 브라운 운동(GBM) + Merton 점프 확산 기반 몬테카를로 시뮬레이션으로 미래 주가 분포를 예측합니다.
C++ 멀티스레드 엔진이 대규모 경로를 빠르게 처리하며, Streamlit 웹 대시보드(Toss 스타일 다크 UI)와 Matplotlib CLI 두 가지 방식으로 결과를 확인할 수 있습니다.

---

## Quick Start

### 1. 빌드 (최초 1회)

> `simulator.cpp` 수정 시에만 재빌드. Python 파일만 수정했다면 빌드 불필요.

**방법 A — CMake 수동 빌드 (로컬 개발)**

```powershell
.\venv\Scripts\cmake -S . -B build
.\venv\Scripts\cmake --build build --config Release
```

**방법 B — `pip install`로 확장 모듈 빌드 (권장: CI / Streamlit Community Cloud)**

C++ 컴파일러와 CMake가 `PATH`에 있어야 합니다. `packages.txt`에 맞춰 Linux에서는 `build-essential`, `cmake`, `git`, `python3-dev`를 설치한 뒤:

```bash
pip install .
```

`gbm_simulator`가 site-packages에 설치되면 `python/loader.py`가 로컬 `build/` 없이도 import합니다.

### 2-A. 웹 대시보드 (권장)

```powershell
.\venv\Scripts\streamlit run python\dashboard.py
```

브라우저에서 `http://localhost:8501`이 열립니다.  
**왼쪽 사이드바**에서 **종목 선택**(주요 티커 프리셋 또는 「직접 입력…」)을 고른 뒤, **본문**에서 **시뮬레이션 시작**을 누릅니다.  
시뮬 경로 수·기간·점프 등은 대시보드 코드에 **고정**되어 있으며(아래 **웹 대시보드 구성** 절 참고), UI에서 바꿀 수 없습니다. 상단 **햄버거(≡)** 로 사이드바를 열고 닫을 수 있습니다.

### 2-B. CLI (터미널)

```powershell
.\venv\Scripts\python python\main.py --ticker <종목코드>
```

**예시:**

```powershell
.\venv\Scripts\python python\main.py --ticker AAPL
.\venv\Scripts\python python\main.py --ticker TSLA --paths 3000000 --years 2.0
.\venv\Scripts\python python\main.py --ticker 005930.KS
.\venv\Scripts\python python\main.py --ticker AAPL --no-plot
```

---

## 실행 모드 비교

| | 웹 대시보드 | CLI |
|:--|:--|:--|
| **실행** | `streamlit run python\dashboard.py` | `python python\main.py --ticker TSLA` |
| **그래프** | Plotly (인터랙티브, 줌/호버) | Matplotlib (정적 이미지) |
| **종목 변경** | 사이드바 프리셋 또는 직접 입력 + **시뮬레이션 시작** 재실행 | 명령어 재입력 |
| **출력** | 브라우저 대시보드 | 터미널 + PNG + summary.md |
| **최적 용도** | 탐색적 분석, 프레젠테이션 | 배치 처리, 자동화 |

---

## CLI 옵션

| 옵션 | 기본값 | 설명 |
|:--|--:|:--|
| `--ticker` | `AAPL` | Yahoo Finance 종목코드 |
| `--paths` | `10,000,000` | 시뮬레이션 횟수 |
| `--fan-paths` | `8,000` | Fan Chart 경로 수 (대시보드와 동일 기본값) |
| `--steps` | `252` | 시점 수 (거래일) |
| `--years` | `1.0` | 예측 기간 (년) |
| `--threads` | `0` | CPU 스레드 수 (0 = 자동) |
| `--no-plot` | — | 그래프 비활성화 |

---

## 프로젝트 구조

```text
Stellar-Quant/
├── src/
│   └── simulator.cpp        # C++23 GBM + Jump Diffusion 멀티스레드 엔진
├── python/
│   ├── dashboard.py          # Streamlit 웹 대시보드 (Toss 스타일 다크 UI)
│   ├── main.py               # CLI 메인 실행
│   ├── data_utils.py         # 데이터 수집 & 파라미터 추정
│   ├── benchmark.py          # 성능 벤치마크
│   └── loader.py             # C++ 모듈 로더
├── build/                    # 빌드 산출물 (자동 생성)
├── CMakeLists.txt            # CMake 빌드 설정
├── setup.py                  # pip install 시 CMake 확장 빌드
├── pyproject.toml            # PEP 517 빌드 설정 (setuptools)
├── MANIFEST.in               # sdist에 소스 포함
├── packages.txt              # Streamlit Cloud apt 패키지
├── requirements.txt          # Python 패키지 의존성
├── summary.md                # 분석 결과 (자동 생성)
└── README.md
```

### 핵심 소스

| 파일 | 역할 |
|:--|:--|
| `src/simulator.cpp` | C++23 GBM + Merton 점프 확산 엔진. `std::thread` 병렬 + **스레드별 독립 `std::mt19937_64`**(공유 RNG·락 없음). `pybind11`로 Python에서 호출, `numpy.ndarray` zero-copy 반환. |
| `python/dashboard.py` | Streamlit + Plotly. 사이드바 **종목 selectbox** + 본문 실행 버튼; 시뮬 파라미터는 모듈 상수로 고정(터미널 1천만 경로·팬 샘플 8k 등). Toss 다크 테마, `Scattergl` 팬 차트(Mean/Median·분위수 밴드), `session_state` 캐시. |
| `python/main.py` | CLI 파이프라인: 데이터 수집 → C++ 시뮬레이션 → 리스크 분석 → 터미널 리포트 → Matplotlib 시각화 → summary.md 저장. |
| `python/data_utils.py` | `yfinance` 데이터 다운로드, μ/σ 추정, 점프 파라미터 추정, 통화 자동 감지, 가격 포맷 유틸리티. |
| `python/benchmark.py` | C++ vs NumPy vs Python for-loop 성능 비교. |
| `python/loader.py` | 빌드 디렉토리에서 `gbm_simulator` 모듈을 찾아 import. |

### 자동 생성 파일

| 파일 | 내용 |
|:--|:--|
| `summary.md` | 실행 설정, 리스크 지표, 벤치마크 결과 (한영 병기) |
| `python/simulation_plot.png` | 히스토그램 + Fan Chart 2-패널 그래프 |
| `python/benchmark_results.json` | 벤치마크 수치 데이터 |

---

## 웹 대시보드 구성 (`python/dashboard.py`)

### 사이드바 (왼쪽)

| 구역 | 내용 |
|:--|:--|
| 브랜드 | Stellar-Quant · 이준협 |
| **종목 선택** | `st.selectbox`: TSLA, AAPL, NVDA, MSFT, GOOGL, AMZN, META, AMD, INTC, `005930.KS`, `000660.KS`, `373220.KQ`, **직접 입력…** |
| 안내 캡션 | 현재 빌드에 고정된 시뮬 요약(경로 수·기간·팬 샘플·스텝)을 한 줄로 표시 |
| 하단 | C++23 엔진 안내; 시뮬 후 **역사적 점프 추정(λ̂, μ̂_J, σ̂_J)** 캡션(`yfinance` 성공 시) |

**대시보드 코드 고정값** (`python/dashboard.py` 상단 `FIXED_*`): 터미널 경로 **10,000,000**, 예측 기간 **1.0년**, 팬 샘플 **8,000** 경로, 시간 스텝 **252**, 스레드 **0**(자동), Merton 점프(하방 왜도 강화) **λ=1.5, μ_J=-0.15, σ_J=0.10**. `data_utils.estimate_gbm_params`는 연율 μ̂에 **시장 prior 8%와 5:5 수축**(`shrink_mu_toward_market_prior`) 후 **μ≤20%, σ≥15%** 클램프를 적용합니다. 수동 폴백도 동일 수축·클램프를 거칩니다. 리스크는 `compute_risk_metrics`의 정규화 Score와 초과첨도 가중으로 산출합니다.

시세는 기본 **최근 1년** 일봉입니다. **직접 입력…** 을 고르면 본문에 티커 입력 필드가 나타납니다.

### 본문 (메인 영역)

| 구역 | 내용 |
|:--|:--|
| 랜딩 전 | 헤더 카피 + (프리셋 선택 시) 선택 종목 표시 또는 (직접 입력 시) 티커 필드 + **시뮬레이션 시작** |
| 시뮬 직후 | 단계 메시지: 최근 **1년** `yfinance` → C++ 연산 → 완료; **토스트** |
| 항상(결과 있을 때) | **Risk Report** 히어로 + **액션 지표**(Kelly 권장 비중·Sortino·Sharpe·상승 확률): 소요 시간·경로 수, **예상 중앙가·상승 확률·VaR·CVaR·Risk Score** |
| **상세 지표와 수학 모델 보기** (접기) | 탭 **차트** / **리스크 지표** / **수학 모델** |
| └ **차트** | 팬 차트만: 분위수 밴드, **Median**·**기대 평균 경로(Mean)** (`Scattergl`), 현재가선 |
| └ **리스크 지표** | 핵심 지표 카드 → 전망 → 시나리오·파라미터 표 |
| └ **수학 모델** | KaTeX 카드 + 파라미터 요약 + 면책 |

> 웹 **차트** 탭에는 히스토그램이 없습니다(`build_hist`는 코드에만 존재). 분포는 CLI PNG에서 확인합니다.

### 실행 후 캐시 동작

결과는 `session_state`에 보관됩니다. **종목이나 고정 파라미터를 바꾼 뒤**에는 반드시 다시 **시뮬레이션 시작**을 눌러야 합니다(파라미터는 코드 상수이므로 재실행 = 코드 수정 후 배포).

---

## 통화 자동 감지

티커 접미사로 통화를 자동 판별하여 모든 출력에 기호를 표시합니다.

| 접미사 | 거래소 | 통화 | 기호 | 예시 |
|:--|:--|:--|:--|:--|
| *(없음)* | NYSE / NASDAQ | USD | `$` | `AAPL`, `TSLA` |
| `.KS` | 코스피 | KRW | `₩` | `005930.KS` |
| `.KQ` | 코스닥 | KRW | `₩` | `373220.KQ` |
| `.T` | 도쿄증권 | JPY | `¥` | `7203.T` |
| `.HK` | 홍콩증권 | HKD | `HK$` | `0700.HK` |
| `.L` | 런던증권 | GBP | `£` | `SHEL.L` |
| `.DE` | 프랑크푸르트 | EUR | `€` | `SAP.DE` |

---

## 수학 모델

### 확률 미분 방정식 (SDE)

$$dS_t = \mu S_t \, dt + \sigma S_t \, dW_t$$

- **μ** : 연율화 기대 수익률 (drift)
- **σ** : 연율화 변동성 (volatility)
- **dW_t** : 위너 과정 증분

### 해석해 (Itô's Lemma)

$$S_T = S_0 \exp\!\left((\mu - \tfrac{1}{2}\sigma^2)T + \sigma\sqrt{T}\,Z\right), \quad Z \sim N(0,1)$$

### 이산화 (Exact Discretization)

$$S_{t+\Delta t} = S_t \exp\!\left((\mu - \tfrac{1}{2}\sigma^2)\Delta t + \sigma\sqrt{\Delta t}\,Z_t + \textstyle\sum_i J_i\right)$$

### 드리프트 시장 수축 (리스크 관리)

표본에서 얻은 연율화 추정치 \(\hat\mu\)를 그대로 쓰지 않고, 시장 평균형 prior \(\mu_m = 8\%\)와 반반 혼합합니다.

$$\mu_{\mathrm{sim}} = \tfrac{1}{2}\hat\mu + \tfrac{1}{2}\mu_m$$

이후 시뮬 입력에는 \(\mu_{\mathrm{sim}} \le 20\%\), \(\sigma \ge 15\%\) 클램프를 추가 적용합니다.

### Volatility Drag (이토 보정)

로그 주가 \(\ln S\) 관점에서 GBM을 풀면 드리프트에 \(-\tfrac{1}{2}\sigma^2\) 항이 들어갑니다. 동일한 표면적 \(\mu\)라도 **σ가 클수록 기대 로그수익이 깎이는** 효과(Volatility drag / convexity effect)입니다. C++ 엔진은 \((\mu - \tfrac{1}{2}\sigma^2)\,dt\) (및 점프 항)으로 이를 반영합니다.

$$\ln\frac{S_{t+\Delta t}}{S_t} \approx \left(\mu - \frac{\sigma^2}{2}\right)\Delta t + \sigma\sqrt{\Delta t}\,Z_t + (\text{점프})$$

### Merton 점프 확산

$$\ln S_T - \ln S_0 = (\mu - \tfrac{1}{2}\sigma^2)T + \sigma\sqrt{T}\,Z + \sum_{i=1}^{N_T} J_i, \quad N_T \sim \text{Poisson}(\lambda T),\; J_i \sim N(\mu_J, \sigma_J^2)$$

### Value at Risk / CVaR

$$\text{VaR}_{95\%} = S_0 - Q_{0.05}(S_T), \qquad \text{CVaR}_{95\%} = \mathbb{E}[\,\text{Loss} \mid \text{Loss} \geq \text{VaR}_{95\%}\,]$$

### Kelly 근사 권장 비중

무위험 이자율을 연 \(r_f = 3\%\)로 두고, 시뮬에 입력되는 연율 \(\mu\), \(\sigma\)로 **전략 자본 대비 레버리지 근사**를 합니다.

$$f^\star = \max\!\left(0,\ \min\!\left(1,\ \frac{\mu - r_f}{\sigma^2}\right)\right)$$

대시보드 **액션 지표**에 \(f^\star\)를 퍼센트로 보여 주며, \(f^\star &lt; 20\%\)이면 변동성 대비 드리프트가 작다는 뜻으로 소액 투자 안내 캡션을 띄웁니다. (실제 매매는 비용·제약을 반영해 축소해야 합니다.)

### Sortino 비율 (하방 변동성)

종료 로그수익 \(r_i = \ln(S_T/S_0)\)에 대해, **손실 경로만** \(r_i &lt; 0\)을 모아 표본 표준편차 \(\sigma_-\)를 구하고, 분자는 평균 로그수익에서 \(\ln(1+r_f)\)를 뺀 초과분을 씁니다.

$$\text{Sortino} = \frac{\bar r - \ln(1+r_f)}{\sigma_-}, \qquad \sigma_- = \mathrm{std}(\{ r_i \mid r_i &lt; 0 \})$$

같은 분자로 **전체** \(\mathrm{std}(r)\)를 분모에 쓰면 샤프에 가까운 비교 지표가 되어, UI에서 Sortino를 먼저 배치해 하방 리스크 대비 수익을 강조합니다.

---

## 아키텍처

```text
[Yahoo Finance]  →  data_utils.py  →  μ, σ, S₀, (추정 점프), Currency
                                              ↓
                                     simulator.cpp (C++23)
                                     ├── std::thread × N cores
                                     ├── std::mt19937_64 per thread  (no lock)
                                     ├── Merton Jump Diffusion
                                     └── pybind11 → numpy (zero-copy)
                                              ↓
                                  ┌───────────┴───────────┐
                             dashboard.py              main.py
                          (Streamlit + Plotly)     (CLI + Matplotlib)
                                  │                       │
                       ┌──────────┴──────────┐     ├── Console report
                       │ Risk Report · 액션 지표 │     ├── Histogram + Fan (PNG)
                       │ 펼침: 차트·리스크·수학│     └── summary.md
                       │ (웹 차트 = 팬 차트) │
                       └─────────────────────┘
                                  │
                    브라우저 http://localhost:8501
```

**핵심 설계:**

- **C++ 코어** : `std::thread` 병렬 연산, 스레드별 **`std::mt19937_64`**(공유 RNG 없음·락 경합 없음)
- **Zero-copy** : `pybind11`로 C++ 배열 → `numpy.ndarray` 메모리 복사 없이 전달
- **GIL 해제** : 시뮬레이션 구간에서 `py::gil_scoped_release`로 Python 병목 제거
- **메모리 분리** : 웹은 터미널 **1천만** 경로 + 팬 샘플 **8k**(코드 상수); CLI는 `--paths` / `--fan-paths`로 조절
- **차트 캐싱** : Plotly figure를 `session_state`에 보관
- **모듈 캐싱** : `@st.cache_resource`로 C++ 공유 라이브러리를 한 번만 import
- **로딩 UI** : 데이터 분석 → 엔진 가동 → 완료 메시지 및 `st.toast`
- **렌더링** : 팬 차트 경로 다운샘플링 + 분위수 밴드 + `Scattergl` 선(중앙값·평균 경로)

---

## 벤치마크

```powershell
.\venv\Scripts\python python\benchmark.py --build-dir build\Release
```

C++ 멀티스레드 vs NumPy 벡터화 vs 순수 Python, 1만~1,000만 경로 규모 비교. 결과는 `python/benchmark_results.json`에 저장됩니다. (Release 빌드 예: 1,000만 경로에서 NumPy 대비 **약 5×대** — 세부는 **최종 보고서 §6.1**, 실제 배속은 환경마다 다름.)

---

## 의존성

```
numpy, pandas, yfinance, matplotlib, pybind11, streamlit, plotly
```

```powershell
.\venv\Scripts\pip install -r requirements.txt
```

ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ

# 2차 보고서

본 절은 Stellar-Quant **1차 개발 성과**와 **2차 고도화 방향**을 학술·기술 보고서 형식으로 요약합니다. 상세 수식은 위 **수학 모델** 절, 구현 세부는 **프로젝트 구조** 및 소스 표를 참고하세요.

### 1. 요약 (Abstract)

**1차 시스템**은 Yahoo Finance 기반 GBM 파라미터 추정과 C++23 병렬 몬테카를로 엔진을 결합해, 단일 종목·유동 예측 구간에 대한 **미래 가격 분포**와 **VaR/CVaR·상승 확률** 등 리스크 지표를 산출합니다. 단순 GBM만으로는 **연속 경로 가정**과 **과거 추정 \(\hat\mu\)의 편향**이 누적되어, 특히 고변동·급등 종목에서 **상승 확률 과대**와 **꼬리 리스크 과소**가 나타날 수 있습니다.

**고도화**에서는 **Merton 점프 확산**으로 **단절적 하락·급등(블랙 스완에 가까운 꼬리)** 을 반영하고, **드리프트 수축(Bayesian-style shrinkage)** 과 **\(\mu\)/\(\sigma\) 클램프**로 통계적 낙관을 완화합니다. UI 측면에서는 Toss 스타일 다크 테마, 종목 프리셋, Plotly `Scattergl` 팬 차트(평균·중앙 경로), 그리고 **Kelly 근사 비중**·**Sortino**·**VaR/CVaR**·**Risk Score**로 **실전 의사결정(자산 배분 참고)** 에 가깝게 지표를 배치했습니다.

**핵심 성과(정성):** 상승 확률과 손실 꼬리(CVaR)가 **점프·수축·클램프**와 맞물려 더 **보수적이고 해석 가능한** 범위로 이동하며, 동일 데이터 창에서도 **“너무 낙관적인 한 장의 숫자”** 에서 벗어나 리스크 리포트 형태로 정리됩니다.

### 2. 연구·개발 목적 및 배경 (고도화 관점)

#### 2.1 목적

- **통계적 편향 완화:** 표본 기반 연율 \(\hat\mu\)를 시뮬 입력에 그대로 쓸 때 발생하는 **과거 급등·추세의 미래 전이**를 줄이기 위해 prior 혼합 및 상한 클램프를 둡니다.
- **블랙 스완 근사:** 연속 GBM만으로는 **짧은 시간 대 손실/급등**을 충분히 재현하기 어려우므로, **Poisson 도착 + 가우시안 로그점프** 구조를 도입합니다.

#### 2.2 배경

단순 시뮬레이션 데모를 넘어, **자산 배분·포지션 크기**를 논할 때 필요한 것은 분포의 **중앙**뿐 아니라 **하위 분위·꼬리 평균(CVaR)** 과 **변동성 대비 초과수익**입니다. 본 프로젝트는 **교육·연구용 퀀트 시스템**으로, Kelly·Sortino 등은 **모형 가정 하의 참고치**이며 매수·매도를 지시하지 않습니다.

### 3. 심화 이론적 기반

#### 3.1 Merton Jump Diffusion (SDE)

확산항에 더해 점프항을 둔 주가 모형으로, 로그가격은 **가우시안 확산**과 **화합 포아송 점프**의 합으로 기술됩니다. 이산화 시 각 스텝에서 **점프 횟수**를 포아송으로 샘플링하고 **로그점프**를 가우시안에서 추출합니다(위 **수학 모델**·`src/simulator.cpp`).

#### 3.2 베이지안 수축 (Drift 보정)

$$\mu_{\mathrm{sim}} = w\,\hat\mu + (1-w)\,\mu_m, \quad w=\tfrac{1}{2},\ \mu_m=8\%$$

이후 \(\mu_{\mathrm{sim}}\le 20\%\), \(\sigma\ge 15\%\) **클램프**로 입력 공간을 제한합니다(`python/data_utils.py`). 이는 엄밀한 베이즈 추론 전체가 아니라 **수축(shrinkage) 해석이 가능한 경량 보정**입니다.

#### 3.3 고도화된 리스크·액션 지표

| 지표 | 역할 |
|:--|:--|
| **CVaR (ES)** | VaR를 넘는 꼬리 구간의 **조건부 기대 손실**로 tail risk 정량화 |
| **Kelly \(f^\star\)** | \(f^\star=\max(0,\min(1,(\mu-r_f)/\sigma^2))\), **무위험율 \(r_f=3\%\)** — 연율 \(\mu,\sigma\) 기준 **근사 최적 비중**(상한 100%) |
| **Sortino** | 종료 **로그수익** 기준, **손실 경로만**의 표준편차를 분모로 **하방 변동 대비 수익성** 표시 |

### 4. 시스템 아키텍처 및 고도화 설계

#### 4.1 데이터 흐름

**데이터 수집** (`yfinance`) → **\(\mu\) 수축·\(\mu,\sigma\) 클램프** (`data_utils.py`) → **Merton 점프 파라미터 결정** → **C++23 엔진** 대량 경로 생성 → **리스크·액션 지표** (`compute_risk_metrics`, Kelly/Sortino) → **Streamlit** 또는 **CLI + summary.md/PNG**.

- **웹 대시보드:** 시뮬에는 코드 상수 **고정 점프**(`python/dashboard.py`의 `FIXED_JUMP_*`)가 쓰이며, `estimate_jump_params`로 뽑은 \(\hat\lambda,\hat\mu_J,\hat\sigma_J\)는 **참고·UX 캡션** 위주입니다.
- **CLI (`main.py`):** 동일 데이터에서 추정한 **\(\lambda,\mu_J,\sigma_J\)** 를 엔진에 넘기는 흐름이 기본입니다.

#### 4.2 엔진·동시성·C++23

- **병렬 구조:** 작업 구간을 스레드 수에 맞게 분할하고, `std::thread` 워커가 구간 단위로 종료가격(또는 경로 행렬)을 채웁니다. Python 구간에서는 `py::gil_scoped_release`로 GIL을 해제합니다.
- **스레드 안전한 난수(Thread-safe RNG 설계):** 전역 **단일** `std::mt19937_64`를 두고 락으로 감싸면, 1,000만 경로 규모에서 난수 호출이 **직렬화**되어 병렬 이득이 사라집니다. 본 엔진은 **워커마다 독립된 `std::mt19937_64`** 를 두되, 시드만 `seed + tid × 7919` 형태로 **스레드 인덱스 기반 오프셋**을 주어 스트림을 분리합니다. 공유 상태·뮤텍스 없이 **Lock contention 없이** 병렬로 `std::normal_distribution` 등을 호출합니다(`src/simulator.cpp`).
- **C++23 채택 이유:** CMake에서 **ISO C++23**을 요구하는 것은 최신 MSVC/GCC 모드와 표준 라이브러리 정합을 맞추기 위함입니다. **현재 핫 루프**는 `<random>`·`<thread>` 중심이나, 진단 로그에는 추후 **`std::format`** 으로 할당 최소화 출력을, 중간 배열 파이프라인에는 **`std::views` / `std::ranges`** 로 불필요한 복사를 줄이는 확장 여지를 둡니다.

#### 4.3 성능·수치·메모리 안전 설계

- **부동소수점:** 경로 누적은 `double`로 수행해 단정밀도 대비 **반올림·언더플로** 민감도를 낮춥니다. 지수 변환 `exp(·)` 구간에서도 GBM 표준 이산화를 따릅니다.
- **메모리 이원화:** 대규모 몬테카를로에서 **전 구간 가격 행렬**을 모두 적재하면 \(O(N \times K)\) 로 RAM이 폭증합니다. 터미널 통계·히스토그램·VaR에는 **종료가 \(S_T\) 벡터만** 유지하고(대시보드 기본 **1,000만** 샘플), 팬 차트 등 시각화에는 **`FIXED_FAN_PATHS`(8,000)** 만 경로 행렬로 얻는 **표본 분리**로 메모리 상한을 제어합니다(`python/dashboard.py`).
- **수치적 관점:** 안티테틱 페어(확산 \(Z\)와 \(-Z\))는 분산 감소와 난수 재사용 효율을 동시에 노립니다.

#### 4.4 UI/UX

- Toss 스타일 **커스텀 CSS**, 사이드바 **종목 프리셋·직접 입력**
- Risk Report + **액션 지표** 카드, 접이식 탭(차트·리스크·수학 모델)

### 5. 구현 상세 (Implementation)

| 구분 | 파일 | 내용 |
|:--|:--|:--|
| **5.1 엔진** | `src/simulator.cpp` | GBM + Merton 점프 이산화, 멀티스레딩·난수 |
| **5.2 통계·제약** | `python/data_utils.py` | 수축·클램프, VaR/CVaR, Risk Score, Kelly/Sortino |
| **5.3 대시보드** | `python/dashboard.py` | Plotly `Scattergl` 팬 차트, Risk 히어로, KaTeX 수학 카드 |

### 6. 실증 분석 및 실험 결과

#### 6.1 벤치마크 (정량 Speed-up)

`python/benchmark.py`는 동일 GBM(본 측정은 **점프 \(\lambda=0\)**, \(\mu=0.1\), \(\sigma=0.2\), \(T=1\), `repeat=3` best-of)에서 **C++ 멀티스레드**·**NumPy 벡터화**·(소규모만) **순수 Python for-loop** 시간을 비교하고 `python/benchmark_results.json`에 기록합니다. **하드웨어·컴파일 옵션에 따라 달라지므로** 보고용 표는 로컬 또는 CI에서 스크립트를 재실행해 갱신하는 것이 바람직합니다.

**참고 측정 예시** (개발자 PC, Windows, MSVC **Release**, AVX2, 위 GBM 설정):

| 구현 | 10,000,000 경로 소요 시간 (s) | NumPy 대비 배속 |
|:--|--:|--:|
| **NumPy** (벡터화, 기준) | 0.2687 | 1.00× |
| **C++23** (멀티스레드, 터미널만) | 0.0515 | **약 5.21×** |
| **Pure Python** (순회) | — | — |

\* 순수 Python은 `benchmark.py` 기본 상한(20만 경로)만 직접 측정하며, 1,000만 경로는 **선형 외삽** 시 최소 **수 초~수분**대로 추정 가능(실측 생략). 소규모에서 C++가 NumPy보다 느리게 나오는 구간은 오버헤드·캐시·벡터화 이득이 지배할 때이며, **경로 수가 커질수록** C++ 병렬+네이티브 루프 이득이 커지는 경향이 있습니다.

#### 6.2 케이스 스터디 (예: TSLA)

고변동 종목에서 **순 GBM + 과도한 표본 드리프트**만 반영한 가상의 1차 설정과 비교할 때, **점프·드리프트 수축·\(\sigma\) 하한**을 포함한 고도화 설정에서는 종종 다음과 같은 **방향**이 관측됩니다: **상승 확률이 완화**되고, **CVaR 등 꼬리 손실 지표가 악화(손실 규모 반영)** 되어 분포가 **더 두꺼운 왼쪽 꼬리**를 가집니다.

아래 수치는 **고도화 전·후를 대비시키기 위한 예시(illustrative)** 이며, **실제 값은 시세 구간·실행 시점·난수에 따라 변동**합니다. 재현을 위해서는 동일 티커로 CLI/대시보드를 각 설정에 맞게 실행해 확인해야 합니다.

| 구분 | 예시: 상승 확률 | 예시: CVaR(95%) 상대 손실 |
|:--|:--|:--|
| 고도화 전(개념: GBM 중심·편향 큰 입력) | 약 **78%** 이상 | (비교 기준) |
| 고도화 후(본 저장소 기본: 점프+수축+클램프) | 약 **57%** | 약 **−44%** (현재가 대비 꼬리 손실률 스케일 예시) |

*해석:* 상승 확률만 보면 “좋아 보이던” 시나리오가, **꼬리 리스크**를 함께 보면 **덜 매력적**으로 보일 수 있음을 보여 주는 것이 목적입니다.

*현실 검증(정성):* 예시로 **57%** 근처로 내려온 상승 확률이 (단기 표본이 과장했던) **78%** 대비 “더 믿을 만한” 쪽으로 느껴지는 이유는, 역사적으로 관측되는 **깊은 조정·MDD**(최대 낙폭)나 **장기 평균 수익** 등과 비교할 때, **수축(Shrinkage)** 과 **\(\sigma\) 하한·점프**가 표본 추세를 **과도하게 미래에 복제하지 않도록** 눌러 주기 때문입니다. 즉 단기 급등 \(\hat\mu\)가 만든 단일 GBM 낙관을 줄여, **평균 회귀(mean reversion) 성격**과 **꼬리 사건**을 조금 더 현실적인 비율로 섞은 효과에 가깝습니다. (엄밀한 **과거 데이터와의 확률적 역검증(out-of-sample backtest)** 은 별도 실험 설계가 필요합니다.)

### 7. 한계 및 윤리적·실무적 주의사항

- **모형 한계:** 점프 강도·분포는 **단순화**되어 있으며, **실제 시장의 구조적 변화·이자·배당·거래비용·유동성**은 반영하지 않습니다.
- **데이터 의존:** 대시보드는 최근 **1년** 일봉, CLI 기본 `fetch_prices`는 **2년** 등 **선택한 창**에 민감합니다.
- **용도:** 본 소프트웨어는 **투자 조언이 아닌 교육·연구용 도구**입니다. Kelly 등은 **이론적·참고용**이며 실무에서는 **분할 매수·포트폴리오 제약·규제**를 반드시 고려해야 합니다.
- **Kelly \(f^\star\)의 실무 한계:** 화면의 \(f^\star\)는 **모형이 참**이라는 가정 하의 근사 비중입니다. \(\mu,\sigma\)·점프·데이터 창 오차가 있으면 \(f^\star\)는 쉽게 과대평가되므로, 운용에서는 **Half-Kelly**(산출치의 **절반 이하** 비중)·**캡**(예: 단일 종목 상한) 등으로 **모형 오차 마진**을 두는 것이 일반적입니다.

### 8. 결론 및 향후 과제

**성과:** C++23 하이브리드 엔진과 **Merton 점프 + 수축/클램프 + 꼬리·액션 지표** 결합으로, 1차 GBM 대비 **해석 가능하고 보수적으로 스케일링된** 리스크 리포트를 한 화면에 모았습니다.

**향후 과제:** **확률적 변동성 모형(예: Heston)** 으로 변동성 군집·레버리지 효과를 반영하고, 다자산·상관 구조·동적 헤징 등으로 **포트폴리오 수준**으로 확장하는 것을 제언합니다. Kelly류 비중은 **불완전 정보**와 **거래 비용**을 반영한 **보수적 스케일링(Half-Kelly 등)** 과 함께 설계해야 합니다.
