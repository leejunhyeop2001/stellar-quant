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
| `src/simulator.cpp` | C++23 GBM + Merton 점프 확산 엔진. `std::thread` 병렬 연산 + `std::mt19937` 스레드별 독립 RNG. `pybind11`로 Python에서 호출, `numpy.ndarray` zero-copy 반환. |
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
| 항상(결과 있을 때) | **Risk Report** 히어로: 위험도, 소요 시간·경로 수, **예상 중앙가·상승 확률·VaR·CVaR** |
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
                                     ├── std::mt19937 per thread  (thread-safe RNG)
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

- **C++ 코어** : `std::thread` 병렬 연산, 스레드별 독립 RNG로 thread-safety 보장
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

C++ 멀티스레드 vs NumPy 벡터화 vs 순수 Python, 1만~1,000만 경로 규모 비교.

---

## 의존성

```
numpy, pandas, yfinance, matplotlib, pybind11, streamlit, plotly
```

```powershell
.\venv\Scripts\pip install -r requirements.txt
```
