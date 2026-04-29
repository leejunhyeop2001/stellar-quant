# Stellar-Quant

**C++17 / Python 하이브리드 고성능 주가 시뮬레이션 엔진**

기하 브라운 운동(GBM) + Merton 점프 확산 기반 몬테카를로 시뮬레이션으로 미래 주가 분포를 예측합니다.
C++ 멀티스레드 엔진이 1,000만 경로를 수 초 내에 처리하며, Streamlit 웹 대시보드(Toss 스타일 다크 UI)와 Matplotlib CLI 두 가지 방식으로 결과를 확인할 수 있습니다.

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

브라우저에서 `http://localhost:8501` 이 자동으로 열립니다.
좌측 사이드바에서 종목코드·시뮬레이션 횟수·예측 기간을 설정하고
**Run Simulation** 버튼을 클릭하면 됩니다.

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
| **종목 변경** | 사이드바에서 즉시 변경 | 명령어 재입력 |
| **출력** | 브라우저 대시보드 | 터미널 + PNG + summary.md |
| **최적 용도** | 탐색적 분석, 프레젠테이션 | 배치 처리, 자동화 |

---

## CLI 옵션

| 옵션 | 기본값 | 설명 |
|:--|--:|:--|
| `--ticker` | `AAPL` | Yahoo Finance 종목코드 |
| `--paths` | `10,000,000` | 시뮬레이션 횟수 |
| `--fan-paths` | `10,000` | Fan Chart 경로 수 |
| `--steps` | `252` | 시점 수 (거래일) |
| `--years` | `1.0` | 예측 기간 (년) |
| `--threads` | `0` | CPU 스레드 수 (0 = 자동) |
| `--no-plot` | — | 그래프 비활성화 |

---

## 프로젝트 구조

```text
Stellar-Quant/
├── src/
│   └── simulator.cpp        # C++17 GBM + Jump Diffusion 멀티스레드 엔진
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
| `src/simulator.cpp` | C++17 GBM + Merton 점프 확산 엔진. `std::thread` 병렬 연산 + `std::mt19937` 스레드별 독립 RNG. `pybind11`으로 Python에서 호출, `numpy.ndarray` zero-copy 반환. |
| `python/dashboard.py` | Streamlit + Plotly 웹 대시보드. Toss 스타일 다크 테마, 3단계 로딩 인디케이터(`st.status`), 차트 figure 캐싱(`session_state`), shimmer 스켈레톤, 인터랙티브 차트. |
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

## 대시보드 UI

Toss 스타일의 현대적인 다크 대시보드로, 실행 시 3단계 진행 상태가 표시됩니다.

| 단계 | 내용 |
|:--|:--|
| 📡 데이터 수집 | Yahoo Finance 2년치 시세 다운로드, μ·σ·점프 파라미터 추정 |
| ⚙️ 엔진 가동 | C++17 멀티스레드로 Monte Carlo 경로 연산 |
| 📊 차트 생성 | 리스크 지표 계산 + Plotly 차트 빌드 (결과는 session_state에 캐싱) |

이후 사이드바 위젯 변경 시에는 C++ 재연산 없이 캐시된 결과를 즉시 렌더링합니다.

**주요 섹션:**
- **핵심 지표** — 최종 예상가(Median) · VaR(95%) · CVaR(95%)
- **차트** — 최종가 분포(히스토그램 + PDF) / 주가 경로 Fan Chart (WebGL 가속)
- **투자 전망** — 현재가 · 평균 예측가 · 상승 확률
- **리스크 시나리오** — Best/Expected/Worst + 추정 파라미터 테이블
- **수학 모델** — GBM SDE · Itô 해석해 · VaR · CVaR · Merton 점프 카드

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

### Merton 점프 확산

$$\ln S_T - \ln S_0 = (\mu - \tfrac{1}{2}\sigma^2)T + \sigma\sqrt{T}\,Z + \sum_{i=1}^{N_T} J_i, \quad N_T \sim \text{Poisson}(\lambda T),\; J_i \sim N(\mu_J, \sigma_J^2)$$

### Value at Risk / CVaR

$$\text{VaR}_{95\%} = S_0 - Q_{0.05}(S_T), \qquad \text{CVaR}_{95\%} = \mathbb{E}[\,\text{Loss} \mid \text{Loss} \geq \text{VaR}_{95\%}\,]$$

---

## 아키텍처

```text
[Yahoo Finance]  →  data_utils.py  →  μ, σ, S₀, λ, μ_J, σ_J, Currency
                                              ↓
                                     simulator.cpp (C++17)
                                     ├── std::thread × N cores
                                     ├── std::mt19937 per thread  (thread-safe RNG)
                                     ├── Merton Jump Diffusion
                                     └── pybind11 → numpy (zero-copy)
                                              ↓
                                  ┌───────────┴───────────┐
                             dashboard.py              main.py
                          (Streamlit + Plotly)     (CLI + Matplotlib)
                                  │                       │
                       ┌──────────┼──────────┐        ├── Console Report
                       │ 3-Phase Status UI   │        ├── Risk Analysis
                       │ Histogram + PDF     │        ├── Visualization
                       │ Fan Chart (WebGL)   │        └── summary.md
                       │ Metric Cards        │
                       │ Risk Table          │
                       └── Math Model Cards  │
                                             │
                                   Browser localhost:8501
```

**핵심 설계:**

- **C++ 코어** : `std::thread` 병렬 연산, 스레드별 독립 RNG로 thread-safety 보장
- **Zero-copy** : `pybind11`로 C++ 배열 → `numpy.ndarray` 메모리 복사 없이 전달
- **GIL 해제** : 시뮬레이션 구간에서 `py::gil_scoped_release`로 Python 병목 제거
- **메모리 분리** : 터미널 가격(최대 1,000만) / Fan Chart 경로(기본 5,000)로 분리
- **차트 캐싱** : Plotly figure를 `session_state`에 보관 — 위젯 변경 시 C++ 재연산 없이 즉시 렌더링
- **모듈 캐싱** : `@st.cache_resource`로 C++ 공유 라이브러리를 한 번만 import
- **3단계 로딩** : `st.status`로 데이터 수집 → 엔진 → 차트 단계별 진행 상태 표시
- **렌더링 최적화** : 데이터 다운샘플링 + `Scattergl` WebGL 렌더링

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
