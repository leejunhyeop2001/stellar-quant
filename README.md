# Stellar-Quant

**C++17 / Python 하이브리드 고성능 주가 시뮬레이션 엔진**

기하 브라운 운동(GBM) 기반 몬테카를로 시뮬레이션으로 미래 주가 분포를 예측합니다.
C++ 멀티스레드 엔진이 1,000만 경로를 수 초 내에 처리하며,
Streamlit 웹 대시보드와 Matplotlib CLI 두 가지 방식으로 결과를 확인할 수 있습니다.

---

## Quick Start

### 1. 빌드 (최초 1회)

> `simulator.cpp` 수정 시에만 재빌드. Python 파일만 수정했다면 빌드 불필요.

```powershell
.\venv\Scripts\cmake -S . -B build
.\venv\Scripts\cmake --build build --config Release
```

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
.\venv\Scripts\python python\main.py --ticker 034020.KS
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
│   └── simulator.cpp        # C++17 GBM 멀티스레드 엔진
├── python/
│   ├── dashboard.py          # Streamlit 웹 대시보드
│   ├── main.py               # CLI 메인 실행
│   ├── data_utils.py         # 데이터 수집 & 파라미터 추정
│   ├── benchmark.py          # 성능 벤치마크
│   └── loader.py             # C++ 모듈 로더
├── build/                    # 빌드 산출물 (자동 생성)
├── CMakeLists.txt            # CMake 빌드 설정
├── requirements.txt          # Python 의존성
├── summary.md                # 분석 결과 (자동 생성)
└── README.md
```

### 핵심 소스

| 파일 | 역할 |
|:--|:--|
| `src/simulator.cpp` | C++17 GBM 엔진. `std::thread` 병렬 연산 + `std::mt19937` 스레드별 독립 RNG. `pybind11`으로 Python에서 호출, `numpy.ndarray` zero-copy 반환. |
| `python/dashboard.py` | Streamlit + Plotly 웹 대시보드. 다크 테마, 인터랙티브 차트, 카드형 메트릭, 리스크 시나리오 테이블. |
| `python/main.py` | CLI 파이프라인: 데이터 수집 → C++ 시뮬레이션 → 리스크 분석 → 터미널 리포트 → Matplotlib 시각화 → summary.md 저장. |
| `python/data_utils.py` | `yfinance` 데이터 다운로드, μ/σ 추정, 통화 자동 감지, 가격 포맷 유틸리티. |
| `python/benchmark.py` | C++ vs NumPy vs Python for-loop 성능 비교. |
| `python/loader.py` | 빌드 디렉토리에서 `gbm_simulator` 모듈을 찾아 import. |

### 자동 생성 파일

| 파일 | 내용 |
|:--|:--|
| `summary.md` | 실행 설정, 리스크 지표, 벤치마크 결과 (한영 병기) |
| `python/simulation_plot.png` | 히스토그램 + Fan Chart 2-패널 그래프 |
| `python/benchmark_results.json` | 벤치마크 수치 데이터 |

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

$$S_{t+\Delta t} = S_t \exp\!\left((\mu - \tfrac{1}{2}\sigma^2)\Delta t + \sigma\sqrt{\Delta t}\,Z_t\right)$$

### Value at Risk

$$\text{VaR}(95\%) = S_0 - Q_{0.05}(S_T)$$

1,000만 경로 비모수적(non-parametric) 추정으로 높은 정밀도 확보.

---

## 아키텍처

```text
[Yahoo Finance]  →  data_utils.py  →  μ, σ, S₀, Currency
                                          ↓
                                   simulator.cpp (C++17)
                                   ├── std::thread × N cores
                                   ├── std::mt19937 per thread
                                   └── pybind11 → numpy (zero-copy)
                                          ↓
                              ┌───────────┴───────────┐
                         dashboard.py              main.py
                      (Streamlit + Plotly)     (CLI + Matplotlib)
                              │                       │
                     ┌────────┼────────┐        ├── Console Report
                     │ Histogram      │        ├── Risk Analysis
                     │ Fan Chart      │        ├── Visualization
                     │ Metric Cards   │        └── summary.md
                     └── Risk Table   │
                                      │
                             Browser localhost:8501
```

**핵심 설계:**

- **C++ 코어** : `std::thread` 병렬 연산, 스레드별 독립 RNG로 thread-safety 보장
- **Zero-copy** : `pybind11`로 C++ 배열 → `numpy.ndarray` 메모리 복사 없이 전달
- **GIL 해제** : 시뮬레이션 구간에서 `py::gil_scoped_release`로 Python 병목 제거
- **메모리 분리** : 터미널 가격(최대 1,000만) / Fan Chart 경로(기본 1만)로 분리
- **렌더링 최적화** : 대시보드에서 데이터 다운샘플링 + `Scattergl` WebGL 렌더링

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
