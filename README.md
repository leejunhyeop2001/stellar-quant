## Stellar-Quant

C++23 / Python 하이브리드 주가 몬테카를로 시뮬레이션 도구입니다.

Yahoo Finance 가격 데이터를 기반으로 GBM(Geometric Brownian Motion)과 Merton 점프 확산을 시뮬레이션하고, 종료 가격 분포·상승 확률·VaR/CVaR·Kelly 참고치·Sortino 같은 리스크 지표를 제공합니다. 계산이 큰 구간은 C++ pybind11 확장 모듈에서 멀티스레드로 처리하고, Python은 데이터 수집·분석·시각화·대시보드·백테스트를 담당합니다.

> 투자 조언 도구가 아니라 교육·연구용 분석 도구입니다. 실제 투자 판단에는 거래비용, 세금, 유동성, 포트폴리오 제약, 모델 오차를 별도로 고려해야 합니다.

## 주요 기능

- Yahoo Finance 일봉 데이터 수집 및 통화 자동 감지
- 연율 드리프트 `mu`, 변동성 `sigma`, 점프 파라미터 추정
- C++23 멀티스레드 GBM / Merton 점프 확산 시뮬레이션
- Streamlit 대시보드: 프리셋 티커, 팬 차트, 리스크 리포트
- CLI: 터미널 리포트, Matplotlib PNG, `summary.md` 생성
- 백테스트: 과거 시점에서 1년 뒤 예측 구간 보정 상태 검증
- 벤치마크: C++ 엔진과 NumPy / 순수 Python 비교

## 프로젝트 구조

```text
Stellar-Quant/
├── src/
│   └── simulator.cpp          # C++23 pybind11 시뮬레이션 엔진
├── python/
│   ├── dashboard.py           # Streamlit 웹 대시보드
│   ├── main.py                # CLI 실행 파이프라인
│   ├── data_utils.py          # 데이터 수집, 파라미터 추정, 리스크 지표
│   ├── loader.py              # gbm_simulator 모듈 로더
│   ├── backtest.py            # 과거 1년 예측 보정 검증
│   └── benchmark.py           # 성능 벤치마크
├── docs/
│   └── SUBMISSION_REPORT.md   # 제출/기술 보고서
├── CMakeLists.txt             # C++ 확장 빌드 설정
├── setup.py                   # pip install용 CMake 빌드 래퍼
├── pyproject.toml             # PEP 517 빌드 설정
├── requirements.txt           # Python 의존성 설치 진입점
├── packages.txt               # Streamlit Cloud apt 패키지
└── README.md
```

자동 생성 파일은 Git에서 제외됩니다: `build/`, `summary.md`, `python/simulation_plot.png`, `python/benchmark_results.json`, `python/benchmark_summary.md`, `python/benchmark_bar_chart.png`, `python/backtest_results.json`, `python/backtest_results.md`.

## 설치

### 1. Python 환경 준비

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt`는 현재 프로젝트(`.`)를 설치합니다. 설치 과정에서 `setup.py`가 CMake를 호출해 `gbm_simulator` C++ 확장 모듈을 빌드합니다.

### 2. 수동 CMake 빌드

로컬 개발 중 C++ 파일만 빠르게 다시 빌드하려면 다음 명령을 사용할 수 있습니다.

```powershell
cmake -S . -B build
cmake --build build --config Release
```

빌드 후 Python 로더는 `site-packages`, `build/Release`, `build/Debug`, `build/` 순서로 모듈을 찾습니다.

## 실행

### Streamlit 대시보드

```powershell
streamlit run python\dashboard.py
```

브라우저에서 `http://localhost:8501`을 열고 사이드바에서 티커를 선택한 뒤 **시뮬레이션 시작**을 누릅니다.

대시보드 기본값은 코드 상수로 고정되어 있습니다.

| 항목 | 기본값 |
|:--|--:|
| 터미널 경로 수 | `3,000,000` |
| 팬 차트 경로 수 | `8,000` |
| 시간 스텝 | `252` |
| 예측 기간 | `1.0`년 |
| 스레드 | `0` = 자동 |
| 점프 모드 | 보수적 스트레스 / 역사적 추정 / 순수 GBM |
| 드리프트 불확실성 | 최근 표본 길이와 변동성 기반, 연율 표준편차 `5%~30%` |

### CLI

```powershell
python python\main.py --ticker AAPL
python python\main.py --ticker TSLA --paths 3000000 --years 2.0
python python\main.py --ticker 005930.KS --no-plot
```

주요 옵션:

| 옵션 | 기본값 | 설명 |
|:--|--:|:--|
| `--ticker` | `AAPL` | Yahoo Finance 티커 |
| `--paths` | `10,000,000` | 종료 가격 시뮬레이션 수 |
| `--fan-paths` | `8,000` | 팬 차트 경로 수 |
| `--steps` | `252` | 팬 차트 시간 스텝 |
| `--years` | `1.0` | 예측 기간 |
| `--threads` | `0` | C++ 워커 스레드 수, `0`은 자동 |
| `--build-dir` | 없음 | 직접 빌드한 확장 모듈 경로 |
| `--no-plot` | 꺼짐 | Matplotlib 창/PNG 생성 비활성화 |

CLI 실행 결과는 터미널 출력과 함께 `summary.md`에 저장됩니다. 플롯을 켜면 `python/simulation_plot.png`도 생성됩니다.

## 백테스트

```powershell
python python\backtest.py --ticker AAPL --period 10y --paths 50000
python python\backtest.py --ticker 005930.KS --market-ticker ^KS11 --period 10y
```

백테스트는 과거 여러 시점에서 직전 2년 데이터를 사용해 1년 뒤 분포를 예측하고, 실제 1년 뒤 가격이 예측 분포의 어느 분위수에 들어갔는지 확인합니다. 비교 모델은 다음과 같습니다.

| 모델 | 설명 |
|:--|:--|
| `stellar` | 현재 Stellar-Quant 추정 방식과 역사적 점프 추정 |
| `market_prior` | 시장 평균 prior 8%와 동일 변동성 기준 |
| `beta_prior` | 시장 프록시 대비 beta × 시장 prior 기준 |
| `flat_median` | 중앙값이 현재가 근처인 랜덤워크성 기준 |

결과는 `python/backtest_results.json`, `python/backtest_results.md`에 저장됩니다.

## 벤치마크

```powershell
python python\benchmark.py
```

결과 파일:

- `python/benchmark_results.json`
- `python/benchmark_summary.md`
- `python/benchmark_bar_chart.png`

벤치마크 수치는 CPU, 컴파일러, 빌드 타입, 백그라운드 부하에 따라 달라집니다. 보고서에 넣을 수치는 실행 환경에서 다시 측정하는 것이 좋습니다.

## 모델 요약

기본 GBM:

```text
dS_t = mu S_t dt + sigma S_t dW_t
```

종료 가격:

```text
S_T = S_0 exp((mu - 0.5 sigma^2)T + sigma sqrt(T) Z)
```

Merton 점프 확산:

```text
log(S_T / S_0) = (mu - 0.5 sigma^2)T + sigma sqrt(T)Z + sum(J_i)
N_T ~ Poisson(lambda T), J_i ~ Normal(mu_J, sigma_J^2)
```

`python/data_utils.py`는 표본 드리프트가 과하게 낙관적으로 반영되는 문제를 줄이기 위해 시장 prior 8%와 5:5로 수축한 뒤, `mu <= 20%`, `sigma >= 15%` 제약을 적용합니다. 대시보드는 추가로 연율 `mu` 추정 불확실성을 경로별로 샘플링해 단일 추정치가 만드는 과도하게 좁은 분포를 완화합니다.

## 통화 감지

티커 접미사로 통화를 추정합니다.

| 예시 | 통화 |
|:--|:--|
| `AAPL`, `TSLA` | USD |
| `005930.KS`, `373220.KQ` | KRW |
| `7203.T` | JPY |
| `0700.HK` | HKD |
| `SHEL.L` | GBP |
| `SAP.DE`, `OR.PA` | EUR |
| `600519.SS`, `000001.SZ` | CNY |

## 문서

제출용 또는 기술 보고서 형태의 긴 설명은 `docs/SUBMISSION_REPORT.md`에 있습니다. README는 실행과 유지보수에 필요한 최소 문서로 유지합니다.
