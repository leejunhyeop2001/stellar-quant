# Stellar-Quant 프로젝트 기술 보고서

**Geometric Brownian Motion 기반 주가 몬테카를로 시뮬레이션 시스템**

---

| 항목 | 내용 |
|:---|:---|
| 프로젝트명 | Stellar-Quant |
| 개발 환경 | C++23, Python 3.x, CMake, pybind11 |
| 주요 라이브러리 | NumPy, Pandas, yfinance, Matplotlib, Streamlit, Plotly |

*(제출 시 기관 요구에 맞게 **작성자·학번·과목명·제출일** 등을 위 표 또는 표지에 추가하세요.)*

---

## 1. 요약 (Abstract)

본 프로젝트는 **기하 브라운 운동(GBM)** 과 **Merton 점프 확산**을 가정한 주가 모델 위에서 **몬테카를로 시뮬레이션**을 수행하여, 미래 시점 주가의 **분포·분위수·상승 확률·VaR(95%)** 등을 추정하는 하이브리드 응용 프로그램이다. 계산 집약 구간은 **C++23 멀티스레드**로 구현하고, 데이터 수집·통계·시각화·UI는 **Python**으로 구성하였다. **pybind11**을 통해 C++ 모듈을 Python에서 호출하며, 시뮬레이션 구간에서는 **GIL 해제**로 병목을 줄였다. 사용자는 **Streamlit 웹 대시보드**(종목 프리셋 selectbox, 시뮬 파라미터는 코드 상수로 고정) 또는 **CLI(Matplotlib)** 로 동일한 엔진 코어 결과를 확인할 수 있다.

---

## 2. 연구·개발 목적 및 배경

### 2.1 목적

- 실제 시장 데이터(야후 파이낸스)로부터 **초기가격 $S_0$**, **연율 드리프트 $\mu$**, **연율 변동성 $\sigma$** 를 추정하고, 이를 GBM 가정 하에 **대규모 난수 시뮬레이션**으로 미래 가격 분포를 근사한다.
- 단순 스크립트 수준이 아니라, **고성능 코어(C++)** 와 **분석·표현 계층(Python)** 을 분리한 구조로 확장성과 실행 속도를 동시에 확보한다.
- 금융 실무에서 자주 쓰는 지표(**분위수, 상승 확률, VaR**)를 **비모수적(시뮬레이션 샘플 기반)** 으로 산출하여 시각적으로 전달한다.

### 2.2 배경

주가 모델링에서 GBM은 **로그수익률의 정규성**을 가정하는 대표적인 확률 모델이다. 해석해를 이용하면 종료 시점 $T$ 의 주가는 로그정규 분포를 따르며, 다단계 경로(Fan Chart)는 **정확 이산화(exact step)** 로 시간축을 따라 시뮬레이션할 수 있다. 본 구현은 Python 순환만으로는 대규모 경로(예: 1천만 개) 처리에 한계가 있으므로, **C++ 병렬 RNG 및 지수 갱신**으로 성능을 확보한다.

---

## 3. 이론적 기반

### 3.1 GBM 확률미분방정식 (SDE)

$$
dS_t = \mu S_t\,dt + \sigma S_t\,dW_t
$$

- $\mu$: 연율화 기대수익률(drift)  
- $\sigma$: 연율화 변동성(volatility)  
- $dW_t$: 위너 과정 증분  

### 3.2 종료 시점 해석해 (Itô 보조정리)

$$
S_T = S_0 \exp\left(\left(\mu - \tfrac{1}{2}\sigma^2\right)T + \sigma\sqrt{T}\,Z\right), \quad Z \sim \mathcal{N}(0,1)
$$

### 3.3 이산화 (다단계 경로용, GBM의 정확 한 스텝)

$$
S_{t+\Delta t} = S_t \exp\left(\left(\mu - \tfrac{1}{2}\sigma^2\right)\Delta t + \sigma\sqrt{\Delta t}\,Z_t\right)
$$

### 3.4 리스크 지표 (본 프로젝트에서의 정의)

- **상승 확률**: $P(S_T > S_0)$ 의 시뮬레이션 비율 추정  
- **VaR(95%) (본 구현)**: $\mathrm{VaR}_{95\%} = S_0 - Q_{0.05}(S_T)$  
  - 즉, 시뮬레이션 분포의 **하위 5% 분위수**를 기준으로 한 **절대 손실 규모** 해석  

---

## 4. 시스템 구성 및 아키텍처

### 4.1 전체 데이터 흐름

```text
[Yahoo Finance (yfinance)]
        ↓
  data_utils.py  —  가격 시계열, S₀, μ, σ, 통화
        ↓
  simulator.cpp  —  터미널 샘플 / 경로 행렬 (멀티스레드)
        ↓
   ┌────┴────┐
dashboard.py   main.py
(Streamlit)    (CLI + Matplotlib)
```

### 4.2 모듈 역할

| 구성요소 | 파일 | 역할 |
|:---|:---|:---|
| 시뮬레이션 엔진 | `src/simulator.cpp` | GBM 터미널 시뮬레이션, 다단계 경로 행렬, `std::thread` 분할, 스레드별 `std::mt19937` |
| 빌드 | `CMakeLists.txt` | `gbm_simulator` 확장 모듈 빌드, pybind11(FetchContent 또는 시스템) |
| 데이터·추정 | `python/data_utils.py` | `yfinance` 다운로드, 로그수익 기반 $\mu$·$\sigma$ 연율화 추정, 통화·포맷 |
| 모듈 로딩 | `python/loader.py` | `build/Release` 등 후보 경로에서 `gbm_simulator` import |
| CLI 파이프라인 | `python/main.py` | 인자 파싱, 리스크 지표, Matplotlib, `summary.md` 등 |
| 웹 UI | `python/dashboard.py` | Streamlit, Plotly, 사이드바 종목 selectbox, `Scattergl` 팬 차트, 고정 시뮬 파라미터(예: 터미널 1천만 경로), 수식·리스크 카드 |
| 성능 비교 | `python/benchmark.py` | C++ vs NumPy vs 순수 Python 루프 등 |

### 4.3 성능·안전 설계 요지

- **GIL 해제**: C++ 시뮬레이션 구간에서 `py::gil_scoped_release` 사용  
- **메모리**: 터미널 샘플(대용량)과 Fan Chart용 경로 수를 분리; 예상 메모리 상한 검토(`main.py` 등)  
- **대시보드**: 다운샘플링, WebGL(`Scattergl`) 등으로 렌더 부담 완화  

---

## 5. 구현 상세

### 5.1 C++ 엔진 (`simulator.cpp`)

- 작업량 `n_paths`를 스레드 수에 맞게 구간 분할(`dispatch_threads`)  
- **터미널 전용**: 각 경로에 대해 한 번의 지수 변환으로 $S_T$ 생성  
- **경로 행렬**: `(n_paths, n_steps+1)` 행렬에 시계열 저장 (Fan Chart용)  
- 난수: 스레드 ID 기반 시드 오프셋으로 **스레드 간 독립성** 확보  

### 5.2 Python 계층

- **파라미터**: 역사적 수익률에서 $\mu$, $\sigma$ 추정(구체식은 `data_utils.py` 구현에 따름)  
- **리스크**: `numpy.quantile`, `numpy.mean`으로 분위수·상승 확률·VaR 계산  
- **시각화**:  
  - CLI: 히스토그램 + Fan Chart + 해석 패널  
  - 대시보드: Plotly **팬 차트**(분위수 밴드, Median/Mean 선, WebGL `Scattergl`); 터미널 분포 히스토그램은 웹 탭에 미연결(코드 `build_hist` 존재)  

### 5.3 국제 시장 지원

티커 접미사(예: `.KS`, `.KQ`)에 따른 **통화 기호 자동 표시**로 국내·해외 종목을 동일 파이프라인에서 처리한다(자세한 매핑은 `README.md` 참고).

---

## 6. 빌드 및 실행 방법 (로컬)

### 6.1 의존성 설치

```powershell
cd <프로젝트_루트>
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 6.2 C++ 모듈 빌드

```powershell
cmake -S . -B build
cmake --build build --config Release
```

### 6.3 실행

- **웹 대시보드**: `streamlit run python\dashboard.py` → 브라우저 `http://localhost:8501`  
- **CLI**: `python python\main.py --ticker AAPL` (옵션: `--paths`, `--years`, `--no-plot` 등)  
- **벤치마크**: `python python\benchmark.py`  

---

## 7. 기대 결과 및 시각적 산출물

- **히스토그램**: 종료 시점 $S_T$ 의 경험 분포  
- **Fan Chart**: 시간에 따른 분위수 밴드(불확실성 시각화)  
- **지표 카드·테이블**: 중앙값, 평균, 상승 확률, VaR, 시나리오(낙관/기준/비관 등)  

---

## 8. 한계 및 윤리적·실무적 주의사항

1. **모델 가정**: GBM은 **실제 시장의 점프·레버리지 효과·구조적 변화**를 반영하지 못한다.  
2. **과거 기반 추정**: $\mu$, $\sigma$ 는 **과거 데이터**에서 나온 추정치이며, 미래를 보장하지 않는다.  
3. **투자 조언 아님**: 본 소프트웨어는 **교육·연구용 시뮬레이션 도구**이며, 투자 권유나 법적 자문이 아니다.  
4. **데이터 의존**: `yfinance` 등 외부 데이터 품질·지연에 따라 결과가 달라질 수 있다.  

---

## 9. 결론 및 향후 과제

본 프로젝트는 GBM 및 Merton 점프 확산 기반 몬테카를로 시뮬레이션을 **C++/Python 하이브리드**로 구현하여, **대규모 샘플 처리**와 **풍부한 시각화·UI**를 동시에 달성하였다. 향후에는 **확률적 변동성**, **리스크 중립 측도와의 구분**, **백테스트 프레임** 등으로 모델을 확장할 수 있다.

---

## 10. 참고·인용

- Black–Scholes–Merton 계열 GBM 모델 및 이토 적분 기초  
- `pybind11`: C++/Python 바인딩  
- `yfinance`: 야후 파이낸스 기반 시세 데이터(비공식 API; 이용 약관 준수 필요)  
- 프로젝트 저장소 내 `README.md` — 설치·옵션·구조 상세  

---

*본 문서는 저장소 `docs/SUBMISSION_REPORT.md`에 포함되어 있으며, 필요 시 Word/PDF로 변환하여 제출 형식에 맞게 표지·목차·그림 캡션을 추가하면 된다.*
