# StellarQuant 전용 코딩 규칙

범용 규칙(`/CLAUDE.md`)에 추가되는 이 프로젝트 전용 제약이다.

---

## C++ 엔진 변경 규칙

**`src/simulator.cpp`를 수정하면 반드시 cmake 빌드 후 테스트한다.**

빌드 없이 Python 코드를 실행하면 이전 바이너리로 동작해 변경 효과를 확인할 수 없다.

```powershell
# 프로젝트 루트에서 실행
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

빌드 성공 여부를 확인한 뒤 `python python/main.py --ticker AAPL` 등으로 검증한다.

C++와 Python 사이에 같은 수식이 두 곳에 존재한다.
- GBM 해석해. `src/simulator.cpp` 의 시뮬 루프
- 파라미터 추정. `python/data_utils.py` 의 `estimate_gbm_params`

`simulator.cpp`의 drift/diffusion 수식을 바꾸면 `data_utils.py`도 동기화해야 한다.

---

## Streamlit 대시보드 규칙

**`GLOBAL_CSS` 블록(약 700줄)은 요청 없이 건드리지 않는다.**

`python/dashboard.py` 상단의 CSS 블록은 Toss 다크 테마 전체를 담고 있다. 무단 수정은 UI 레이아웃 파괴로 이어진다. 디자인 변경을 요청받지 않았으면 이 블록은 읽기 전용으로 취급한다.

**session_state 패턴을 유지한다.**

대시보드는 Streamlit rerun 사이 상태를 `st.session_state`로 관리한다. 새로운 상태를 추가할 때 기존 패턴(`_store_dashboard_result`, `_result_from_session_state`)을 따른다. 직접 `st.session_state`에 임의 키를 추가하면 rerun 시 충돌한다.

**유닛 테스트가 없다. UI 변경은 직접 확인한다.**

```powershell
streamlit run python/dashboard.py
```

시뮬레이션 버튼 클릭 → 결과 렌더링 → 탭 전환까지 직접 확인한 뒤 완료로 표시한다.

---

## 시뮬레이션 파라미터 규칙

**새로운 시뮬 코드는 반드시 `clamp_gbm_for_simulation`과 `shrink_mu_toward_market_prior`를 통과시킨다.**

직접 μ·σ 값을 엔진에 넣지 않는다. 우회하면 μ > 20% 또는 σ < 15% 입력이 들어가 분포가 비현실적으로 왜곡된다.

```python
from data_utils import clamp_gbm_for_simulation, shrink_mu_toward_market_prior

mu = shrink_mu_toward_market_prior(mu_raw)
mu, sigma = clamp_gbm_for_simulation(mu, sigma)
```

**`FIXED_N_PATHS = 3_000_000` 기본값을 올리지 않는다.**

모델 오차가 MC 오차보다 크다. 경로 수를 늘려도 예측 정확도가 오르지 않고 속도만 느려진다. 백테스트·시나리오 분석에 계산 여유를 쓰는 것이 우선이다.

---

## yfinance 데이터 규칙

**fetch 실패는 코드 버그가 아닐 수 있다.**

Yahoo Finance는 IP 제한·세션 만료로 간헐적으로 빈 응답을 반환한다. `YahooFinanceFetchError`가 발생해도 즉시 코드를 수정하지 않는다. `fetch_prices`에는 이미 3회 재시도와 period 축소 폴백이 구현되어 있다. 네트워크 이슈인지 먼저 확인한다.

**한국 주식 티커는 접미사가 필요하다.**

- KRX 상장. `005930.KS` (KOSPI), `373220.KQ` (KOSDAQ)
- 접미사 없이 입력하면 Yahoo가 미국 주식으로 인식하거나 오류를 낸다.

---

## 백테스트 규칙

**인라인 백테스트(`_run_inline_backtest`)의 paths를 늘리지 않는다.**

10K paths, 126-day stride 설정은 속도(≈20초)와 정밀도의 균형점이다. 대시보드 안에서 돌리는 용도로, 정밀 분석은 CLI(`python/backtest.py --paths 50000`)를 쓴다.

**`_summarize` 함수는 `backtest.py` 내부 함수다.**

`from backtest import _summarize` 로 임포트할 수 있지만 공개 API가 아니다. 시그니처가 바뀔 수 있으므로 `backtest.py` 수정 시 `_render_backtest_inline`에서의 사용과 함께 확인한다.

---

## 핵심 파일 지도

| 파일 | 역할 | 수정 시 주의 |
|---|---|---|
| `src/simulator.cpp` | C++23 MC 엔진 | 빌드 필수, data_utils와 수식 동기화 |
| `python/data_utils.py` | GBM 파라미터 추정·리스크 지표 | clamp/shrink 로직 보존 |
| `python/dashboard.py` | Streamlit UI | CSS 블록 무단 수정 금지 |
| `python/backtest.py` | 롤링 백테스트 CLI | \_summarize 시그니처 변경 시 dashboard와 동기화 |
| `python/loader.py` | C++ 모듈 로더 + 폴백 | 빌드 경로 변경 시 업데이트 |
