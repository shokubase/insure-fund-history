# fund-scraper

생명보험협회 공시실(`pub.insure.or.kr`)에서 변액보험 펀드의 일별 기준가/순자산액 데이터를 받아 SQLite에 적재하는 GitHub Actions 배치.

## 구조

```
.
├── .github/workflows/
│   ├── daily.yml          # 매일 KST 08시 cron (평일)
│   └── backfill.yml       # 수동 실행, 전체 기간 채움
├── src/
│   ├── scraper.py         # HTTP 요청 + HTML 파싱
│   ├── storage.py         # SQLite upsert
│   ├── backfill.py        # 풀히스토리 백필 CLI
│   ├── daily.py           # 증분 업데이트 CLI
│   └── smoke.py           # 응답 형식 확인용
├── data/
│   └── fund_history.db    # SQLite (커밋되어 버전관리됨)
├── fund_list.csv          # 받아올 펀드 목록
└── requirements.txt
```

## 셋업

### 1) GitHub 리포 만들기

이 폴더 통째로 새 리포에 푸시.

### 2) Actions 권한

리포 Settings → Actions → General → Workflow permissions:
**Read and write permissions** 체크 (워크플로가 `data/` 커밋해야 함).

### 3) 펀드 목록 작성

`fund_list.csv` 편집:

```csv
memberCd,fundCd,name
L34,KLVL34001NQ,교보 글로벌 어쩌고 펀드
L34,KLVL34002NQ,교보 또다른 펀드
L01,KLVL01001NQ,삼성 뭐시기 펀드
```

`memberCd` = 보험사 코드, `fundCd` = 펀드 코드. 공시실 페이지 URL에서 그대로 복붙.

### 4) 첫 백필

리포의 **Actions 탭 → "Backfill fund NAV history" → Run workflow**.
시작일 기본값 2018-01-01. 끝나면 `data/fund_history.db`가 커밋됨.

### 5) 일일 cron 활성화

`daily.yml`은 푸시되면 자동으로 활성화됨. UTC 23시(KST 08시) 평일 실행.

## 로컬 실행

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 응답 형식 확인 (최초 1회 권장)
python -m src.smoke --member-cd L34 --fund-cd KLVL34001NQ --days 30

# 백필
python -m src.backfill --start 2018-01-01

# 증분
python -m src.daily

# 분석 대시보드 생성
python -m src.dashboard                         # → data/dashboard.html
python -m src.dashboard --risk-free 4.0          # 무위험수익률 변경
python -m src.dashboard --top-drawdowns 10       # 하락 이벤트 표시 개수
```

## DB 스키마

```sql
CREATE TABLE fund_nav (
    member_cd   TEXT NOT NULL,
    fund_cd     TEXT NOT NULL,
    std_ymd     TEXT NOT NULL,      -- 'YYYY-MM-DD'
    nav         REAL NOT NULL,      -- 기준가(원)
    net_assets  REAL,               -- 순자산액(억원)
    change_pct  REAL,               -- 기준가 등락률(%)
    fetched_at  TEXT NOT NULL,      -- UTC ISO timestamp
    PRIMARY KEY (member_cd, fund_cd, std_ymd)
);
```

PK가 `(member_cd, fund_cd, std_ymd)`라 동일 행을 여러 번 받아도 upsert됨 → idempotent.

## 알려진 위험

### GH Actions IP 차단 가능성
`pub.insure.or.kr`은 한국 협회 사이트. GH Actions 러너는 US/EU 리전이라
403, 타임아웃, 캡차 등이 날 수 있음. 처음 daily.yml 돌렸을 때 막히면:

1. **Korea 리전 self-hosted runner** — Naver Cloud/AWS Seoul에 작은 VM 띄우고 self-hosted runner 등록 → 워크플로의 `runs-on`만 `[self-hosted]`로 변경
2. **Korea 리전 프록시** — `scraper.py`의 `Session`에 `proxies=` 추가
3. **본인 맥북 self-hosted runner** — 켜져 있을 때만 동작

### 응답 형식 가정
`parse_response_html`은 `<table>` → `<tr>` → 4컬럼(기준일, 순자산액, 기준가, 등락률) 가정.
실제 응답이 다르면 `src/smoke.py`로 원본 떠보고 `parse_response_html` 조정.
스모크 결과가 0행이면 거기서 멈추고 HTML 구조 확인 필요.

### 큰 범위 응답이 잘릴 가능성
첫 백필에서 2018~오늘 (8년) 한 방에 호출. 만약 서버가 일부만 돌려주면
`--chunk-years 2` 옵션으로 2년씩 끊어서 호출 가능.

## 펀드 코드 추가 방법

공시실 본 화면에서 펀드 검색 → 상세 팝업의 URL에서 `memberCd=...&fundCd=...` 복사
→ `fund_list.csv`에 한 줄 추가 → 다음 daily 실행 때 자동으로 백필됨
(DB에 그 펀드 데이터가 없으면 FALLBACK_START 2018-01-01부터 채움).
