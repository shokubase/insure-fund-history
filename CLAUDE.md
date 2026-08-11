# insure-fund-history

보험사 변액보험 펀드 NAV 이력을 수집·저장하고 정적 대시보드로 시각화하는 프로젝트.

- `src/scraper.py` — NAV 수집
- `src/storage.py` — 데이터 저장/로드
- `src/daily.py`, `src/backfill.py` — 수집 진입점
- `src/benchmarks.py` — 벤치마크 지수
- `src/dashboard.py` — 정적 HTML 대시보드 생성 (`data/dashboard.html`, gitignored)
- `.github/workflows/` — daily / backfill / GitHub Pages 배포

## 작업 방침 (자동 진행)

이 저장소는 개인용·저위험 프로젝트다. **확인을 위해 멈추지 말고 끝까지 진행할 것.**

- 커밋, `git push`, 브랜치 병합, PR 생성·머지 모두 사전 확인 없이 바로 수행한다.
- 파일 생성·수정·삭제도 확인 없이 진행한다.
- `main`에 직접 커밋·푸시해도 된다. 브랜치를 팔 필요 없다.
- 작업이 끝나면 무엇을 했는지 결과만 보고한다. 중간에 선택지를 늘어놓고 묻지 않는다.
- 예외: 되돌릴 수 없는 파괴적 작업(`git push --force`, 히스토리 재작성, `data/` 대량 삭제)은 먼저 알린다.

`git add` 는 변경한 경로를 명시해서 스테이징한다 (`git add .` / `-A` 는 쓰지 않는다).

## 개발

```bash
source .venv/bin/activate
python -m src.dashboard   # 대시보드 재생성
```

대시보드는 Jinja/문자열 템플릿이 아니라 파이썬에서 HTML을 생성하는 구조이므로,
UI 를 고칠 때는 `src/dashboard.py` 안의 HTML/CSS/JS 문자열을 수정한 뒤 재생성해
`data/dashboard.html` 을 브라우저로 열어 확인한다.
