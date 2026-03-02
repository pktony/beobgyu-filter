# Beobgyu Filter

MediaPipe Hands + KNN 기반 손 제스처 인식 필터. 가운데 손가락(beobgyu) 제스처를 감지하면 해당 영역을 모자이크 처리합니다.

## 주요 기능

- 웹캠 실시간 손 랜드마크 검출 (MediaPipe Hands)
- 관절 각도 기반 KNN 제스처 분류 (10종)
- 가운데 손가락 감지 시 모자이크 자동 적용
- 제스처 학습 데이터 수집/삭제 도구 포함

## 지원 제스처

| Label | 제스처 |
|-------|--------|
| 0 | beobgyu (가운데 손가락) |
| 1 | ok |
| 2 | ddabong (엄지 척) |
| 3 | bad |
| 4 | one |
| 5 | two |
| 6 | three |
| 7 | four |
| 8 | five |
| 9 | fist (주먹) |

## 폴더 구조

```
beobgyu-filter/
├── beobgyu_filter.py      # 메인 필터 (제스처 인식 + 모자이크)
├── main.py                # 손 랜드마크 검출 테스트
├── hand_tracker.py        # MediaPipe HandLandmarker 래퍼 클래스
├── gestures.py            # 제스처 이름 ↔ 라벨 매핑
├── collect_dataset.py     # 제스처 학습 데이터 수집
├── delete_gesture.py      # 특정 제스처 데이터 삭제
├── hand_landmarker.task   # MediaPipe 손 랜드마크 모델
├── pyproject.toml         # 프로젝트 설정 (uv)
├── data/
│   ├── gesture_train.csv  # 학습 데이터 (관절 각도 + 라벨)
│   ├── images/            # 수집 시 저장된 이미지
│   └── captures/          # 실행 중 캡처 이미지
└── runs/
    ├── run.sh             # 손 검출 테스트 실행
    ├── beobgyu_filter.sh  # 필터 실행
    └── collect_data.sh    # 데이터 수집 실행
```

## 요구사항

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/) 패키지 매니저
- 웹캠

## 설치

```bash
uv sync
```

## 실행

### 필터 실행 (모자이크 적용)

```bash
uv run beobgyu_filter.py
```

### 손 검출 테스트

```bash
uv run main.py
```

### 제스처 데이터 수집

```bash
uv run collect_dataset.py <gesture_name>
# 예: uv run collect_dataset.py beobgyu
```

- 스페이스바: 현재 프레임 데이터 저장
- q: 종료

### 제스처 데이터 삭제

```bash
uv run delete_gesture.py <gesture_name>
# 예: uv run delete_gesture.py beobgyu
```

## 조작법

| 키 | 동작 |
|----|------|
| Space | 현재 프레임 캡처 저장 (`data/captures/`) |
| q | 종료 |

## 동작 원리

1. MediaPipe HandLandmarker로 손 21개 관절 좌표 검출
2. 인접 관절 간 벡터를 구하고 15개 각도 계산
3. KNN(k=3)으로 학습 데이터 기반 제스처 분류
4. `beobgyu` 제스처 감지 시 손 영역 모자이크 처리
