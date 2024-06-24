# SatelliteLaneDet2024


## 구조 초기화

```text
├── config
│   └── config.py
├── data
│   └── __init__.py
├── example
│   └── __init__.py
├── log
│   └── logger.py
├── model
│   ├── decoder.py
│   ├── encoder.py
│   ├── loss.py
│   ├── matcher.py
│   ├── metric.py
│   └── nms.py
├── README.md
├── tools
│   └── __init__.py
└── train
    └── train.py
```

- config: 프로젝트의 다양한 파라미터들을 설정
  - config.py: 실제 파라미터가 들어있는 파일을 읽어와서 프로젝트에서 쓸 수 있는 config 객체 생성
  - yaml or py: 실제 파라미터가 들어있는 파일
- data: 데이터셋과 관련된 처리
  - xxx_dataset.py: 특정 데이터셋을 읽어오는 모듈
  - preprocessor.py: 전처리
  - augmentation.py: 데이터 증강
- example: 만들고 있는 특정 클래스를 따로 테스트해 볼 수 있는 곳, 예제를 잘 만들고 보관하자.
- log: 데이터 로깅
- model: 모델 안에서 처리과정
  - 지금은 개별 파일로 여러개 있지만 종류가 다양해지면 각각이 폴더가 되어야 함
- tools: 다양한 유틸리티 함수, 클래스를 정의
