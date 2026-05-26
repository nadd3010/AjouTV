# AjouTV
HW Attention 기반 반도체 공정 이상 AI 감지 센서

# 디렉토리 구조
```
project-root/
├── rtl/                  # Verilog 소스 (.v)
├── tb/                   # 테스트벤치 (.v)
├── sim/                  # ModelSim .do 스크립트
├── mem/                  # ROM 초기값 (.mem)
├── python/               # PyTorch 학습·변환·UART 스크립트
├── constraints/          # FPGA 핀 배치 (.xdc 또는 .qsf)
├── docs/                 # 인터페이스 정의서, 발표 자료
├── .gitignore            # 아래 설명
└── README.md
```

# rtl 모듈 구조