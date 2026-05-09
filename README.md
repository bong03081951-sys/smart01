# 총괄생산계획 최적화 대시보드

원예장비 제조업체의 총괄생산계획(APP)을 Pyomo + GLPK 솔버로 최적화하는 웹앱입니다.

## 실행 방법

### 로컬 실행
```bash
pip install -r requirements.txt
sudo apt-get install glpk-utils   # GLPK 솔버 설치
streamlit run app.py
```

### Streamlit Cloud 배포
1. GitHub에 이 폴더 업로드
2. share.streamlit.io 에서 배포
