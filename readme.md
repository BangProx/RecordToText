# 이제 노트 필기 고통은 그만
openai whisper 기반 TTS 서비스
고통 -> 행복 ... 이 아니라 음성 -> 노트
다만.. 시간이 조.금 글린다.
### 실행 방법
streamlit run main.py

### 버그 Fix
1. 기존에 Text가 너무 긴 경우 번역이 수행되지 않는 오류 발생
    -> Chunk로 잘라서 해결
2. 다운로드 버튼 누르면 main 화면으로 돌아가는 문제
    -> session 도입해서 해결

### 버그
번역본을 다운로드하면 요약이 다시 수행되는 버그가 있습니다.
우상단의 stop을 누르고 나머지 다시 다운받으시면 됩니다.

### 개선 방향
1. PPT Slide를 전달해서 Slide 별로 요약하면 좋을 듯
