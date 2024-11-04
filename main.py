import streamlit as st
import openai
import os
from tempfile import NamedTemporaryFile
import time

# OpenAI API 키 설정
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY가 설정되지 않았습니다.")
    st.stop()
    
def transcribe_audio(audio_file, language):
    """음성 파일을 텍스트로 변환"""
    try:
        transcript = openai.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language=language
        )
        return transcript.text
    except Exception as e:
        st.error(f"음성 변환 중 오류가 발생했습니다: {str(e)}")
        return None

def summarize_text(text, is_english=False):
    """텍스트 내용 정리 및 필요시 번역"""
    try:
        if is_english:
            prompt = f"""다음 영어 텍스트를 한국어로 번역하고 주요 내용을 요약해주세요:
            {text}"""
        else:
            prompt = f"""다음 한국어 텍스트의 주요 내용을 요약해주세요:
            {text}"""

        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"텍스트 처리 중 오류가 발생했습니다: {str(e)}")
        return None

def main():
    st.title("음성 파일 변환 및 요약")
    
    # 언어 선택
    language = st.radio(
        "음성 파일의 언어를 선택하세요:",
        ("한국어", "영어"),
        index=0
    )
    
    # 파일 업로드
    audio_file = st.file_uploader("음성 파일을 업로드하세요 (mp3, wav, m4a)", 
                                 type=["mp3", "wav", "m4a"])
    
    if audio_file and st.button("변환 및 요약 시작"):
        with st.spinner("음성을 처리하는 중..."):
            # 임시 파일 생성 및 저장
            with NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.name)[1]) as tmp_file:
                tmp_file.write(audio_file.getvalue())
                tmp_file_path = tmp_file.name

            # 음성을 텍스트로 변환
            with open(tmp_file_path, "rb") as audio:
                lang_code = "en" if language == "영어" else "ko"
                transcribed_text = transcribe_audio(audio, lang_code)

            # 임시 파일 삭제
            os.unlink(tmp_file_path)

            if transcribed_text:
                st.subheader("변환된 텍스트:")
                st.write(transcribed_text)

                # 텍스트 요약 및 필요시 번역
                with st.spinner("텍스트를 분석하는 중..."):
                    summary = summarize_text(transcribed_text, language == "영어")
                    if summary:
                        st.subheader("요약 및 정리:")
                        st.write(summary)

if __name__ == "__main__":
    main()
