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

def translate_text(text):
    """영어 텍스트를 한국어로 번역"""
    try:
        prompt = f"""다음 영어 텍스트를 한국어로 번역해주세요:
        {text}"""

        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"번역 중 오류가 발생했습니다: {str(e)}")
        return None

def create_study_notes(text):
    """텍스트를 학생 노트 필기 형태로 정리"""
    try:
        prompt = f"""다음 텍스트를 학생의 노트필기 형태로 정리해주세요. 
        - 중요 개념은 굵은 글씨로 표시
        - 핵심 내용은 번호나 기호로 구분
        - 부가 설명은 들여쓰기로 구분
        - 필요한 경우 도식화나 분류를 활용
        
        텍스트: {text}"""

        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"노트 정리 중 오류가 발생했습니다: {str(e)}")
        return None

def main():
    st.title("음성 파일 변환 및 번역")
    
    # 언어 선택
    language = st.radio(
        "음성 파일의 언어를 선택하세요:",
        ("한국어", "영어"),
        index=0
    )
    
    # 기능 선택
    col1, col2 = st.columns(2)
    with col1:
        do_summary = st.checkbox("내용 요약하기")
    with col2:
        do_notes = st.checkbox("노트 필기 형태로 정리하기")
    
    # 파일 업로드
    audio_file = st.file_uploader("음성 파일을 업로드하세요 (mp3, wav, m4a)", 
                                 type=["mp3", "wav", "m4a"])
    
    if audio_file and st.button("변환 시작"):
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
                
                # 변환된 텍스트 다운로드 버튼
                st.download_button(
                    label="변환된 텍스트 다운로드",
                    data=transcribed_text,
                    file_name="transcribed_text.txt",
                    mime="text/plain"
                )

                # 영어인 경우 한국어 번역
                if language == "영어":
                    with st.spinner("텍스트를 번역하는 중..."):
                        translated_text = translate_text(transcribed_text)
                        if translated_text:
                            st.subheader("한국어 번역:")
                            st.write(translated_text)
                            
                            # 번역본 다운로드 버튼
                            st.download_button(
                                label="번역본 다운로드",
                                data=translated_text,
                                file_name="translated_text.txt",
                                mime="text/plain"
                            )

                # 요약 옵션이 선택된 경우
                if do_summary:
                    with st.spinner("텍스트를 요약하는 중..."):
                        text_to_summarize = translated_text if language == "영어" else transcribed_text
                        summary = summarize_text(text_to_summarize)
                        if summary:
                            st.subheader("요약:")
                            st.write(summary)
                            
                            # 요약본 다운로드 버튼
                            st.download_button(
                                label="요약본 다운로드",
                                data=summary,
                                file_name="summary.txt",
                                mime="text/plain"
                            )

                # 노트 필기 정리 옵션이 선택된 경우
                if do_notes:
                    with st.spinner("노트 필기 형태로 정리하는 중..."):
                        text_to_notes = translated_text if language == "영어" else transcribed_text
                        study_notes = create_study_notes(text_to_notes)
                        if study_notes:
                            st.subheader("노트 필기 정리:")
                            st.markdown(study_notes)
                            
                            # 노트 필기 다운로드 버튼
                            st.download_button(
                                label="노트 필기 다운로드",
                                data=study_notes,
                                file_name="study_notes.txt",
                                mime="text/plain"
                            )

if __name__ == "__main__":
    main()
