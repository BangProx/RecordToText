import streamlit as st
import openai
import os
from tempfile import NamedTemporaryFile
import time
import tiktoken
from typing import List

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

def split_text_into_chunks(text: str, chunk_size: int = 3000, chunk_overlap: int = 500) -> List[str]:
    """텍스트를 토큰 기반으로 청크로 분할"""
    try:
        # GPT-3.5-turbo 토크나이저 사용
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        tokens = encoding.encode(text)
        
        chunks = []
        i = 0
        while i < len(tokens):
            # 청크 크기만큼 토큰 추출
            chunk_tokens = tokens[i:i + chunk_size]
            chunk_text = encoding.decode(chunk_tokens)
            chunks.append(chunk_text)
            
            # 오버랩을 고려하여 다음 시작 위치 계산
            i += (chunk_size - chunk_overlap)
            
        return chunks
    except Exception as e:
        st.error(f"텍스트 분할 중 오류가 발생했습니다: {str(e)}")
        return [text]

def translate_text(text: str, chunk_size: int = 3000, chunk_overlap: int = 500) -> str:
    """영어 텍스트를 한국어로 번역 (청크 단위로 처리)"""
    try:
        # 텍스트를 청크로 분할
        chunks = split_text_into_chunks(text, chunk_size, chunk_overlap)
        translated_chunks = []
        
        # 각 청크별로 번역 수행
        for i, chunk in enumerate(chunks):
            with st.spinner(f"텍스트 번역 중... ({i+1}/{len(chunks)})"):
                prompt = f"""다음 영어 텍스트를 한국어로 번역해주세요. 
                이 텍스트는 더 긴 문서의 일부일 수 있으므로, 문맥을 고려하여 자연스럽게 번역해주세요:
                {chunk}"""

                response = openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}]
                )
                translated_chunks.append(response.choices[0].message.content.strip())  # 각 청크의 앞뒤 공백 제거
        
        # 번역된 청크들을 하나로 합침 (각 청크 사이에 줄바꿈 추가)
        combined_translation = "\n\n".join(translated_chunks)
        return combined_translation
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
    # 세션 상태 초기화
    if 'transcribed_text' not in st.session_state:
        st.session_state.transcribed_text = None
    if 'translated_text' not in st.session_state:
        st.session_state.translated_text = None
    if 'summary' not in st.session_state:
        st.session_state.summary = None
    if 'study_notes' not in st.session_state:
        st.session_state.study_notes = None
    if 'language' not in st.session_state:
        st.session_state.language = None

    st.title("음성 파일 변환 및 번역")
    
    # 언어 선택
    language = st.radio(
        "음성 파일의 언어를 선택하세요:",
        ("한국어", "영어"),
        index=0,
        key="language_radio"
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
    
    # 변환 시작 버튼이 눌렸거나 이전 결과가 있는 경우
    if (audio_file and st.button("변환 시작")) or st.session_state.transcribed_text:
        if audio_file:  # 새로운 파일이 업로드된 경우
            with st.spinner("음성을 처리하는 중..."):
                with NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.name)[1]) as tmp_file:
                    tmp_file.write(audio_file.getvalue())
                    tmp_file_path = tmp_file.name

                with open(tmp_file_path, "rb") as audio:
                    lang_code = "en" if language == "영어" else "ko"
                    st.session_state.transcribed_text = transcribe_audio(audio, lang_code)
                    st.session_state.language = language

                os.unlink(tmp_file_path)

        # 결과 표시 (새로운 변환이나 저장된 결과)
        if st.session_state.transcribed_text:
            st.subheader("변환된 텍스트:")
            st.write(st.session_state.transcribed_text)
            
            st.download_button(
                label="변환된 텍스트 다운로드",
                data=st.session_state.transcribed_text,
                file_name="transcribed_text.txt",
                mime="text/plain",
                key="transcribed_download"
            )

            # 영어인 경우 한국어 번역
            if st.session_state.language == "영어":
                if not st.session_state.get('translated_text'):  # 번역이 아직 안된 경우에만
                    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
                    token_length = len(encoding.encode(st.session_state.transcribed_text))
                    st.info(f"텍스트 길이: {token_length} 토큰")
                    
                    st.session_state.translated_text = translate_text(
                        st.session_state.transcribed_text,
                        chunk_size=3000,
                        chunk_overlap=500
                    )
                
                if st.session_state.translated_text:
                    st.subheader("한국어 번역:")
                    st.write(st.session_state.translated_text)
                    
                    # 번역본 다운로드 버튼 (전체 번역본을 하나의 파일로)
                    st.download_button(
                        label="번역본 다운로드",
                        data=st.session_state.translated_text,
                        file_name="translated_text.txt",
                        mime="text/plain",
                        key="translated_download"
                    )

            # 요약 옵션이 선택된 경우
            if do_summary:
                if not st.session_state.get('summary'):  # 요약이 아직 안된 경우에만
                    with st.spinner("텍스트를 요약하는 중..."):
                        text_to_summarize = st.session_state.translated_text if st.session_state.language == "영어" else st.session_state.transcribed_text
                        st.session_state.summary = summarize_text(text_to_summarize)
                
                if st.session_state.summary:
                    st.subheader("요약:")
                    st.write(st.session_state.summary)
                    
                    st.download_button(
                        label="요약본 다운로드",
                        data=st.session_state.summary,
                        file_name="summary.txt",
                        mime="text/plain",
                        key="summary_download"
                    )

            # 노트 필기 정리 옵션이 선택된 경우
            if do_notes:
                if not st.session_state.get('study_notes'):  # 노트가 아직 안만들어진 경우에만
                    with st.spinner("노트 필기 형태로 정리하는 중..."):
                        text_to_notes = st.session_state.translated_text if st.session_state.language == "영어" else st.session_state.transcribed_text
                        st.session_state.study_notes = create_study_notes(text_to_notes)
                
                if st.session_state.study_notes:
                    st.subheader("노트 필기 정리:")
                    st.markdown(st.session_state.study_notes)
                    
                    st.download_button(
                        label="노트 필기 다운로드",
                        data=st.session_state.study_notes,
                        file_name="study_notes.txt",
                        mime="text/plain",
                        key="notes_download"
                    )

    # 초기화 버튼
    if st.session_state.transcribed_text and st.button("새로운 파일 변환하기"):
        for key in ['transcribed_text', 'translated_text', 'summary', 'study_notes', 'language']:
            if key in st.session_state:
                del st.session_state[key]
        st.experimental_rerun()

if __name__ == "__main__":
    main()
