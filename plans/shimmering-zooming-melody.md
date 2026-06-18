# 참고문서 파일명 표시 수정

## Context

PDF를 업로드하면 `PyPDFLoader`가 임시 파일(`/tmp/tmpXXXXXX.pdf`)을 만들어 로드하는데, LangChain이 이 임시 경로를 `source` 메타데이터로 자동 기록한다. 결과적으로 참고문서 섹션 최상단에 임시 경로 문자열이 노출된다. TXT는 `metadata={"source": uf.name}`을 명시하므로 문제없다.

## 수정 대상

**파일 1개, 3줄 변경:**

`app.py` 라인 46-48

```python
# 현재
loader = PyPDFLoader(tmp_path)
docs.extend(loader.load())
os.unlink(tmp_path)

# 변경 후
loader = PyPDFLoader(tmp_path)
loaded = loader.load()
for doc in loaded:
    doc.metadata["source"] = uf.name  # 임시 경로 → 업로드 파일명으로 덮어쓰기
docs.extend(loaded)
os.unlink(tmp_path)
```

`PyPDFLoader`는 page별로 Document를 나눠 반환하고 각 Document에 `{"source": "/tmp/...", "page": 0}` 형태로 메타데이터를 붙인다. `source`만 `uf.name`으로 교체하면 `page` 번호는 그대로 보존된다.

## 검증

1. `streamlit run app.py` 실행
2. PDF 파일 업로드 후 질문
3. "참고 문서" 익스팬더 열기 → `[1] 실제파일명.pdf` 형식으로 표시되는지 확인
4. TXT 업로드도 동일하게 확인 (기존 동작 유지)
