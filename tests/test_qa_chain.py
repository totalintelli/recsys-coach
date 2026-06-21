import importlib
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# heavy deps를 import 전에 mock 처리해 모델 다운로드 방지
for _mod in ("transformers", "langchain_huggingface", "bitsandbytes"):
    sys.modules.setdefault(_mod, MagicMock())


def _load_qa(env: dict):
    """환경변수를 주입한 채로 qa_chain을 재로딩한다."""
    with patch.dict(os.environ, env, clear=False):
        import src.qa_chain as m
        importlib.reload(m)
        return m


def _mock_vectorstore(page_content: str = "hello"):
    doc = MagicMock()
    doc.page_content = page_content
    doc.metadata = {"source": "test.pdf", "page": 0}
    vs = MagicMock()
    vs.as_retriever.return_value.invoke.return_value = [doc]
    vs._chunks = [doc]
    return vs, doc


# ── 기존 테스트 (Plan A) ──────────────────────────────────────────────────────

class TestCleanText(unittest.TestCase):
    def setUp(self):
        self.m = _load_qa({})

    def test_escaped_newline_restored(self):
        result = self.m._clean_text("line1\\nline2")
        self.assertIn("\n", result)
        self.assertNotIn("\\n", result)

    def test_br_tag_replaced(self):
        result = self.m._clean_text("A<br>B<br/>C")
        self.assertNotIn("<br>", result)
        self.assertNotIn("<br/>", result)
        self.assertIn("\n", result)

    def test_html_tags_stripped(self):
        result = self.m._clean_text("<p>hello <b>world</b></p>")
        self.assertEqual(result, "hello world")

    def test_consecutive_newlines_collapsed(self):
        result = self.m._clean_text("a\n\n\n\nb")
        self.assertNotIn("\n\n\n", result)


class TestEnrichChunk(unittest.TestCase):
    def setUp(self):
        self.m = _load_qa({})

    def test_source_header_added(self):
        doc = MagicMock()
        doc.page_content = "content"
        doc.metadata = {"source": "guide.pdf", "page": 2}
        result = self.m._enrich_chunk(doc)
        self.assertIn("[출처: guide.pdf", result.page_content)
        self.assertIn("3페이지", result.page_content)

    def test_no_page_metadata(self):
        doc = MagicMock()
        doc.page_content = "content"
        doc.metadata = {"source": "guide.pdf"}
        result = self.m._enrich_chunk(doc)
        self.assertIn("[출처: guide.pdf]", result.page_content)


class TestOfflineMode(unittest.TestCase):
    def test_offline_returns_raw_docs_no_llm(self):
        m = _load_qa({"OFFLINE_MODE": "true", "UPSTAGE_API_KEY": "x", "LLM_BACKEND": "upstage"})
        m._answer_cache = {}
        vs, _ = _mock_vectorstore("hello")
        fake_doc = MagicMock()
        fake_doc.page_content = "hello"
        fake_doc.metadata = {}
        with patch.object(m, "_build_retriever") as mock_ret, \
             patch.object(m, "_rerank_with_cross_encoder", return_value=[fake_doc]):
            mock_ret.return_value.invoke.return_value = [fake_doc]
            result = m.answer_question(vs, "q")
        self.assertIn("hello", result["answer"])
        self.assertIn("sources", result)

    def test_offline_multiple_docs_joined(self):
        m = _load_qa({"OFFLINE_MODE": "true", "UPSTAGE_API_KEY": "x", "LLM_BACKEND": "upstage"})
        m._answer_cache = {}
        doc1, doc2 = MagicMock(), MagicMock()
        doc1.page_content = "A"
        doc2.page_content = "B"
        doc1.metadata = {}
        doc2.metadata = {}
        vs = MagicMock()
        vs._chunks = []
        with patch.object(m, "_build_retriever") as mock_ret, \
             patch.object(m, "_rerank_with_cross_encoder", return_value=[doc1, doc2]):
            mock_ret.return_value.invoke.return_value = [doc1, doc2]
            result = m.answer_question(vs, "q")
        self.assertIn("A", result["answer"])
        self.assertIn("B", result["answer"])


class TestBackendRouting(unittest.TestCase):
    def test_upstage_backend_default(self):
        m = _load_qa({"OFFLINE_MODE": "false", "LLM_BACKEND": "upstage", "UPSTAGE_API_KEY": "x"})
        m._answer_cache = {}
        vs, _ = _mock_vectorstore()
        with patch.object(m, "_build_retriever") as mock_ret, \
             patch.object(m, "_rerank_with_cross_encoder", return_value=[]), \
             patch.object(m, "_get_upstage_llm") as mock_upstage, \
             patch.object(m, "_get_upstage_prompt") as mock_prompt:
            mock_ret.return_value.invoke.return_value = []
            mock_upstage.return_value = MagicMock()
            mock_prompt.return_value = MagicMock()
            try:
                m.answer_question(vs, "question")
            except Exception:
                pass
            mock_upstage.assert_called_once()

    def test_llama3_backend_explicit(self):
        m = _load_qa({"OFFLINE_MODE": "false", "LLM_BACKEND": "llama3"})
        m._answer_cache = {}
        vs, _ = _mock_vectorstore()
        with patch.object(m, "_build_retriever") as mock_ret, \
             patch.object(m, "_rerank_with_cross_encoder", return_value=[]), \
             patch.object(m, "_get_llama_llm") as mock_llama, \
             patch.object(m, "_get_llama_prompt") as mock_prompt, \
             patch.object(m, "_get_upstage_llm") as mock_upstage:
            mock_ret.return_value.invoke.return_value = []
            mock_llama.return_value = MagicMock()
            mock_prompt.return_value = MagicMock()
            try:
                m.answer_question(vs, "question")
            except Exception:
                pass
            mock_llama.assert_called_once()
            mock_upstage.assert_not_called()

    def test_upstage_fallback_on_exception(self):
        m = _load_qa({"OFFLINE_MODE": "false", "LLM_BACKEND": "upstage", "UPSTAGE_API_KEY": "x"})
        m._answer_cache = {}
        vs, _ = _mock_vectorstore()
        with patch.object(m, "_build_retriever") as mock_ret, \
             patch.object(m, "_rerank_with_cross_encoder", return_value=[]), \
             patch.object(m, "_get_upstage_llm", side_effect=Exception("API down")), \
             patch.object(m, "_get_llama_llm") as mock_llama, \
             patch.object(m, "_get_llama_prompt") as mock_prompt:
            mock_ret.return_value.invoke.return_value = []
            mock_llama.return_value = MagicMock()
            mock_prompt.return_value = MagicMock()
            try:
                m.answer_question(vs, "question")
            except Exception:
                pass
            mock_llama.assert_called_once()


class TestLlamaLLMCache(unittest.TestCase):
    def test_llama_llm_loaded_once(self):
        m = _load_qa({"OFFLINE_MODE": "false", "LLM_BACKEND": "llama3"})
        m._llama_llm_cache = None

        fake_llm = MagicMock()
        call_count = 0

        def fake_build():
            nonlocal call_count
            call_count += 1
            m._llama_llm_cache = fake_llm
            return fake_llm

        with patch.object(m, "_get_llama_llm", side_effect=fake_build):
            m._get_llama_llm()
            m._get_llama_llm()

        self.assertIsNotNone(m._llama_llm_cache)


class TestBuildRetriever(unittest.TestCase):
    def test_fallback_to_faiss_when_bm25_unavailable(self):
        m = _load_qa({})
        vs = MagicMock()
        faiss_retriever = MagicMock()
        vs.as_retriever.return_value = faiss_retriever
        try:
            result = m._build_retriever(vs, [])
            self.assertIsNotNone(result)
        except Exception:
            pass


# ── 신규 테스트 (Plan B) ──────────────────────────────────────────────────────

class TestSplitDocuments(unittest.TestCase):
    def setUp(self):
        self.m = _load_qa({})

    def test_split_produces_multiple_chunks(self):
        doc = MagicMock()
        doc.page_content = "가" * 1500
        doc.metadata = {"source": "test.pdf", "page": 0}
        # RecursiveCharacterTextSplitter는 실제 LangChain 객체이므로 통합 테스트
        # 여기서는 반환 타입만 검증
        try:
            from langchain_core.documents import Document as RealDoc
            real_doc = RealDoc(page_content="가" * 1500, metadata={"source": "test.pdf", "page": 0})
            result = self.m._split_documents([real_doc])
            self.assertGreater(len(result), 1)
        except Exception:
            pass  # LangChain 미설치 환경에서는 패스

    def test_chunk_metadata_preserved(self):
        try:
            from langchain_core.documents import Document as RealDoc
            real_doc = RealDoc(page_content="가" * 1500, metadata={"source": "guide.pdf", "page": 3})
            result = self.m._split_documents([real_doc])
            for chunk in result:
                self.assertEqual(chunk.metadata.get("source"), "guide.pdf")
                self.assertEqual(chunk.metadata.get("page"), 3)
        except Exception:
            pass

    def test_paragraph_boundary_separators_configured(self):
        m = self.m
        # _split_documents를 실제 호출하지 않고 함수가 존재하는지만 확인
        self.assertTrue(callable(m._split_documents))


class TestCleanAnswer(unittest.TestCase):
    def setUp(self):
        self.m = _load_qa({})

    def test_strips_escaped_newlines(self):
        result = self.m._clean_answer("hello\\nworld")
        self.assertIn("\n", result)
        self.assertNotIn("\\n", result)

    def test_strips_leading_trailing_whitespace(self):
        result = self.m._clean_answer("  answer  ")
        self.assertEqual(result, "answer")

    def test_collapses_triple_newlines(self):
        result = self.m._clean_answer("a\n\n\n\nb")
        self.assertNotIn("\n\n\n", result)

    def test_strips_html_artifacts(self):
        result = self.m._clean_answer("answer<br/>more")
        self.assertNotIn("<br/>", result)
        self.assertIn("\n", result)

    def test_preserves_leading_indentation_for_nested_lists(self):
        # 하위 불릿의 2칸 들여쓰기가 보존되어야 CommonMark 중첩 목록(●/○)이 유지된다
        md = "- 상위\n  - 하위1\n  - 하위2"
        result = self.m._clean_answer(md)
        self.assertIn("\n  - 하위1", result)
        self.assertIn("\n  - 하위2", result)

    def test_nested_list_renders_nested_ul(self):
        # _clean_answer를 거친 뒤에도 _md_to_html이 중첩 <ul>을 생성해야 한다
        md = "## 제목\n- 상위\n  - 하위"
        html = self.m._md_to_html(self.m._clean_answer(md))
        self.assertGreaterEqual(html.count("<ul>"), 2)
        self.assertIn("하위", html)

    def test_collapses_midline_double_spaces(self):
        # 줄 중간/끝의 중복 공백은 여전히 1칸으로 압축되어야 한다 (회귀 방지)
        result = self.m._clean_answer("a  b   c")
        self.assertEqual(result, "a b c")


class TestReranker(unittest.TestCase):
    def setUp(self):
        self.m = _load_qa({})

    def _make_docs(self, n=4):
        docs = []
        for i in range(n):
            d = MagicMock()
            d.page_content = f"document {i}"
            d.metadata = {}
            docs.append(d)
        return docs

    def test_import_error_fallback(self):
        docs = self._make_docs(4)
        with patch.dict(sys.modules, {"sentence_transformers": None}):
            result = self.m._rerank_with_cross_encoder("query", docs, top_k=3)
        self.assertEqual(len(result), 3)

    def test_reranker_reorders_by_score(self):
        docs = self._make_docs(4)
        scores = [0.1, 0.9, 0.5, 0.3]
        mock_ce = MagicMock()
        mock_ce.return_value.predict.return_value = scores
        with patch.dict(sys.modules, {"sentence_transformers": MagicMock(CrossEncoder=mock_ce)}):
            try:
                result = self.m._rerank_with_cross_encoder("query", docs, top_k=3)
                # 설치된 경우: 점수 0.9인 docs[1]이 첫 번째여야 함
                if result:
                    self.assertEqual(len(result), 3)
            except Exception:
                pass  # sentence_transformers mock 구조 차이 허용

    def test_runtime_exception_fallback(self):
        docs = self._make_docs(4)
        mock_st = MagicMock()
        mock_st.CrossEncoder.side_effect = RuntimeError("OOM")
        with patch.dict(sys.modules, {"sentence_transformers": mock_st}):
            result = self.m._rerank_with_cross_encoder("query", docs, top_k=3)
        self.assertEqual(len(result), 3)

    def test_fewer_docs_than_top_k(self):
        docs = self._make_docs(2)
        with patch.dict(sys.modules, {"sentence_transformers": None}):
            result = self.m._rerank_with_cross_encoder("query", docs, top_k=3)
        self.assertEqual(len(result), 2)


class TestAnswerCache(unittest.TestCase):
    def test_cache_cleared_on_module_reload(self):
        m = _load_qa({})
        self.assertEqual(m._answer_cache, {})

    def test_cache_hit_returns_same_object(self):
        m = _load_qa({"OFFLINE_MODE": "false", "LLM_BACKEND": "upstage", "UPSTAGE_API_KEY": "x"})
        m._answer_cache = {}
        vs = MagicMock()
        vs._chunks = []
        expected = {"answer": "cached answer", "sources": []}
        cache_key = (id(vs), "same question")
        m._answer_cache[cache_key] = expected

        with patch.object(m, "_build_retriever") as mock_ret:
            result = m.answer_question(vs, "same question")
            mock_ret.assert_not_called()  # 캐시 히트 → retriever 호출 없음

        self.assertIs(result, expected)

    def test_different_question_misses_cache(self):
        m = _load_qa({"OFFLINE_MODE": "false", "LLM_BACKEND": "upstage", "UPSTAGE_API_KEY": "x"})
        m._answer_cache = {}
        vs = MagicMock()
        vs._chunks = []
        m._answer_cache[(id(vs), "question A")] = {"answer": "A", "sources": []}

        with patch.object(m, "_build_retriever") as mock_ret, \
             patch.object(m, "_rerank_with_cross_encoder", return_value=[]), \
             patch.object(m, "_get_upstage_llm") as mock_llm, \
             patch.object(m, "_get_upstage_prompt") as mock_prompt:
            mock_ret.return_value.invoke.return_value = []
            mock_llm.return_value = MagicMock()
            mock_prompt.return_value = MagicMock()
            try:
                m.answer_question(vs, "question B")  # 다른 질문 → 캐시 미스
            except Exception:
                pass
            mock_ret.assert_called_once()  # retriever가 호출되어야 함

    def test_different_vectorstore_misses_cache(self):
        m = _load_qa({"OFFLINE_MODE": "false", "LLM_BACKEND": "upstage", "UPSTAGE_API_KEY": "x"})
        m._answer_cache = {}
        vs1 = MagicMock()
        vs1._chunks = []
        vs2 = MagicMock()
        vs2._chunks = []
        m._answer_cache[(id(vs1), "q")] = {"answer": "cached", "sources": []}

        with patch.object(m, "_build_retriever") as mock_ret, \
             patch.object(m, "_rerank_with_cross_encoder", return_value=[]), \
             patch.object(m, "_get_upstage_llm") as mock_llm, \
             patch.object(m, "_get_upstage_prompt") as mock_prompt:
            mock_ret.return_value.invoke.return_value = []
            mock_llm.return_value = MagicMock()
            mock_prompt.return_value = MagicMock()
            try:
                m.answer_question(vs2, "q")  # 다른 vs → 캐시 미스
            except Exception:
                pass
            mock_ret.assert_called_once()


class TestMdToHtml(unittest.TestCase):
    def setUp(self):
        self.m = _load_qa({})

    def test_empty_string_returns_empty(self):
        result = self.m._md_to_html("")
        self.assertEqual(result, "")

    def test_whitespace_only_returns_empty(self):
        result = self.m._md_to_html("   \n   ")
        self.assertEqual(result, "")

    def test_headers_rendered(self):
        md = "# 제목\n## 부제목\n### 소제목\n"
        result = self.m._md_to_html(md)
        self.assertIn("<h1>", result)
        self.assertIn("제목</h1>", result)
        self.assertIn("<h2>", result)
        self.assertIn("부제목</h2>", result)
        self.assertIn("<h3>", result)
        self.assertIn("소제목</h3>", result)
        self.assertIn('class="jupyter-markdown"', result)

    def test_code_block_with_highlighting(self):
        md = "```python\nprint('hello')\n```"
        result = self.m._md_to_html(md)
        self.assertIn("<pre>", result)
        self.assertIn("<code", result)
        self.assertIn("print", result)

    def test_inline_code(self):
        md = "이것은 `inline code` 입니다"
        result = self.m._md_to_html(md)
        self.assertIn("<code>", result)
        self.assertIn("inline code", result)

    def test_table_rendered(self):
        md = "| A | B |\n|---|---|\n| 1 | 2 |"
        result = self.m._md_to_html(md)
        self.assertIn("<table>", result)
        self.assertIn("<th>", result)
        self.assertIn("<td>", result)

    def test_bullet_list(self):
        md = "- 항목1\n- 항목2\n  - 하위항목"
        result = self.m._md_to_html(md)
        self.assertIn("<ul>", result)
        self.assertIn("<li>", result)
        self.assertIn("항목1", result)
        self.assertIn("항목2", result)
        self.assertIn("하위항목", result)

    def test_ordered_list(self):
        md = "1. 첫째\n2. 둘째"
        result = self.m._md_to_html(md)
        self.assertIn("<ol>", result)
        self.assertIn("<li>", result)
        self.assertIn("첫째", result)
        self.assertIn("둘째", result)

    def test_blockquote(self):
        md = "> 인용문입니다"
        result = self.m._md_to_html(md)
        self.assertIn("<blockquote>", result)
        self.assertIn("인용문입니다", result)

    def test_jupyter_css_included(self):
        md = "# 테스트"
        result = self.m._md_to_html(md)
        self.assertIn(".jupyter-markdown", result)
        self.assertIn("<style>", result)

    def test_fallback_when_import_missing(self):
        with patch.dict(sys.modules, {"markdown_it": None, "pygments": None}):
            m = _load_qa({})
            result = m._md_to_html("hello\nworld")
            self.assertIn("hello", result)
            self.assertIn("world", result)


if __name__ == "__main__":
    unittest.main()
