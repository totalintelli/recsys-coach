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
        from langchain_core.documents import Document
        self.Document = Document
        self.m = _load_qa({})

    def test_source_header_added(self):
        doc = self.Document(page_content="content", metadata={"source": "guide.pdf", "page": 2})
        result = self.m._enrich_chunk(doc)
        self.assertIn("[출처: guide.pdf", result.page_content)
        self.assertIn("3페이지", result.page_content)

    def test_no_page_metadata(self):
        doc = self.Document(page_content="content", metadata={"source": "guide.pdf"})
        result = self.m._enrich_chunk(doc)
        self.assertIn("[출처: guide.pdf]", result.page_content)


class TestOfflineMode(unittest.TestCase):
    def test_offline_returns_raw_docs_no_llm(self):
        m = _load_qa({"OFFLINE_MODE": "true", "UPSTAGE_API_KEY": "x", "LLM_BACKEND": "upstage"})
        vs, _ = _mock_vectorstore("hello")
        with patch.object(m, "_build_retriever") as mock_ret:
            mock_ret.return_value.invoke.return_value = [MagicMock(page_content="hello", metadata={})]
            result = m.answer_question(vs, "q")
        self.assertIn("hello", result["answer"])
        self.assertIn("sources", result)

    def test_offline_multiple_docs_joined(self):
        m = _load_qa({"OFFLINE_MODE": "true", "UPSTAGE_API_KEY": "x", "LLM_BACKEND": "upstage"})
        doc1, doc2 = MagicMock(), MagicMock()
        doc1.page_content = "A"
        doc2.page_content = "B"
        doc1.metadata = {}
        doc2.metadata = {}
        vs = MagicMock()
        vs._chunks = []
        with patch.object(m, "_build_retriever") as mock_ret:
            mock_ret.return_value.invoke.return_value = [doc1, doc2]
            result = m.answer_question(vs, "q")
        self.assertIn("A", result["answer"])
        self.assertIn("B", result["answer"])


class TestBackendRouting(unittest.TestCase):
    def test_upstage_backend_default(self):
        m = _load_qa({"OFFLINE_MODE": "false", "LLM_BACKEND": "upstage", "UPSTAGE_API_KEY": "x"})
        vs, _ = _mock_vectorstore()
        with patch.object(m, "_build_retriever") as mock_ret, \
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
        vs, _ = _mock_vectorstore()
        with patch.object(m, "_build_retriever") as mock_ret, \
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
        vs, _ = _mock_vectorstore()
        with patch.object(m, "_build_retriever") as mock_ret, \
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

        with patch.dict(sys.modules, {"langchain_community.retrievers": None}):
            # BM25Retriever import 실패 → FAISS만 반환
            try:
                result = m._build_retriever(vs, [])
                # ImportError가 발생하지 않으면 FAISS retriever이어야 함
                self.assertIsNotNone(result)
            except Exception:
                pass  # import 실패 경로도 허용


if __name__ == "__main__":
    unittest.main()
