"""Offline checks using in-memory Qdrant and deterministic embeddings."""
import hashlib
import io
import json
import os
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import streamlit as st
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from qdrant_client import QdrantClient

import app
import evaluate_rag


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [([0.1, 1.0] if "needle" in text else [1.0, 0.0]) + [0.0] * 382 for text in texts]

    def embed_query(self, text):
        return [1.0, 0.0] + [0.0] * 382


class Upload(io.BytesIO):
    def __init__(self, name, content):
        super().__init__(content)
        self.name = name


class AppRegressionTests(unittest.TestCase):
    def setUp(self):
        self.client = QdrantClient(":memory:")
        self.store = app.get_vectorstore(self.client, FakeEmbeddings())

    def tearDown(self):
        self.client.close()

    def test_greeting_does_not_swallow_document_question(self):
        self.assertIsNone(app.detect_chit_chat("hi refund policy"))
        self.assertEqual(app.detect_chit_chat(" Hello! "), app.GREETINGS_MAP["hello"])

    def test_repeated_text_retains_page_identity(self):
        chunks = app.split_documents([Document(page_content="Repeated heading", metadata={"source": "a.pdf", "page": page})
                                     for page in (0, 1)])
        self.assertNotEqual(chunks[0].metadata["chunk_id"], chunks[1].metadata["chunk_id"])
        self.assertEqual(app.format_doc_location({"row": 0}), "Row 1")

    def test_loading_validation_and_utf8(self):
        docs = app.load_any_file(Upload("a.txt", "\ufeffA real document".encode("utf-8")))
        self.assertEqual(docs[0].page_content, "A real document")
        for uploaded in [Upload("bad.doc", b"legacy"), Upload("empty.txt", b"")]:
            with self.assertRaises(ValueError):
                app.load_any_file(uploaded)

    def test_tuple_scores_and_formatting(self):
        docs = [Document(page_content=str(i)) for i in range(3)]
        ordered = app.reorder_for_llm([(docs[0], 0.2), (docs[1], 0.9), (docs[2], 0.7)])
        self.assertEqual([d.page_content for d, _ in ordered], ["1", "0", "2"])
        self.assertIn("Document Source", app.format_docs((docs[0], 0.2)))

    def test_vectorstore_uses_requested_client(self):
        other = QdrantClient(":memory:")
        try:
            self.assertIs(app.get_vectorstore(other, FakeEmbeddings()).client, other)
            self.assertIs(self.store.client, self.client)
        finally:
            other.close()

    def test_hybrid_real_scores_filter_and_k(self):
        dense = [Document(page_content=f"ordinary {i}", metadata={"source": "a", "chunk_id": str(i)}) for i in range(25)]
        needle = Document(page_content="needle", metadata={"source": "a", "chunk_id": "needle"})
        other = Document(page_content="needle other", metadata={"source": "b", "chunk_id": "other"})
        corpus = dense + [needle, other]
        self.store.add_documents(corpus)
        docs, _ = app.retrieve_docs_with_scores(self.store, "needle", 2, "Hybrid", 0.0, "a", corpus)
        self.assertEqual(len(docs), 2)
        self.assertTrue(all(d.metadata["source"] == "a" for d in docs))
        sparse = next(d for d in docs if d.metadata["chunk_id"] == "needle")
        self.assertAlmostEqual(sparse.metadata["score"], 0.1 / np.sqrt(1.01))
        self.assertNotIn("score", needle.metadata)
        docs, _ = app.retrieve_docs_with_scores(self.store, "needle", 2, "Hybrid", 0.4, "a", corpus)
        self.assertTrue(all(d.metadata["chunk_id"] != "needle" for d in docs))

    def test_native_hybrid_fuses_and_filters(self):
        self.store.add_documents([Document(page_content="needle", metadata={"source": "a"}),
                                  Document(page_content="ordinary", metadata={"source": "b"})])
        app.prepare_native_index(self.client)
        docs, _ = app.retrieve_docs_with_scores(self.store, "needle", 3, "Native Hybrid (Qdrant RRF)", 0, "a")
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].metadata["source"], "a")
        self.assertAlmostEqual(docs[0].metadata["score"], 0.1 / np.sqrt(1.01))
        expected = int.from_bytes(hashlib.blake2s(b"needle", digest_size=4).digest(), "big")
        self.assertEqual(app.create_sparse_vector("needle").indices, [expected])

    def test_replacement_and_idempotent_upload(self):
        old = app.split_documents(app.load_any_file(Upload("a.txt", b"old content")))
        new = app.split_documents(app.load_any_file(Upload("a.txt", b"new content")))
        app.index_chunks(self.store, old)
        app.index_chunks(self.store, new)
        app.index_chunks(self.store, new)
        recovered = app.recover_documents(self.client)
        self.assertEqual([d.page_content for d in recovered], ["new content"])

    def test_failed_upload_rolls_back_new_points(self):
        old = app.split_documents(app.load_any_file(Upload("a.txt", b"old content")))
        app.index_chunks(self.store, old)
        new = app.split_documents(app.load_any_file(Upload("a.txt", b"new content longer than one chunk")), 10, 0)
        original = self.store.add_documents
        calls = 0

        def fail_after_first(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls > 1:
                raise RuntimeError("simulated network error")
            return original(*args, **kwargs)

        with patch.object(app, "BATCH_SIZE", 1), patch.object(self.store, "add_documents", side_effect=fail_after_first):
            with self.assertRaises(RuntimeError):
                app.index_chunks(self.store, new)
        self.assertEqual([d.page_content for d in app.recover_documents(self.client)], ["old content"])

    def test_recovery_paginates(self):
        self.store.add_documents([Document(page_content=f"chunk {i}") for i in range(1005)])
        self.assertEqual(len(app.recover_documents(self.client)), 1005)

    def test_semantic_and_mmr_respect_scope(self):
        self.store.add_documents([Document(page_content="ordinary", metadata={"source": "a"}),
                                  Document(page_content="needle", metadata={"source": "b"})])
        for strategy in ["Semantic Similarity", "Score Threshold (Cutoff)", "MMR (Diverse)"]:
            docs, maximum = app.retrieve_docs_with_scores(self.store, "policy", 3, strategy, 0.4, "a")
            self.assertEqual(len(docs), 1)
            self.assertEqual(docs[0].metadata["source"], "a")
            self.assertAlmostEqual(maximum, 1.0)


class EvaluationTests(unittest.TestCase):
    def test_judge_rejects_invalid_scores_without_final_sleep(self):
        for value in [float("nan"), 1.5, True, "0.5"]:
            scores = dict.fromkeys(["faithfulness", "answer_relevancy", "context_recall", "context_precision"], value)
            chain = types.SimpleNamespace(invoke=lambda payload: json.dumps(scores))
            with patch.object(evaluate_rag.time, "sleep") as sleep:
                with self.assertRaises(RuntimeError):
                    evaluate_rag.evaluate_sample_with_judge(chain, {}, max_retries=1)
                sleep.assert_not_called()

    def test_judge_accepts_fenced_json(self):
        scores = dict.fromkeys(["faithfulness", "answer_relevancy", "context_recall", "context_precision"], 0.5)
        chain = types.SimpleNamespace(invoke=lambda payload: "```json\n" + json.dumps(scores) + "\n```")
        self.assertEqual(evaluate_rag.evaluate_sample_with_judge(chain, {})["faithfulness"], 0.5)


class StreamlitTests(unittest.TestCase):
    def test_startup_greeting_history_preview_and_reset(self):
        from streamlit.testing.v1 import AppTest
        client = QdrantClient(":memory:")
        app.get_vectorstore(client, FakeEmbeddings()).add_documents([
            Document(page_content="Our refund policy is 30 days.", metadata={"source": "policy.txt"})])
        st.cache_resource.clear()
        try:
            with patch.dict(os.environ, {"QDRANT_URL": "", "GROQ_API_KEY": "", "OPENAI_API_KEY": ""}), \
                    patch("langchain_huggingface.HuggingFaceEmbeddings", return_value=FakeEmbeddings()), \
                    patch("qdrant_client.QdrantClient", return_value=client):
                ui = AppTest.from_file(str(Path(app.__file__))).run(timeout=30)
                self.assertEqual(len(ui.exception), 0)
                self.assertEqual(ui.session_state["indexed_files"], {"policy.txt"})
                ui.selectbox[0].select("Preview Mode (No LLM)").run()
                ui.chat_input[0].set_value("hello").run()
                self.assertEqual(len(ui.session_state["chat_history"]), 2)
                ui.chat_input[0].set_value("What is the refund policy?").run()
                self.assertEqual(len(ui.exception), 0)
                self.assertIn("30 days", ui.session_state["chat_history"][-1][1])
                ui.selectbox[0].select("Groq").run()
                ui.text_input[0].set_value("test-key").run()
                generation_prompts = []

                def fake_llm(prompt):
                    if prompt.messages[0].content.startswith("Rewrite"):
                        return AIMessage(content="What is the refund policy?")
                    generation_prompts.append(prompt.messages[-1].content)
                    return AIMessage(content="Refunds are available within 30 days.")

                with patch("langchain_openai.ChatOpenAI", return_value=RunnableLambda(fake_llm)):
                    ui.chat_input[0].set_value("How long is it?").run()
                self.assertEqual(len(ui.exception), 0)
                self.assertIn("Question: What is the refund policy?", generation_prompts[0])
                self.assertIn("30 days", ui.session_state["chat_history"][-1][1])
                previous = list(ui.session_state["chat_history"])
                with patch("langchain_openai.ChatOpenAI", side_effect=RuntimeError("provider unavailable")):
                    ui.chat_input[0].set_value("A failing follow-up").run()
                self.assertEqual(ui.session_state["chat_history"], previous)
                self.assertEqual(len(ui.exception), 0)
                self.assertTrue(ui.error)
                reset = next(button for button in ui.button if button.label == "🗑️ Reset Docs")
                reset.click().run()
                self.assertEqual(len(ui.exception), 0)
                self.assertEqual(ui.session_state["indexed_files"], set())
                self.assertEqual(ui.session_state["chat_history"], [])
        finally:
            st.cache_resource.clear()
            client.close()


if __name__ == "__main__":
    unittest.main()
