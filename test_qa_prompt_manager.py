import hashlib
import json
import re
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import numpy as np

from chatbot_v04_keywords import TyphoonChatbot
from qa_prompt_manager import QAPromptManager
from query_normalizer import QueryNormalizer


class DeterministicEmbeddingModel:
    """Deterministic embedding stub for repeatable unit tests."""

    def __init__(self, dimension: int = 128):
        self.dimension = dimension

    def encode(self, texts, normalize_embeddings: bool = True):
        if isinstance(texts, str):
            texts = [texts]

        vectors = [self._vectorize(text) for text in texts]
        matrix = np.vstack(vectors)
        if normalize_embeddings:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            matrix = matrix / norms
        return matrix

    def _vectorize(self, text: str) -> np.ndarray:
        lowered = text.lower()
        vector = np.zeros(self.dimension, dtype=float)
        tokens = re.findall(r"[a-z0-9\u0E00-\u0E7F]+", lowered)

        for token in tokens:
            index = int(hashlib.sha1(token.encode("utf-8")).hexdigest(), 16) % self.dimension
            vector[index] += 1.0

        phrase_boosts = [
            "เปลี่ยนตอนเรียน",
            "ถอนวิชาเรียน",
            "การขอลาพักการเรียน",
            "ปฏิทินการศึกษา",
        ]
        for phrase in phrase_boosts:
            if phrase in lowered:
                index = int(hashlib.sha1(f"phrase:{phrase}".encode("utf-8")).hexdigest(), 16) % self.dimension
                vector[index] += 3.0

        if not np.any(vector):
            index = int(hashlib.sha1(lowered.encode("utf-8")).hexdigest(), 16) % self.dimension
            vector[index] = 1.0

        return vector


class FakeKnowledgeBase:
    """Minimal KB stub for TyphoonChatbot routing smoke tests."""

    def __init__(self, model: DeterministicEmbeddingModel, synonym_path: str):
        self.model = model
        self._normalizer = QueryNormalizer(synonym_path=synonym_path)

    def normalize_query(self, query: str) -> Dict[str, Any]:
        return self._normalizer.normalize_query(query)

    def search_knowledge(self, query: str, n_results: int = 10) -> List[Dict[str, Any]]:
        normalized = self.normalize_query(query)["normalized_text"] or query
        return [
            {
                "text": "ข้อมูลตัวอย่างจาก vector",
                "score": 0.8,
                "rank": 1,
                "bm25_score": 0.7,
                "vector_score": 0.9,
                "raw_vector_similarity": 0.9,
                "max_raw_vector_similarity": 0.9,
                "keyword_score": 0.5,
                "metadata": {},
                "matched_keywords": [],
                "normalized_query": normalized,
            }
        ][:n_results]

    def get_context_string(self, relevant_items: List[Dict[str, Any]]) -> str:
        if not relevant_items:
            return ""
        return f"ข้อมูลที่เกี่ยวข้องจากฐานความรู้:\n\n1. {relevant_items[0]['text']}\n"


class TestQAPromptManager(unittest.TestCase):
    def setUp(self):
        self.embedding_model = DeterministicEmbeddingModel()
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir.name)
        self.qa_path = self.tmp_path / "qa_prompt.json"
        self.synonym_path = self.tmp_path / "query_synonyms.json"
        self.excel_path = self.tmp_path / "QA.xlsx"

        self.qa_items = [
            {
                "id": "q1",
                "source_document": "คู่มือนักศึกษา",
                "category": "การเปลี่ยนตอนเรียน",
                "question": "ขอข้อมูลการการเปลี่ยนตอนเรียนหน่อย",
                "answer": "รายละเอียดการเปลี่ยนตอนเรียน",
                "answer_updated_at": None,
            },
            {
                "id": "q2",
                "source_document": "คู่มือนักศึกษา",
                "category": "การถอนวิชาเรียน",
                "question": "ฉันต้องการถอนวิชาเรียนทำได้ไหม ต้องทำอะไรบ้าง",
                "answer": "รายละเอียดการถอนวิชาเรียน",
                "answer_updated_at": None,
            },
            {
                "id": "q3",
                "source_document": "คู่มือนักศึกษา",
                "category": "การลาพักการศึกษา",
                "question": "การขอลาพักการเรียน",
                "answer": "รายละเอียดการลาพักการเรียน",
                "answer_updated_at": None,
            },
            {
                "id": "q4",
                "source_document": "คู่มือนักศึกษา",
                "category": "การลงทะเบียน",
                "question": "ระเบียบการลงทะเบียน",
                "answer": "รายละเอียดการลงทะเบียน",
                "answer_updated_at": None,
            },
        ]
        self._write_json(self.qa_path, self.qa_items)

        synonym_payload = {
            "version": 1,
            "aliases": [
                {
                    "canonical": "เปลี่ยนตอนเรียน",
                    "terms": ["เปลี่ยนเซค", "เปลี่ยน section", "เปลี่ยนกลุ่มเรียน", "ย้ายตอน"],
                },
                {
                    "canonical": "ถอนวิชาเรียน",
                    "terms": ["ดรอปวิชา", "ดรอปเรียน", "drop วิชา", "ถอนรายวิชา"],
                },
                {
                    "canonical": "การขอลาพักการเรียน",
                    "terms": ["พักการศึกษา", "พักการเรียน", "ลาพัก", "ขอพักการศึกษาชั่วคราว"],
                },
            ],
        }
        self._write_json(self.synonym_path, synonym_payload)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_synonym_maps_to_change_section_question(self):
        manager = self._build_manager()
        match = manager.find_best_match("เปลี่ยนเซคเรียนได้ไหม")

        self.assertIsNotNone(match)
        self.assertEqual(match["category"], "การเปลี่ยนตอนเรียน")
        self.assertIn("เปลี่ยนตอนเรียน", match["matched_canonicals"])
        self.assertIn("เปลี่ยนตอนเรียน", match["normalized_query"])
        self.assertGreaterEqual(match["score_gap"], 0.06)

    def test_synonym_maps_to_leave_of_absence_question(self):
        manager = self._build_manager()
        match = manager.find_best_match("ขอพักการศึกษาชั่วคราว")

        self.assertIsNotNone(match)
        self.assertEqual(match["question"], "การขอลาพักการเรียน")
        self.assertIn("การขอลาพักการเรียน", match["matched_canonicals"])

    def test_withdraw_drop_query_matches_withdraw_intent(self):
        manager = self._build_manager()
        match = manager.find_best_match("ถอนดรอปวิชาจะขึ้น W ไหม")

        self.assertIsNotNone(match)
        self.assertEqual(match["category"], "การถอนวิชาเรียน")
        self.assertNotEqual(match["category"], "การเปลี่ยนตอนเรียน")

    def test_returns_none_when_top_scores_too_close(self):
        ambiguous_items = [
            {
                "id": "a1",
                "source_document": "คู่มือ",
                "category": "เปลี่ยนตอน A",
                "question": "เปลี่ยนตอนเรียนอย่างไร",
                "answer": "คำตอบ A",
                "answer_updated_at": None,
            },
            {
                "id": "a2",
                "source_document": "คู่มือ",
                "category": "เปลี่ยนตอน B",
                "question": "เปลี่ยนตอนเรียนต้องทำอะไร",
                "answer": "คำตอบ B",
                "answer_updated_at": None,
            },
        ]
        ambiguous_qa_path = self.tmp_path / "qa_ambiguous.json"
        self._write_json(ambiguous_qa_path, ambiguous_items)

        manager = QAPromptManager(
            excel_path=str(self.excel_path),
            json_path=str(ambiguous_qa_path),
            embedding_model=self.embedding_model,
            match_threshold=0.40,
            match_min_gap=0.06,
            synonym_path=str(self.synonym_path),
        )
        manager.sync_from_excel(force=False, preserve_answers=True)

        match = manager.find_best_match("เปลี่ยนตอนเรียน")
        self.assertIsNone(match)

    def test_missing_synonym_file_falls_back_without_exception(self):
        manager = QAPromptManager(
            excel_path=str(self.excel_path),
            json_path=str(self.qa_path),
            embedding_model=self.embedding_model,
            match_threshold=0.40,
            match_min_gap=0.06,
            synonym_path=str(self.tmp_path / "missing_synonyms.json"),
        )
        manager.sync_from_excel(force=False, preserve_answers=True)

        match = manager.find_best_match("ขอข้อมูลการการเปลี่ยนตอนเรียนหน่อย")
        self.assertIsNotNone(match)
        self.assertEqual(match["category"], "การเปลี่ยนตอนเรียน")

    def test_typhoon_prepare_prompt_context_adds_routing_gate_metadata(self):
        kb = FakeKnowledgeBase(model=self.embedding_model, synonym_path=str(self.synonym_path))
        with patch.object(TyphoonChatbot, "setup_typhoon", lambda self: setattr(self, "client", object())):
            bot = TyphoonChatbot(
                api_key="test-key",
                knowledge_base=kb,
                use_compression=False,
                qa_excel_path=str(self.excel_path),
                qa_json_path=str(self.qa_path),
                qa_match_threshold=0.40,
                qa_match_min_gap=0.06,
                synonym_path=str(self.synonym_path),
            )

        context = bot._prepare_prompt_context("เปลี่ยนเซคเรียนได้ไหม")
        routing = bot.get_last_routing_info()

        self.assertIn("QA_MATCH", context)
        self.assertIn("normalized_query", routing)
        self.assertIn("qa_gate_passed", routing)
        self.assertIn("qa_score_gap", routing)
        self.assertTrue(routing["qa_gate_passed"])
        self.assertIn("เปลี่ยนตอนเรียน", routing["normalized_query"])

    def _build_manager(self) -> QAPromptManager:
        manager = QAPromptManager(
            excel_path=str(self.excel_path),
            json_path=str(self.qa_path),
            embedding_model=self.embedding_model,
            match_threshold=0.40,
            match_min_gap=0.06,
            synonym_path=str(self.synonym_path),
        )
        manager.sync_from_excel(force=False, preserve_answers=True)
        return manager

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    unittest.main()
