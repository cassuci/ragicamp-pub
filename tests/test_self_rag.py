"""Tests for SelfRAGAgent.

Tests the adaptive retrieval agent with three-phase batched execution:
1. Phase 1 (Assess): Batch assess all queries for retrieval confidence
2. Phase 2 (Retrieve): Embed+search only for queries needing retrieval
3. Phase 3 (Generate): Batch generate all answers, optionally verify, fallback

Key behaviors tested:
- Confidence parsing from LLM responses
- Verification verdict parsing
- High-confidence queries skip retrieval (direct generation)
- Low-confidence queries use RAG
- NOT_SUPPORTED answers trigger fallback to direct generation
- Result metadata includes confidence, used_retrieval, num_docs, verification
"""

import pytest

from ragicamp.agents.base import Query
from ragicamp.agents.self_rag import SelfRAGAgent
from tests.shared_mocks import FakeEmbedder, FakeGenerator, FakeIndex, FakeProvider


class TestConfidenceParsing:
    def test_parse_confidence_valid(self):
        response = "I need to think about this. CONFIDENCE: 0.7\nReasoning: This is clear."
        confidence = SelfRAGAgent._parse_confidence(response)
        assert confidence == 0.7

    def test_parse_confidence_clamped_high(self):
        response = "CONFIDENCE: 1.5"
        confidence = SelfRAGAgent._parse_confidence(response)
        assert confidence == 1.0

    def test_parse_confidence_clamped_low(self):
        response = "CONFIDENCE: -0.3"
        confidence = SelfRAGAgent._parse_confidence(response)
        assert confidence == 0.3

    def test_parse_confidence_missing(self):
        response = "No score here, just text"
        confidence = SelfRAGAgent._parse_confidence(response)
        assert confidence == 0.3

    def test_parse_confidence_case_insensitive(self):
        response = "confidence: 0.8 — I'm pretty sure"
        confidence = SelfRAGAgent._parse_confidence(response)
        assert confidence == 0.8

    def test_parse_confidence_decimal_variations(self):
        assert SelfRAGAgent._parse_confidence("CONFIDENCE: 0.5") == 0.5
        assert SelfRAGAgent._parse_confidence("CONFIDENCE: .9") == 0.9
        assert SelfRAGAgent._parse_confidence("CONFIDENCE: 1") == 1.0
        assert SelfRAGAgent._parse_confidence("CONFIDENCE: 0") == 0.0


class TestVerificationParsing:
    def test_parse_verification_supported(self):
        response = "The answer is SUPPORTED by the context provided."
        verdict = SelfRAGAgent._parse_verification(response)
        assert verdict == "SUPPORTED"

    def test_parse_verification_not_supported(self):
        response = "This is NOT_SUPPORTED because the context contradicts it."
        verdict = SelfRAGAgent._parse_verification(response)
        assert verdict == "NOT_SUPPORTED"

    def test_parse_verification_partial(self):
        response = "The answer is PARTIALLY_SUPPORTED — some info is correct."
        verdict = SelfRAGAgent._parse_verification(response)
        assert verdict == "PARTIALLY_SUPPORTED"

    def test_parse_verification_unknown(self):
        response = "I don't know, unclear from the context."
        verdict = SelfRAGAgent._parse_verification(response)
        assert verdict == "UNKNOWN"

    def test_parse_verification_case_insensitive(self):
        response = "This answer is supported by evidence."
        verdict = SelfRAGAgent._parse_verification(response)
        assert verdict == "SUPPORTED"

    def test_parse_verification_priority_order(self):
        response = "NOT_SUPPORTED but PARTIALLY_SUPPORTED is also mentioned"
        verdict = SelfRAGAgent._parse_verification(response)
        assert verdict == "NOT_SUPPORTED"


class TestAllQueriesNeedRetrieval:
    def test_all_queries_need_retrieval(self):
        gen = FakeGenerator(
            responses={
                "Analyze this question and decide": "CONFIDENCE: 0.2\nDefinitely need retrieval.",
                "Answer": "generated answer with context",
            }
        )
        agent = SelfRAGAgent(
            name="test",
            embedder_provider=FakeProvider(FakeEmbedder()),
            generator_provider=FakeProvider(gen),
            index=FakeIndex(),
            top_k=3,
            retrieval_threshold=0.5,
        )

        queries = [
            Query(idx=0, text="What is the capital?"),
            Query(idx=1, text="Who invented the lightbulb?"),
        ]
        results = agent.run(queries)

        assert len(results) == 2
        for result in results:
            assert result.metadata["used_retrieval"] is True
            assert result.metadata["confidence"] == 0.2
            assert result.metadata["num_docs"] == 3

    def test_low_confidence_triggers_retrieval(self):
        gen = FakeGenerator(
            responses={
                "Analyze this question": "CONFIDENCE: 0.1\nNeed external info.",
            }
        )
        agent = SelfRAGAgent(
            name="test",
            embedder_provider=FakeProvider(FakeEmbedder()),
            generator_provider=FakeProvider(gen),
            index=FakeIndex(),
            top_k=2,
            retrieval_threshold=0.5,
        )

        queries = [Query(idx=0, text="Obscure factual question?")]
        results = agent.run(queries)

        assert results[0].metadata["used_retrieval"] is True
        assert results[0].metadata["confidence"] == 0.1


class TestHighConfidenceSkipsRetrieval:
    def test_high_confidence_skips_retrieval(self):
        gen = FakeGenerator(
            responses={
                "Analyze this question": "CONFIDENCE: 0.9\nI know this already.",
                "Answer": "direct answer without context",
            }
        )
        agent = SelfRAGAgent(
            name="test",
            embedder_provider=FakeProvider(FakeEmbedder()),
            generator_provider=FakeProvider(gen),
            index=FakeIndex(),
            top_k=3,
            retrieval_threshold=0.5,
        )

        queries = [Query(idx=0, text="What is 2+2?")]
        results = agent.run(queries)

        assert len(results) == 1
        assert results[0].metadata["used_retrieval"] is False
        assert results[0].metadata["confidence"] == 0.9
        assert results[0].metadata["num_docs"] == 0

    def test_confidence_equal_threshold_uses_retrieval(self):
        gen = FakeGenerator(responses={"Analyze this question": "CONFIDENCE: 0.5\nOn the fence."})
        agent = SelfRAGAgent(
            name="test",
            embedder_provider=FakeProvider(FakeEmbedder()),
            generator_provider=FakeProvider(gen),
            index=FakeIndex(),
            top_k=3,
            retrieval_threshold=0.5,
        )

        queries = [Query(idx=0, text="Edge case question?")]
        results = agent.run(queries)

        assert results[0].metadata["used_retrieval"] is True


class TestMixedRetrievalAndDirect:
    def test_mixed_retrieval_and_direct(self):
        class StatefulGenerator:
            def __init__(self):
                self.assess_call = 0

            def batch_generate(self, prompts, **kwargs):
                results = []
                for p in prompts:
                    if "Analyze this question" in p:
                        if self.assess_call == 0:
                            results.append("CONFIDENCE: 0.2\nNeed retrieval.")
                        elif self.assess_call == 1:
                            results.append("CONFIDENCE: 0.95\nI know this.")
                        elif self.assess_call == 2:
                            results.append("CONFIDENCE: 0.3\nUnsure.")
                        self.assess_call += 1
                    else:
                        results.append("answer")
                return results

        agent = SelfRAGAgent(
            name="test",
            embedder_provider=FakeProvider(FakeEmbedder()),
            generator_provider=FakeProvider(StatefulGenerator()),
            index=FakeIndex(),
            top_k=2,
            retrieval_threshold=0.5,
        )

        queries = [
            Query(idx=0, text="Q1 needs retrieval"),
            Query(idx=1, text="Q2 is direct"),
            Query(idx=2, text="Q3 needs retrieval"),
        ]
        results = agent.run(queries)

        assert len(results) == 3
        assert results[0].metadata["used_retrieval"] is True
        assert results[0].metadata["confidence"] == 0.2
        assert results[1].metadata["used_retrieval"] is False
        assert results[1].metadata["confidence"] == 0.95
        assert results[2].metadata["used_retrieval"] is True
        assert results[2].metadata["confidence"] == 0.3


class TestVerificationFallback:
    def test_verification_fallback_on_not_supported(self):
        gen = FakeGenerator(
            responses={
                "Analyze this question": "CONFIDENCE: 0.2\nNeed retrieval.",
                "Verify if the following answer": "NOT_SUPPORTED\nThis contradicts context.",
                "Question:": "fallback direct answer",
            }
        )
        agent = SelfRAGAgent(
            name="test",
            embedder_provider=FakeProvider(FakeEmbedder()),
            generator_provider=FakeProvider(gen),
            index=FakeIndex(),
            top_k=3,
            retrieval_threshold=0.5,
            verify_answer=True,
            fallback_to_direct=True,
        )

        queries = [Query(idx=0, text="Test question?")]
        results = agent.run(queries)

        assert len(results) == 1
        assert results[0].metadata["verification"] == "NOT_SUPPORTED"
        assert results[0].answer == "fallback direct answer"
        assert results[0].metadata["num_docs"] == 0

    def test_verification_no_fallback_if_supported(self):
        gen = FakeGenerator(
            responses={
                "Analyze this question": "CONFIDENCE: 0.1\nNeed retrieval.",
                "Verify if the following answer": "SUPPORTED\nCorrect based on context.",
                "Answer": "rag answer",
            }
        )
        agent = SelfRAGAgent(
            name="test",
            embedder_provider=FakeProvider(FakeEmbedder()),
            generator_provider=FakeProvider(gen),
            index=FakeIndex(),
            top_k=3,
            retrieval_threshold=0.5,
            verify_answer=True,
            fallback_to_direct=True,
        )

        queries = [Query(idx=0, text="Test question?")]
        results = agent.run(queries)

        assert results[0].metadata["verification"] == "SUPPORTED"
        assert results[0].answer == "rag answer"
        assert results[0].metadata["num_docs"] == 3

    def test_verification_partial_no_fallback(self):
        gen = FakeGenerator(
            responses={
                "Analyze this question": "CONFIDENCE: 0.2\nNeed info.",
                "Verify if the following answer": "PARTIALLY_SUPPORTED\nSome correct.",
                "Answer": "rag answer",
            }
        )
        agent = SelfRAGAgent(
            name="test",
            embedder_provider=FakeProvider(FakeEmbedder()),
            generator_provider=FakeProvider(gen),
            index=FakeIndex(),
            top_k=2,
            retrieval_threshold=0.5,
            verify_answer=True,
            fallback_to_direct=True,
        )

        queries = [Query(idx=0, text="Test question?")]
        results = agent.run(queries)

        assert results[0].metadata["verification"] == "PARTIALLY_SUPPORTED"
        assert results[0].answer == "rag answer"
        assert results[0].metadata["num_docs"] == 2

    def test_verification_disabled(self):
        gen = FakeGenerator(
            responses={
                "Analyze this question": "CONFIDENCE: 0.2\nNeed retrieval.",
                "Answer": "rag answer",
            }
        )
        agent = SelfRAGAgent(
            name="test",
            embedder_provider=FakeProvider(FakeEmbedder()),
            generator_provider=FakeProvider(gen),
            index=FakeIndex(),
            top_k=3,
            retrieval_threshold=0.5,
            verify_answer=False,
        )

        queries = [Query(idx=0, text="Test question?")]
        results = agent.run(queries)

        assert results[0].metadata["verification"] is None
        assert results[0].answer == "rag answer"


class TestResultMetadata:
    def test_result_metadata_includes_confidence(self):
        gen = FakeGenerator(
            responses={
                "Analyze this question": "CONFIDENCE: 0.7\nUncertain.",
            }
        )
        agent = SelfRAGAgent(
            name="test",
            embedder_provider=FakeProvider(FakeEmbedder()),
            generator_provider=FakeProvider(gen),
            index=FakeIndex(),
            top_k=5,
            retrieval_threshold=0.5,
        )

        queries = [Query(idx=0, text="Question?")]
        results = agent.run(queries)

        metadata = results[0].metadata
        assert "confidence" in metadata
        assert "used_retrieval" in metadata
        assert "num_docs" in metadata
        assert "verification" in metadata

        assert metadata["confidence"] == 0.7
        assert metadata["used_retrieval"] is False
        assert metadata["num_docs"] == 0
        assert metadata["verification"] is None

    def test_metadata_for_retrieval_query(self):
        gen = FakeGenerator(
            responses={
                "Analyze this question": "CONFIDENCE: 0.25\nNeed docs.",
            }
        )
        agent = SelfRAGAgent(
            name="test",
            embedder_provider=FakeProvider(FakeEmbedder()),
            generator_provider=FakeProvider(gen),
            index=FakeIndex(),
            top_k=4,
            retrieval_threshold=0.5,
        )

        queries = [Query(idx=0, text="Complex question?")]
        results = agent.run(queries)

        metadata = results[0].metadata
        assert metadata["confidence"] == 0.25
        assert metadata["used_retrieval"] is True
        assert metadata["num_docs"] == 4


class TestMultipleFallbacks:
    def test_multiple_queries_fallback(self):
        class ConditionalGenerator:
            def __init__(self):
                self.batch_call = 0

            def batch_generate(self, prompts, **kwargs):
                self.batch_call += 1
                results = []
                for p in prompts:
                    if "Analyze this question" in p:
                        results.append("CONFIDENCE: 0.1\nNeed retrieval.")
                    elif "Verify if the following answer" in p:
                        if "Q1" in p:
                            results.append("NOT_SUPPORTED\nWrong.")
                        else:
                            results.append("SUPPORTED\nCorrect.")
                    elif self.batch_call == 4 and "Context:" not in p:
                        results.append("fallback answer")
                    else:
                        results.append("rag answer")
                return results

        agent = SelfRAGAgent(
            name="test",
            embedder_provider=FakeProvider(FakeEmbedder()),
            generator_provider=FakeProvider(ConditionalGenerator()),
            index=FakeIndex(),
            top_k=2,
            retrieval_threshold=0.5,
            verify_answer=True,
            fallback_to_direct=True,
        )

        queries = [
            Query(idx=0, text="Q1 fallback?"),
            Query(idx=1, text="Q2 no fallback?"),
        ]
        results = agent.run(queries)

        assert results[0].metadata["verification"] == "NOT_SUPPORTED"
        assert results[0].answer == "fallback answer"
        assert results[0].metadata["num_docs"] == 0

        assert results[1].metadata["verification"] == "SUPPORTED"
        assert results[1].answer == "rag answer"
        assert results[1].metadata["num_docs"] == 2


class TestEdgeCases:
    def test_empty_queries_list(self):
        agent = SelfRAGAgent(
            name="test",
            embedder_provider=FakeProvider(FakeEmbedder()),
            generator_provider=FakeProvider(FakeGenerator()),
            index=FakeIndex(),
            top_k=3,
            retrieval_threshold=0.5,
        )

        results = agent.run([])
        assert len(results) == 0

    def test_single_query(self):
        gen = FakeGenerator(responses={"Analyze this question": "CONFIDENCE: 0.5\nOn edge."})
        agent = SelfRAGAgent(
            name="test",
            embedder_provider=FakeProvider(FakeEmbedder()),
            generator_provider=FakeProvider(gen),
            index=FakeIndex(),
            top_k=1,
            retrieval_threshold=0.5,
        )

        queries = [Query(idx=0, text="Single query")]
        results = agent.run(queries)

        assert len(results) == 1
        assert results[0].metadata["confidence"] == 0.5

    def test_custom_retrieval_threshold(self):
        gen = FakeGenerator(
            responses={"Analyze this question": "CONFIDENCE: 0.6\nFairly confident."}
        )
        agent = SelfRAGAgent(
            name="test",
            embedder_provider=FakeProvider(FakeEmbedder()),
            generator_provider=FakeProvider(gen),
            index=FakeIndex(),
            top_k=3,
            retrieval_threshold=0.7,
        )

        queries = [Query(idx=0, text="Test")]
        results = agent.run(queries)

        assert results[0].metadata["used_retrieval"] is True
        assert results[0].metadata["confidence"] == 0.6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
