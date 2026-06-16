from ragicamp.utils.prompts import FewShotExample, PromptBuilder, PromptConfig


def test_build_direct_includes_query_and_answer_marker():
    builder = PromptBuilder(PromptConfig())
    prompt = builder.build_direct("What is the capital of France?")

    assert "Question: What is the capital of France?" in prompt
    assert "Answer:" in prompt


def test_build_direct_with_system_instruction():
    config = PromptConfig(system_instruction="You are a helpful geography tutor.")
    builder = PromptBuilder(config)
    prompt = builder.build_direct("What is Paris?")

    assert "You are a helpful geography tutor." in prompt
    assert "Question: What is Paris?" in prompt


def test_build_rag_default_includes_context_and_query():
    builder = PromptBuilder(PromptConfig())
    context = "Paris is the capital of France. It has 2 million residents."
    prompt = builder.build_rag("What is Paris?", context)

    assert "Retrieved Passages:" in prompt
    assert context in prompt
    assert "Question: What is Paris?" in prompt
    assert "Answer:" in prompt


def test_build_rag_structured_has_delimited_sections():
    config = PromptConfig(use_delimiters=True)
    builder = PromptBuilder(config)
    context = "The Eiffel Tower is in Paris."
    prompt = builder.build_rag("Where is the Eiffel Tower?", context)

    assert "### Context" in prompt
    assert "### Question" in prompt
    assert "### Answer" in prompt
    assert context in prompt
    assert "Where is the Eiffel Tower?" in prompt


def test_build_rag_extractive_has_strict_extraction_language():
    config = PromptConfig(strict_extraction=True)
    builder = PromptBuilder(config)
    context = "Mount Everest is the tallest mountain."
    prompt = builder.build_rag("What is the tallest mountain?", context)

    assert "MUST be extracted" in prompt
    assert "Do NOT use any knowledge outside" in prompt
    assert context in prompt


def test_build_rag_cot_has_chain_of_thought_language():
    config = PromptConfig(include_reasoning=True)
    builder = PromptBuilder(config)
    context = "Albert Einstein was born in 1879."
    prompt = builder.build_rag("When was Einstein born?", context)

    assert "step by step" in prompt
    assert "Reasoning:" in prompt
    assert context in prompt


def test_build_rag_cited_has_citation_instructions():
    config = PromptConfig(require_citation=True)
    builder = PromptBuilder(config)
    context = "Passage 1: Water boils at 100C. Passage 2: Ice melts at 0C."
    prompt = builder.build_rag("What temperature does water boil?", context)

    assert "[1], [2]" in prompt
    assert "citation" in prompt.lower()
    assert context in prompt


def test_build_direct_with_fewshot_examples_includes_qa_format():
    examples = [
        FewShotExample(question="What is 2+2?", answer="4"),
        FewShotExample(question="What is the sky color?", answer="blue"),
    ]
    config = PromptConfig(examples=examples)
    builder = PromptBuilder(config)
    prompt = builder.build_direct("What is 3+3?")

    assert "Q: What is 2+2?" in prompt
    assert "A: 4" in prompt
    assert "Q: What is the sky color?" in prompt
    assert "A: blue" in prompt
    assert "Question: What is 3+3?" in prompt


def test_from_config_default_returns_working_builder():
    builder = PromptBuilder.from_config("default")

    assert isinstance(builder, PromptBuilder)
    assert builder.config.style == "Give only the answer, no explanations."
    assert "passages" in builder.config.knowledge_instruction.lower()

    prompt = builder.build_direct("Test question")
    assert "Question: Test question" in prompt


def test_from_config_unknown_type_returns_builder_with_empty_config():
    builder = PromptBuilder.from_config("unknown_type_xyz")

    assert isinstance(builder, PromptBuilder)
    assert builder.config.style == ""
    assert builder.config.knowledge_instruction == ""
    assert builder.config.use_delimiters is False
    assert builder.config.strict_extraction is False

    prompt = builder.build_direct("Test")
    assert "Question: Test" in prompt


def test_from_config_concise_returns_minimal_style():
    builder = PromptBuilder.from_config("concise")

    assert builder.config.style == "Reply with just the answer."
    prompt = builder.build_rag("What is AI?", "AI is artificial intelligence.")
    assert "Retrieved Passages:" in prompt


def test_from_config_structured_sets_delimiters_flag():
    builder = PromptBuilder.from_config("structured")

    assert builder.config.use_delimiters is True
    prompt = builder.build_rag("Test query", "Test context")
    assert "### Context" in prompt


def test_from_config_extractive_sets_strict_extraction_flag():
    builder = PromptBuilder.from_config("extractive")

    assert builder.config.strict_extraction is True
    prompt = builder.build_rag("Test query", "Test context")
    assert "exact answer" in prompt.lower()


def test_from_config_cot_sets_reasoning_flag():
    builder = PromptBuilder.from_config("cot")

    assert builder.config.include_reasoning is True
    prompt = builder.build_rag("Test query", "Test context")
    assert "step by step" in prompt.lower()


def test_from_config_cited_sets_citation_flag():
    builder = PromptBuilder.from_config("cited")

    assert builder.config.require_citation is True
    prompt = builder.build_rag("Test query", "Test context")
    assert "[1]" in prompt


def test_format_examples_returns_q_a_format():
    examples = [
        FewShotExample(question="First question", answer="First answer"),
        FewShotExample(question="Second question", answer="Second answer"),
    ]
    config = PromptConfig(examples=examples)
    builder = PromptBuilder(config)

    formatted = builder._format_examples()

    assert "Examples:" in formatted
    assert "Q: First question" in formatted
    assert "A: First answer" in formatted
    assert "Q: Second question" in formatted
    assert "A: Second answer" in formatted


def test_build_rag_default_with_knowledge_instruction():
    config = PromptConfig(knowledge_instruction="Focus on dates and locations.")
    builder = PromptBuilder(config)
    prompt = builder.build_rag("When?", "Historical context here.")

    assert "Focus on dates and locations." in prompt
    assert "Retrieved Passages:" in prompt


def test_build_direct_default_instruction_when_no_system_instruction():
    builder = PromptBuilder(PromptConfig())
    prompt = builder.build_direct("What is X?")

    assert "Answer the question using your knowledge." in prompt


def test_build_direct_includes_style_and_stop_instruction():
    config = PromptConfig(style="Be concise.", stop_instruction="Do not add explanations.")
    builder = PromptBuilder(config)
    prompt = builder.build_direct("What is Y?")

    assert "Be concise." in prompt
    assert "Do not add explanations." in prompt
    assert "Answer:" in prompt
