from pathlib import Path


class LazyPromptTemplate:
    """String-like prompt template that reads the prompt file only when used."""

    def __init__(self, filename: str = "prompt.txt") -> None:
        self.filename = filename
        self._template: str | None = None

    def _load(self) -> str:
        if self._template is None:
            self._template = load_prompt_template(self.filename)
        return self._template

    def format(self, *args, **kwargs) -> str:
        return self._load().format(*args, **kwargs)

    def __str__(self) -> str:
        return self._load()

    def __repr__(self) -> str:
        return repr(self._load())

    def __eq__(self, other) -> bool:
        return self._load() == other


def load_prompt_template(filename: str = "prompt.txt") -> str:
    """Load the CoT prompt template from the repository root."""
    prompt_path = Path(__file__).resolve().parent.parent / filename
    return prompt_path.read_text(encoding="utf-8").rstrip("\n")


DIRECT_PROMPT_TEMPLATE = """Please answer with ONLY the answer enclosed in <answer> </answer> tags.
Question: {question}
"""

COT_PROMPT_TEMPLATE = LazyPromptTemplate()
