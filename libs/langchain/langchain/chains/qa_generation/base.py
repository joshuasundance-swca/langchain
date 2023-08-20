from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from pydantic import Field

from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chains.qa_generation.prompt import PARSER, PROMPT_SELECTOR
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import BasePromptTemplate, OutputParserException
from langchain.schema.language_model import BaseLanguageModel
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter


class QAGenerationChain(Chain):
    """Base class for question-answer generation chains."""

    llm_chain: LLMChain
    """LLM Chain that generates responses from user input and context."""
    text_splitter: TextSplitter = Field(
        default=RecursiveCharacterTextSplitter(chunk_overlap=500)
    )
    """Text splitter that splits the input into chunks."""
    input_key: str = "text"
    """Key of the input to the chain."""
    output_key: str = "questions"
    """Key of the output of the chain."""
    k: int = 1
    """Number of questions to generate."""

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        prompt: Optional[BasePromptTemplate] = None,
        output_parser: Optional[PydanticOutputParser] = None,
        **kwargs: Any,
    ) -> QAGenerationChain:
        """
        Create a QAGenerationChain from a language model.

        Args:
            llm: a language model
            prompt: a prompt template
            output_parser: an output parser
            **kwargs: additional arguments

        Returns:
            a QAGenerationChain class
        """
        _prompt = prompt or PROMPT_SELECTOR.get_prompt(llm)
        _output_parser = output_parser or PARSER
        chain = LLMChain(llm=llm, prompt=_prompt, output_parser=_output_parser)
        return cls(llm_chain=chain, **kwargs)

    @property
    def _chain_type(self) -> str:
        raise NotImplementedError

    @property
    def input_keys(self) -> List[str]:
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, List]:
        docs = self.text_splitter.create_documents([inputs[self.input_key]])
        results = self.llm_chain.generate(
            [{"text": d.page_content, "k": self.k} for d in docs],
            run_manager=run_manager,
        )

        def _gen() -> Iterable[str]:
            for res in results.generations:
                text = res[0].text
                try:
                    yield self.llm_chain.output_parser.parse(text)  # type: ignore
                except OutputParserException as e:
                    raise e

        qa = list(_gen())
        return {self.output_key: qa}
