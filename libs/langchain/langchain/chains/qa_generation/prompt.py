# flake8: noqa

from __future__ import annotations

from typing import List

from pydantic import BaseModel, validator, Field

from langchain.chains.prompt_selector import ConditionalPromptSelector, is_chat_model
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.prompts.prompt import PromptTemplate


class QuestionAnswerPair(BaseModel):
    question: str = Field(..., description="The question that will be answered.")
    answer: str = Field(..., description="The answer to the question that was asked.")

    @validator("question")
    def validate_question(cls, v: str) -> str:
        if not v.endswith("?"):
            raise ValueError("Question must end with a question mark.")
        return v


class QuestionAnswerPairList(BaseModel):
    QuestionAnswerPairs: List[QuestionAnswerPair]


PARSER: PydanticOutputParser = PydanticOutputParser(
    pydantic_object=QuestionAnswerPairList
)


templ1 = """You are a smart assistant designed to help high school teachers come up with reading comprehension questions.
Given a piece of text, you must come up with {k} question and answer pairs that can be used to test a student's reading comprehension abilities.
When coming up with the question/answer pairs, you must respond in the following format:
{format_instructions}

Do not provide additional commentary and do not wrap your response in Markdown formatting. Return RAW, VALID JSON.
"""
templ2 = """Please create {k} question/answer pairs, in the specified JSON format, for the following text:
----------------
{text}"""
CHAT_PROMPT_t = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            templ1,
            partial_variables={"format_instructions": PARSER.get_format_instructions()},
        ),
        HumanMessagePromptTemplate.from_template(templ2),
    ]
)
CHAT_PROMPT = CHAT_PROMPT_t.partial(
    format_instructions=PARSER.get_format_instructions()
)

templ = """You are a smart assistant designed to help high school teachers come up with reading comprehension questions.
Given a piece of text, you must come up with {k} question and answer pairs that can be used to test a student's reading comprehension abilities.
When coming up with the question/answer pairs, you must respond in the following format:
{format_instructions}

Do not provide additional commentary and do not wrap your response in Markdown formatting. Return RAW, VALID JSON.

Please create {k} question/answer pairs, in the specified JSON format, for the following text:
----------------
{text}"""
PROMPT_t = PromptTemplate.from_template(
    templ, partial_variables={"format_instructions": PARSER.get_format_instructions()}
)
PROMPT = PROMPT_t.partial(format_instructions=PARSER.get_format_instructions())
PROMPT_SELECTOR = ConditionalPromptSelector(
    default_prompt=PROMPT, conditionals=[(is_chat_model, CHAT_PROMPT)]
)
