"""
Runs the STORM-style 'interview': each persona asks a research agent
a series of questions about the topic, and the agent answers by
searching the web and synthesizing sources. This is the agentic core
of the pipeline — the agent decides when and how to call the search
tool, rather than us hardcoding a fixed retrieve-then-answer sequence.

Each turn's search results and synthesized answer are embedded and
written into the vector store (if one is provided), so article.py can
later retrieve only the chunks relevant to a given section instead of
reading the full transcript every time.
"""
from dataclasses import dataclass, field
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from app.config import settings
from app.retrieval.vector_store import index_text
from app.retrieval.web_search import google_search_tool


@dataclass
class Interview:
    persona: str
    questions: List[str] = field(default_factory=list)
    answers: List[str] = field(default_factory=list)

    def as_transcript(self) -> str:
        turns = []
        for q, a in zip(self.questions, self.answers):
            turns.append(f"Q ({self.persona}): {q}\nA: {a}")
        return "\n\n".join(turns)


def _question_writer() -> ChatOpenAI:
    return ChatOpenAI(model=settings.fast_model, api_key=settings.require_openai())


def _build_research_agent():
    llm = ChatOpenAI(model=settings.strong_model, api_key=settings.require_openai())
    return create_react_agent(llm, [google_search_tool])


def _extract_tool_outputs(messages) -> List[str]:
    """Pull raw search-tool results out of the agent's message trace,
    so we can index what was actually found, not just what the agent
    said about it."""
    outputs = []
    for message in messages:
        if getattr(message, "type", None) == "tool":
            outputs.append(message.content)
    return outputs


def run_interview(
    topic: str,
    persona: str,
    turns: int = 5,
    store: Optional[Chroma] = None,
) -> Interview:
    """One persona's multi-turn conversation with the research agent.
    If `store` is given, search results and answers are embedded and
    indexed as they're produced."""
    question_writer = _question_writer()
    research_agent = _build_research_agent()

    interview = Interview(persona=persona)
    history: List[str] = []

    for turn in range(turns):
        question_prompt = (
            f"You are a Wikipedia writer working on an article about {topic}. "
            f"Ask one question from the perspective of a {persona} to deepen "
            f"understanding of the topic, given this conversation so far: {history}. "
            "Respond with only the question."
        )
        question = question_writer.invoke([{"role": "user", "content": question_prompt}]).content
        interview.questions.append(question)
        history.append(question)

        agent_prompt = (
            f"Answer this question by searching the web and citing at most 3 "
            f"credible sources: {question}"
        )
        response = research_agent.invoke({"messages": [{"role": "user", "content": agent_prompt}]})
        messages = response["messages"]
        answer = messages[-1].content
        interview.answers.append(answer)
        history.append(answer)

        if store is not None:
            base_metadata = {"topic": topic, "persona": persona, "question": question, "turn": turn}
            for i, tool_output in enumerate(_extract_tool_outputs(messages)):
                index_text(store, tool_output, {**base_metadata, "kind": "search_result", "chunk": i})
            index_text(store, answer, {**base_metadata, "kind": "agent_answer"})

    return interview


def run_all_interviews(
    topic: str,
    personas: List[str],
    turns: int = 5,
    store: Optional[Chroma] = None,
) -> List[Interview]:
    return [run_interview(topic, persona, turns=turns, store=store) for persona in personas]
