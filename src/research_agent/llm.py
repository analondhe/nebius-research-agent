"""Model inference behind a small interface: Nebius Token Factory and an offline fake.

Both implementations return an OpenAI-compatible `chat.completion` body
(https://docs.tokenfactory.nebius.com/api-reference/inference/create-chat-completion).
"""
from __future__ import annotations

import json
import re
import time
import uuid
from collections import deque
from typing import Any, Callable, Iterable, Protocol, Union

from openai import OpenAI, OpenAIError

from . import prompts

DEFAULT_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct"

Message = dict[str, str]


class LLMError(Exception):
    """A model call failed or returned no usable text."""


class LLMClient(Protocol):
    model: str

    def complete(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        json_mode: bool = False,
    ) -> dict[str, Any]:
        """Run one chat completion and return the `chat.completion` body."""
        ...


class NebiusLLMClient:
    """Chat completions on Nebius Token Factory through its OpenAI-compatible API."""

    def __init__(
        self,
        api_key: str,
        *,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 120.0,
        max_retries: int = 2,
    ) -> None:
        self.model = model
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=max_retries)

    def complete(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        json_mode: bool = False,
    ) -> dict[str, Any]:
        extra: dict[str, Any] = {"response_format": {"type": "json_object"}} if json_mode else {}
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **extra,
            )
        except OpenAIError as exc:
            raise LLMError(f"Nebius Token Factory call failed: {type(exc).__name__}: {exc}") from exc
        return response.model_dump()


def completion_text(body: dict[str, Any]) -> str:
    """The assistant text of the first choice; raises LLMError when there is none."""
    try:
        choice = body["choices"][0]
        content = choice["message"].get("content")
    except (KeyError, IndexError, TypeError, AttributeError) as exc:
        raise LLMError("malformed chat completion body") from exc
    if not isinstance(content, str) or not content.strip():
        raise LLMError(f"empty completion (finish_reason={choice.get('finish_reason')})")
    return content.strip()


def make_chat_completion(content: str, *, model: str, prompt_tokens: int = 0) -> dict[str, Any]:
    """A `chat.completion` body with every field the Token Factory schema marks required."""
    completion_tokens = len(content.split())
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content, "tool_calls": None},
                "finish_reason": "stop",
                "logprobs": None,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "service_tier": "default",
    }


FakeReply = Union[str, Exception, Callable[[list[Message]], str]]


class FakeLLMClient:
    """Offline stand-in for NebiusLLMClient.

    With `replies`, each call consumes the next one: a string is returned as the
    assistant text, an exception is raised, a callable gets the messages. With
    no `replies`, a rule-based responder plans, refines and writes cited answers
    from the prompts the pipeline sends.
    """

    model = "fake/research-llm"

    def __init__(self, replies: Iterable[FakeReply] | None = None) -> None:
        self._replies = deque(replies) if replies is not None else None
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        json_mode: bool = False,
    ) -> dict[str, Any]:
        self.calls.append({"messages": messages, "max_tokens": max_tokens, "json_mode": json_mode})
        if self._replies is None:
            content = rule_based_reply(messages)
        else:
            if not self._replies:
                raise AssertionError("FakeLLMClient ran out of scripted replies")
            reply = self._replies.popleft()
            if isinstance(reply, Exception):
                raise reply
            content = reply(messages) if callable(reply) else reply
        prompt_tokens = sum(len(m["content"].split()) for m in messages)
        return make_chat_completion(content, model=self.model, prompt_tokens=prompt_tokens)


_QUESTION_LINE = re.compile(r"^Question: (.+)$", re.MULTILINE)
_SOURCE_BLOCK = re.compile(r"<source id=\"(\d+)\"[^>]*>\n(.*?)\n</source>", re.DOTALL)


def rule_based_reply(messages: list[Message]) -> str:
    system = messages[0]["content"]
    match = _QUESTION_LINE.search(messages[1]["content"])
    question = match.group(1) if match else messages[1]["content"]
    if system == prompts.PLANNER_SYSTEM:
        return json.dumps({"queries": [question, f"{question} documentation", f"{question} pricing"]})
    if system == prompts.REFINE_SYSTEM:
        return json.dumps({"queries": [f"{question} changelog", f"{question} migration guide"]})
    if system == prompts.WRITER_SYSTEM:
        claims = []
        for sid, content in _SOURCE_BLOCK.findall(messages[1]["content"])[:3]:
            first_sentence = re.split(r"(?<=[.!?])\s", content.strip(), maxsplit=1)[0].rstrip(".!?")
            claims.append(f"{first_sentence} [{sid}].")
        return " ".join(claims) if claims else prompts.INSUFFICIENT_EVIDENCE
    raise AssertionError("FakeLLMClient got a prompt it does not recognise")
