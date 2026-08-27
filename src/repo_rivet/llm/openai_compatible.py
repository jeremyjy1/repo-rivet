"""OpenAI-compatible Chat Completions model adapter."""

from typing import Any, cast

from openai import OpenAI

from repo_rivet.config import ApiConfig
from repo_rivet.llm.base import ModelResponse
from repo_rivet.llm.parser import ResponseParser


class OpenAICompatibleClient:
    """Call a configured Chat Completions endpoint using native tool calling."""

    def __init__(
        self,
        config: ApiConfig,
        *,
        client: Any | None = None,
        parser: ResponseParser | None = None,
    ) -> None:
        self._config = config
        self._client = client or OpenAI(
            api_key=config.api_key.get_secret_value(),
            base_url=str(config.base_url),
            timeout=config.timeout_seconds,
            max_retries=config.max_retries,
        )
        self._parser = parser or ResponseParser()

    def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        response = self._client.chat.completions.create(
            model=self._config.model,
            messages=cast(Any, messages),
            tools=cast(Any, tools),
            tool_choice="auto",
        )
        return self._parser.parse(response)
