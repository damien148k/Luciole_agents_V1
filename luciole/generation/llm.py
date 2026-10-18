"""LLM local par Ollama (``POST /api/chat``), injectable.

Chaque appel est reproductible : température et graine du profil de cas
d'usage, fenêtre (``num_ctx``) et modèle du profil matériel. Une réponse
arrêtée par la limite de tokens est marquée ``truncated``.

``extract`` demande une sortie JSON conforme à un schéma pydantic (Ollama
contraint le décodage avec ``format``) puis la **valide** : un JSON invalide
lève une erreur, il n'est jamais deviné ni réparé.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

import httpx
from pydantic import BaseModel, ValidationError


@dataclass
class Completion:
    text: str
    model: str
    truncated: bool = False
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    options: dict = field(default_factory=dict)


class LLM(Protocol):
    model: str

    def generate(
        self,
        system: str,
        prompt: str,
        *,
        options: Mapping[str, object] | None = None,
        schema: dict | None = None,
    ) -> Completion: ...


def ollama_url() -> str:
    return os.environ.get("LUCIOLE_OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")


class OllamaLLM:
    def __init__(
        self,
        model: str,
        *,
        num_ctx: int,
        base_options: Mapping[str, object] | None = None,
        url: str | None = None,
        client: httpx.Client | None = None,
        timeout_s: float = 600.0,
    ) -> None:
        self.model = model
        self.num_ctx = num_ctx
        self.base_options = dict(base_options or {})
        self.url = (url or ollama_url()).rstrip("/")
        self._client = client or httpx.Client(timeout=timeout_s)

    def generate(
        self,
        system: str,
        prompt: str,
        *,
        options: Mapping[str, object] | None = None,
        schema: dict | None = None,
    ) -> Completion:
        merged = {**self.base_options, "num_ctx": self.num_ctx, **(options or {})}
        body: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": merged,
        }
        if schema is not None:
            body["format"] = schema
        try:
            response = self._client.post(f"{self.url}/api/chat", json=body)
        except httpx.HTTPError as e:
            raise RuntimeError(
                f"Ollama injoignable sur {self.url} ({e}) : est-il lancé, et le modèle "
                f"« {self.model} » récupéré (ollama pull {self.model}) ?"
            ) from e
        if response.status_code != 200:
            raise RuntimeError(
                f"Ollama a refusé l'appel ({response.status_code}) : {response.text[:300]}"
            )
        data = response.json()
        return Completion(
            text=(data.get("message") or {}).get("content", ""),
            model=data.get("model", self.model),
            truncated=data.get("done_reason") == "length",
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
            options=merged,
        )


def extract[T: BaseModel](
    llm: LLM,
    system: str,
    prompt: str,
    model: type[T],
    *,
    options: Mapping[str, object] | None = None,
) -> T:
    """Sortie typée : JSON contraint par le schéma de ``model``, puis validé."""
    completion = llm.generate(system, prompt, options=options, schema=model.model_json_schema())
    try:
        return model.model_validate(json.loads(completion.text))
    except (json.JSONDecodeError, ValidationError) as e:
        raise ValueError(f"sortie du LLM non conforme au schéma {model.__name__} : {e}") from e
