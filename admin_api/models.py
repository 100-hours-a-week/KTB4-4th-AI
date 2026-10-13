from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, model_validator

from app.api.schemas.chat import ChatMessageRequest
from app.domain.conversation.policy import MAX_TURNS


class Connection(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    provider: Literal["lmstudio", "ollama", "openai", "huggingface"] = "lmstudio"
    environment: Literal["local", "runpod", "other"] = "local"
    base_url: str = Field(min_length=8, max_length=500)
    model: str = Field(default="", max_length=200)
    revision: str = Field(default="", max_length=200)
    precision: str = Field(default="", max_length=100)
    hardware: str = Field(default="", max_length=200)
    api_key_env: str = Field(default="", pattern=r"^$|^[A-Za-z_][A-Za-z0-9_]*$")

    @model_validator(mode="after")
    def validate_url(self):
        parts = urlsplit(self.base_url)
        if (
            parts.scheme not in {"http", "https"}
            or not parts.hostname
            or parts.username
            or parts.password
            or parts.query
            or parts.fragment
        ):
            raise ValueError("인증정보·쿼리가 없는 http(s) API URL을 입력하세요.")
        self.base_url = self.base_url.rstrip("/")
        if self.provider == "huggingface" and self.base_url != "https://router.huggingface.co/v1":
            raise ValueError(
                "Hugging Face Inference Providers 주소는 https://router.huggingface.co/v1입니다. "
                "전용 엔드포인트는 OpenAI 호환 서버로 등록하세요."
            )
        if self.provider == "huggingface":
            self.api_key_env = self.api_key_env or "HF_TOKEN"
            self.environment = "other"
        return self


class RunSpec(BaseModel):
    connection_id: str = "lmstudio"
    mode: Literal["quality", "load", "needu"] = "needu"
    prompt: str = Field(default="", max_length=12000)
    system: str = Field(
        default="너는 니쥬의 대화 도우미다. 한국어로 자연스럽고 간결하게 답하라.", max_length=12000
    )
    temperature: float = Field(default=0.2, ge=0, le=2)
    max_tokens: int = Field(default=256, ge=1, le=512)
    requests: int = Field(default=10, ge=1, le=100)
    concurrency: int = Field(default=1, ge=1, le=8)
    duration_seconds: int = Field(default=120, ge=5, le=1800)
    timeout_seconds: int = Field(default=60, ge=1, le=120)
    note: str = Field(default="", max_length=1000)
    response_connection_id: str | None = None
    extraction_connection_id: str | None = None
    judgment_mode: Literal["environment", "disabled"] = "environment"
    recommendation_stage: Literal["queries", "embedding", "recommendation"] = "queries"
    persona_id: str | None = None
    dialogue_mode: Literal["fixed", "simulated"] = "fixed"
    max_turns: int = Field(default=10, ge=1, le=MAX_TURNS)

    @model_validator(mode="after")
    def check_prompt(self):
        if self.mode != "needu" and self.duration_seconds > 300:
            raise ValueError("단일 모델 테스트는 최대 300초까지 실행할 수 있습니다.")
        if self.mode != "needu" and self.dialogue_mode != "fixed":
            raise ValueError("페르소나 자동 대화는 니쥬 서비스 테스트에서만 지원합니다.")
        if self.mode == "needu" and self.dialogue_mode == "simulated":
            if not self.persona_id:
                raise ValueError("자동 대화할 페르소나를 선택하세요.")
            self.requests = self.max_turns + 2
            self.concurrency = 1
            return self
        if self.mode != "quality" and not self.prompt.strip():
            raise ValueError("테스트할 입력을 작성하세요.")
        if self.mode == "needu":
            lines = [line.strip() for line in self.prompt.splitlines() if line.strip()]
            if not 1 <= len(lines) <= MAX_TURNS:
                raise ValueError(
                    f"니쥬 대화는 한 줄에 한 발화씩, 최대 {MAX_TURNS}턴으로 입력하세요."
                )
            for line in lines:
                ChatMessageRequest(user_id=1, message=line)
            self.requests = len(lines) + 2  # Opening, user turns, final analysis/query stage.
            self.concurrency = 1
        return self


class ChatSpec(BaseModel):
    title: str = Field(default="새 테스트 대화", min_length=1, max_length=80)
    response_connection_id: str | None = None
    extraction_connection_id: str | None = None
    judgment_mode: Literal["environment", "disabled"] = "environment"
    persona_id: str | None = None
