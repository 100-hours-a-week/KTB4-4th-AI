from datetime import datetime
from typing import Annotated, Literal

from pydantic import Field, HttpUrl

from app.api.schemas.common import ApiModel
from app.api.schemas.profile import TasteField, TasteProfile

RecommendationMode = Literal["self", "gift"]


class FeedbackSummary(ApiModel):
    liked_product_ids: list[str]
    disliked_product_ids: list[str]
    already_have_product_ids: list[str]
    hidden_product_ids: list[str]
    disliked_signals: list[str]


class PriceRange(ApiModel):
    min: int = Field(ge=0)
    max: int = Field(ge=0)
    currency: Literal["KRW"]


class MatchedSignal(ApiModel):
    field: TasteField
    value: str | None
    label: str | None
    taste_node_id: str | None
    via: Literal["vector_content", "vector_usage", "vector_gift", "taxonomy", "feedback"]
    similarity: float | None
    contribution: float


class RankingBreakdown(ApiModel):
    mode: RecommendationMode
    factors: dict[str, float]
    ranker_version: str


class RecommendationFunnel(ApiModel):
    retrieved: int = Field(ge=0)
    after_hard_filter: int = Field(ge=0)
    after_score_floor: int = Field(ge=0)
    returned: int = Field(ge=0)


class RecommendedItem(ApiModel):
    product_id: str
    title: str
    price: int = Field(ge=0)
    image_url: HttpUrl
    product_url: HttpUrl
    category: str
    price_band: int = Field(ge=0)
    rank: int = Field(ge=0)
    score: float
    reason: str
    matched_signals: list[MatchedSignal]
    ranking: RankingBreakdown


class RecommendationResult(ApiModel):
    recommendation_id: str
    generated_at: datetime
    mode: RecommendationMode
    price_range: PriceRange | None
    items: list[RecommendedItem] = Field(max_length=20)
    empty_reason: Literal[
        "NO_PROFILE_SIGNAL",
        "ALL_EXCLUDED",
        "NO_RELEVANT_CANDIDATE",
        "CATALOG_GAP",
    ] | None
    suggestion: str | None = None
    funnel: RecommendationFunnel


class CreateRecommendationJobRequest(ApiModel):
    user_id: str = Field(min_length=1)
    session_id: str | None
    profile: TasteProfile
    feedback_summary: FeedbackSummary | None = None


class CreateRecommendationJobResponse(ApiModel):
    job_id: str
    status: Literal["pending"]
    poll_after_ms: int = Field(ge=0)
    estimated_ms: int | None = Field(default=None, ge=0)


class RecommendationLists(ApiModel):
    self: RecommendationResult
    gift: RecommendationResult


class RecommendationJobError(ApiModel):
    code: Literal["RECOMMENDATION_FAILED"]
    message: str
    retryable: bool


class PendingRecommendationJob(ApiModel):
    job_id: str
    status: Literal["pending"]
    poll_after_ms: int = Field(ge=0)


class RunningRecommendationJob(ApiModel):
    job_id: str
    status: Literal["running"]
    progress: float = Field(ge=0.0, le=1.0)
    poll_after_ms: int = Field(ge=0)


class SucceededRecommendationJob(ApiModel):
    job_id: str
    status: Literal["succeeded"]
    lists: RecommendationLists


class FailedRecommendationJob(ApiModel):
    job_id: str
    status: Literal["failed"]
    error: RecommendationJobError


RecommendationJobResponse = Annotated[
    PendingRecommendationJob
    | RunningRecommendationJob
    | SucceededRecommendationJob
    | FailedRecommendationJob,
    Field(discriminator="status"),
]


class CreateRecommendationRequest(ApiModel):
    user_id: str = Field(min_length=1)
    mode: RecommendationMode
    profile: TasteProfile
    exclude_categories: list[str] = Field(default_factory=list)
    exclude_product_ids: list[str] = Field(default_factory=list)
    feedback_summary: FeedbackSummary | None = None
    limit: int = Field(default=20, ge=1, le=20)


class FeedbackMatchedSignal(ApiModel):
    field: TasteField
    value: str
    taste_node_id: str | None


class CreateRecommendationFeedbackRequest(ApiModel):
    product_id: str
    action: Literal[
        "like",
        "dislike",
        "already_have",
        "hidden",
        "purchased",
        "gift_satisfied",
        "gift_unsatisfied",
    ]
    matched_signals: list[FeedbackMatchedSignal] = Field(default_factory=list)


class ProductFeedbackEffect(ApiModel):
    target: Literal["product"]
    product_id: str
    change: Literal["exclude"]


class ProfileItemFeedbackEffect(ApiModel):
    target: Literal["profile_item"]
    value: str
    change: Literal["move_to_owned"]
    confidence_change: float | None = None


FeedbackEffect = Annotated[
    ProductFeedbackEffect | ProfileItemFeedbackEffect,
    Field(discriminator="target"),
]


class CreateRecommendationFeedbackResponse(ApiModel):
    accepted: bool
    applied_to_ranking: bool
    effects: list[FeedbackEffect]
