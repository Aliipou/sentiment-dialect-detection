"""
Pydantic schemas for API request/response validation
"""
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class TextInput(BaseModel):
    """Single text input for analysis"""
    text: str = Field(..., min_length=1, max_length=10000, description="Text to analyze")
    language: str = Field(default="fa", description="Language code (fa, en, etc.)")

    @field_validator('text')
    @classmethod
    def text_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Text cannot be empty')
        return v.strip()

    @field_validator('language')
    @classmethod
    def language_valid(cls, v):
        valid_languages = ['fa', 'en', 'fr', 'de', 'es', 'it', 'pt', 'nl', 'sv', 'no', 'da', 'fi', 'el', 'ru', 'pl', 'cs', 'hu', 'ro', 'tr']
        if v not in valid_languages:
            raise ValueError(f'Language must be one of: {", ".join(valid_languages)}')
        return v


class BatchTextInput(BaseModel):
    """Batch text input for analysis"""
    texts: List[str] = Field(..., min_length=1, max_length=100, description="List of texts to analyze")
    language: str = Field(default="fa", description="Language code")

    @field_validator('texts')
    @classmethod
    def texts_not_empty(cls, v):
        if not v:
            raise ValueError('Texts list cannot be empty')
        cleaned = [t.strip() for t in v if t and t.strip()]
        if not cleaned:
            raise ValueError('All texts are empty')
        return cleaned


class SentimentResponse(BaseModel):
    """Sentiment analysis response"""
    text: str
    sentiment: str
    sentiment_score: float = Field(..., ge=0.0, le=1.0)
    processing_time: float


class DialectResponse(BaseModel):
    """Dialect detection response"""
    text: str
    dialect: str
    dialect_score: float = Field(..., ge=0.0, le=1.0)
    processing_time: float


class CombinedResponse(BaseModel):
    """Combined sentiment and dialect analysis response"""
    text: str
    sentiment: str
    sentiment_score: float = Field(..., ge=0.0, le=1.0)
    dialect: Optional[str] = None
    dialect_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    processing_time: float


class BatchResponse(BaseModel):
    """Batch analysis response"""
    results: List[CombinedResponse]
    total_texts: int
    total_processing_time: float


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    models_loaded: dict


class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    detail: Optional[str] = None
    status_code: int
