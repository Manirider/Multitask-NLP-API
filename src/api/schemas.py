from pydantic import BaseModel, Field
from typing import List

class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1, json_schema_extra={
                      "example": "The movie was absolutely fantastic!"})

class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    score: float

class NERRequest(BaseModel):
    text: str = Field(..., min_length=1, json_schema_extra={
                      "example": "Apple is looking at buying U.K. startup for $1 billion."})

class Entity(BaseModel):
    text: str
    type: str
    score: float
    start_char: int
    end_char: int

class NERResponse(BaseModel):
    text: str
    entities: List[Entity]

class QARequest(BaseModel):
    context: str = Field(..., min_length=1, json_schema_extra={
                         "example": "The capital of France is Paris."})
    question: str = Field(..., min_length=1, json_schema_extra={
                          "example": "What is the capital of France?"})

class QAResponse(BaseModel):
    answer: str
    start_char: int
    end_char: int
    score: float

class HealthResponse(BaseModel):
    status: str
