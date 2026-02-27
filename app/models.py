from __future__ import annotations

from typing import List
from pydantic import BaseModel, Field


class JobBrochureData(BaseModel):
    company_name: str = Field(description="Official company name")
    job_title: str = Field(description="Internship role title")
    core_requirements: List[str] = Field(description="5-10 core requirements as short bullet phrases")
    deadline: str = Field(description="Application deadline in ISO format YYYY-MM-DD when possible; otherwise raw text")


class ResearchSummary(BaseModel):
    strategic_points: List[str] = Field(description="Exactly 3 concise strategic bullets from web context")


class TailoredCV(BaseModel):
    tailored_cv_markdown: str = Field(description="Rewritten and filtered CV content in Markdown")
    highlighted_match_reasons: List[str] = Field(description="Top 3 reasons this CV now matches the role")


class CoverLetterDraft(BaseModel):
    cover_letter: str = Field(description="Final cover letter under 300 words")
