from __future__ import annotations

from typing import List
from pydantic import BaseModel, Field


class JobBrochureData(BaseModel):
    company_name: str = Field(description="Official company name")
    job_title: str = Field(description="Internship role title")
    core_requirements: List[str] = Field(description="5-10 core requirements as short bullet phrases")
    deadline: str = Field(description="Application deadline in ISO format YYYY-MM-DD when possible; otherwise raw text")


class JobBrochureList(BaseModel):
    jobs: List[JobBrochureData] = Field(description="List of all extracted internship opportunities found in the brochure")




class ResearchSummary(BaseModel):
    strategic_points: List[str] = Field(description="Exactly 3 concise strategic bullets from web context")
    reasoning_trace: List[str] = Field(description="List of thoughts/actions taken to find these points", default_factory=list)


class TailoredCV(BaseModel):
    tailored_cv_markdown: str = Field(description="Rewritten and filtered CV content in Markdown")
    highlighted_match_reasons: List[str] = Field(description="Top 3 reasons this CV now matches the role")
    role_classification: str = Field(description="Classified role type (Quant/Academic, SWE/Quant Dev, Hybrid)")


class ATSEvaluation(BaseModel):
    match_percentage: int = Field(description="Percentage of keywords matched (0-100)")
    missing_keywords: List[str] = Field(description="List of important keywords missing from CV")
    stuffing_detected: bool = Field(description="True if keyword stuffing is detected")
    feedback: str = Field(description="Constructive feedback on how to improve ATS score")




class ResearchAction(BaseModel):
    action: str = Field(description="Action to take: 'search' or 'finish'")
    search_query: str = Field(description="Search query if action is 'search', empty otherwise")
    thought: str = Field(description="Reasoning for this action")

class RoleClassification(BaseModel):
    role_type: str = Field(description="One of: 'Quant/Academic', 'SWE/Quant Dev', 'Hybrid'")

class CoverLetterDraft(BaseModel):
    cover_letter: str = Field(description="Final cover letter under 300 words")
