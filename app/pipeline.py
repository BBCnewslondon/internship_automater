from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import fitz
import requests
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from .models import CoverLetterDraft, JobBrochureData, ResearchSummary, TailoredCV
from .prompts import (
    EXTRACTION_SYSTEM_PROMPT,
    FILTER_SYSTEM_PROMPT,
    RESEARCH_SYSTEM_PROMPT,
    SYNTHESIS_SYSTEM_PROMPT,
)
from .utils import (
    append_job_to_csv,
    flatten_search_results,
    prompt_human_review,
    read_text,
    retry_with_exponential_backoff,
    save_text,
)


@dataclass
class PipelineConfig:
    openai_api_key: str
    tavily_api_key: str
    model_name: str = "gpt-4o-mini"


class LLMClient:
    def __init__(self, model_name: str, openai_api_key: str) -> None:
        os.environ["OPENAI_API_KEY"] = openai_api_key
        self._llm = ChatOpenAI(model=model_name, temperature=0)

    @retry_with_exponential_backoff(max_retries=5, initial_delay=1.0, backoff_factor=2.0)
    def invoke_structured(self, system_prompt: str, user_prompt: str, schema: Any) -> Any:
        structured_llm = self._llm.with_structured_output(schema)
        return structured_llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )


class BrochureExtractor:
    def __init__(self, llm_client: LLMClient, tracking_csv: Path) -> None:
        self.llm_client = llm_client
        self.tracking_csv = tracking_csv

    @staticmethod
    def _extract_pdf_text(pdf_path: Path) -> str:
        chunks: List[str] = []
        with fitz.open(pdf_path) as doc:
            for page in doc:
                raw_text = page.get_text("text")
                if isinstance(raw_text, str) and raw_text:
                    chunks.append(raw_text)
        return "\n".join(chunks).strip()

    def run(self, brochure_pdf: Path) -> JobBrochureData:
        brochure_text = self._extract_pdf_text(brochure_pdf)
        user_prompt = (
            "Extract job brochure information into schema.\n\n"
            f"Brochure Text:\n{brochure_text}"
        )
        extracted: JobBrochureData = self.llm_client.invoke_structured(
            EXTRACTION_SYSTEM_PROMPT,
            user_prompt,
            JobBrochureData,
        )
        append_job_to_csv(self.tracking_csv, extracted.model_dump())
        return extracted


class WebResearcher:
    def __init__(self, llm_client: LLMClient, tavily_api_key: str) -> None:
        self.llm_client = llm_client
        self.tavily_api_key = tavily_api_key

    @retry_with_exponential_backoff(max_retries=5, initial_delay=1.0, backoff_factor=2.0)
    def _search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        response = requests.post(
            "https://api.tavily.com/search",
            headers={"Content-Type": "application/json"},
            json={
                "api_key": self.tavily_api_key,
                "query": query,
                "search_depth": "advanced",
                "include_answer": False,
                "max_results": max_results,
            },
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("results", [])

    def run(self, extracted: JobBrochureData) -> ResearchSummary:
        query = f"{extracted.company_name} core values 2024 initiatives {extracted.job_title}"
        results = self._search(query)
        findings = flatten_search_results(results)
        user_prompt = (
            f"Company: {extracted.company_name}\n"
            f"Role: {extracted.job_title}\n"
            f"Search Query: {query}\n\n"
            f"Search Findings:\n{findings}\n\n"
            "Return exactly 3 strategic points."
        )
        summary: ResearchSummary = self.llm_client.invoke_structured(
            RESEARCH_SYSTEM_PROMPT,
            user_prompt,
            ResearchSummary,
        )
        if len(summary.strategic_points) > 3:
            summary.strategic_points = summary.strategic_points[:3]
        elif len(summary.strategic_points) < 3:
            summary.strategic_points.extend(["No additional verified strategic point found."] * (3 - len(summary.strategic_points)))
        return summary


class CVFilter:
    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    def run(self, master_cv_path: Path, extracted: JobBrochureData) -> TailoredCV:
        master_cv = read_text(master_cv_path)
        requirements = "\n".join(f"- {item}" for item in extracted.core_requirements)
        user_prompt = (
            f"Role: {extracted.job_title} at {extracted.company_name}\n"
            f"Core Requirements:\n{requirements}\n\n"
            f"Master CV (Markdown):\n{master_cv}\n"
        )
        return self.llm_client.invoke_structured(
            FILTER_SYSTEM_PROMPT,
            user_prompt,
            TailoredCV,
        )


class CoverLetterSynthesizer:
    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    def run(
        self,
        extracted: JobBrochureData,
        research: ResearchSummary,
        tailored_cv: TailoredCV,
    ) -> CoverLetterDraft:
        research_points = "\n".join(f"- {point}" for point in research.strategic_points)
        user_prompt = (
            f"Company: {extracted.company_name}\n"
            f"Role: {extracted.job_title}\n"
            f"Deadline: {extracted.deadline}\n\n"
            f"Research Strategic Points:\n{research_points}\n\n"
            f"Tailored CV:\n{tailored_cv.tailored_cv_markdown}\n\n"
            "Write final cover letter under 300 words."
        )
        drafted: CoverLetterDraft = self.llm_client.invoke_structured(
            SYNTHESIS_SYSTEM_PROMPT,
            user_prompt,
            CoverLetterDraft,
        )
        words = drafted.cover_letter.split()
        if len(words) > 300:
            drafted.cover_letter = " ".join(words[:300])
        return drafted


class InternshipApplicationPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.llm_client = LLMClient(config.model_name, config.openai_api_key)
        self.extractor = BrochureExtractor(self.llm_client, Path("tracking.csv"))
        self.researcher = WebResearcher(self.llm_client, config.tavily_api_key)
        self.filter = CVFilter(self.llm_client)
        self.synthesizer = CoverLetterSynthesizer(self.llm_client)

    def run(self, brochure_pdf: Path, master_cv_path: Path, output_dir: Path) -> Dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)

        extracted = self.extractor.run(brochure_pdf)
        research = self.researcher.run(extracted)
        tailored_cv = self.filter.run(master_cv_path, extracted)
        draft = self.synthesizer.run(extracted, research, tailored_cv)

        save_text(output_dir / "extracted_job.json", json.dumps(extracted.model_dump(), indent=2))
        save_text(output_dir / "research_summary.json", json.dumps(research.model_dump(), indent=2))
        save_text(output_dir / "tailored_cv.md", tailored_cv.tailored_cv_markdown)
        save_text(output_dir / "cover_letter_draft.txt", draft.cover_letter)

        approved = prompt_human_review(draft.cover_letter)
        if approved:
            save_text(output_dir / "cover_letter_final.txt", draft.cover_letter)
            print("Final cover letter approved and saved.")
        else:
            print("Draft rejected in human review. Revise inputs/prompts and rerun.")

        return {
            "extracted": extracted.model_dump(),
            "research": research.model_dump(),
            "tailored_cv": tailored_cv.model_dump(),
            "cover_letter_draft": draft.model_dump(),
            "approved": approved,
        }


def load_pipeline_config() -> PipelineConfig:
    load_dotenv()

    openai_api_key = os.getenv("OPENAI_API_KEY", "")
    tavily_api_key = os.getenv("TAVILY_API_KEY", "")
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    missing = []
    if not openai_api_key:
        missing.append("OPENAI_API_KEY")
    if not tavily_api_key:
        missing.append("TAVILY_API_KEY")
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

    return PipelineConfig(
        openai_api_key=openai_api_key,
        tavily_api_key=tavily_api_key,
        model_name=model_name,
    )
