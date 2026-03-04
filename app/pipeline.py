from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import fitz
import requests
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from .models import (
    ATSEvaluation,
    CoverLetterDraft,
    JobBrochureData,
    JobBrochureList,
    ResearchAction,
    ResearchSummary,
    RoleClassification,
    TailoredCV,
)
from .prompts import (
    ATS_SCORING_PROMPT,
    EXTRACTION_SYSTEM_PROMPT,
    FILTER_SYSTEM_PROMPT,
    RESEARCH_REACT_SYSTEM_PROMPT,
    RESEARCH_SYSTEM_PROMPT,
    ROLE_CLASSIFICATION_PROMPT,
    SYNTHESIS_SYSTEM_PROMPT,
    VERIFICATION_SYSTEM_PROMPT,
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
    google_api_key: str
    tavily_api_key: str
    model_name: str = "gemini-1.5-flash"


class LLMClient:
    def __init__(self, model_name: str, google_api_key: str) -> None:
        os.environ["GOOGLE_API_KEY"] = google_api_key
        # For ReAct loops and multi-step reasoning, higher temperature (though 0 is safer for structured output)
        self._llm = ChatGoogleGenerativeAI(model=model_name, temperature=0, convert_system_message_to_human=True)

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

    def run(self, brochure_pdf: Path) -> JobBrochureList:
        brochure_text = self._extract_pdf_text(brochure_pdf)
        user_prompt = (
            "Extract ALL job brochure information into schema.\n\n"
            f"Brochure Text:\n{brochure_text}"
        )
        extracted: JobBrochureList = self.llm_client.invoke_structured(
            EXTRACTION_SYSTEM_PROMPT,
            user_prompt,
            JobBrochureList,
        )
        for job in extracted.jobs:
            append_job_to_csv(self.tracking_csv, job.model_dump())
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
        # ReAct Loop Implementation
        search_history = []
        max_iterations = 3
        
        # Initial context
        context = f"Company: {extracted.company_name}\nRole: {extracted.job_title}"
        
        for i in range(max_iterations):
            user_prompt = (
                f"Context so far:\n{context}\n\n"
                f"Iteration {i+1}/{max_iterations}. What is the next step?"
            )
            
            # Decide next action
            action: ResearchAction = self.llm_client.invoke_structured(
                RESEARCH_REACT_SYSTEM_PROMPT,
                user_prompt,
                ResearchAction
            )
            
            if action.action == "finish":
                break
                
            # Execute search
            print(f"Researcher Thought: {action.thought}")
            print(f"Researcher Action: Searching for '{action.search_query}'")
            try:
                if action.search_query:
                    results = self._search(action.search_query)
                    findings = flatten_search_results(results)
                    context += f"\n\nSearch '{action.search_query}' Results:\n{findings}"
                    search_history.append(f"Thought: {action.thought}\nQuery: {action.search_query}")
                else:
                    break
            except Exception as e:
                print(f"Search failed: {e}")
                break

        # Final Synthesis
        synthesis_prompt = (
            f"Context:\n{context}\n\n"
            "Based on the research above, synthesize exactly 3 strategic points."
        )
        summary: ResearchSummary = self.llm_client.invoke_structured(
            RESEARCH_SYSTEM_PROMPT,
            synthesis_prompt,
            ResearchSummary
        )
        summary.reasoning_trace = search_history
        return summary


class ATSScorer:
    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    def run(self, cv_text: str, job: JobBrochureData) -> ATSEvaluation:
        job_desc = f"Role: {job.job_title}\nCompany: {job.company_name}\nRequirements: {job.core_requirements}"
        user_prompt = (
            f"Job Description:\n{job_desc}\n\n"
            f"Candidate CV:\n{cv_text}\n"
        )
        return self.llm_client.invoke_structured(
            ATS_SCORING_PROMPT,
            user_prompt,
            ATSEvaluation
        )


class CVFilter:
    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    def _classify_role(self, job: JobBrochureData) -> RoleClassification:
        user_prompt = (
            f"Job Title: {job.job_title}\n"
            f"Requirements: {job.core_requirements}"
        )
        role = self.llm_client.invoke_structured(
            ROLE_CLASSIFICATION_PROMPT,
            user_prompt,
            RoleClassification
        )
        return role

    def run(self, master_cv_path: Path, extracted: JobBrochureData) -> TailoredCV:
        master_cv = read_text(master_cv_path)
        requirements = "\n".join(f"- {item}" for item in extracted.core_requirements)
        
        # Step 1: Classify Role
        role_class = self._classify_role(extracted)
        print(f"Role Classified as: {role_class.role_type}")

        # Step 2: Generate Initial CV
        user_prompt = (
            f"Role: {extracted.job_title} at {extracted.company_name}\n"
            f"Core Requirements:\n{requirements}\n\n"
            f"Master CV:\n{master_cv}\n"
        )
        # We need to inject the role type into the system prompt effectively
        # Since invoke_structured takes a static system prompt, we format it here
        formatted_system_prompt = FILTER_SYSTEM_PROMPT.format(role_type=role_class.role_type)
        
        initial_cv: TailoredCV = self.llm_client.invoke_structured(
            formatted_system_prompt,
            user_prompt,
            TailoredCV,
        )
        
        # Manually set role classification in the output object if not populated by LLM (since LLM generates markdown + reasons)
        initial_cv.role_classification = role_class.role_type

        # Step 3: Chain-of-Verification (CoVe)
        # Verify the generated CV against the Master CV
        verification_prompt = (
            f"Master CV (Ground Truth):\n{master_cv}\n\n"
            f"Generated CV (To Verify):\n{initial_cv.tailored_cv_markdown}\n"
        )
        
        verified_cv: TailoredCV = self.llm_client.invoke_structured(
            VERIFICATION_SYSTEM_PROMPT,
            verification_prompt,
            TailoredCV
        )
        
        # Merge metadata (keep classification and reasons, update markdown if verification changed it)
        verified_cv.role_classification = role_class.role_type
        if initial_cv.highlighted_match_reasons:
             verified_cv.highlighted_match_reasons = initial_cv.highlighted_match_reasons
             
        return verified_cv


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
        self.llm_client = LLMClient(config.model_name, config.google_api_key)
        self.extractor = BrochureExtractor(self.llm_client, Path("tracking.csv"))
        self.researcher = WebResearcher(self.llm_client, config.tavily_api_key)
        self.filter = CVFilter(self.llm_client)
        self.synthesizer = CoverLetterSynthesizer(self.llm_client)
        self.scorer = ATSScorer(self.llm_client)

    def run(self, brochure_pdf: Path, master_cv_path: Path, output_dir: Path) -> List[Dict[str, Any]]:
        output_dir.mkdir(parents=True, exist_ok=True)

        extracted_list = self.extractor.run(brochure_pdf)
        all_results = []
        
        print(f"DEBUG: Found {len(extracted_list.jobs)} job(s) in the brochure.")

        for job in extracted_list.jobs:
            # Create a unique folder for each job to avoid file collision
            safe_company = "".join(c for c in job.company_name if c.isalnum() or c in (' ', '_', '-')).strip()
            safe_title = "".join(c for c in job.job_title if c.isalnum() or c in (' ', '_', '-')).strip()
            job_output_dir = output_dir / f"{safe_company} - {safe_title}"
            job_output_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"Processing: {job.company_name} - {job.job_title} -> {job_output_dir}")

            research = self.researcher.run(job)
            tailored_cv = self.filter.run(master_cv_path, job)
            
            # ATS Scoring Validation Loop
            ats_score = self.scorer.run(tailored_cv.tailored_cv_markdown, job)
            print(f"ATS Score: {ats_score.match_percentage}%")
            if ats_score.match_percentage < 80:
                print(f"WARNING: Low ATS Score ({ats_score.match_percentage}%). Missing: {ats_score.missing_keywords}")
                print(f"Feedback: {ats_score.feedback}")
            
            draft = self.synthesizer.run(job, research, tailored_cv)

            save_text(job_output_dir / "extracted_job.json", json.dumps(job.model_dump(), indent=2))
            save_text(job_output_dir / "research_summary.json", json.dumps(research.model_dump(), indent=2))
            save_text(job_output_dir / "tailored_cv.md", tailored_cv.tailored_cv_markdown)
            save_text(job_output_dir / "ats_evaluation.json", json.dumps(ats_score.model_dump(), indent=2))
            save_text(job_output_dir / "cover_letter_draft.txt", draft.cover_letter)

            approved = prompt_human_review(draft.cover_letter)
            if approved:
                save_text(job_output_dir / "cover_letter_final.txt", draft.cover_letter)
                print(f"Final cover letter approved and saved for {job.company_name}.")
            else:
                print(f"Draft rejected for {job.company_name}.")

            all_results.append({
                "job": job.model_dump(),
                "research": research.model_dump(),
                "tailored_cv": tailored_cv.model_dump(),
                "cover_letter_draft": draft.model_dump(),
                "ats_evaluation": ats_score.model_dump(),
                "approved": approved,
            })
        
        return all_results

def load_pipeline_config() -> PipelineConfig:
    load_dotenv()

    google_api_key = os.getenv("GOOGLE_API_KEY", "")
    tavily_api_key = os.getenv("TAVILY_API_KEY", "")
    model_name = os.getenv("GOOGLE_MODEL", "gemini-1.5-flash")

    missing = []
    if not google_api_key:
        missing.append("GOOGLE_API_KEY")
    if not tavily_api_key:
        missing.append("TAVILY_API_KEY")
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

    return PipelineConfig(
        google_api_key=google_api_key,
        tavily_api_key=tavily_api_key,
        model_name=model_name,
    )
