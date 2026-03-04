# Automated Internship Application Pipeline Builder

A modular 4-phase pipeline that transforms:
- Internship brochure PDF
- Master CV (Markdown/Text)
- Web search context

into a tailored CV and cover letter draft with a human approval gate.

## Phases

1. **Extractor** (`BrochureExtractor`)
   - Uses **PyMuPDF** (`fitz`) to read brochure text.
   - Uses LangChain + structured output to extract:
     - company_name
     - job_title
     - core_requirements
     - deadline
   - Appends results to `tracking.csv`.

2. **Researcher** (`WebResearcher`)
   - Builds query like: `"{Company} core values 2024 initiatives {Role}"`.
   - Calls Tavily Search API.
   - Summarizes to exactly 3 strategic points with LLM.

3. **Filter** (`CVFilter`)
   - Tailors master CV using role requirements.
   - Drops irrelevant content and rewrites bullets for role fit.

4. **Synthesizer** (`CoverLetterSynthesizer`)
   - Drafts final cover letter under 300 words.
   - Enforces explicit connection between CV project and company initiative.

## Safety & Reliability

- `.env` secrets loaded via `python-dotenv`
- Exponential backoff on all API calls (LLM + Tavily)
- Human-in-the-loop review step before finalization

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Configure environment:

```bash
copy .env.example .env
```

Fill in keys in `.env`:
- `GOOGLE_API_KEY`
- `TAVILY_API_KEY`
- `GOOGLE_MODEL` (optional, defaults to `gemini-1.5-flash`)

## Run

```bash
python main.py --brochure path/to/brochure.pdf --cv path/to/master_cv.md --output-dir outputs
```

## Outputs

Generated in `--output-dir`:
- `extracted_job.json`
- `research_summary.json`
- `tailored_cv.md`
- `cover_letter_draft.txt`
- `cover_letter_final.txt` (only if approved)

Also updates:
- `tracking.csv`
