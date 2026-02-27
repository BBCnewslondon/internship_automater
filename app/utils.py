from __future__ import annotations

import csv
import json
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Iterable


def retry_with_exponential_backoff(
    max_retries: int = 5,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    retry_exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable:
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            delay = initial_delay
            attempt = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except retry_exceptions:
                    attempt += 1
                    if attempt > max_retries:
                        raise
                    time.sleep(delay)
                    delay *= backoff_factor

        return wrapper

    return decorator


def append_job_to_csv(csv_path: Path, row: Dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["company_name", "job_title", "core_requirements", "deadline"]
    file_exists = csv_path.exists()

    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        csv_row = {
            "company_name": row.get("company_name", "Unknown"),
            "job_title": row.get("job_title", "Unknown"),
            "core_requirements": json.dumps(row.get("core_requirements", []), ensure_ascii=False),
            "deadline": row.get("deadline", "Unknown"),
        }
        writer.writerow(csv_row)


def flatten_search_results(results: Iterable[Dict[str, Any]]) -> str:
    lines = []
    for idx, item in enumerate(results, start=1):
        title = item.get("title", "")
        url = item.get("url", "")
        content = item.get("content", "")
        lines.append(f"[{idx}] {title}\nURL: {url}\nSnippet: {content}\n")
    return "\n".join(lines).strip()


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def prompt_human_review(draft: str) -> bool:
    print("\n=== Human-in-the-loop Review ===")
    print(draft)
    print("\nApprove this draft? Type 'yes' to finalize, anything else to reject.")
    decision = input("Decision: ").strip().lower()
    return decision == "yes"
