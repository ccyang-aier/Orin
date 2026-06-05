"""Collect arXiv AI paper candidates for the Orin daily paper radar."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


API_URL = "https://export.arxiv.org/api/query"
OAI_URL = "https://oaipmh.arxiv.org/oai"
USER_AGENT = "OrinArxivPaperRadar/0.1 (https://github.com/ccyang-aier/Orin)"
TIMEZONE = dt.timezone(dt.timedelta(hours=8))
DEFAULT_STATE = Path("rag/arxiv_papers/processed_papers.json")
DEFAULT_OUTPUT_DIR = Path("code/arxiv_paper_radar/.tmp")

CORE_CATEGORIES = [
    "cs.AI",
    "cs.CL",
    "cs.LG",
    "cs.CV",
    "cs.NE",
    "cs.IR",
    "cs.DC",
    "cs.PF",
    "cs.AR",
    "stat.ML",
]

OAI_SET_SPECS = {
    "cs.AI": "cs:cs:AI",
    "cs.CL": "cs:cs:CL",
    "cs.LG": "cs:cs:LG",
    "cs.CV": "cs:cs:CV",
    "cs.NE": "cs:cs:NE",
    "cs.IR": "cs:cs:IR",
    "cs.DC": "cs:cs:DC",
    "cs.PF": "cs:cs:PF",
    "cs.AR": "cs:cs:AR",
    "stat.ML": "stat:stat:ML",
}

OAI_NAMESPACES = {
    "oai": "http://www.openarchives.org/OAI/2.0/",
    "arxivraw": "http://arxiv.org/OAI/arXivRaw/",
}

KEYWORD_TERMS = [
    "AI",
    "artificial intelligence",
    "machine learning",
    "deep learning",
    "large language model",
    "LLM",
    "LLMs",
    "foundation model",
    "generative AI",
    "transformer",
    "vision language model",
    "VLM",
    "multimodal model",
    "reasoning model",
    "LLM reasoning",
    "inference",
    "LLM inference",
    "inference engine",
    "model serving",
    "serving system",
    "continuous batching",
    "speculative decoding",
    "decoding",
    "KV cache",
    "KVCache",
    "key-value cache",
    "PagedAttention",
    "FlashAttention",
    "attention algorithm",
    "long context",
    "context window",
    "quantization",
    "model compression",
    "pruning",
    "distillation",
    "Mixture of Experts",
    "MoE",
    "sparse model",
    "retrieval augmented generation",
    "RAG",
    "AI agent",
]

SEARCH_BATCHES = [
    [
        "large language model",
        "LLM",
        "LLMs",
        "foundation model",
        "generative AI",
        "transformer",
        "vision language model",
        "VLM",
        "multimodal model",
    ],
    [
        "LLM reasoning",
        "reasoning model",
        "chain of thought",
        "inference",
        "LLM inference",
        "inference engine",
        "model serving",
        "continuous batching",
        "speculative decoding",
        "decoding",
    ],
    [
        "KV cache",
        "KVCache",
        "key-value cache",
        "PagedAttention",
        "FlashAttention",
        "attention algorithm",
        "long context",
        "context window",
    ],
    [
        "quantization",
        "model compression",
        "pruning",
        "distillation",
        "Mixture of Experts",
        "MoE",
        "sparse model",
        "retrieval augmented generation",
        "RAG",
        "AI agent",
    ],
]

CODE_HOSTS = [
    "github.com",
    "gitlab.com",
    "bitbucket.org",
    "huggingface.co",
    "colab.research.google.com",
    "paperswithcode.com",
]

IGNORED_CODE_LINK_PATTERNS = [
    "github.com/arxiv/html_feedback",
    "github.com/brucemiller/latexml",
    "huggingface.co/docs/",
    "huggingface.co/huggingface",
]

MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}


def previous_day(now: dt.datetime | None = None) -> dt.date:
    current = now or dt.datetime.now(TIMEZONE)
    return (current.astimezone(TIMEZONE).date() - dt.timedelta(days=1))


def parse_gmt_datetime(value: str) -> dt.datetime:
    return dt.datetime.strptime(value, "%a, %d %b %Y %H:%M:%S GMT").replace(tzinfo=dt.timezone.utc)


def gmt_to_shanghai_day(value: str) -> str:
    return parse_gmt_datetime(value).astimezone(TIMEZONE).date().isoformat()


def build_oai_query_days(
    target_date: dt.date,
    now: dt.datetime | None = None,
    lookahead_days: int = 2,
) -> list[dt.date]:
    current = (now or dt.datetime.now(TIMEZONE)).astimezone(TIMEZONE).date()
    end = min(current, target_date + dt.timedelta(days=lookahead_days))
    if end < target_date:
        return []
    return [target_date + dt.timedelta(days=offset) for offset in range((end - target_date).days + 1)]


def html_to_text(value: str) -> str:
    value = re.sub(r"<script\b.*?</script>", " ", value, flags=re.I | re.S)
    value = re.sub(r"<style\b.*?</style>", " ", value, flags=re.I | re.S)
    value = re.sub(r"<[^>]+>", " ", value)
    return " ".join(html.unescape(value).split())


def parse_human_date(value: str) -> str | None:
    value = html_to_text(value)
    match = re.search(r"(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})", value)
    if not match:
        return None
    day = int(match.group(1))
    month = MONTHS.get(match.group(2).lower())
    year = int(match.group(3))
    if not month:
        return None
    return dt.date(year, month, day).isoformat()


def normalize_arxiv_id(value: str) -> str:
    value = value.strip().rstrip("/")
    value = value.rsplit("/", 1)[-1]
    return re.sub(r"v\d+$", "", value)


def quote_term(term: str) -> str:
    if " " in term or "-" in term:
        return f'all:"{term}"'
    return f"all:{term}"


def build_api_query(target_date: dt.date) -> str:
    date_clause = f"submittedDate:[{target_date:%Y%m%d}0000 TO {target_date:%Y%m%d}2359]"
    category_clause = " OR ".join(f"cat:{category}" for category in CORE_CATEGORIES)
    keyword_clause = " OR ".join(quote_term(term) for term in KEYWORD_TERMS)
    return f"{date_clause} AND ({category_clause} OR {keyword_clause})"


def extract_urls(*texts: str) -> list[str]:
    joined = "\n".join(texts)
    urls = re.findall(r"https?://[^\s<)\]}>,;\"\\]+", joined)
    cleaned = [url.rstrip(".,;:)]}\"'") for url in urls]
    return sorted(set(cleaned))


def classify_code_links(urls: list[str]) -> dict[str, Any]:
    code_links = []
    for url in urls:
        lowered = url.lower()
        if any(pattern in lowered for pattern in IGNORED_CODE_LINK_PATTERNS):
            continue
        if any(host in lowered for host in CODE_HOSTS):
            code_links.append(url)
    return {
        "has_open_source_demo_code": "yes" if code_links else "not_found",
        "code_links": sorted(set(code_links)),
    }


def infer_tags(text: str, categories: list[str] | None = None) -> list[str]:
    lowered = text.lower()
    tags: set[str] = set(categories or [])
    checks = {
        "LLM推理": ["llm inference", "inference engine", "serving", "decoding", "batching"],
        "KVCache": ["kv cache", "kvcache", "key-value cache", "pagedattention"],
        "注意力算法": ["attention", "flashattention"],
        "长上下文": ["long context", "context window"],
        "模型架构": ["architecture", "transformer", "mixture of experts", "moe", "sparse model"],
        "模型压缩": ["quantization", "compression", "pruning", "distillation"],
        "LLM推理能力": ["reasoning", "chain of thought", "search tree"],
        "RAG": ["retrieval augmented generation", "rag"],
        "AI Agent": ["agent", "tool use"],
        "多模态": ["multimodal", "vision language", "vlm", "video generation"],
    }
    for tag, needles in checks.items():
        if any(needle in lowered for needle in needles):
            tags.add(tag)
    return sorted(tags)


def http_get(url: str, timeout: int = 45) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def parse_search_results(page_html: str) -> list[dict[str, Any]]:
    items = re.findall(r'<li class="arxiv-result">(.*?)</li>', page_html, re.S)
    papers: list[dict[str, Any]] = []
    for item in items:
        id_match = re.search(r"https://arxiv\.org/abs/(\d{4}\.\d{4,5}(?:v\d+)?)", item)
        if not id_match:
            continue
        arxiv_id = normalize_arxiv_id(id_match.group(1))
        title_match = re.search(r'<p class="title is-5 mathjax">(.*?)</p>', item, re.S)
        authors_block = re.search(r'<p class="authors">(.*?)</p>', item, re.S)
        abstract_match = re.search(r'<p class="abstract mathjax">(.*?)</p>', item, re.S)
        submitted_match = re.search(r"<span[^>]*>\s*Submitted\s*</span>\s*([^;<]+)", item, re.I | re.S)
        categories = re.findall(r'data-tooltip="[^"]*">\s*([^<]+)\s*</span>', item)
        authors = re.findall(r"<a[^>]*>(.*?)</a>", authors_block.group(1), re.S) if authors_block else []
        abstract = html_to_text(abstract_match.group(1)) if abstract_match else ""
        abstract = re.sub(r"^Abstract\s*:?\s*", "", abstract, flags=re.I)
        paper = {
            "arxiv_id": arxiv_id,
            "title": html_to_text(title_match.group(1)) if title_match else "",
            "authors": [html_to_text(author) for author in authors],
            "categories": [html_to_text(category) for category in categories],
            "submitted_date": parse_human_date(submitted_match.group(1)) if submitted_match else None,
            "abstract": abstract,
            "paper_url": f"https://arxiv.org/abs/{arxiv_id}",
            "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
        }
        papers.append(paper)
    return papers


def parse_abs_page(page_html: str, arxiv_id: str) -> dict[str, Any]:
    dateline_match = re.search(r'<div class="dateline">(.*?)</div>', page_html, re.S)
    abstract_match = re.search(r'<blockquote class="abstract[^"]*".*?</blockquote>', page_html, re.S)
    authors_match = re.search(r'<div class="authors">(.*?)</div>', page_html, re.S)
    subjects_match = re.search(r'<td class="tablecell subjects">(.*?)</td>', page_html, re.S)
    html_link_match = re.search(r'href="(https://arxiv\.org/html/[^"]+|/html/[^"]+)"', page_html)

    categories: list[str] = []
    if subjects_match:
        subjects = html_to_text(subjects_match.group(1))
        categories = re.findall(r"\(([a-z-]+\.[A-Z]+)\)", subjects)

    authors: list[str] = []
    if authors_match:
        authors = [html_to_text(author) for author in re.findall(r"<a[^>]*>(.*?)</a>", authors_match.group(1), re.S)]

    abstract = html_to_text(abstract_match.group(0)) if abstract_match else ""
    abstract = re.sub(r"^Abstract\s*:?\s*", "", abstract, flags=re.I)

    html_url = None
    if html_link_match:
        html_url = urllib.parse.urljoin("https://arxiv.org", html_link_match.group(1))

    return {
        "arxiv_id": normalize_arxiv_id(arxiv_id),
        "submitted_date": parse_human_date(dateline_match.group(1)) if dateline_match else None,
        "abstract": abstract,
        "authors": authors,
        "categories": categories,
        "html_url": html_url,
    }


def parse_api_feed(feed_xml: bytes) -> list[dict[str, Any]]:
    namespaces = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }
    root = ET.fromstring(feed_xml)
    papers: list[dict[str, Any]] = []
    for entry in root.findall("atom:entry", namespaces):
        raw_id = entry.findtext("atom:id", default="", namespaces=namespaces)
        arxiv_id = normalize_arxiv_id(raw_id)
        links = {}
        for link in entry.findall("atom:link", namespaces):
            key = link.attrib.get("title") or link.attrib.get("rel") or "link"
            links[key] = link.attrib.get("href", "")
        published = entry.findtext("atom:published", default="", namespaces=namespaces)
        categories = [node.attrib.get("term", "") for node in entry.findall("atom:category", namespaces)]
        papers.append(
            {
                "arxiv_id": arxiv_id,
                "title": html_to_text(entry.findtext("atom:title", default="", namespaces=namespaces)),
                "authors": [
                    html_to_text(author.findtext("atom:name", default="", namespaces=namespaces))
                    for author in entry.findall("atom:author", namespaces)
                ],
                "categories": [category for category in categories if category],
                "submitted_date": published[:10] if published else None,
                "abstract": html_to_text(entry.findtext("atom:summary", default="", namespaces=namespaces)),
                "paper_url": f"https://arxiv.org/abs/{arxiv_id}",
                "pdf_url": links.get("pdf") or f"https://arxiv.org/pdf/{arxiv_id}",
            }
        )
    return papers


def parse_oai_record(record: str | ET.Element) -> dict[str, Any]:
    node = ET.fromstring(record) if isinstance(record, str) else record
    meta = node.find("oai:metadata/arxivraw:arXivRaw", OAI_NAMESPACES)
    if meta is None:
        raise ValueError("OAI record missing arXivRaw metadata")

    arxiv_id = normalize_arxiv_id(meta.findtext("arxivraw:id", default="", namespaces=OAI_NAMESPACES))
    versions = meta.findall("arxivraw:version", OAI_NAMESPACES)
    if not arxiv_id or not versions:
        raise ValueError("OAI record missing id or version history")

    first_version_gmt = versions[0].findtext("arxivraw:date", default="", namespaces=OAI_NAMESPACES)
    categories = (meta.findtext("arxivraw:categories", default="", namespaces=OAI_NAMESPACES) or "").split()
    authors_raw = html_to_text(meta.findtext("arxivraw:authors", default="", namespaces=OAI_NAMESPACES))
    authors = [part.strip() for part in authors_raw.split(",") if part.strip()]
    abstract = html_to_text(meta.findtext("arxivraw:abstract", default="", namespaces=OAI_NAMESPACES))
    datestamp = node.findtext("oai:header/oai:datestamp", default="", namespaces=OAI_NAMESPACES)

    return {
        "arxiv_id": arxiv_id,
        "title": html_to_text(meta.findtext("arxivraw:title", default="", namespaces=OAI_NAMESPACES)),
        "authors": authors,
        "categories": categories,
        "submitted_date": gmt_to_shanghai_day(first_version_gmt),
        "abstract": abstract,
        "paper_url": f"https://arxiv.org/abs/{arxiv_id}",
        "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
        "oai_datestamp": datestamp,
        "oai_first_version_gmt": first_version_gmt,
    }


def fetch_oai_papers(
    target_date: dt.date,
    delay_seconds: float,
    lookahead_days: int,
    now: dt.datetime | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    query_days = build_oai_query_days(target_date, now=now, lookahead_days=lookahead_days)
    papers: list[dict[str, Any]] = []
    day_statuses: list[dict[str, Any]] = []

    for query_day in query_days:
        set_statuses: list[dict[str, Any]] = []
        for category in CORE_CATEGORIES:
            set_spec = OAI_SET_SPECS[category]
            token = None
            pages = 0
            records = 0
            kept = 0
            errors: list[str] = []

            while True:
                if token:
                    url = f"{OAI_URL}?verb=ListRecords&resumptionToken={urllib.parse.quote(token)}"
                else:
                    encoded_set = urllib.parse.quote(set_spec)
                    day_text = query_day.isoformat()
                    url = (
                        f"{OAI_URL}?verb=ListRecords&metadataPrefix=arXivRaw&set={encoded_set}"
                        f"&from={day_text}&until={day_text}"
                    )
                try:
                    time.sleep(delay_seconds)
                    root = ET.fromstring(http_get(url, timeout=90))
                    pages += 1
                    for record in root.findall(".//oai:record", OAI_NAMESPACES):
                        records += 1
                        try:
                            paper = parse_oai_record(record)
                        except ValueError as error:
                            errors.append(str(error))
                            continue
                        if paper["submitted_date"] == target_date.isoformat():
                            papers.append(paper)
                            kept += 1
                    token_node = root.find(".//oai:resumptionToken", OAI_NAMESPACES)
                    token = (token_node.text or "").strip() if token_node is not None else ""
                    if not token:
                        break
                except urllib.error.HTTPError as error:
                    errors.append(f"HTTPError {error.code}: {error.reason}")
                    break
                except Exception as error:  # noqa: BLE001
                    errors.append(f"{type(error).__name__}: {error}")
                    break

            set_statuses.append(
                {
                    "category": category,
                    "set_spec": set_spec,
                    "pages": pages,
                    "records": records,
                    "kept": kept,
                    "errors": errors,
                }
            )
        day_statuses.append({"query_day": query_day.isoformat(), "sets": set_statuses})

    availability_window_end = target_date + dt.timedelta(days=lookahead_days)
    status = {
        "status": "ok",
        "query_days": [day.isoformat() for day in query_days],
        "lookahead_days": lookahead_days,
        "availability_window_complete": (now or dt.datetime.now(TIMEZONE)).astimezone(TIMEZONE).date() >= availability_window_end,
        "days": day_statuses,
        "count": len(merge_papers(papers)),
    }
    return merge_papers(papers), status


def fetch_api_papers(target_date: dt.date, max_results: int, delay_seconds: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    query = build_api_query(target_date)
    encoded = urllib.parse.urlencode(
        {
            "search_query": query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
    )
    url = f"{API_URL}?{encoded}"
    try:
        time.sleep(delay_seconds)
        papers = parse_api_feed(http_get(url, timeout=60))
        return papers, {"status": "ok", "url": url, "count": len(papers)}
    except urllib.error.HTTPError as error:
        return [], {"status": "http_error", "code": error.code, "reason": error.reason, "url": url}
    except Exception as error:  # External API failures should not break the whole job.
        return [], {"status": "error", "error": f"{type(error).__name__}: {error}", "url": url}


def make_search_query(terms: list[str]) -> str:
    rendered = []
    for term in terms:
        rendered.append(f'"{term}"' if " " in term else term)
    return " OR ".join(rendered)


def fetch_html_search_papers(
    target_date: dt.date,
    size: int,
    pages_per_batch: int,
    delay_seconds: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    papers: dict[str, dict[str, Any]] = {}
    batches = []
    for terms in SEARCH_BATCHES:
        query = make_search_query(terms)
        for page in range(pages_per_batch):
            start = page * size
            url = "https://arxiv.org/search/?" + urllib.parse.urlencode(
                {
                    "query": query,
                    "searchtype": "all",
                    "abstracts": "show",
                    "order": "-announced_date_first",
                    "size": str(size),
                    "start": str(start),
                }
            )
            try:
                time.sleep(delay_seconds)
                parsed = parse_search_results(http_get(url, timeout=60).decode("utf-8", "replace"))
                kept = 0
                for paper in parsed:
                    if paper.get("submitted_date") == target_date.isoformat():
                        papers[paper["arxiv_id"]] = paper
                        kept += 1
                batches.append({"status": "ok", "url": url, "parsed": len(parsed), "kept": kept})
                if len(parsed) < size:
                    break
            except Exception as error:
                batches.append({"status": "error", "url": url, "error": f"{type(error).__name__}: {error}"})
                break
    return list(papers.values()), {"status": "ok", "batches": batches, "count": len(papers)}


def merge_papers(*paper_lists: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for paper_list in paper_lists:
        for paper in paper_list:
            arxiv_id = paper.get("arxiv_id")
            if not arxiv_id:
                continue
            current = merged.get(arxiv_id, {})
            combined = {**current, **{key: value for key, value in paper.items() if value not in (None, "", [])}}
            merged[arxiv_id] = combined
    return sorted(merged.values(), key=lambda item: (item.get("submitted_date") or "", item.get("arxiv_id") or ""), reverse=True)


def enrich_from_abs_page(paper: dict[str, Any], delay_seconds: float) -> dict[str, Any]:
    try:
        time.sleep(delay_seconds)
        abs_html = http_get(paper["paper_url"], timeout=45).decode("utf-8", "replace")
        meta = parse_abs_page(abs_html, paper["arxiv_id"])
        urls = extract_urls(abs_html)
        enriched = {**paper}
        for key in ["submitted_date", "abstract", "authors", "categories", "html_url"]:
            if meta.get(key) and not enriched.get(key):
                enriched[key] = meta[key]
        enriched["source_urls"] = sorted(set(urls))
        return enriched
    except Exception as error:
        enriched = {**paper}
        enriched["metadata_error"] = f"{type(error).__name__}: {error}"
        enriched.setdefault("source_urls", [])
        return enriched


def extract_html_fulltext(url: str, timeout: int) -> tuple[str, list[str]]:
    page = http_get(url, timeout=timeout).decode("utf-8", "replace")
    title_match = re.search(r"<body[^>]*>(.*?)</body>", page, re.I | re.S)
    body = title_match.group(1) if title_match else page
    return html_to_text(body), extract_urls(page)


def extract_pdf_fulltext(url: str, timeout: int) -> tuple[str, list[str]]:
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError as error:
        raise RuntimeError("pypdf is required for PDF fallback; install requirements.txt") from error
    import io

    data = http_get(url, timeout=timeout)
    reader = PdfReader(io.BytesIO(data))
    text = "\n".join((page.extract_text() or "") for page in reader.pages)
    return html_to_text(text), extract_urls(text)


def enrich_fulltext(
    paper: dict[str, Any],
    timeout: int,
    max_chars: int,
) -> dict[str, Any]:
    enriched = {**paper}
    text = ""
    urls: list[str] = []
    status = "failed"
    error = None
    html_url = paper.get("html_url") or f"https://arxiv.org/html/{paper['arxiv_id']}"

    try:
        text, urls = extract_html_fulltext(html_url, timeout=timeout)
        status = "html"
        enriched["html_url"] = html_url
    except Exception as html_error:
        try:
            text, urls = extract_pdf_fulltext(paper.get("pdf_url") or f"https://arxiv.org/pdf/{paper['arxiv_id']}", timeout=timeout)
            status = "pdf"
            error = f"html failed: {type(html_error).__name__}: {html_error}"
        except Exception as pdf_error:
            error = f"html failed: {type(html_error).__name__}: {html_error}; pdf failed: {type(pdf_error).__name__}: {pdf_error}"

    fulltext_hash = hashlib.sha256(text.encode("utf-8")).hexdigest() if text else None
    truncated = bool(max_chars and len(text) > max_chars)
    if truncated:
        text = text[:max_chars]

    all_urls = sorted(set((paper.get("source_urls") or []) + extract_urls(paper.get("abstract", ""))))
    code_info = classify_code_links(all_urls)
    basis = "\n".join([paper.get("title", ""), paper.get("abstract", ""), text])
    enriched.update(
        {
            "tags": infer_tags(basis, paper.get("categories") or []),
            "source_urls": all_urls,
            "fulltext_reference_urls": sorted(set(urls)),
            "has_open_source_demo_code": code_info["has_open_source_demo_code"],
            "code_links": code_info["code_links"],
            "fulltext_status": status,
            "fulltext_error": error,
            "fulltext_chars": len(text),
            "fulltext_truncated": truncated,
            "fulltext_sha256": fulltext_hash,
            "fulltext": text,
        }
    )
    return enriched


def enrich_without_fulltext(paper: dict[str, Any], reason: str) -> dict[str, Any]:
    all_urls = sorted(set((paper.get("source_urls") or []) + extract_urls(paper.get("abstract", ""))))
    code_info = classify_code_links(all_urls)
    enriched = {**paper}
    enriched.update(
        {
            "tags": infer_tags("\n".join([paper.get("title", ""), paper.get("abstract", "")]), paper.get("categories") or []),
            "source_urls": all_urls,
            "has_open_source_demo_code": code_info["has_open_source_demo_code"],
            "code_links": code_info["code_links"],
            "fulltext_status": reason,
            "fulltext_chars": 0,
            "fulltext_truncated": False,
            "fulltext": "",
        }
    )
    return enriched


def split_sentences(text: str) -> list[str]:
    if not text:
        return []
    normalized = " ".join(text.split())
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", normalized) if part.strip()]


def find_relevant_sentence(text: str, patterns: list[str]) -> str:
    for sentence in split_sentences(text):
        lowered = sentence.lower()
        if any(pattern in lowered for pattern in patterns):
            return sentence
    return ""


def analysis_confidence(paper: dict[str, Any]) -> str:
    status = paper.get("fulltext_status")
    if status in {"html", "pdf"}:
        return "高"
    if status in {"skipped_scale", "skipped"}:
        return "中低"
    return "低"


def describe_fulltext_status(paper: dict[str, Any]) -> str:
    status = paper.get("fulltext_status")
    chars = paper.get("fulltext_chars", 0)
    if status == "html":
        return f"已抽取 HTML 正文，字符数 {chars}"
    if status == "pdf":
        return f"已抽取 PDF 正文，字符数 {chars}"
    if status == "skipped_scale":
        return "未抽取全文（规模控制策略），当前仅基于摘要与元数据分析"
    if status == "skipped":
        return "未抽取全文（本次运行显式跳过），当前仅基于摘要与元数据分析"
    if status == "failed":
        error = paper.get("fulltext_error") or "未记录失败原因"
        return f"全文抽取失败（{error}），当前仅基于摘要与元数据分析"
    return "全文抽取状态未记录，当前仅基于摘要与元数据分析"


def primary_orin_connection(paper: dict[str, Any]) -> list[str]:
    tags = set(paper.get("tags") or [])
    connections = []
    mapping = {
        "LLM推理": "可连接到 Orin 里的 vLLM / 推理服务链路",
        "KVCache": "可连接到 Orin 里的 KVCache 与长上下文主题",
        "注意力算法": "可连接到 Orin 里的注意力优化与算子设计主题",
        "长上下文": "可连接到 Orin 里的长上下文与上下文窗口主题",
        "模型架构": "可连接到 Orin 里的模型架构与 MoE 主题",
        "模型压缩": "可连接到 Orin 里的量化、压缩、部署成本主题",
        "RAG": "可连接到 Orin 里的 RAG / 检索增强主题",
        "AI Agent": "可连接到 Orin 里的 Agent、工具调用与工作流主题",
        "多模态": "可连接到 Orin 里的多模态模型与应用主题",
        "LLM推理能力": "可连接到 Orin 里的推理能力与评测主题",
    }
    for tag, sentence in mapping.items():
        if tag in tags:
            connections.append(sentence)
    if not connections:
        categories = ", ".join(paper.get("categories") or []) or "当前分类"
        connections.append(f"当前更像通用 {categories} 研究，可作为 Orin 里 AI 论文观察池的背景材料")
    return connections


def importance_rating(paper: dict[str, Any]) -> str:
    tags = set(paper.get("tags") or [])
    score = 0
    if tags & {"LLM推理", "AI Agent", "KVCache", "RAG", "LLM推理能力"}:
        score += 2
    if tags & {"模型架构", "模型压缩", "多模态", "长上下文"}:
        score += 1
    if paper.get("has_open_source_demo_code") == "yes":
        score += 1
    if paper.get("fulltext_status") in {"html", "pdf"}:
        score += 1
    if paper.get("fulltext_status") in {"failed", "skipped_scale", "skipped"}:
        score -= 1
    if score >= 4:
        return "A"
    if score >= 2:
        return "B"
    return "C"


def build_paper_analysis(paper: dict[str, Any]) -> dict[str, str]:
    abstract = paper.get("abstract", "")
    title = paper.get("title", paper.get("arxiv_id", "论文"))
    tags = paper.get("tags") or []
    confidence = analysis_confidence(paper)
    area = tags[0] if tags else (paper.get("categories") or ["AI 研究"])[0]
    problem_sentence = split_sentences(abstract)[0] if split_sentences(abstract) else "摘要没有给出足够清晰的问题定义"
    idea_sentence = find_relevant_sentence(abstract, ["propose", "introduce", "present", "develop", "design"])
    evidence_sentence = find_relevant_sentence(abstract, ["experiment", "benchmark", "result", "evaluate", "outperform", "performance"])

    if not idea_sentence:
        sentences = split_sentences(abstract)
        idea_sentence = sentences[1] if len(sentences) > 1 else problem_sentence

    if not evidence_sentence:
        evidence_sentence = "摘要未直接给出明确的定量结果，需要阅读全文或代码仓库进一步确认"

    architecture_sentence = "从标题、标签与摘要看，方法主线是“问题建模 -> 关键模块设计 -> 基准验证”"
    if "AI Agent" in tags:
        architecture_sentence = "从摘要与标签看，系统更像技能、工具、规划器协同的 Agent 工作流，而不是单一模型增量"
    elif "LLM推理" in tags or "LLM推理能力" in tags:
        architecture_sentence = "方法重点落在大模型推理链路上，核心流程通常围绕推理步骤、搜索结构或解码策略展开"
    elif "多模态" in tags:
        architecture_sentence = "方法形态更接近多模态编码、模态对齐或跨模态融合，再接下游任务头做验证"
    elif "模型压缩" in tags:
        architecture_sentence = "方法重心偏向模型压缩链路，通常包含压缩策略、误差控制与部署收益评估"

    confidence_note = f"当前分析置信度：{confidence}"
    if paper.get("fulltext_status") not in {"html", "pdf"}:
        confidence_note += "；正文未完整抽取，本条判断主要基于标题、摘要、分类和链接信号"

    conclusion = f"{title} 更偏向 {area} 方向，当前最值得关注的是其是否能在公开基准或真实系统中复现摘要声称的收益"
    follow_up = []
    if paper.get("has_open_source_demo_code") != "yes":
        follow_up.append("作者是否会补公开代码或最小可复现实验脚本")
    if paper.get("fulltext_status") not in {"html", "pdf"}:
        follow_up.append("需要在正文可用后复核方法细节与实验设置")
    if "AI Agent" in tags:
        follow_up.append("需要关注任务拆分、工具选择与多轮反馈机制是否有消融实验")
    elif "LLM推理" in tags or "LLM推理能力" in tags:
        follow_up.append("需要确认是否报告了时延、吞吐、成本或准确率之间的权衡")
    elif "模型压缩" in tags:
        follow_up.append("需要确认压缩收益是否在不同模型规模与硬件上都成立")
    else:
        follow_up.append("需要确认摘要中的收益是否覆盖更强基线或更复杂场景")

    return {
        "core_problem": f"论文聚焦于 {area}，摘要显示其核心问题是：{problem_sentence}",
        "key_idea": f"从摘要可见，作者的关键思路是：{idea_sentence}",
        "architecture": f"{architecture_sentence}；{confidence_note}",
        "evidence": f"当前能直接拿到的证据是：{evidence_sentence}",
        "rating": importance_rating(paper),
        "connections": "；".join(primary_orin_connection(paper)),
        "questions": "；".join(follow_up),
        "conclusion": conclusion,
    }


def load_state(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {"schema_version": 1, "last_run_at": None, "papers": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(path: str | Path, state: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def filter_unprocessed(papers: list[dict[str, Any]], state: dict[str, Any]) -> list[dict[str, Any]]:
    processed = state.get("papers", {})
    return [paper for paper in papers if paper.get("arxiv_id") not in processed]


def mark_analyzed(
    state_path: str | Path,
    papers: list[dict[str, Any]],
    report_path: str,
    now: str | None = None,
) -> None:
    state = load_state(state_path)
    timestamp = now or dt.datetime.now(TIMEZONE).isoformat(timespec="seconds")
    state["last_run_at"] = timestamp
    state.setdefault("schema_version", 1)
    state.setdefault("papers", {})
    for paper in papers:
        arxiv_id = paper["arxiv_id"]
        state["papers"][arxiv_id] = {
            "status": "analyzed",
            "title": paper.get("title", ""),
            "last_analyzed_at": timestamp,
            "report_path": report_path,
            "paper_url": paper.get("paper_url") or f"https://arxiv.org/abs/{arxiv_id}",
            "fulltext_sha256": paper.get("fulltext_sha256"),
        }
    save_state(state_path, state)


def collect(args: argparse.Namespace) -> int:
    target_date = dt.date.fromisoformat(args.target_date) if args.target_date else previous_day()
    state = load_state(args.state)
    now = dt.datetime.now(TIMEZONE)
    papers, oai_status = fetch_oai_papers(
        target_date,
        delay_seconds=args.delay_seconds,
        lookahead_days=args.oai_lookahead_days,
        now=now,
    )
    papers = [paper for paper in papers if paper.get("submitted_date") in (None, target_date.isoformat())]
    pending_reason = None
    if not papers and not oai_status.get("availability_window_complete"):
        pending_reason = "target day may not be fully exposed in public arXiv feeds yet"

    unprocessed = filter_unprocessed(papers, state)
    enriched: list[dict[str, Any]] = []
    for index, paper in enumerate(unprocessed):
        if args.max_papers and index >= args.max_papers:
            break
        paper = enrich_from_abs_page(paper, args.delay_seconds)
        if paper.get("submitted_date") and paper["submitted_date"] != target_date.isoformat():
            continue
        if not args.skip_fulltext and index < args.fulltext_limit:
            paper = enrich_fulltext(paper, timeout=args.fulltext_timeout, max_chars=args.max_fulltext_chars)
        else:
            reason = "skipped" if args.skip_fulltext else "skipped_scale"
            paper = enrich_without_fulltext(paper, reason=reason)
        enriched.append(paper)

    output = {
        "schema_version": 1,
        "generated_at": dt.datetime.now(TIMEZONE).isoformat(timespec="seconds"),
        "target_date": target_date.isoformat(),
        "query": {
            "collector": "oai_pmh_v1_shanghai_filter",
            "categories": CORE_CATEGORIES,
            "keyword_terms": KEYWORD_TERMS,
            "oai_set_specs": OAI_SET_SPECS,
            "oai_query_days": oai_status.get("query_days", []),
        },
        "source_status": {
            "oai": oai_status,
            "pending_reason": pending_reason,
        },
        "counts": {
            "candidate_papers": len(papers),
            "already_processed": len(papers) - len(unprocessed),
            "new_papers": len(enriched),
        },
        "papers": enriched,
    }
    output_path = Path(args.output) if args.output else DEFAULT_OUTPUT_DIR / f"{target_date.isoformat()}-papers.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output_path)
    return 0


def render_report(args: argparse.Namespace) -> int:
    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    target_date = data["target_date"]
    generated_at = dt.datetime.now(TIMEZONE).date().isoformat()
    source_status = data.get("source_status", {})
    oai_status = source_status.get("oai", {})
    fulltext_ready = sum(1 for paper in data["papers"] if paper.get("fulltext_status") in {"html", "pdf"})
    fulltext_skipped = sum(1 for paper in data["papers"] if paper.get("fulltext_status") not in {"html", "pdf"})
    lines = [
        "---",
        "tags: [papers, arxiv, ai, llm-inference]",
        f"updated: {generated_at}",
        f"description: {target_date} arXiv AI、LLM 推理与 AI 模型论文日报。",
        "---",
        "",
        f"# 1 {target_date} arXiv AI 论文雷达",
        "",
        "## 1.1 运行摘要",
        "",
        f"- 抓取目标日期：{target_date}；",
        f"- 候选论文：{data['counts']['candidate_papers']} 篇；",
        f"- 新增待分析论文：{data['counts']['new_papers']} 篇；",
        f"- 已跳过重复论文：{data['counts']['already_processed']} 篇；",
        f"- 采集入口：OAI-PMH 单日轮询 + `v1 GMT -> Asia/Shanghai` 日期过滤；",
        f"- 完整正文抽取：{fulltext_ready} 篇；",
        f"- 仅基于摘要/元数据分析：{fulltext_skipped} 篇；",
        "",
        "## 1.2 抓取状态",
        "",
        f"- OAI 查询日：{', '.join(oai_status.get('query_days', [])) or '未记录'}；",
        f"- 可见性窗口是否完整：{'是' if oai_status.get('availability_window_complete') else '否'}；",
    ]
    if not oai_status.get("availability_window_complete"):
        lines.append("- 版本说明：当前报告为首版，后续可能因 arXiv 公共可见性延迟而补录；")
    if source_status.get("pending_reason"):
        lines.append(f"- 公开可见性提醒：{source_status['pending_reason']}；")
    lines.extend(
        [
            "",
            "## 1.3 今日论文总表",
            "",
            "| 评级 | arXiv ID | 标题 | 标签 | 论文链接 | 开源Demo/代码 | 一句话结论 |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for paper in data["papers"]:
        analysis = build_paper_analysis(paper)
        code = "有：" + ", ".join(paper.get("code_links", [])) if paper.get("code_links") else "未发现"
        tags = ", ".join(paper.get("tags", []))
        title = (paper.get("title") or "").replace("|", "\\|")
        conclusion = analysis["conclusion"].replace("|", "\\|")
        lines.append(
            f"| {analysis['rating']} | {paper['arxiv_id']} | {title} | {tags} | {paper.get('paper_url')} | {code} | {conclusion} |"
        )
    lines.extend(["", "## 1.4 论文详析", ""])
    for index, paper in enumerate(data["papers"], 1):
        analysis = build_paper_analysis(paper)
        code = "有：" + ", ".join(paper.get("code_links", [])) if paper.get("code_links") else "未发现"
        lines.extend(
            [
                f"### 1.4.{index} [{analysis['rating']}] {paper.get('title', paper['arxiv_id'])}",
                "",
                "**基础信息**",
                "",
                f"- arXiv ID：{paper['arxiv_id']}；",
                f"- 作者：{', '.join(paper.get('authors', [])) or '未解析'}；",
                f"- 分类：{', '.join(paper.get('categories', [])) or '未解析'}；",
                f"- 自动标签：{', '.join(paper.get('tags', [])) or '未解析'}；",
                f"- 论文链接：{paper.get('paper_url')}；",
                f"- 开源Demo/代码：{code}；",
                f"- 正文抽取：{describe_fulltext_status(paper)}；",
                "",
                "**摘要**",
                "",
                paper.get("abstract", "未解析"),
                "",
                "**AI 分析**",
                "",
                f"- 核心问题：{analysis['core_problem']}；",
                f"- 原理与核心思想：{analysis['key_idea']}；",
                f"- 架构与流程：{analysis['architecture']}；",
                f"- 实验与证据：{analysis['evidence']}；",
                f"- 重要性评级：{analysis['rating']}；",
                f"- Orin 知识连接：{analysis['connections']}；",
                f"- 后续问题：{analysis['questions']}；",
                "",
            ]
        )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(output_path)
    return 0


def mark(args: argparse.Namespace) -> int:
    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    mark_analyzed(args.state, data.get("papers", []), args.report)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect_parser = subparsers.add_parser("collect", help="Collect previous-day arXiv paper candidates")
    collect_parser.add_argument("--target-date", help="Date to collect, YYYY-MM-DD. Defaults to yesterday in Asia/Shanghai.")
    collect_parser.add_argument("--state", default=str(DEFAULT_STATE), help="Processed-paper state JSON path.")
    collect_parser.add_argument("--output", help="Output JSON path. Defaults to code/arxiv_paper_radar/.tmp/<date>-papers.json.")
    collect_parser.add_argument("--max-results", type=int, default=1000, help="Maximum arXiv API results.")
    collect_parser.add_argument("--search-size", type=int, default=200, help="HTML search page size per keyword batch.")
    collect_parser.add_argument("--search-pages", type=int, default=2, help="HTML search pages per keyword batch.")
    collect_parser.add_argument("--delay-seconds", type=float, default=0.2, help="Delay before external arXiv requests.")
    collect_parser.add_argument("--fulltext-timeout", type=int, default=60, help="Per-paper fulltext fetch timeout.")
    collect_parser.add_argument("--max-fulltext-chars", type=int, default=0, help="0 keeps complete extracted full text in the temporary JSON.")
    collect_parser.add_argument("--fulltext-limit", type=int, default=20, help="Only the first N papers attempt fulltext extraction; the rest stay abstract-only.")
    collect_parser.add_argument("--max-papers", type=int, default=0, help="Optional cap for validation runs.")
    collect_parser.add_argument("--oai-lookahead-days", type=int, default=2, help="Single-day OAI query window extends this many days beyond the target date.")
    collect_parser.add_argument("--skip-fulltext", action="store_true", help="Skip fulltext extraction for smoke tests.")
    collect_parser.set_defaults(func=collect)

    render_parser = subparsers.add_parser("render-report", help="Render an AI-fillable Markdown report skeleton")
    render_parser.add_argument("--input", required=True)
    render_parser.add_argument("--output", required=True)
    render_parser.set_defaults(func=render_report)

    mark_parser = subparsers.add_parser("mark-analyzed", help="Record papers from a completed report as analyzed")
    mark_parser.add_argument("--input", required=True)
    mark_parser.add_argument("--state", default=str(DEFAULT_STATE))
    mark_parser.add_argument("--report", required=True)
    mark_parser.set_defaults(func=mark)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
