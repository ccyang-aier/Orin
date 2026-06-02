import argparse
import datetime as dt
import json
from pathlib import Path

import orin_arxiv_radar as radar


def test_build_api_query_uses_previous_day_and_broad_or_terms():
    query = radar.build_api_query(dt.date(2026, 5, 31))

    assert "submittedDate:[202605310000 TO 202605312359]" in query
    assert "cat:cs.AI" in query
    assert "cat:stat.ML" in query
    assert 'all:"KV cache"' in query
    assert " OR " in query


def test_parse_search_results_extracts_links_dates_and_categories():
    html = """
    <li class="arxiv-result">
      <p class="list-title is-inline-block"><a href="https://arxiv.org/abs/2605.13111">arXiv:2605.13111</a>
        <span>[<a href="https://arxiv.org/pdf/2605.13111">pdf</a>]</span>
      </p>
      <div class="tags is-inline-block">
        <span class="tag is-small is-link tooltip is-tooltip-top" data-tooltip="Computer Vision and Pattern Recognition">cs.CV</span>
        <span class="tag is-small is-link tooltip is-tooltip-top" data-tooltip="Machine Learning">cs.LG</span>
      </div>
      <p class="title is-5 mathjax">Pyramid Forcing: Head-Aware Pyramid KV Cache Policy</p>
      <p class="authors"><span>Authors:</span><a>Jiayu Chen</a>, <a>Junbei Tang</a></p>
      <p class="abstract mathjax"><span>Abstract</span>: Full abstract text.</p>
      <p class="is-size-7"><span>Submitted</span> 31 May, 2026;
        <span>originally announced</span> June 2026.</p>
    </li>
    """

    papers = radar.parse_search_results(html)

    assert papers == [
        {
            "arxiv_id": "2605.13111",
            "title": "Pyramid Forcing: Head-Aware Pyramid KV Cache Policy",
            "authors": ["Jiayu Chen", "Junbei Tang"],
            "categories": ["cs.CV", "cs.LG"],
            "submitted_date": "2026-05-31",
            "abstract": "Full abstract text.",
            "paper_url": "https://arxiv.org/abs/2605.13111",
            "pdf_url": "https://arxiv.org/pdf/2605.13111",
        }
    ]


def test_parse_abs_page_extracts_html_link_and_metadata():
    html = """
    <div class="dateline">[Submitted on 31 May 2026]</div>
    <blockquote class="abstract mathjax"><span class="descriptor">Abstract:</span>
      A method for long-context LLM inference.
    </blockquote>
    <div class="authors"><span class="descriptor">Authors:</span>
      <a>Author One</a>, <a>Author Two</a>
    </div>
    <td class="tablecell subjects">Machine Learning (cs.LG); Artificial Intelligence (cs.AI)</td>
    <a href="https://arxiv.org/html/2605.00001v1">HTML</a>
    """

    meta = radar.parse_abs_page(html, "2605.00001")

    assert meta["submitted_date"] == "2026-05-31"
    assert meta["abstract"] == "A method for long-context LLM inference."
    assert meta["authors"] == ["Author One", "Author Two"]
    assert meta["categories"] == ["cs.LG", "cs.AI"]
    assert meta["html_url"] == "https://arxiv.org/html/2605.00001v1"


def test_code_link_detection_marks_open_source_candidates():
    urls = [
        "https://example.com/project",
        "https://github.com/acme/paper-code",
        "https://huggingface.co/spaces/acme/demo",
    ]

    result = radar.classify_code_links(urls)

    assert result["has_open_source_demo_code"] == "yes"
    assert result["code_links"] == [
        "https://github.com/acme/paper-code",
        "https://huggingface.co/spaces/acme/demo",
    ]


def test_code_link_detection_ignores_arxiv_html_infrastructure_links():
    urls = radar.extract_urls(
        """
        <a href="https://github.com/arXiv/html_feedback/issues">feedback</a>
        <a href="https://github.com/brucemiller/LaTeXML/issues">latexml</a>
        <a href="https://github.com/acme/research-demo</a">broken html</a>
        """
    )

    result = radar.classify_code_links(urls)

    assert result["code_links"] == ["https://github.com/acme/research-demo"]


def test_state_marks_only_new_papers_and_records_report(tmp_path):
    state_path = tmp_path / "processed_papers.json"
    state_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "papers": {
                    "2605.00001": {
                        "status": "analyzed",
                        "last_analyzed_at": "2026-06-01T10:00:00+08:00",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    papers = [{"arxiv_id": "2605.00001"}, {"arxiv_id": "2605.00002"}]
    new_papers = radar.filter_unprocessed(papers, radar.load_state(state_path))
    radar.mark_analyzed(
        state_path,
        [{"arxiv_id": "2605.00002", "title": "New paper"}],
        "notes/papers/arxiv-ai-daily/2026-06-01 arxiv ai papers.md",
        now="2026-06-01T10:30:00+08:00",
    )

    saved = json.loads(state_path.read_text(encoding="utf-8"))
    assert new_papers == [{"arxiv_id": "2605.00002"}]
    assert saved["papers"]["2605.00002"]["status"] == "analyzed"
    assert saved["papers"]["2605.00002"]["title"] == "New paper"
    assert saved["papers"]["2605.00002"]["report_path"].endswith("2026-06-01 arxiv ai papers.md")


def test_parse_oai_record_uses_v1_timestamp_in_shanghai_timezone():
    xml = """
    <record xmlns="http://www.openarchives.org/OAI/2.0/">
      <header>
        <identifier>oai:arXiv.org:2606.01313</identifier>
        <datestamp>2026-06-02</datestamp>
      </header>
      <metadata>
        <arXivRaw xmlns="http://arxiv.org/OAI/arXivRaw/">
          <id>2606.01313</id>
          <title>PSG-Nav: Probabilistic Scene Graph Navigation via Multiverse Decision Making</title>
          <authors>Author One, Author Two</authors>
          <categories>cs.AI</categories>
          <abstract>Introduces a navigation method for embodied agents.</abstract>
          <version version="v1">
            <date>Sun, 31 May 2026 16:00:19 GMT</date>
          </version>
        </arXivRaw>
      </metadata>
    </record>
    """

    paper = radar.parse_oai_record(xml)

    assert paper["arxiv_id"] == "2606.01313"
    assert paper["submitted_date"] == "2026-06-01"
    assert paper["oai_datestamp"] == "2026-06-02"
    assert paper["oai_first_version_gmt"] == "Sun, 31 May 2026 16:00:19 GMT"


def test_build_oai_query_days_caps_at_current_shanghai_date():
    target = dt.date(2026, 6, 1)
    now = dt.datetime(2026, 6, 2, 10, 0, tzinfo=radar.TIMEZONE)

    days = radar.build_oai_query_days(target, now=now, lookahead_days=2)

    assert days == [dt.date(2026, 6, 1), dt.date(2026, 6, 2)]


def test_render_report_replaces_placeholder_with_auto_analysis(tmp_path):
    input_path = tmp_path / "papers.json"
    output_path = tmp_path / "report.md"
    input_path.write_text(
        json.dumps(
            {
                "target_date": "2026-06-01",
                "counts": {"candidate_papers": 1, "new_papers": 1, "already_processed": 0},
                "source_status": {"oai": {"status": "ok"}},
                "papers": [
                    {
                        "arxiv_id": "2606.01314",
                        "title": "SkillSmith: Co-Evolving Skills and Tools for Self-Improving Agent Systems",
                        "authors": ["A", "B"],
                        "categories": ["cs.AI", "cs.LG"],
                        "tags": ["AI Agent", "LLM推理能力"],
                        "paper_url": "https://arxiv.org/abs/2606.01314",
                        "abstract": "This paper studies self-improving agent systems. It introduces a co-evolution workflow for skills and tools. Experiments show strong gains on agent benchmarks.",
                        "code_links": ["https://github.com/acme/skillsmith"],
                        "has_open_source_demo_code": "yes",
                        "fulltext_status": "skipped_scale",
                        "fulltext_chars": 0,
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = radar.render_report(
        argparse.Namespace(
            input=str(input_path),
            output=str(output_path),
        )
    )

    report = output_path.read_text(encoding="utf-8")
    assert result == 0
    assert "AI 分析待填" not in report
    assert "- 核心问题：" in report
    assert "- 重要性评级：" in report
    assert "S/A/B/C" not in report
