---
name: "orin-tutorial-style"
description: "Write, revise, and iteratively improve source-backed Chinese tutorial notes in the C:\\AIWorks\\Orin knowledge base, especially AI/LLM engineering tutorials that need Obsidian Markdown frontmatter, numbered headings, stable long-term wording, high-quality imagegen diagrams, explicit source links, and user-feedback-driven style refinement."
---

# Orin Tutorial Style

## Core Workflow

Use this skill when creating or revising Orin tutorial notes, especially under `notes/`, when the user expects a polished long-term learning artifact rather than a one-off answer.

1. Read repo rules first:
   - Load `AGENTS.md` for Orin-wide writing and git conventions;
   - Open the target note and preserve any user-provided outline or existing structure;
   - If this skill is being refined, also read `references/tutorial-style-playbook.md`;

2. Build evidence before writing:
   - Use web search when the topic depends on external technical facts;
   - Prefer a balanced source set: papers/arXiv, official docs, GitHub repositories or community docs, and high-quality blogs/tutorials;
   - Keep official facts, paper claims, community practice, and local interpretation clearly separated;

3. Plan the teaching path:
   - Start with the real problem the concept solves;
   - Give a compact definition after the reader has a mental hook;
   - Move from intuition to mechanism, then to implementation and decision rules;
   - Include common misconceptions and deployment/practice guidance when the topic is engineering-facing;

4. Use AI-era visuals deliberately:
   - Prefer generated raster diagrams via the `imagegen` skill for spatial, architectural, conceptual, or comparison-heavy explanations;
   - Copy final project-bound images into the note's local `imgs/` folder before referencing them;
   - Use Mermaid/PlantUML only when exact syntax, deterministic graph text, or source-controlled diagram diffability matters more than visual quality;
   - Do not leave generated images only under `$CODEX_HOME/generated_images`;

5. Write in Orin tutorial style:
   - Use Simplified Chinese unless the user asks otherwise;
   - Add Obsidian YAML frontmatter with `tags`, `updated`, and `description`;
   - Use numbered H2/H3 headings such as `## 1. ...` and `### 1.1 ...`;
   - Avoid process narration such as "本次检索" or "刚才生成";
   - Make the note self-contained outside the current conversation;
   - Use concise tables where they sharpen contrast, not as decoration;
   - End list items with `；` when following Orin list punctuation conventions;

6. Close with sources and validation:
   - Add a final reference section with links actually used;
   - Verify image paths exist and are repo-local;
   - Run `git diff --check -- <target-file>` on edited Markdown files;
   - Run `quick_validate.py` when this skill itself changes;

## Style Refinement Loop

When the user critiques the output, treat the critique as training signal for this skill.

1. Identify whether the feedback is about:
   - structure;
   - depth;
   - tone;
   - visual strategy;
   - source standards;
   - Markdown/Obsidian format;
   - examples and analogies;

2. Apply the feedback to the current note first.

3. If the user explicitly wants the preference to become reusable, update `references/tutorial-style-playbook.md` with a short durable rule and, if needed, tighten this `SKILL.md`.

4. Keep the skill concise. Put detailed evolving style rules in the reference file, not in the main skill body.

## Reference

Read `references/tutorial-style-playbook.md` whenever the task is a substantial tutorial, a rewrite after user dissatisfaction, or a new note where style fidelity matters.
