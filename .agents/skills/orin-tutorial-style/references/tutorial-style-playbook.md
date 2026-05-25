# Orin Tutorial Style Playbook

## 1. Desired Reader Experience

A strong Orin tutorial should feel like a durable learning artifact inside a personal AI knowledge base. It should not read like a chat transcript, execution log, or generic encyclopedia entry.

The preferred experience is:

- The reader quickly understands why the topic matters；
- The first concept hook is concrete, not abstract；
- Definitions arrive after intuition, not before；
- Mechanism is explained with diagrams, formulas, and examples working together；
- Engineering judgment is explicit: when to use it, when not to use it, and what can go wrong；
- Sources are visible at the end so claims can be traced later；

## 2. Default Tutorial Structure

Use this structure as a starting point, then adapt to the topic.

1. Frontmatter:
   - `tags`；
   - `updated`；
   - `description` in one or two sentences；

2. Title:
   - Keep the user's requested title；
   - Use a single H1；

3. Lead-in callout:
   - Use an Obsidian `[!Quote]` or similar callout when useful；
   - State the core learning promise in plain language；

4. Problem section:
   - Explain what practical or conceptual problem forces this idea to exist；
   - Avoid starting with a dictionary definition；

5. Definition section:
   - Give the compact definition；
   - Contrast adjacent concepts immediately；

6. Mechanism section:
   - Explain the core mechanism step by step；
   - Use formulas only when they clarify the mechanism；
   - Pair formulas with prose explanation；

7. Engineering section:
   - Show how the concept appears in mainstream tools, commands, code paths, or deployment patterns；
   - Make constraints and trade-offs explicit；

8. Misconceptions section:
   - Include common wrong mental models；
   - Correct them with crisp distinctions；

9. Summary:
   - Rebuild the whole concept in one or two paragraphs；
   - Prefer a reusable mental model over a slogan；

10. References:
   - Group by papers, official docs/GitHub, and blogs/tutorials when the source set is large；

Keep top-level content chapters to 6-7 or fewer, excluding references. Too many major chapters make the tutorial feel fragmented and make it harder for readers to see the article's full map.

Keep headings short. A heading should be a navigation handle, not a compressed paragraph. Avoid sentence-length headings such as "TP usually shards hidden dimension, output dimension, head dimension, or vocab dimension"; use a short heading like "Shard Dimensions" and move the detail into body text.

## 3. Visual Strategy

Prefer generated raster images when the concept needs visual intuition.

Good imagegen candidates:

- distributed architecture；
- tensor/data flow；
- before/after comparison；
- memory layout；
- system topology；
- conceptual maps；
- lifecycle diagrams；

Figure rules:

- Every figure needs nearby reading guidance: an ordered caption, a "读图顺序", or a short explanation of the key objects and arrows；
- For long technical tutorials, three figures is usually too sparse unless the note is short. Add figures at major conceptual transitions, but only when the figure teaches a concrete point；
- Keep figure style consistent across a note；
- Use Chinese for reader-facing explanatory labels that reduce comprehension cost；
- Keep professional terms and standard phrases in English when they are the terms readers will see in docs or code, for example `Tensor Parallelism`, `all-reduce`, `hidden states`, `input activations`, `Column Parallel`, `Row Parallel`；
- Avoid all-English diagrams in Chinese tutorials unless the figure is a screenshot or an external source image；
- When a diagram teaches implementation mechanics, make output layout and communication semantics explicit. For example, distinguish `all-reduce -> replicated output` from `reduce-scatter -> sharded output`；

Use generated diagrams with caution when exact text fidelity is critical. In those cases:

- Keep in-image labels short；
- Put detailed labels and exact terminology in Markdown captions；
- Prefer formulas and tables outside the image；

Use Mermaid or PlantUML only for:

- deterministic flow charts；
- source-controlled architecture diagrams that will be edited frequently；
- exact protocol/state-machine diagrams；
- quick local sketches where visual polish is not important；

## 4. Source Standards

For technical tutorials, gather sources before drafting.

Use a mix of:

- original or canonical papers；
- official framework documentation；
- GitHub repositories or source-linked docs；
- high-quality community explanations or long-form blogs；

Do not blur source boundaries:

- "Paper says" is not the same as "framework currently supports"；
- "Official docs recommend" is not the same as "community often does"；
- "Local interpretation" should be written as explanation, not citation-backed fact；

When facts are likely to drift, prefer current official docs and record the access context through stable links.

## 5. Tone And Depth

The preferred tone is clear, structured, and technically confident.

Use:

- short conceptual paragraphs；
- concrete examples；
- small toy examples when formulas introduce a new algebraic distinction；
- comparison tables；
- precise terms with Chinese explanation；
- "why this design exists" framing；

Avoid:

- hype language；
- vague adjectives without mechanism；
- overlong historical detours；
- excessive H4/H5 nesting；
- decorative horizontal rules；
- ungrounded "事实标准" claims unless sourced and qualified；
- writing about the conversation or the agent's process；

## 6. Iteration Notes

Current durable preferences learned from Orin work:

- Tutorial notes should use numbered headings；
- Long-term notes need YAML frontmatter and stable wording；
- The user values source-backed explanations over unsupported confidence；
- When a note is empty or outline-like, expand it into a complete artifact rather than creating a separate side document；
- For AI-era tutorials, imagegen diagrams should be first-class educational assets, not optional decoration；
- Figure captions are mandatory when a diagram introduces unfamiliar objects such as `shard`, `hidden states`, `input activations`, or communication collectives；
- Bilingual diagram labels should be deliberate: Chinese lowers reading cost, English preserves standard technical terminology；
- Top-level chapters should normally stay within 6-7 content sections, excluding references；
- The quality gate should use one content-accuracy reviewer and one reader-experience reviewer, each scoring out of 100. A combined score above 190 is the target for high-quality tutorials；
- If reviewers find formula-to-concept jumps, add a tiny worked example instead of adding more abstract prose；
- For TP and similar systems tutorials, explicitly state where communication happens and what layout the output has after each collective；
- At least one realistic deployment or usage scenario should connect theory to operational judgment when the topic is engineering-facing；

When the user gives new correction, append only durable preferences here. Do not add transient task details.
