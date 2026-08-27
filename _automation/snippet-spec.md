# Snippet spec — the contract for automated news digests

**This file is authoritative.** The scheduled Cowork jobs read it on every run. If a
job's prompt disagrees with this file, this file wins. Tune the output by editing
here, not by editing the task prompt.

---

## 1. What a digest is

One Markdown file per category per week, containing several short snippets. Never
one file per story. Never an article-length write-up.

| Category | Filename | Items per digest | Cadence |
|---|---|---|---|
| `ai` | `_posts/YYYY-MM-DD-ai-weekly.md` | 5–8 | Weekly |
| `breakthroughs` | `_posts/YYYY-MM-DD-breakthroughs-weekly.md` | 1–4 | Weekly, **skipped only if nothing clears the bar** |

`YYYY-MM-DD` is the Sunday the digest covers up to, in `America/Los_Angeles`.

**Lookback window.** `ai` covers the 7 days ending on that Sunday. `breakthroughs`
covers a rolling **30 days**, because significant results arrive irregularly and a
7-day window misses ones that landed a fortnight earlier. The dedup ledger in
section 6 is what stops a result being reported twice, not the window.

## 2. Front matter

Exactly these keys, in this order. All are required except `tags`.

```yaml
---
title: "AI and LLM Weekly — 30 August 2026"
date: 2026-08-30
categories: [ai]
summary: "One sentence, under 25 words, naming the week's most notable item."
item_count: 6
tags: [model-releases, agents]
---
```

Rules:

- `categories` is a one-element array. Only `ai` or `breakthroughs`. Lowercase.
  Never `AI`, never a string, never two categories.
- `date` must match the date in the filename.
- `title` uses an em-dash and the date spelled as `30 August 2026`.
- **Never put `&` in a title.** Write `AI and LLM Weekly`, not `AI & LLM Weekly`.
  An ampersand gets double-escaped by jekyll-feed and shows up in feed readers
  as a literal `&amp;`.
- `summary` is plain text, no Markdown, no quotes inside.
- `item_count` must equal the number of snippets in the body.
- `layout` is set automatically by `_config.yml`. Do not include it.

## 3. Snippet format

Each snippet is exactly this shape, separated by a blank line:

```markdown
### [Headline in your own words](https://source-url)

**Source name** · 27 Aug 2026 · *Topic tag*

Two to four sentences, entirely paraphrased. What happened, what is actually new
about it, and one clause on why it matters. No hype adjectives.
```

Rules:

- The `###` heading link is the **only** link in a snippet. No "read more", no
  secondary links, no footnotes.
- The link goes to the **primary source** where one exists — the lab's own blog
  post, the arXiv abstract page, the journal announcement — not to coverage of it.
  Use secondary coverage only when there is no primary source.
- Order snippets most significant first.
- No images, no embedded video, no tables.

## 4. Copyright — non-negotiable

This site publishes without a human reading every word before it goes live, so
this rule carries more weight than usual.

- Everything is **paraphrased in your own words**. Do not lift sentences.
- At most **one quotation per snippet**, under 15 words, in quotation marks. Prefer
  zero. Quote only when the exact wording carries meaning a paraphrase would lose
  (a formal claim, a legal commitment, a named result).
- Never reproduce abstracts, opening paragraphs, or pull-quotes.
- Never reproduce a source's own structure — do not walk through its sections.

## 5. Verification — non-negotiable

- Include **only** items you actually retrieved during this run. If you did not
  fetch the page, it does not go in.
- Every URL must be one you opened this run and that resolved. Never reconstruct a
  URL from memory or from a pattern.
- If a claim is a number, a date, or an attribution, it must appear on the page you
  fetched. Otherwise leave it out.
- **Drop, never guess.** A digest with four verified items beats one with six where
  two are shaky.

## 6. Deduplication

Before researching, read `_data/covered.yml` and the two most recent digests in the
same category. Build an exclusion set.

- Exclude by **underlying event**, not by URL. The same launch covered by three
  outlets is one story, and if it appeared last week it does not appear again.
- A genuine follow-up is allowed (a paper that was rumoured now published, a
  reversal, a result now peer-reviewed). Say explicitly in the snippet what changed.
- Append every URL used to `_data/covered.yml` **in the same commit** as the digest:

```yaml
  - url: "https://example.com/story"
    title: "Headline as used in the digest"
    date: "2026-08-30"
    category: "ai"
```

## 7. Quiet weeks

- `ai`: if fewer than 5 items clear the bar, publish what you have. Do not pad. Note
  the thin week in `summary`.
- `breakthroughs`: publish whenever **at least one** item clears the bar. A digest
  with a single genuinely significant result is a good digest — do not hold it back
  waiting for company. Only when **nothing** qualifies do you skip: open no PR,
  report "nothing qualified" as the run's output, and stop.
- The bar in section 8 is the only quality gate. Never treat the item count as a
  second gate — if it cleared section 8, it publishes.

## 8. What clears the bar

**`ai`** — a new or updated model, a research result with a public artefact, a
notable tooling or standards change, a funding or policy move that changes what
builders can do. Not: opinion pieces, funding rounds with no product, listicles,
vendor marketing, rehashed benchmarks.

**`breakthroughs`** — a resolved open problem, a major conjecture settled or
disproved, a first experimental confirmation, a significant complexity-theoretic
result, a major prize citing specific work. Preprints count only if a recognised
expert has publicly vouched for the result. Not: incremental improvements, press
releases, "scientists may have found", speculative preprints with no scrutiny.

**Dating a breakthrough.** Judge the lookback window by whichever is later: when
the result was published, or when it first reached general attention through
credible coverage or expert commentary. A result that surfaces via Quanta, a
Terence Tao post, or a formalisation months after the original preprint is news at
the point it surfaced, not at the preprint's timestamp. When you admit something on
this basis, say so plainly in the snippet — "published in March, brought to wider
attention this month by ...".

Two limits on that. Prize announcements are not a route in: a Fields Medal or Abel
Prize citing work from previous years is not a breakthrough this month, and the
underlying result qualifies only if it is genuinely new on its own terms. And
coverage alone is not attention — a rehash of an old story by an aggregator does
not restart the clock. What counts is the result becoming known to people who did
not previously know it, evidenced by expert engagement.

## 9. Suggested sources

Starting points, not a closed list. Follow through to primary sources freely.

**`ai`** — Anthropic, OpenAI, Google DeepMind, Meta AI, Mistral, DeepSeek and Qwen
blogs; Hugging Face papers; arXiv `cs.LG` / `cs.CL` / `cs.AI` new listings;
VentureBeat AI; TechCrunch AI; Simon Willison's weblog; Import AI.

**`breakthroughs`** — Quanta Magazine; Nature and Science news; APS Physics /
Physical Review Letters highlights; arXiv `math.*` and `quant-ph` listings;
Terence Tao's blog; the n-Category Café; Clay, Fields, Abel and Breakthrough Prize
announcements.

## 10. Pull request conventions

- Branch: `news/<category>-YYYY-MM-DD`
- If that branch already exists (a re-run), update it rather than opening a second PR.
- PR title: same as the digest `title`.
- PR body: the `summary`, then a bullet list of each item's headline and source
  domain, then a line stating how many URLs were added to `covered.yml`.
- Two files change per PR and no others: the new digest, and `_data/covered.yml`.
  **Never modify `_config.yml`, `_layouts/`, `news/`, `index.md`, or any existing
  post.** If something looks like it needs changing there, say so in the PR body
  and leave it alone.

---

## Appendix — worked example

Copy this into `_posts/` once to smoke-test the build before wiring up the jobs,
then delete it.

```markdown
---
title: "AI and LLM Weekly — 24 August 2026"
date: 2026-08-24
categories: [ai]
summary: "Sample digest used to verify the build. Delete once the first real run lands."
item_count: 2
tags: [sample]
---

### [Example lab publishes a smaller reasoning model](https://example.com/a)

**Example Lab** · 21 Aug 2026 · *Model releases*

A sentence describing what shipped. A sentence on what is actually new about it
relative to the previous version. A clause on why it matters for people building
on top of it.

### [Open problem in graph colouring reported settled](https://example.com/b)

**Example Journal** · 22 Aug 2026 · *Combinatorics*

A sentence on the result. A sentence on the method. A note on what remains
unverified.
```
