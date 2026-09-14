# Task C — Data and Orchestration weekly digest

- **Name:** `Data and Orchestration weekly digest`
- **Frequency:** Weekly, Sunday, 10:00
- **Folder:** *(leave empty — must stay remote-only)*
- **Connector:** enable GitHub in this task's own config

## Prompt

```
You maintain the data and orchestration section of the Jekyll site at
github.com/pvelua/pvelua.github.io (default branch: master). It covers the
substrate underneath AI systems: data infrastructure, retrieval and memory, agent
orchestration frameworks, and the protocols that wire models to tools and to each
other.

Work exclusively through the GitHub connector. You have no local files, no clone,
and no shell.

STEP 1 — LOAD THE CONTRACT
Read _automation/snippet-spec.md from master. It is the authoritative format
specification. Where anything below conflicts with it, the spec file wins.

Pay particular attention to two parts. Section 8 defines what clears the bar for
`data`, and contains the `ai` / `data` boundary table - read that table before you
judge any candidate, because roughly a third of what you find will sit near the
line. Section 6 defines the ledger layout.

STEP 2 — LOAD WHAT IS ALREADY COVERED
Read EVERY ledger: _data/covered-ai.yml, _data/covered-data.yml,
_data/covered-breakthroughs.yml, and the legacy _data/covered.yml. Dedup is
global - a URL used by any digest is spent for all of them, and the ai job runs
the day before you, so its picks will already be in its ledger.

Then list _posts/ and read the two most recent files ending in -data-weekly.md.

Build one exclusion set from all of it. Exclude by underlying event, not by URL.

STEP 3 — RESEARCH
Determine today's date in America/Los_Angeles. Cover the seven days ending today.
Search and fetch across the sources in section 9 of the spec under `data`, plus
anything else credible you find. Follow through to primary sources - a project's
own release notes beat an article about the release.

Apply the bar in section 8. Then apply the boundary test to every survivor: is
this about what a model can do, or about how information reaches a model and how
models are composed into systems? Only the second belongs to you. If a story is a
genuine coin flip, leave it to `ai` and note that you did so.

Shortlist, then keep the 4 to 7 most significant. If fewer than 4 clear the bar,
publish what you have and say so in the summary. Do not pad, and do not reach
across the boundary to fill space.

For every item you keep, you must have actually opened the page this run. If you
did not fetch it, drop it.

STEP 4 — COMPOSE
Write the digest as _posts/YYYY-MM-DD-data-weekly.md, dated today, category
[data], titled "Data and Orchestration Weekly — D Month YYYY", in exactly the
format the spec defines. Paraphrase everything. At most one quote per snippet,
under 15 words, and prefer none. No ampersand in the title.

Append every URL you used to _data/covered-data.yml only, preserving the existing
entries and the file's comments. Do not touch the other ledgers.

STEP 5 — OPEN A PULL REQUEST
Create branch news/data-YYYY-MM-DD off master. If that branch already exists,
update it in place rather than opening a second pull request.

Commit exactly two files: the new digest and _data/covered-data.yml. Change
nothing else.

Open a pull request into master. Title it the same as the digest title. In the
body: the summary line, then a bullet per item giving the headline and the source
domain, then a line stating how many URLs you added to covered-data.yml, then a
"Left to the AI digest" section listing any boundary calls you made and why.

Do not merge. A human reviews and merges.

FINALLY
Report back: the pull request URL, the item count, the boundary calls you made,
and anything that went wrong or looked off.
```

---

## Amendment needed to Task A (AI and LLM weekly digest)

The AI job now has a neighbour, so its prompt needs two changes. Everything else
stays as it is.

**In Step 2**, replace the ledger instruction with:

```
Read EVERY ledger: _data/covered-ai.yml, _data/covered-data.yml,
_data/covered-breakthroughs.yml, and the legacy _data/covered.yml. Dedup is
global. Append your new URLs to _data/covered-ai.yml only.
```

**Add to Step 3**, after the existing bar instruction:

```
Then apply the `ai` / `data` boundary test from section 8 of the spec. Data
infrastructure, retrieval architectures, agent orchestration frameworks and
interop protocols belong to the data digest, not to you - even when a model lab
published them. You run first in the week, so on a genuine coin flip the story is
yours; but do not claim clear data stories just because you saw them first.
```

**In Step 4**, change `_data/covered.yml` to `_data/covered-ai.yml`.

---

## Amendment needed to Task B (Breakthroughs weekly digest)

Only the ledger paths change.

**In Step 2**, read all four ledger files as above. **In Step 5**, append to
`_data/covered-breakthroughs.yml` instead of `_data/covered.yml`, and commit that
file rather than the old one.
