---
title: "Breakthroughs Weekly — 14 September 2026"
date: 2026-09-14
categories: [breakthroughs]
summary: "A near-linear-time algorithm delivers a new, more efficient proof of the four-color theorem, cutting coloring time from quadratic to O(n log n)."
item_count: 1
tags: [graph-theory, algorithms]
---

### [A new algorithmic proof of the four-color theorem cuts coloring time from quadratic to near-linear](https://arxiv.org/abs/2603.24880)

**arXiv** · 25 Mar 2026 · *Graph theory*

Mikkel Thorup, Carsten Thomassen, Ken-ichi Kawarabayashi, Bojan Mohar and two graduate students spent nearly a decade building a genuinely different proof of the 1976 four-color theorem, organized around 8,202 small interchangeable map patterns rather than the 1,482 used in the standard proof. Because the new patterns also cover flatter regions of a map instead of only the sharply curved ones the older approach relied on, many of them can be resolved in parallel rather than one at a time. That parallelism turns the widely used 1996 quadratic-time algorithm for four-coloring a planar map into a near-linear, O(n log n) one, a real complexity gain rather than a rephrasing of the existing theorem, and the result is due to be presented at the Foundations of Computer Science conference in November. The preprint drew little notice beyond a few blogs after it was posted in March, until Quanta Magazine profiled it this month, prompting Inria's Georges Gonthier, who formalized the original four-color proof in a computer proof assistant, to call it "really cool to see a real result for once."
