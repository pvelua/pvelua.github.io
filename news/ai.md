---
title: AI & LLM Weekly
description: What's new in AI and LLMs, in short snippets
permalink: /news/ai/
---

# AI &amp; LLM Weekly

Short snippets on model releases, research, tooling and industry moves — each one a couple of sentences and a link to the source. Collected weekly.

[Subscribe via RSS]({{ '/feed/ai.xml' | relative_url }})

{% assign digests = site.categories.ai %}
{% if digests.size == 0 %}
*No digests published yet.*
{% else %}
{% for post in digests %}
## [{{ post.title }}]({{ post.url | relative_url }})

<small>{{ post.date | date: "%-d %B %Y" }}{% if post.item_count %} · {{ post.item_count }} items{% endif %}</small>

{{ post.summary }}

{% endfor %}
{% endif %}
