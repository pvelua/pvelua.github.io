---
title: AI and LLM Weekly
description: What's new in AI and LLMs, in short snippets
permalink: /news/ai/
---

# AI and LLM Weekly

Short snippets on model releases, research, tooling and industry moves — each one a couple of sentences and a link to the source. Collected weekly.

[Subscribe via RSS]({{ '/feed/ai.xml' | relative_url }})

{% include news-styles.html %}
{% assign digests = site.categories.ai %}
{% if digests.size == 0 %}

*No digests published yet.*

{% else %}
{% assign latest = digests.first %}
<div class="digest-latest">
<p class="digest-eyebrow">Latest digest</p>
<h2><a class="digest-link" href="{{ latest.url | relative_url }}">{{ latest.title }} <span class="arrow" aria-hidden="true">&rarr;</span></a></h2>
<p class="digest-meta">{{ latest.date | date: "%-d %B %Y" }}{% if latest.item_count %} &middot; {{ latest.item_count }} items{% endif %}</p>
{{ latest.content }}
</div>
{% if digests.size > 1 %}
<div class="digest-archive">
<h2>Earlier digests</h2>
{% for post in digests offset: 1 %}
<h3><a class="digest-link" href="{{ post.url | relative_url }}">{{ post.title }} <span class="arrow" aria-hidden="true">&rarr;</span></a></h3>
<p class="digest-meta">{{ post.date | date: "%-d %B %Y" }}{% if post.item_count %} &middot; {{ post.item_count }} items{% endif %}</p>
<p>{{ post.summary }}</p>
{% endfor %}
</div>
{% endif %}
{% endif %}
