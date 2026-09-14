---
title: Data and Orchestration Weekly
description: Data infrastructure, retrieval and agent orchestration, in short snippets
permalink: /news/data/
---

# Data and Orchestration Weekly

Short snippets on the substrate underneath AI systems — data infrastructure, retrieval and memory, agent frameworks, and the protocols that wire models to tools and to each other. Collected weekly.

[Subscribe via RSS]({{ '/feed/data.xml' | relative_url }})

{% include news-styles.html %}
{% assign digests = site.categories.data %}
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
