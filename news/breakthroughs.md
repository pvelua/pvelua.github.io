---
title: Breakthroughs in Mathematics, CS and Physics
description: Significant results, collected as they land
permalink: /news/breakthroughs/
---

# Breakthroughs

Significant results in mathematics, computer science and physics. The bar is deliberately high, so quiet weeks produce nothing at all rather than filler.

[Subscribe via RSS]({{ '/feed/breakthroughs.xml' | relative_url }})

For the earlier hand-written round-ups, see [2026 so far]({{ '/maths-2026-breakthroughs.html' | relative_url }}) and [2025]({{ '/maths-2025-breakthroughs.html' | relative_url }}).

{% include news-styles.html %}
{% assign digests = site.categories.breakthroughs %}
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
{% assign earlier = digests | slice: 1, 500 %}
{% assign years = earlier | group_by_exp: "post", "post.date | date: '%Y'" %}
<div class="digest-archive">
<h2>Earlier digests</h2>
{% for year in years %}
<h3>{{ year.name }}</h3>
{% for post in year.items %}
<h4><a class="digest-link" href="{{ post.url | relative_url }}">{{ post.title }} <span class="arrow" aria-hidden="true">&rarr;</span></a></h4>
<p class="digest-meta">{{ post.date | date: "%-d %B %Y" }}{% if post.item_count %} &middot; {{ post.item_count }} items{% endif %}</p>
<p>{{ post.summary }}</p>
{% endfor %}
{% endfor %}
</div>
{% endif %}
{% endif %}
