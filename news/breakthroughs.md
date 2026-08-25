---
title: Breakthroughs in Mathematics, CS and Physics
description: Significant results, collected as they land
permalink: /news/breakthroughs/
---

# Breakthroughs

Significant results in mathematics, computer science and physics. The bar is deliberately high, so quiet weeks produce nothing at all rather than filler.

[Subscribe via RSS]({{ '/feed/breakthroughs.xml' | relative_url }})

For the earlier hand-written round-ups, see [2026 so far]({{ '/maths-2026-breakthroughs.html' | relative_url }}) and [2025]({{ '/maths-2025-breakthroughs.html' | relative_url }}).

{% assign digests = site.categories.breakthroughs %}
{% if digests.size == 0 %}
*No digests published yet.*
{% else %}
{% assign years = digests | group_by_exp: "post", "post.date | date: '%Y'" %}
{% for year in years %}
## {{ year.name }}

{% for post in year.items %}
### [{{ post.title }}]({{ post.url | relative_url }})

<small>{{ post.date | date: "%-d %B %Y" }}{% if post.item_count %} · {{ post.item_count }} items{% endif %}</small>

{{ post.summary }}

{% endfor %}
{% endfor %}
{% endif %}
