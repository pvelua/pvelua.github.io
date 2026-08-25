---
title: News
description: Weekly snippets, collected automatically
---

# News

Two weekly digests, assembled automatically and published after review.

- **[AI &amp; LLM Weekly]({{ '/news/ai/' | relative_url }})** — model releases, research, tooling and industry moves. Published most Sundays.
- **[Breakthroughs]({{ '/news/breakthroughs/' | relative_url }})** — significant results in mathematics, computer science and physics. Published only when something genuinely qualifies.

Feeds: [everything]({{ '/feed.xml' | relative_url }}) · [AI only]({{ '/feed/ai.xml' | relative_url }}) · [breakthroughs only]({{ '/feed/breakthroughs.xml' | relative_url }})

## Latest digests

{% if site.posts.size == 0 %}
*Nothing published yet.*
{% else %}
{% for post in site.posts limit: 12 %}
### [{{ post.title }}]({{ post.url | relative_url }})

<small>{{ post.date | date: "%-d %B %Y" }}{% if post.item_count %} · {{ post.item_count }} items{% endif %}</small>

{{ post.summary }}

{% endfor %}
{% endif %}
