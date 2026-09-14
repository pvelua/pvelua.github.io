---
title: News
description: Weekly snippets, collected automatically
---

# News

Three weekly digests, assembled automatically and published after review.

- **[AI and LLM Weekly]({{ '/news/ai/' | relative_url }})** — model releases, research, tooling and industry moves. Published most Saturdays.
- **[Data and Orchestration Weekly]({{ '/news/data/' | relative_url }})** — data infrastructure, retrieval and memory, agent frameworks and interop protocols. Published most Sundays.
- **[Breakthroughs in Mathematics, CS and Physics]({{ '/news/breakthroughs/' | relative_url }})** — significant results. Published only when something genuinely qualifies.

Feeds: [everything]({{ '/feed.xml' | relative_url }}) · [AI]({{ '/feed/ai.xml' | relative_url }}) · [data and orchestration]({{ '/feed/data.xml' | relative_url }}) · [breakthroughs]({{ '/feed/breakthroughs.xml' | relative_url }})

## Latest digests

{% include news-styles.html %}
{% if site.posts.size == 0 %}

*Nothing published yet.*

{% else %}
{% for post in site.posts limit: 12 %}
<h3><a class="digest-link" href="{{ post.url | relative_url }}">{{ post.title }} <span class="arrow" aria-hidden="true">&rarr;</span></a></h3>
<p class="digest-meta">{{ post.date | date: "%-d %B %Y" }}{% if post.categories.size > 0 %} &middot; {{ post.categories | join: ", " }}{% endif %}{% if post.item_count %} &middot; {{ post.item_count }} items{% endif %}</p>
<p>{{ post.summary }}</p>
{% endfor %}
{% endif %}
