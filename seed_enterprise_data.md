[← Back to Home](/)

# Seed Data for Enterprise Systems

## Oracle Applications

## Salesforce

Building a **GraphRAG** (Retrieval-Augmented Generation) system for Salesforce is particularly challenging because standard RAG often fails to capture the "Account Hierarchy" or the "Opportunity-Contact-Activity" web that defines enterprise sales.

To seed your local database with the density required to test multi-hop traversal (e.g., *"Show me all VPs involved in lost deals who are also currently active on high-priority cases"*), I recommend a tiered approach.

### 1. The "Golden Standard" for Density: Salesforce Snowfakery
Salesforce open-sourced a tool called **[Snowfakery](https://github.com/SFDO-Tooling/Snowfakery)**. Unlike a simple CSV generator, it allows you to define relationships in a YAML recipe so that a "Contact" created is automatically linked to an "Account," and "Tasks" are child records of both.

* **Why it works:** You can specify that 20% of Contacts should have `OpportunityContactRole` entries across different Accounts, creating the "density" you need.
* **Where to find it:** Check the [Salesforce Recipes GitHub](https://github.com/SFDO-Tooling/Snowfakery) for their standard CRM recipe. 
* **Action:** I can help you draft a `recipe.yml` that matches your specific volume requirements (50 Accounts, 200 Contacts, etc.) if you'd like to run it locally.

### 2. Pre-Packaged Sample Apps (Quick Start)
If you don't want to generate data from scratch, Salesforce maintains sample applications with pre-populated datasets designed specifically to show off relationships:

| Dataset | Best For... | Source |
| :--- | :--- | :--- |
| **Dreamhouse-LWC** | Basic Account/Contact/Case relations. | [GitHub](https://github.com/trailheadapps/dreamhouse-lwc) |
| **Easy Spaces**    | Complex scheduling and lead-to-contact flows. | [GitHub](https://github.com/trailheadapps/easy-spaces-lwc) |
| **E-Bikes**        | Product-heavy data (Products, Orders, Cases). | [GitHub](https://github.com/trailheadapps/ebikes-lwc) |

### 3. Public Industry Datasets (CSV)
For the 3 industry verticals (Finance, Tech, Healthcare), you can pull "Backbone" data from these sources and map them to Salesforce objects:
* **Healthcare:** Use the [CMS.gov Synthetic Data](https://www.cms.gov/data-research/statistics-trends-and-reports/medicare-claims-synthetic-public-use-files) or **Synthea** to get realistic patient/provider names to map to Accounts/Contacts.
* **Financial Services:** The [LendingClub Dataset](https://www.kaggle.com/datasets/wordsforthewise/lending-club) is excellent for seeding "Opportunities" with realistic amounts and "Closed Lost/Won" statuses.
* **Technology:** Use the [Crunchbase Open Data](https://www.crunchbase.com/home) to get real company names and industry categorizations for your 50+ Accounts.

---

### Suggested Local Graph Schema
Since you are using GraphRAG, your local seed should likely follow this structure to ensure your queries can "hop":



* **Node: Account** (Properties: Industry, Tier, Revenue)
* **Node: Contact** (Properties: Title, Department)
* **Edge: [WORKS_FOR]** (Contact → Account)
* **Edge: [INVOLVED_IN]** (Contact → Opportunity) — *This is where you'll find your "Winning vs. Losing" insights.*
* **Edge: [LOGGED_AGAINST]** (Activity → Case/Opportunity)

### Next Step
Would you like me to **write a Python script** using a library like `Faker` that generates these CSVs for you, specifically ensuring the "Relationship Density" requirements (like linking one Contact to multiple Opportunities) are met?


---
The Salesforce test database should contain:
Minimum Data Volume
•      50+ Accounts across at
least 3 industry verticals (Financial Services, Technology, Healthcare
recommended).
•      200+ Contacts with
realistic distribution of titles (C-level, VP, Director, Manager) and
department assignments.
•      100+ Opportunities across
various stages (Prospecting through Closed Won/Lost) with realistic revenue
amounts.
•      150+ Cases with varied
priorities, statuses, and resolution times.
•      500+ Activities (Tasks and
Events) distributed across accounts and contacts.
•      30+ Products organized into
product families.
•      20+ Campaigns with
CampaignMember records linking to contacts.
Relationship Density Requirements
The
key differentiator of GraphRAG over standard RAG is its ability to traverse
relationships. The test data must include sufficient relationship density:
•      Contacts should be
associated with multiple accounts via Opportunity Contact Roles (not just their
primary account).
•      Some contacts should appear
on both winning and losing opportunities across different accounts.
•      Support cases should have
realistic escalation chains (reopened cases, multiple related cases per
account).
•      Activity records should
show cross-entity engagement (same contact discussed in tasks on different
opportunities).
---