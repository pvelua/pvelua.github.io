[← Back to Home](/)

# Neo4j's Cypher Quick Start

Cypher is Neo4j's declarative graph query language. This Quick Start page provide infomation that you need to get started quering or adding data to a Knowledge Graph.

Table of Contents:
- [Cypher Syntax Guide](#cypher-syntax-guide)
- [Questions & Answers](#questions--answers):
    - [Is 'p' in "MATCH (p:Person {name: 'Alice'}) RETURN p" just an alias?](#is-p-in-match-pperson-name-alice-return-p-just-an-alias)
    - [In "MATCH (p:Person {name: 'Alice'})" can I use a wildcard to match several name?](#is-p-in-match-pperson-name-alice-return-p-just-an-alias)
    - [Query didn't return any data - how to troublkeshoot?](#query-didnt-return-any-data---how-to-troublkeshoot)
    - [Explain query: MATCH (a:Person {name: 'Alice'})-[:KNOWS*1..3]->(friend) RETURN friend.name](#explain-query-match-aperson-name-alice-knows13-friend-return-friendname)
    - [What is the rule for using single quote 'value' vs. double "value" in property values?](#what-is-the-rule-for-using-single-quote-value-vs-double-value-in-property-values)
    - [What relationship's directions are supported in MATCH?](#what-relationships-directions-are-supported-in-match)
    - [Can I use multiple predicates sequentially?](#can-i-use-multiple-predicates-sequentially)
    - [Can I combine several predicates with AND or OR in MATCH?](#can-i-combine-several-predicates-with-and-or-or-in-match)

---

## Cypher Syntax Guide

### Basic Syntax

Cypher uses ASCII art to represent graph patterns:
- `()` represents nodes
- `[]` represents relationships
- `-->` shows relationship direction

### Creating Data

**Create a node:**
```cypher
CREATE (p:Person {name: 'Alice', age: 30})
```

**Create a relationship:**
```cypher
MATCH (a:Person {name: 'Alice'})
MATCH (b:Person {name: 'Bob'})
CREATE (a)-[:KNOWS]->(b)
```

**Create everything at once:**
```cypher
CREATE (a:Person {name: 'Alice'})-[:KNOWS]->(b:Person {name: 'Bob'})
```

### Reading Data

**Find all nodes with a label:**
```cypher
MATCH (p:Person)
RETURN p
```

**Find specific nodes:**
```cypher
MATCH (p:Person {name: 'Alice'})
RETURN p
```

**Find relationships:**
```cypher
MATCH (a:Person)-[r:KNOWS]->(b:Person)
RETURN a.name, b.name
```

**Pattern matching with depth:**
```cypher
MATCH (a:Person {name: 'Alice'})-[:KNOWS*1..3]->(friend)
RETURN friend.name
```

### Filtering with WHERE

```cypher
MATCH (p:Person)
WHERE p.age > 25
RETURN p.name, p.age
```

### Updating Data

**Update properties:**
```cypher
MATCH (p:Person {name: 'Alice'})
SET p.age = 31
```

**Add labels:**
```cypher
MATCH (p:Person {name: 'Alice'})
SET p:Developer
```

### Deleting Data

**Delete relationships:**
```cypher
MATCH (a:Person)-[r:KNOWS]->(b:Person)
DELETE r
```

**Delete nodes (must delete relationships first):**
```cypher
MATCH (p:Person {name: 'Alice'})
DETACH DELETE p  // DETACH deletes relationships too
```

### Common Clauses

- `MATCH` - find patterns
- `CREATE` - create nodes/relationships
- `RETURN` - specify what to output
- `WHERE` - filter results
- `SET` - update properties
- `DELETE` - remove data
- `ORDER BY` - sort results
- `LIMIT` - restrict number of results

### Quick Tips

1. Use `RETURN *` to see all matched variables
2. Always create indexes for frequently queried properties
3. Use `EXPLAIN` or `PROFILE` before queries to optimize performance
4. Node labels use PascalCase, relationship types use UPPER_CASE

---

## Questions & Answers

### Is 'p' in "MATCH (p:Person {name: 'Alice'}) RETURN p" just an alias?

Yes, exactly! `p` is a variable (or alias) that represents the matched node. You can think of it like a variable in programming. When you write:
```cypher
MATCH (p:Person {name: 'Alice'})
RETURN p
```

The `p` is just a name you're giving to the node that matches the pattern. You could use any valid identifier:
```cypher
MATCH (person:Person {name: 'Alice'})
RETURN person
```
```cypher
MATCH (x:Person {name: 'Alice'})
RETURN x
```
```cypher
MATCH (alice:Person {name: 'Alice'})
RETURN alice
```

All of these do the same thing. The variable lets you:

1. **Reference the node later** in the same query:
```cypher
MATCH (p:Person {name: 'Alice'})
SET p.age = 31
RETURN p
```
2. **Access its properties:**
```cypher
MATCH (p:Person {name: 'Alice'})
RETURN p.name, p.age
```
3. **Use it in relationship patterns:**
```cypher
MATCH (p:Person {name: 'Alice'})-[:KNOWS]->(friend)
RETURN p, friend
```

### In "MATCH (p:Person {name: 'Alice'})" can I use a wildcard to match several name?

Yes! You can use wildcards, but not directly in the curly braces `{}`. You need to use the `WHERE` clause with string matching operators.

#### Using Wildcards

**Starts with:**
```cypher
MATCH (p:Person)
WHERE p.name STARTS WITH 'Ali'
RETURN p
```

**Ends with:**
```cypher
MATCH (p:Person)
WHERE p.name ENDS WITH 'ice'
RETURN p
```

**Contains:**
```cypher
MATCH (p:Person)
WHERE p.name CONTAINS 'lic'
RETURN p
```

#### Regular Expressions

For more complex patterns, use regex with `=~`:

```cypher
MATCH (p:Person)
WHERE p.name =~ 'Ali.*'
RETURN p
```

```cypher
MATCH (p:Person)
WHERE p.name =~ '(?i)alice'  // case-insensitive
RETURN p
```

#### Why Not in Curly Braces?

The `{name: 'Alice'}` syntax is for **exact matches** only. It's actually shorthand for:

```cypher
MATCH (p:Person)
WHERE p.name = 'Alice'
RETURN p
```

For any pattern matching or wildcards, you need the full `WHERE` clause with the appropriate string operators.

**Note:** These string operations are case-sensitive by default. Use `(?i)` in regex or convert to lowercase for case-insensitive matching:

```cypher
MATCH (p:Person)
WHERE toLower(p.name) CONTAINS 'alice'
RETURN p
```

### Query didn't return any data - how to troublkeshoot?

Let's troubleshoot why a query returns no results. Here are the most common causes:

#### 1. Check if the Label Exists

First, verify that you have nodes with the `Entity` label:
```cypher
MATCH (e:Entity)
RETURN count(e)
```
If this returns 0, you don't have any `Entity` nodes.

#### 2. Check Property Names

The property might be named differently. See what properties exist:
```cypher
MATCH (e:Entity)
RETURN e LIMIT 5
```

Or list all property keys:
```cypher
MATCH (e:Entity)
RETURN keys(e)
LIMIT 10
```

#### 3. Check Property Values

See what values `entity_type` actually has:
```cypher
MATCH (e:Entity)
RETURN DISTINCT e.entity_type
```

Maybe it's:
- `'author'` (lowercase)
- `'AUTHOR'` (uppercase)
- Stored differently altogether

#### 4. Case-Sensitive Match

If the value is lowercase, try:
```cypher
MATCH (e:Entity)
WHERE toLower(e.entity_type) = 'author'
RETURN e LIMIT 25
```

#### 5. Check All Your Data

See what labels and nodes you actually have:
```cypher
CALL db.labels()
```

```cypher
MATCH (n)
RETURN labels(n), count(n)
```

Try these diagnostic queries - they will help identify exactly where the mismatch is.

### Explain query: MATCH (a:Person {name: 'Alice'})-[:KNOWS*1..3]->(friend) RETURN friend.name

**Query syntax breakdown**:
```cypher
MATCH (a:Person {name: 'Alice'})-[:KNOWS*1..3]->(friend)
RETURN friend.name
```

Let's break this down piece by piece:

- `(a:Person {name: 'Alice'})` - Find a node labeled `Person` with name 'Alice', call it `a`
- `-[:KNOWS*1..3]->` - Follow `KNOWS` relationships, with **variable length**:
  - `*1..3` means "1 to 3 hops away"
  - `*1` would be direct friends only
  - `*2` would be friends-of-friends only
  - `*1..3` captures friends, friends-of-friends, and friends-of-friends-of-friends
  - `->` indicates direction (outgoing from Alice)
- `(friend)` - The destination nodes at the end of these paths, called `friend`
- `RETURN friend.name` - Return the names of all these people

**Example:**
```
Alice -KNOWS-> Bob -KNOWS-> Carol -KNOWS-> David

This query would return: Bob (1 hop), Carol (2 hops), David (3 hops)
```

If you want unlimited depth, use `*` without numbers: `-[:KNOWS*]->` (but be careful - this can be slow on large graphs!)

### What is the rule for using single quote 'value' vs. double "value" in property values?

**In Cypher (Neo4j), you MUST use single quotes for string values:**

```cypher
✅ MATCH (p:Person {name: 'Alice'})
❌ MATCH (p:Person {name: "Alice"})
```

**Double quotes are for identifiers (property names, labels, variables) that contain special characters or spaces:**

```cypher
// Normal identifier - no quotes needed
MATCH (p:Person)
RETURN p.name

// Identifier with spaces - needs double quotes
MATCH (p:Person)
RETURN p.`first name`  // or p."first name"

// Label with special characters
MATCH (p:`Special-Label`)
RETURN p
```

**The Rule:**
- **Single quotes `'...'`** → String values/literals
- **Double quotes `"..."` or backticks `` `...` ``** → Identifiers with special characters
- **No quotes** → Normal identifiers (labels, properties, variables)

This is different from some SQL databases where both can be used for strings, so it's easy to mix up!


### What relationship's directions are supported in MATCH?

You have three options for direction:
```cypher
// Directed: left to right (outgoing)
MATCH (p:Person)-[r:KNOWS]->(f:Friend)

// Directed: right to left (incoming)
MATCH (p:Person)<-[r:KNOWS]-(f:Friend)

// Undirected: either direction
MATCH (p:Person)-[r:KNOWS]-(f:Friend)
```

The arrow `>` or `<` determines direction, not left-right position!

**Example:**
```cypher
// Find people Alice knows
MATCH (a:Person {name: 'Alice'})-[:KNOWS]->(friend)

// Find people who know Alice
MATCH (a:Person {name: 'Alice'})<-[:KNOWS]-(friend)
```

### Can I use multiple predicates sequentially?

Absolutely! You can chain patterns together:

```cypher
MATCH (p:Person)-[:KNOWS]->(f:Friend)-[:WORKS_AT]->(o:Organization {name: 'Anthropic'})
RETURN p, f, o
```

This finds: Person → knows → Friend → works at → Anthropic

You can chain as many as you need:
```cypher
MATCH (p:Person)-[:KNOWS]->(f)-[:LIVES_IN]->(c:City)-[:LOCATED_IN]->(country:Country)
RETURN p.name, country.name
```

### Can I combine several predicates with AND or OR in MATCH?

You **cannot** use AND/OR directly in the MATCH pattern, but you use multiple MATCH clauses or WHERE:

**Option A: Multiple MATCH clauses (implicit AND)**
```cypher
MATCH (p:Person)-[:KNOWS]->(alice:Person {name: 'Alice'})
MATCH (p)-[:WORKS_AT]->(o:Organization {name: 'Google'})
RETURN p
```

**Option B: Single MATCH with WHERE (AND)**
```cypher
MATCH (p:Person)-[:KNOWS]->(alice:Person),
      (p)-[:WORKS_AT]->(o:Organization)
WHERE alice.name = 'Alice' AND o.name = 'Google'
RETURN p
```

**Option C: OR conditions**
```cypher
MATCH (p:Person)-[:KNOWS]->(friend:Person)
WHERE friend.name = 'Alice' OR friend.name = 'Bob'
RETURN p
```

**Option D: Complex conditions**
```cypher
MATCH (p:Person)-[:KNOWS]->(friend),
      (p)-[:WORKS_AT]->(org)
WHERE (friend.name = 'Alice' OR friend.name = 'Bob')
  AND org.name = 'Google'
  AND p.age > 25
RETURN p
```

**Key point:** The comma `,` in MATCH acts as AND - both patterns must match the same starting node(s).

---