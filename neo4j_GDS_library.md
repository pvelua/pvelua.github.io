[← Back to Home](/)

# Neo4j Graph Data Science (GDS) Library

## What is the Neo4j GDS Library?

The GDS library is a plugin for the Neo4j Graph Database that enables you to perform **analytics** and **machine learning** directly on your graph data.

While the standard Neo4j database is optimized for transactional operations (finding specific neighbors, simple patterns, or CRUD operations), the GDS library is optimized for **global computations**—analyzing the entire graph (or large subgraphs) to find broad patterns, trends, and structures.

### **1. The Three Core Pillars**

GDS capabilities can be grouped into three main categories:

* **Graph Algorithms:** A suite of 65+ highly optimized algorithms to answer specific questions about your graph structure.
* *Examples:* **PageRank** (identifying influencers), **Louvain** (finding communities), **Dijkstra** (finding shortest paths).


* **Machine Learning (ML) Pipelines:** Tools to train predictive models using the relationships in your data.
* *Examples:* **Link Prediction** (predicting future connections) and **Node Classification** (predicting missing labels or categories).


* **Graph Embeddings:** Techniques to transform nodes and relationships into numerical vectors (lists of numbers). This allows you to use your graph data as input for traditional ML models (like neural networks or random forests) outside of Neo4j.

### **2. How It Works: The In-Memory Graph**

This is the most critical concept to understand early on.
To achieve high performance, GDS **does not** run algorithms directly on the database stored on your disk. Instead, it uses an **In-Memory Graph Projection**.

1. **Read:** You select the nodes and relationships relevant to your analysis from the database.
2. **Project:** GDS loads this subgraph into RAM in a highly optimized format.
3. **Compute:** Algorithms run on this in-memory graph at lightning speed.
4. **Write/Stream:** You either stream the results to your application or write them back to the database as new properties.

### **3. Key Use Cases**

* **Fraud Detection:** Identifying rings of suspicious users.
* **Recommendation Engines:** Finding similar products or users based on behavior.
* **Supply Chain:** Finding bottlenecks or critical points of failure in a network.

---

## In-memory Graph Projections

### **1. The Concept: "A Graph within a Graph"**

Imagine your main Neo4j database is a warehouse full of detailed files. It's great for storage and retrieving specific records, but it's too cluttered for running complex math across everything at once.

A **Graph Projection** is like taking a snapshot of only the relevant connections from that warehouse and loading them into a high-speed workspace (RAM).

* **Source:** The Neo4j Database (stored on disk).
* **Destination:** The GDS In-Memory Graph (stored in RAM).
* **Benefit:** Algorithms run roughly 10-100x faster because the data structure is optimized purely for math, not storage.

### **2. How to Create a Projection (Native Projection)**

The most common way to load data is using a "Native Projection." This method selects nodes by Label and relationships by Type.

**The Syntax:**
`CALL gds.graph.project(graphName, nodeProjection, relationshipProjection, configuration)`

**Example Scenario:**
Imagine you have a social network with `Person` nodes and `FRIEND` relationships. You want to analyze who has the most influence.

**The Cypher Code:**

```cypher
CALL gds.graph.project(
  'socialGraph',       // 1. Name of the graph in memory
  'Person',            // 2. Node Label to include
  'FRIEND',            // 3. Relationship Type to include
  {                    // 4. Configuration (Optional)
    relationshipProperties: 'weight' 
  }
)
```

### **3. Essential Configuration Options**

Simply loading nodes isn't always enough. You often need to tweak the graph structure during the projection phase:

* **Relationship Orientation:**
    * **NATURAL (Default):** Keeps the direction as is (A -> B).
    * **REVERSE:** Flips the direction (B -> A).
    * **UNDIRECTED:** Treats connections as two-way (A <-> B). *Crucial for algorithms like "Community Detection" where direction might not matter.*

* **Node/Relationship Properties:**
    * If you want to calculate a weighted shortest path (e.g., distance between cities), you **must** project the property (e.g., `distance`) into memory. If you don't load it here, the algorithm can't see it later.

### **4. Managing the Lifecycle**

Since these graphs live in RAM, they consume memory. You must manage them manually.

* **Check existing graphs:**
```cypher
CALL gds.graph.list()

```

* **Delete a graph (Free up RAM):**
```cypher
CALL gds.graph.drop('socialGraph')

```

> **Pro Tip:** One of the most common beginner mistakes is changing data in the database (e.g., adding a new `FRIEND` link) and expecting the GDS algorithm to see it immediately. **It won't.** You must drop the projection and re-project the graph to update the in-memory version.

---

## Part 1: Centrality Algorithms (Finding the VIPs)

Centrality algorithms answer the question: *"Which nodes are the most important in this network?"* Importance can be defined in different ways depending on the algorithm.

### **The Big Three Centrality Algorithms**

1. **Degree Centrality:**
    * **Logic:** Counts incoming or outgoing connections.
    * **Use Case:** Finding the most connected user (popularity) or the busiest airport.

2. **PageRank:**
    * **Logic:** Measures influence. A node is important if it is connected to *other* important nodes.
    * **Use Case:** Google Search ranking, finding influential thought leaders, or key species in a food web.

3. **Betweenness Centrality:**
    * **Logic:** Measures control over information flow. It identifies nodes that act as "bridges" or "gatekeepers" on the shortest path between other nodes.
    * **Use Case:** Finding bottlenecks in a supply chain or critical transfer points in a transit network.



### **Example: Running PageRank**

Using the `socialGraph` we projected in the previous step, let's find the most influential people.

**Note on Execution Modes:**

* `stream`: Returns the results to your screen immediately (good for testing).
* `write`: Saves the results back to the Neo4j database as a node property.

**The Code (Stream Mode):**

```cypher
CALL gds.pageRank.stream('socialGraph')
YIELD nodeId, score
RETURN gds.util.asNode(nodeId).name AS person, score
ORDER BY score DESC
LIMIT 5

```

---

## Part 2: Community Detection (Finding the Groups)

Community detection algorithms answer the question: *"How is the graph clustered?"* They partition the graph into groups where nodes are densely connected to each other but sparsely connected to other groups.

### The Big Three Community Algorithms

1. **Weakly Connected Components (WCC):**
    * **Logic:** Finds islands. If you can get from Node A to Node B (ignoring direction), they are in the same component.
    * **Use Case:** Understanding if your network is fully connected or fragmented (e.g., identifying disconnected distinct groups of ID records in fraud analysis).

2. **Louvain:**
    * **Logic:** Maximizes "Modularity." It recursively groups nodes to find a hierarchy of communities. It creates very stable, distinct clusters.
    * **Use Case:** Marketing segmentation, detecting "rings" in fraud, or social circles.

3. **Label Propagation:**
    * **Logic:** A voting system. A node adopts the label (community) shared by the majority of its neighbors. It’s very fast but slightly less deterministic than Louvain.
    * **Use Case:** Rapidly clustering massive graphs.

#### Example: Running Louvain

Let's find the social circles in our graph and **write** the result back to the database so we can query it later.

**The Code (Write Mode):**

```cypher
CALL gds.louvain.write('socialGraph', {
    writeProperty: 'communityId' 
})
YIELD communityCount, modularity, strategies

```

*After running this, every node in your database will have a new property called `communityId`. You can then run `MATCH (p:Person) RETURN p.communityId, count(*)` to see the size of each group.*

---

### Summary Table: When to use what?

| Goal | Algorithm Category | Specific Algorithm | Real-World Example |
| :--- | :--- | :--- | :--- |
| **Find popularity** | Centrality | Degree | Who has the most Twitter followers? |
| **Find influence** | Centrality | PageRank | Which doctor's referral is most valued? |
| **Find bottlenecks** | Centrality | Betweenness | Which bridge failure cuts off the city? |
| **Find disjoint groups** | Community | WCC | Do we have isolated user accounts? |
| **Find tight clusters** | Community | Louvain | What are the different political factions? |

---

## Part 3: Similarity Algorithms (The Recommendation Engine)

Similarity algorithms answer the question: *"Which nodes are most alike?"* which is the backbone of any recommendation system (e.g., "People who bought X also bought Y") or Entity Resolution (finding duplicate records).

### The Two Heavyweights

1. **Node Similarity (Jaccard Index):**
* **Logic:** It looks at **shared neighbors**. If User A and User B have both purchased the same 5 items, they have a high Jaccard score.
* **Best For:** Categorical data or explicit connections.
* **Scale:** Can be computationally expensive on massive graphs because it compares every pair.

2. **K-Nearest Neighbors (KNN):**
* **Logic:** It finds the "K" most similar nodes for every node based on **properties** (like age, spend) or **node embeddings**.
* **Best For:** Continuous data or when you have run Graph Embeddings first. It uses approximation to be extremely fast on large datasets.
* **Scale:** Highly scalable; the standard for modern recommendation engines.

#### Example: finding similar users with Jaccard

Let's find pairs of people who have very similar friend groups in our `socialGraph`.

```cypher
CALL gds.nodeSimilarity.stream('socialGraph')
YIELD node1, node2, similarity
RETURN gds.util.asNode(node1).name AS Person1,
       gds.util.asNode(node2).name AS Person2,
       similarity
ORDER BY similarity DESC
LIMIT 5
```
*Note: This effectively generates a "similarity graph" where edges represent how alike two nodes are.*

---

## Part 4: Graph Machine Learning (Predicting the Future)

While algorithms like PageRank calculate a score, **Graph ML** involves training a predictive model. The GDS library allows you to build these Machine Learning Pipelines **entirely inside Neo4j**, without needing to export data to Python tools like Scikit-Learn (though you can do that too).

### The Two Main Prediction Tasks

1. **Node Classification:**
* **The Goal:** Predict a missing label or property.
* **Scenario:** You have a graph of bank transactions. Some are labeled "Fraud," others "Legit," but most are "Unknown."
* **The Method:** The model analyzes the network features (degree, PageRank) of the labeled nodes to predict the labels of the unknown nodes.


2. **Link Prediction:**
* **The Goal:** Predict if a relationship *should* exist or *will* exist in the future.
* **Scenario:** Social media "People you may know" or predicting drug-target interactions in biology.
* **The Method:** The model looks at the proximity and structural similarity of two nodes to calculate the probability of a link.



### **How the GDS ML Pipeline Works (The Workflow)**

Unlike simple algorithms, ML requires a multistep "Pipeline" approach. You define the recipe, and GDS executes it.

1. **Create Pipeline:** Initialize a blank pipeline (e.g., for Link Prediction).
2. **Add Node Properties:** Tell the pipeline which raw data to use (e.g., User Age).
3. **Add Graph Features:** This is the "secret sauce." You tell the pipeline to automatically calculate graph algorithms (like FastRP embeddings or PageRank) and add them as input features for the model.
4. **Split Data:** GDS automatically splits your graph into Train and Test sets.
5. **Train:** GDS trains a model (e.g., Logistic Regression or Random Forest) using the features.
6. **Predict:** You use the trained model to predict new data.

#### **Conceptual Example: Link Prediction Pipeline**

```cypher
// 1. Create the pipeline
CALL gds.beta.pipeline.linkPrediction.create('myPipe');

// 2. Add a graph feature (FastRP Embeddings) to the pipeline
CALL gds.beta.pipeline.linkPrediction.addNodeProperty('myPipe', 'fastRP', {
  mutateProperty: 'embedding',
  embeddingDimension: 256
});

// 3. Train the model (Using the 'socialGraph')
CALL gds.beta.pipeline.linkPrediction.train('socialGraph', {
  pipeline: 'myPipe',
  modelName: 'myTrainedModel',
  randomSeed: 42
});

// 4. Predict new links
CALL gds.beta.pipeline.linkPrediction.predict.stream('socialGraph', {
  modelName: 'myTrainedModel'
})
```

---

### **Summary Table: Complexity Levels**

| Level | Feature | What it does | Complexity |
| :--- | :--- | :--- | :--- |
| **Basic** | Centrality / Community | Describes the *current* state of the graph. | Low |
| **Intermediate** | Similarity | Compares nodes to find overlaps. | Medium |
| **Advanced** | Graph ML Pipelines | Trains models to predict *missing* data using graph features. | High |

---

## Part 5: Graph Embeddings (The Translator)

Graph Embeddings are arguably the most powerful feature in the GDS library for modern data science.

### 1. The Problem

Traditional Machine Learning models (Neural Networks, Random Forests, Logistic Regression) require input data to be in a fixed format: **vectors** (lists of numbers).

* Your graph data is nodes and relationships (topology).
* **The Conflict:** You cannot feed a "relationship" directly into a Neural Network.

### 2. The Solution

Graph Embeddings act as a translator. They learn the shape of your graph and translate every node into a **fixed-length vector** (e.g., a list of 128 numbers).

* **Crucial Concept:** Nodes that are "similar" in the graph (connected to similar people, or structurally similar) will have vectors that are mathematically close to each other.

### 3. The "Gold Standard" Algorithm: FastRP

While GDS supports several embedding algorithms (like Node2Vec or GraphSAGE), the one you will use 90% of the time is **FastRP (Fast Random Projection)**.

* **Why FastRP?**
* **Speed:** It is incredibly fast (linear time complexity). It can embed millions of nodes in seconds, whereas Node2Vec might take hours.
* **Accuracy:** It performs as well or better than slower methods for most classification/prediction tasks.
* **Properties:** It can include node properties (e.g., `Age`, `Price`) alongside the graph structure in the embedding calculations.

### 4. Example: Creating Embeddings

This code creates a new property on every node (e.g., `fastRP_embedding`) containing a list of numbers.

```cypher
CALL gds.fastRP.write('socialGraph', {
  embeddingDimension: 128,  // Length of the vector (commonly 128 or 256)
  writeProperty: 'embedding'
})
```

*Result: User A now has a property `embedding: [0.12, -0.55, 0.89, ...]`.*

---

## Part 6: The GDS Python Client (The Control Center)

Now that you know *what* GDS can do, let's look at *how* you should actually run it. While you can run everything via Cypher in the Neo4j Browser, that is rarely done in production data science.

The **GDS Python Client** (`graphdatascience`) allows you to interact with the GDS library directly from your Python environment (Jupyter Notebooks, VS Code).

### 1. Why use the Python Client?

* **Pure Pythonic Syntax:** No need to write complex Cypher strings for every command.
* **Pandas Integration:** This is the killer feature. You can pull graph results directly into a **Pandas DataFrame** for analysis or visualization.
* **Scikit-Learn Integration:** You can train a model in GDS, stream the embeddings to Python, and feed them immediately into a Scikit-Learn or PyTorch model.

### 2. How it works (Comparison)

**Option A: The Old Way (Cypher in Browser)**

```cypher
CALL gds.graph.project('myGraph', 'Person', 'KNOWS');
CALL gds.pageRank.stream('myGraph') YIELD nodeId, score RETURN nodeId, score;
```

**Option B: The New Way (Python Client)**
The Python client wraps the operations in clean objects.

```python
from graphdatascience import GraphDataScience

# 1. Connect
gds = GraphDataScience("bolt://localhost:7687", auth=("neo4j", "password"))

# 2. Project Graph (Pythonic way)
G, result = gds.graph.project(
    "socialGraph",    # Graph Name
    "Person",         # Node
    "KNOWS"           # Relationship
)

# 3. Run Algorithm (Result is a Pandas DataFrame!)
df = gds.pageRank.stream(G)

# 4. View Top 5
print(df.head(5))
```

### 3. The "Drop to Cypher" Capability

Even when using the Python client, you aren't restricted. If you need to run a specific Cypher query that isn't covered by a convenience method, you can simply run:

```python
gds.run_cypher("MATCH (n) RETURN n LIMIT 5")
```

---

### **Summary of our Learning Journey**

We have now covered the full GDS stack:

1. **Projections:** Loading data from Disk to Memory.
2. **Algorithms:** Analyzing the graph (Centrality, Community).
3. **Embeddings:** Translating the graph into vectors for ML.
4. **Python Client:** The tool to orchestrate it all and integrate with the rest of your Data Science stack.

---

## A Complete End-to-End "Recipe" using the GDS Python Client

This script simulates a real-world workflow:

1. **Connect** to the database.
2. **Seed** it with some dummy social data.
3. **Project** the graph into memory.
4. **Analyze** importance (Centrality).
5. **Translate** nodes to vectors (Embeddings).
6. **Find** similar users based on those vectors (KNN).

### The Prerequisites

You will need the python library installed:

```bash
pip install graphdatascience pandas

```

### The Full Recipe (Python)

```python
import pandas as pd
from graphdatascience import GraphDataScience

# 1. CONNECT
# Replace with your actual URI and password
gds = GraphDataScience("bolt://localhost:7687", auth=("neo4j", "password"))

# ---------------------------------------------------------
# 2. SEED DATA (Resetting DB for this example)
# We create a small network: 
# - Group A: Alice, Bob, Charlie (connected to each other)
# - Group B: Dave, Eve, Frank (connected to each other)
# - Bridge: Charlie connects to Dave
# ---------------------------------------------------------
gds.run_cypher("""
    MATCH (n) DETACH DELETE n
""")
gds.run_cypher("""
    CREATE (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}), (c:Person {name: 'Charlie'}),
           (d:Person {name: 'Dave'}), (e:Person {name: 'Eve'}), (f:Person {name: 'Frank'})
    CREATE (a)-[:KNOWS]->(b), (b)-[:KNOWS]->(c), (c)-[:KNOWS]->(a),
           (d)-[:KNOWS]->(e), (e)-[:KNOWS]->(f), (f)-[:KNOWS]->(d),
           (c)-[:KNOWS]->(d)
""")
print("✅ Data seeded successfully.")

# ---------------------------------------------------------
# 3. GRAPH PROJECTION
# Loading the 'Person' nodes and 'KNOWS' relationships into memory
# ---------------------------------------------------------
G, result = gds.graph.project(
    "socialGraph",      # Graph Name
    "Person",           # Node Label
    "KNOWS"             # Relationship Type
)
print(f"✅ Graph 'socialGraph' projected with {G.node_count()} nodes.")

# ---------------------------------------------------------
# 4. ANALYZE (Centrality)
# Who is the most critical bridge? (Betweenness Centrality)
# ---------------------------------------------------------
df_centrality = gds.betweenness.stream(G)

# Join with actual node properties to see names
# Note: 'gds.util.asNode' is a helper to fetch node props
names = gds.run_cypher("MATCH (n:Person) RETURN id(n) as nodeId, n.name as name")
df_centrality = df_centrality.merge(names, on="nodeId")

print("\n--- Top Influencers (Bridges) ---")
print(df_centrality[['name', 'score']].sort_values(by='score', ascending=False))

# ---------------------------------------------------------
# 5. EMBEDDINGS (FastRP)
# Turn every node into a list of numbers (Vector)
# We 'mutate' the in-memory graph to add the embedding property there
# ---------------------------------------------------------
gds.fastRP.mutate(
    G,
    embeddingDimension=4,   # Small dimension for this small example
    mutateProperty='embedding'
)
print("\n✅ Embeddings created in memory.")

# ---------------------------------------------------------
# 6. SIMILARITY (KNN)
# Use the embeddings to find who is similar to whom
# ---------------------------------------------------------
df_similarity = gds.knn.stream(
    G,
    nodeProperties=['embedding'], # Use the vector we just created
    topK=1                        # Find the 1 closest match for each person
)

# Merge names again for readability
df_similarity = df_similarity.merge(names, left_on="node1", right_on="nodeId").rename(columns={'name': 'Person_A'})
df_similarity = df_similarity.merge(names, left_on="node2", right_on="nodeId").rename(columns={'name': 'Person_B'})

print("\n--- Who is most similar to whom? (Based on Structure) ---")
print(df_similarity[['Person_A', 'Person_B', 'similarity']])

# ---------------------------------------------------------
# 7. CLEANUP
# Free up the RAM
# ---------------------------------------------------------
G.drop()
print("\n✅ In-memory graph dropped.")

```

### What just happened?

1. **Projection:** We loaded the data.
2. **Betweenness:** You likely saw **Charlie** or **Dave** at the top. Because Charlie connects Group A to Group B, he is structurally critical.
3. **FastRP:** We didn't define "features" manually. The algorithm looked at the shape of the graph (Group A vs Group B) and generated vectors.
4. **KNN:** The system successfully identified that Alice is similar to Bob (same group), and Dave is similar to Eve (same group), purely based on the embeddings we generated.

---

