[← Back to Home](/)

## Google Vertex AI - Concise Guide

### Part A: Infrastructure & Capabilities

Vertex AI is Google Cloud's fully managed, unified AI development platform. It consolidates the entire ML workflow—from data preparation to model training, deployment, and monitoring—into a single environment.

**1. The "Model Garden" (Generative AI)**

* **What it is:** A curated library of 130+ foundation models.
* **Key Capabilities:** Instant access to Google’s models (Gemini 1.5 Pro/Flash, Imagen), and open-source models (Llama 3, Mistral, Claude) via managed endpoints.
* **Infrastructure:** Models are optimized for Google's TPUs (Tensor Processing Units) and NVIDIA GPUs, handling the scaling automatically.

**2. Vector Search (Crucial for RAG)**

* **What it is:** A high-scale, low-latency vector database (formerly Matching Engine).
* **Key Capabilities:** Can search billions of vectors in milliseconds. It powers RAG (Retrieval Augmented Generation) by finding relevant context for your LLM prompts.

**3. MLOps & Pipelines**

* **What it is:** Tools to automate and monitor ML workflows.
* **Key Capabilities:**
* **Vertex Pipelines:** Orchestrates workflows (serverless execution of Kubeflow/TFX).
* **Feature Store:** A centralized repository to serve features for training and inference consistently.
* **Experiments:** Tracks parameters and metrics across different training runs.

**4. AutoSx & Custom Training**

* **What it is:** Flexible training options.
* **Key Capabilities:**
* **AutoML:** Train models on tabular, image, or video data with zero code.
* **Custom Training:** Bring your own code (Python/TensorFlow/PyTorch) and run it on managed container clusters.

---

### Part B: How to Use Vertex AI (Developer Workflow)

Since you are using `uv`, here is the modern Python-first workflow.

#### 1. Setup

Ensure the API is enabled and you are authenticated.

```bash
# 1. Enable API in your project
gcloud services enable aiplatform.googleapis.com

# 2. Authenticate (creates local credentials)
gcloud auth application-default login

```

#### 2. Installation

Add the SDK to your project.

```bash
uv add google-cloud-aiplatform langchain-google-vertexai

```

#### 3. Core Usage Patterns

**Scenario A: Using a Foundation Model (GenAI)**
This is the most common "Day 1" task—calling Gemini.

```python
import vertexai
from vertexai.generative_models import GenerativeModel

# Initialize
vertexai.init(project="your-project-id", location="us-central1")

# Load model
model = GenerativeModel("gemini-1.5-flash")

# Generate
response = model.generate_content("Explain quantum computing in one sentence.")
print(response.text)

```

**Scenario B: Deploying a Custom Model**
If you have a trained model (e.g., Scikit-Learn), you upload it to the **Model Registry** and deploy it to an **Endpoint**.

```python
from google.cloud import aiplatform

# 1. Upload Model
model = aiplatform.Model.upload(
    display_name="my-custom-model",
    artifact_uri="gs://my-bucket/model-dir",
    serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest"
)

# 2. Deploy to Endpoint (Provision Compute)
endpoint = model.deploy(
    machine_type="n1-standard-4",
    min_replica_count=1,
    max_replica_count=3
)

# 3. Predict
prediction = endpoint.predict(instances=[[1.2, 2.3, 3.4]])
```

### Summary Checklist

* **Development:** Use the **Vertex AI SDK** for Python.
* **Orchestration:** Use **Vertex Pipelines** to chain steps.
* **Data:** Use **Vertex Feature Store** or direct BigQuery integration.
* **Deployment:** Use **Endpoints** for real-time APIs or **Batch Prediction** for bulk processing.

---

### Part C: How to Use Vertex AI Session Service in Google ADK

### (a) Capabilities of `VertexAiSessionService`

The `VertexAiSessionService` is a storage backend for the ADK that moves your agent's memory out of RAM (where it is lost if the app restarts) and into **Google's managed infrastructure** (specifically the **Vertex AI Agent Engine**).

1. **Persistence (No Amnesia):** Unlike `InMemorySessionService`, this service saves conversation history to the cloud. If you redeploy your agent or it crashes, the user can pick up exactly where they left off.
2. **Stateless Scaling:** It allows your agent code to run on serverless platforms (like **Cloud Run** or **Cloud Functions**). Since the state is stored remotely in Vertex AI, you can spin up 100 copies of your agent, and they will all share the correct context for a user.
3. **Memory Integration:** It pairs natively with **Memory Bank**. While `SessionService` handles the short-term conversation turns, it works with Vertex AI to extract long-term facts (e.g., "User prefers dark mode") automatically.

---

### (b) How to Use It

To use this service, you need to have a **Vertex AI Agent Engine** instance running in your Google Cloud project (this acts as the database).

#### 1. Setup (One-time)

You need the ID of your Agent Engine. You can get this via CLI or a setup script:

```python
from google.cloud import discoveryengine_v1

# Create/Get the Agent Engine (Managed State Store)
client = discoveryengine_v1.AgentEnginesClient()
# ... (creation logic returns an ID, e.g., "my-engine-id")
```

#### 2. Implementation in ADK

Here is how you swap the default in-memory session for the Vertex AI version in your agent code.

```python
from google.adk import Agent, Runner
from google.adk.sessions import VertexAiSessionService

# 1. Configuration
PROJECT_ID = "your-project-id"
LOCATION = "us-central1"
ENGINE_ID = "your-agent-engine-id" # From step 1

# 2. Initialize the Service
# This connects your code to the managed cloud storage
session_service = VertexAiSessionService(
    project=PROJECT_ID,
    location=LOCATION,
    agent_engine_id=ENGINE_ID
)

# 3. Inject into the Runner
# Now, every time 'runner.run()' is called, it fetches state from Vertex AI
runner = Runner(
    agent=my_agent,
    session_service=session_service
)

# 4. Run (The state is now persistent)
runner.run(
    session_id="session_123", 
    input="Hello, I'm back!"
)
```

### Key Difference from Standard SDK

* **Standard `ChatSession`:** Stores history in a temporary, simple list (good for scripts).
* **ADK `VertexAiSessionService`:** Stores history in a scalable, enterprise-grade backend (good for production apps with millions of users).

---