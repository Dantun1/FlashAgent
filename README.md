# FlashAgent

## *🥈 Second Place Winner for "Best Research" in Anthropic Edinburgh AI Expo 2026. 🥈*

## Optimised Cache Architecture for Agentic Applications

### Description

We built a complete caching pipeline for agentic applications. Current ReAct agents spend significant compute reasoning for every query they receive. Previous attempts at addressing this issue include Semantic Caching ([GPTCache](https://github.com/zilliztech/GPTCache)) and Plan Caching ([Agentic Plan Caching](https://arxiv.org/abs/2506.14852)). While semantic caching completely fails to generalize across variable changes, Agentic Plan Caching (APC) solves this by retrieving an abstracted plan template and passing it to a lightweight LLM for variable adaptation. However, this LLM-based adaptation phase introduces a significant latency bottleneck, typically taking several hundred milliseconds to over a second per cache hit.

**FlashAgent** solves this bottleneck by entirely removing the generative LLM from the adaptation step.

### How It Works

Instead of relying on a language model to adapt cached plans, FlashAgent treats agentic cache retrieval as a deterministic parsing problem:

1. **Ultra-Fast Variable Extraction:** We use GLiNER (Generalist Model for Named Entity Recognition) to instantly extract distinct entities and variables from the incoming user query.
2. **Vector-Based Template Retrieval:** The core semantic intent of the query is embedded and matched against a vector database of successful, abstracted plan templates (blueprints).
3. **Deterministic Prompt Construction** The variables extracted by GLiNER are deterministically appended to the agent prompt after the generic plan. This allows optimal KV Cache usage by SGLang as it results in late divergence.

### Key Results & Trade-Offs



<img width="1063" height="567" alt="Screenshot 2026-03-05 at 20 31 45" src="https://github.com/user-attachments/assets/3af810ee-1fdd-42a7-b76a-9f9ee23f7d2d" />

*Hit rate + savings across dataset of queries with identical intent but different variables*

<img width="971" height="569" alt="Screenshot 2026-03-05 at 20 34 52" src="https://github.com/user-attachments/assets/d9a16892-2f30-4f71-ad8e-977632290fb2" />

*Latency to produce reasoning of no cache vs cache hit in FlashAgent*

<img width="1116" height="566" alt="Screenshot 2026-03-05 at 20 38 04" src="https://github.com/user-attachments/assets/3e5f404f-5584-46a1-af97-2d782025a564" />

*Accuracy across complex FinanceBench Queries*

<img width="1033" height="447" alt="Screenshot 2026-03-08 at 01 39 11" src="https://github.com/user-attachments/assets/d593a765-3834-48cc-b46e-1a6405fd58aa" />

*Cumulative Impact of FlashAgent across FinanceBench Queries*

### Highlights

By replacing the LLM decoder ring with deterministic logic, FlashAgent operates on the extreme edge of the efficiency-accuracy Pareto frontier for agent applications. 

* **~100ms Cache Hit Latency:** A 10x to 20x speedup compared to LLM-based template adaptation frameworks.
* **Near-Zero Marginal Cost:** Completely eliminates token generation costs during a cache hit. Vector lookups and GLiNER inference are computationally trivial.
* **Complex Question Accuracy** Plan based caching enables our 8B param local model to accurately answer complex data dependent tasks within seconds.

**The Trade-Off:** This architecture is designed for high-volume, structured agentic workflows. Because the variable plugging is deterministic, it sacrifices some of the semantic flexibility of an LLM. It favors raw speed and cost-efficiency over the ability to dynamically adapt a template to mismatched variable counts or complex edge cases.

---

### Quick Start

**1. Prerequisites**
* Python 3.10+
* SGLang inference engine running your model of choice.
* Google Cloud Vertex AI configured (required for embeddings/agent fallback).

**2. Installation**
Clone the repository and install the required dependencies:
```bash
git clone [https://github.com/yourusername/FlashAgent.git](https://github.com/Dantun1/FlashAgent.git)
cd FlashAgent
pip install -r requirements.txt
```
**3. Environment Setup**
Create a `.env` file in the root directory and add your Google Cloud Project ID for Vertex AI authentication:
```env
PROJECT_ID="hephaestus-488415"
```
*Note: Ensure your local environment is authenticated with Google Cloud by running `gcloud auth application-default login`.*

**4. Start Your Inference Engine**
FlashAgent defaults to expecting an OpenAI-compatible server running on port `30000`. Start your SGLang server with your chosen model (e.g., Llama 3.1 8B FP8 - what we used):
```bash
python -m sglang.launch_server --model-path neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8 --port 30000
```

**5. Basic Usage Example**

We have kept test scripts in the testscripts directory, you should be able to run these to see just how effective it is.

*(Note: Cache hit/miss metrics and token usage are automatically logged to `kv_tracking.csv` and the timestamped `.log` files in your directory.)*











