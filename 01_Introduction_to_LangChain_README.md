# Introduction to LangChain (Beginner to Pro)

> **What you will learn in this README**
>
> 1. What LangChain is and **why it came into the picture**
> 2. What **LLMOps** is and the LLMOps tools in the LangChain ecosystem
> 3. **Alternatives to LangChain** and why we still learn LangChain
> 4. **Alternatives to LangGraph** and which one to choose
> 5. **LangChain evolution** (from 2022 to the 1.x era)
> 6. **Completion model vs Chat model**
> 7. **Legacy LangChain** and **LCEL (LangChain Expression Language)**

---

## 0. New terms used in this README (read this first)

| Term | Simple meaning |
|---|---|
| **LLM** (Large Language Model) | An AI model trained on huge text data that can understand and generate language (GPT, Claude, Gemini, Llama). |
| **Provider** | The company/service that hosts a model: OpenAI, Anthropic, Google, Groq, AWS Bedrock, Azure OpenAI, etc. |
| **API** | A "door" your code uses to send a request to a service and get a reply. |
| **Framework** | A ready-made toolkit with a structure and building blocks, so you don't write everything from scratch. |
| **Abstraction** | Hiding complex details behind a simple common interface. Example: one `.invoke()` method that works for any model. |
| **Integration** | A ready-made connector between LangChain and an outside service (a model, database, vector store, tool). |
| **Vendor lock-in** | Being stuck with one provider because your code only works with that provider. |
| **Prompt** | The input/instruction you give a model. |
| **RAG** (Retrieval-Augmented Generation) | Fetching relevant information (from documents/databases) and giving it to the LLM so it can answer from that data. |
| **Tool** | A function the AI can ask to run, such as "search the web" or "query a database". |
| **Tool calling** | The model replies with a structured request like "call `get_weather` with city=Hyderabad", and your code runs it. |
| **Agent** | An AI system that uses an LLM to decide which steps/tools to use to reach a goal, in a loop. |
| **Orchestration** | Controlling the order and flow of steps (who runs when, what happens next, what happens on failure). |
| **State** | The data an app/agent keeps track of while working (messages so far, results, decisions). |
| **Observability** | Being able to see what happened inside your app: each step, input, output, time, cost, errors. |
| **Tracing** | Recording every step of one request end to end, so you can debug it later. |
| **Evaluation (evals)** | Testing the quality of AI outputs in a systematic way, like test cases for AI. |
| **Deployment** | Putting your app on a server so real users can use it. |
| **Deprecated** | Still works for now, but no longer recommended and may be removed in the future. |
| **Middleware** | Code that runs "in between" steps to add behavior (e.g., trimming history, adding guardrails, asking human approval). |

---

## 1. What is LangChain?

**LangChain is an open-source framework (Python and JavaScript/TypeScript) for building applications and agents powered by LLMs.**

In one line:

> LangChain gives you **standard building blocks** (models, prompts, messages, tools, retrievers, agents) and **hundreds of integrations**, so you can connect LLMs to your data and tools without writing all the glue code yourself.

### 1.1 Why did LangChain come into the picture?

Think about late 2022 / early 2023. ChatGPT had just exploded. Developers wanted to build real apps like:

- "Chat with my PDF"
- "Answer questions from my company's database"
- "A bot that searches the web and then summarizes"

But calling an LLM API directly gave you only **text in → text out**. Everything else you had to build yourself:

| Problem developers faced | What LangChain provides |
|---|---|
| Every provider (OpenAI, Anthropic, Google...) has a **different API** | One common interface: `model.invoke(...)` works for all |
| Switching models meant **rewriting code** | Change one line to switch provider (avoids vendor lock-in) |
| Building prompts with variables was messy string work | **Prompt templates** |
| LLMs don't remember previous messages | **Message history / memory** patterns |
| LLMs don't know your private data | **Document loaders, text splitters, embeddings, vector stores, retrievers** (RAG building blocks) |
| LLMs can't take actions by themselves | **Tools** and **agents** |
| Model outputs are free text, hard to use in code | **Output parsers** and **structured output** |
| Hard to debug multi-step AI apps | Integrated **tracing/observability** with LangSmith |

### 1.2 Real-world analogy

> An LLM is like a powerful **engine**.
> LangChain is like a **car kit**: steering, wheels, fuel pipes, dashboard — so you can turn that engine into a usable vehicle quickly, and even swap the engine (model) without rebuilding the car.

### 1.3 Without vs with LangChain

**Without LangChain (direct provider SDK):**

```python
from openai import OpenAI

client = OpenAI()
resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is S3?"}],
)
print(resp.choices[0].message.content)
```

If you switch to Anthropic or Groq, the SDK, method names and response format all change.

**With LangChain:**

```python
from langchain.chat_models import init_chat_model

model = init_chat_model("openai:gpt-4o-mini")      # swap to "anthropic:..." or "groq:..."
print(model.invoke("What is S3?").content)
```

The rest of your app stays the same.

### 1.4 Main building blocks (high-level)

| Building block | What it does |
|---|---|
| **Chat models** | Talk to LLMs with one common interface |
| **Messages** | System / Human / AI / Tool messages that form a conversation |
| **Prompt templates** | Reusable prompts with placeholders |
| **Output parsers / structured output** | Turn model output into JSON / Python objects |
| **Document loaders & text splitters** | Read files and break them into chunks |
| **Embeddings & vector stores** | Convert text to vectors and search by meaning |
| **Retrievers** | Fetch relevant chunks for RAG |
| **Tools** | Functions the model can call |
| **Agents** (`create_agent`) | LLM + tools in a loop to reach a goal |
| **Middleware** | Hooks to customize agent behavior (guardrails, summarization, human approval) |

### 1.5 The LangChain ecosystem (family of products)

```text
                ┌──────────────────────────────────────────┐
                │              LangChain ecosystem         │
                └──────────────────────────────────────────┘
   BUILD (open source)                    OPERATE (LLMOps platform)
   ───────────────────                    ─────────────────────────
   langchain-core  → base interfaces      LangSmith Observability → tracing
   langchain       → agents, high-level   LangSmith Evaluation    → testing quality
   langchain-xxx   → integrations         LangSmith Prompt Hub    → manage prompts
   (openai, groq, anthropic, aws...)      LangSmith Deployment    → run agents in prod
   langgraph       → low-level agent      LangGraph Studio        → visual agent debugging
                     orchestration
   Deep Agents     → ready-made "agent
                     harness" for long,
                     complex tasks
```

- **LangChain** = quick, high-level way to build LLM apps and agents.
- **LangGraph** = low-level framework and runtime for custom, controllable, long-running agents (state, loops, branching, human-in-the-loop, durable execution). LangChain's agents run on top of LangGraph.
- **LangSmith** = the platform to observe, evaluate, and deploy those apps.

---

## 2. What is LLMOps?

### 2.1 Simple definition

**LLMOps (Large Language Model Operations)** = the practices and tools to **build, test, deploy, monitor, and improve LLM applications reliably in production**.

It is the LLM version of **MLOps** (and MLOps is the ML version of **DevOps**).

```text
DevOps  → reliably ship normal software
MLOps   → reliably ship ML models (training, versioning, monitoring)
LLMOps  → reliably ship LLM apps (prompts, traces, evals, cost, safety)
```

### 2.2 Why do we need LLMOps?

LLM apps behave differently from normal code:

| Normal software | LLM app |
|---|---|
| Same input → same output | Same input can give **different** outputs |
| Bugs show as errors | "Bugs" can be **wrong-but-confident answers** (hallucinations) |
| Cost is mostly fixed infra | Cost depends on **tokens per request** |
| Unit tests check exact output | Need **evaluations** that judge quality |
| Code changes cause regressions | Even a **prompt change or model upgrade** can cause regressions |

So you need to answer questions like:
- Which step of my agent failed, and why?
- How many tokens and how much money did this request cost?
- Did my new prompt make answers better or worse?
- Is latency (response time) acceptable?

### 2.3 LLMOps lifecycle

```text
Design prompt / agent
        ↓
Trace & debug  (see every step)
        ↓
Evaluate       (test on a dataset, compare versions)
        ↓
Deploy         (put into production)
        ↓
Monitor        (latency, cost, errors, user feedback)
        ↓
Improve        (update prompts/models) → back to Evaluate
```

### 2.4 LLMOps tools in the LangChain ecosystem (short and quick)

| Tool | What it is | One-line example |
|---|---|---|
| **LangSmith (Observability / Tracing)** | Records every step of your app: inputs, outputs, tool calls, tokens, latency, errors | "Why did my RAG bot answer wrong?" → open the trace and see the retrieved chunks were wrong |
| **LangSmith Evaluation** | Run your app on a test dataset and score results (human, rule-based, or LLM-as-judge) | Compare prompt v1 vs v2 on 100 questions before release |
| **LangSmith Prompt Hub / Prompt management** | Store, version and test prompts outside code; prompt playground | Product team edits a prompt without a code change |
| **LangSmith Monitoring & Feedback** | Dashboards for cost, latency, error rate; collect user thumbs-up/down | Alert when error rate jumps after a model change |
| **LangSmith Deployment** (formerly **LangGraph Platform / LangGraph Cloud**) | Managed infrastructure to run agents in production with durable execution, streaming, memory, human-in-the-loop, scaling | Deploy a long-running support agent with one command |
| **LangGraph Studio** | Visual IDE to see and debug an agent graph step by step | Watch which node ran, pause, edit state, re-run |
| **LangServe** *(legacy)* | Older way to expose a LangChain chain as a REST API (built on FastAPI) | Mostly replaced by LangSmith Deployment for new projects |

> **Enable tracing in 2 lines** (LangSmith has a free developer tier):
> ```text
> # .env
> LANGSMITH_TRACING=true
> LANGSMITH_API_KEY=your-langsmith-key
> ```
> After this, every LangChain/LangGraph call is automatically traced in LangSmith.

> **Note:** LangSmith is **framework-agnostic** — it can trace and deploy apps not built with LangChain too.

### 2.5 LLMOps tools outside LangChain (for awareness)

| Tool | Focus |
|---|---|
| **Langfuse** | Open-source tracing, evals, prompt management (self-hostable) |
| **Arize Phoenix** | Open-source observability and evaluation |
| **Weights & Biases Weave** | Tracing and evals from the W&B ecosystem |
| **MLflow** | MLOps platform with LLM tracing/evaluation features |
| **Ragas / DeepEval** | Evaluation libraries, especially for RAG quality |
| **Azure AI Foundry / AWS Bedrock** | Cloud platforms with built-in evaluation, monitoring and guardrails |

---

## 3. Alternatives to LangChain

| Alternative | What it is | Best for |
|---|---|---|
| **Direct provider SDKs** (OpenAI, Anthropic, Google) | Call the model API directly | Simple apps, full control, one provider |
| **LlamaIndex** | Framework focused on connecting LLMs to your data | Data-heavy RAG, document indexing and querying |
| **Haystack** (deepset) | Pipeline-based framework for search, RAG and agents | Production RAG/search pipelines |
| **Microsoft Semantic Kernel / Microsoft Agent Framework** | Microsoft's SDKs for AI apps and agents (C#, Python, Java) | .NET/Azure-centric enterprises |
| **DSPy** | "Programming, not prompting" — optimizes prompts automatically | Research, prompt optimization |
| **Pydantic AI** | Type-safe agent framework from the Pydantic team | Python developers who love type safety and validation |
| **Vercel AI SDK** | TypeScript toolkit for AI apps | Web/frontend developers (Next.js, React) |
| **Spring AI** | AI framework for Java/Spring | Java enterprises |

### 3.1 Why learn LangChain over others?

1. **Most widely used and most recognized** — very common in job descriptions, tutorials, courses and interviews.
2. **Provider-agnostic** — learn once, use with OpenAI, Anthropic, Gemini, Groq, Bedrock, Azure, local models.
3. **Huge integration library** — models, vector databases, loaders, tools are ready.
4. **Natural path to agents** — LangChain (`create_agent`) → LangGraph (custom agents) → LangSmith (LLMOps). One ecosystem covers the full journey.
5. **Concepts transfer** — prompts, messages, tools, RAG, memory, agents are the same ideas in every framework. Once you know them in LangChain, other frameworks feel easy.
6. **Stable since 1.0** — LangChain and LangGraph reached 1.0 in October 2025 with a promise of no breaking changes until 2.0.

### 3.2 Honest criticism (good to know for interviews)

- Older versions were called **too abstract and heavy** ("too much magic"), with fast-changing APIs.
- Many classes were deprecated over time, so old tutorials often don't run today.
- LangChain 1.0 addressed much of this by **slimming down the package** and focusing on the core agent loop.
- For a very simple single-provider app, the **direct SDK may be enough**.

---

## 4. Alternatives to LangGraph

LangGraph is for **agent orchestration**: building agents as a **graph** of steps (nodes) connected by edges, with shared **state**, loops, branching, checkpoints and human-in-the-loop.

| Framework | Core idea (mental model) | Strengths | Weaknesses |
|---|---|---|---|
| **LangGraph** | Explicit **graph + state machine** | Fine-grained control, durable execution, checkpointing, human-in-the-loop, model-agnostic, strong production use | Steeper learning curve, more code for simple tasks |
| **CrewAI** | A **crew** of role-based agents (researcher, writer, reviewer) doing tasks | Very fast to prototype, easy to understand | Less control for complex branching/recovery logic |
| **Microsoft Agent Framework** (successor that unifies **AutoGen** + **Semantic Kernel**) | Agents + workflows, enterprise features | Strong Azure/.NET support, enterprise focus | Newer; best fit inside the Microsoft ecosystem |
| **AutoGen / AG2** | Agents **talking to each other** in a conversation | Good for multi-agent research and experiments | Harder to make deterministic for production |
| **OpenAI Agents SDK** | Lightweight agents with **handoffs** and guardrails | Fastest to start, simple code | Best experience with OpenAI models |
| **Google ADK** (Agent Development Kit) | Google's agent framework | Good fit for Gemini / Google Cloud | Google-ecosystem centric |
| **AWS Strands Agents / Bedrock Agents** | AWS's agent SDK / managed agents | Good fit for AWS-based teams | AWS-ecosystem centric |
| **Claude Agent SDK** | Anthropic's agent harness (powers Claude Code style agents) | Strong for coding/computer-use style agents | Best with Claude models |
| **LlamaIndex Workflows** | Event-driven workflows for agents | Great when the agent is data/RAG-heavy | Smaller agent ecosystem than LangGraph |
| **Pydantic AI** | Type-safe agents, with graph support | Clean Python, validation built in | Smaller ecosystem |

### 4.1 Which one is better and why?

There is **no single best** — pick based on the problem:

| Your situation | Good choice |
|---|---|
| Complex, branching, long-running agent that must survive failures and pause for human approval | **LangGraph** |
| Quick prototype of a "team of agents" with clear roles | **CrewAI** |
| You are fully on OpenAI and want the simplest start | **OpenAI Agents SDK** |
| Enterprise on Azure / .NET | **Microsoft Agent Framework** |
| Data/RAG-heavy assistant | **LlamaIndex** (+ workflows) |
| Simple tool-calling agent | **LangChain `create_agent`** (runs on LangGraph underneath) |

**Recommendation for you as a learner:** learn **LangChain → LangGraph**. Reasons:
- LangGraph is widely treated as the **production default for complex, stateful agents** and is model-agnostic.
- It teaches the **core agent concepts explicitly** (state, nodes, edges, loops, checkpoints, human-in-the-loop). Once you understand these, CrewAI or OpenAI Agents SDK will take you days, not weeks.
- As a data engineer, the **graph / DAG thinking** will feel familiar (similar to Airflow DAGs, but with loops and LLM decisions).

---

## 5. LangChain Evolution

| Period | What happened | Key idea |
|---|---|---|
| **Oct 2022** | LangChain launched as an open-source project by Harrison Chase (just before ChatGPT went viral) | Chain LLM calls with prompts and tools |
| **Late 2022 – 2023** | Explosive growth; many **Chains** (`LLMChain`, `RetrievalQA`, `ConversationalRetrievalChain`, `SequentialChain`) and **legacy agents** (`initialize_agent`, ReAct, `AgentExecutor`) | "A chain for every use case" |
| **2023** | Mostly **completion models** early on, then shift to **chat models**; **LangSmith** launched for tracing/evals | From text-in/text-out to messages |
| **Aug 2023** | **LCEL (LangChain Expression Language)** introduced — the `|` pipe syntax | Compose components like Lego blocks |
| **Late 2023 – Jan 2024** | Package split: **`langchain-core`** (base), **`langchain-community`** (community integrations), partner packages like **`langchain-openai`**; **LangChain 0.1** (first stable) | Lighter, modular packages |
| **Early 2024** | **LangGraph** released for cyclical, stateful agents | Agents as graphs |
| **2024** | **LangChain 0.2 / 0.3**; old chains and `AgentExecutor` steadily deprecated in favor of LCEL and LangGraph; native **tool calling** everywhere | Move agent logic to LangGraph |
| **2025** | LangGraph Platform (deployment) becomes generally available; **MCP** (Model Context Protocol) adapters; Deep Agents | Production agents |
| **Oct 2025** | **LangChain 1.0** and **LangGraph 1.0**: new `create_agent` (built on LangGraph), **middleware**, **standard content blocks**, slimmer package; old chains moved to **`langchain-classic`**; LangGraph Platform renamed **LangSmith Deployment** | Focus on the core agent loop; stability (no breaking changes until 2.0) |
| **2026 (now)** | 1.x versions continue with incremental improvements | Mature, stable ecosystem |

### 5.1 The evolution in one picture

```text
2022-23            2023-24              2024              2025 → now
Chains     →       LCEL pipes     →     LangGraph   →     LangChain 1.x
(LLMChain,         (prompt | model      (stateful         create_agent on
 RetrievalQA)       | parser)            agent graphs)     LangGraph + middleware
```

### 5.2 Why this matters to you

- Many YouTube videos and blogs use **old syntax** (`LLMChain`, `initialize_agent`, `from langchain.llms import OpenAI`). If code fails, it is often because it is legacy.
- Always check: **"Is this tutorial for LangChain 1.x?"**

---

## 6. Completion Model vs Chat Model

### 6.1 Completion model (text-in → text-out)

A **completion model** takes **one plain text string** and **continues** it.

```text
Input:  "The capital of France is"
Output: " Paris."
```

- Original style of LLMs (e.g., GPT-3 era, `gpt-3.5-turbo-instruct`).
- No built-in concept of "who is speaking".
- You must design the whole prompt text yourself, even for conversations.
- In LangChain these were called **LLMs** (e.g., `OpenAI` class).

```python
# Completion-style model in LangChain (legacy style, rarely used now)
from langchain_openai import OpenAI

llm = OpenAI(model="gpt-3.5-turbo-instruct")
text = llm.invoke("The capital of France is")
print(text)            # returns a plain string
```

### 6.2 Chat model (messages-in → message-out)

A **chat model** takes a **list of messages with roles** and returns an **AI message**.

```text
Input:
  System: You are a helpful geography tutor.
  Human:  What is the capital of France?
Output:
  AI:     The capital of France is Paris.
```

- Modern standard (GPT-4o, Claude, Gemini, Llama chat models).
- Understands roles: **system**, **human/user**, **AI/assistant**, **tool**.
- Supports **tool calling**, **structured output**, **multimodal** input (images, audio) depending on the model.
- In LangChain: `ChatOpenAI`, `ChatAnthropic`, `ChatGroq`, `init_chat_model(...)`.

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

chat = ChatOpenAI(model="gpt-4o-mini")
response = chat.invoke([
    SystemMessage(content="You are a helpful geography tutor."),
    HumanMessage(content="What is the capital of France?"),
])
print(response.content)   # response is an AIMessage object
```

### 6.3 Side-by-side

| Point | Completion model | Chat model |
|---|---|---|
| Input | One text string | List of role-labeled messages |
| Output | Plain string | `AIMessage` (content + metadata + tool calls) |
| Conversation support | Manual (you format history yourself) | Built-in with roles |
| System instructions | Just part of the text | Dedicated system message |
| Tool calling | Not native | Native |
| LangChain class | `OpenAI` (LLM) | `ChatOpenAI`, `ChatGroq`, ... |
| Today's usage | Mostly legacy | **Default for almost everything** |

> **Takeaway:** Use **chat models** for all new work. Agents depend on them.

---

## 7. Legacy LangChain and LCEL

### 7.1 What is "Legacy LangChain"?

**Legacy LangChain** means the **older style and classes** from the pre-1.0 era (mostly 2022–2024), which are now **deprecated** or moved out of the main package.

Common legacy items:

| Legacy item | What it did | Modern replacement |
|---|---|---|
| `LLMChain` | Prompt + model as one "chain" object | `prompt | model` (LCEL) or direct `model.invoke` |
| `SequentialChain`, `SimpleSequentialChain` | Run chains one after another | LCEL pipes or LangGraph |
| `RetrievalQA`, `ConversationalRetrievalChain` | Ready-made RAG chains | Custom retrieval with LCEL, or an agent with a retriever tool |
| `ConversationChain`, `ConversationBufferMemory` | Built-in chat memory | Message history / LangGraph checkpointer memory |
| `initialize_agent`, `AgentExecutor` | Old agent runner | `create_agent` (LangChain 1.x) or LangGraph |
| `from langchain.llms import OpenAI` | Old import paths | `from langchain_openai import ChatOpenAI` |
| `.run()`, `.predict()`, `.__call__()` | Old ways to execute | `.invoke()`, `.stream()`, `.batch()` |

In LangChain 1.0, these old chains/agents were moved to a separate package, **`langchain-classic`**, so old projects can keep running while the main `langchain` package stays lean.

**Legacy example (do not use for new code):**

```python
# ❌ Legacy style
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

prompt = PromptTemplate.from_template("Explain {topic} simply.")
chain = LLMChain(llm=OpenAI(), prompt=prompt)
print(chain.run(topic="Kafka"))
```

**Why it was replaced:**
- Too many specialized chain classes to learn.
- Hidden logic ("magic") made debugging hard.
- Hard to customize when your use case didn't fit the prebuilt chain.
- No native streaming/async/batch consistency across all classes.

### 7.2 What is LCEL (LangChain Expression Language)?

**LCEL** is a way to build chains by **connecting components with the pipe `|` operator**, like Linux pipes.

```text
Linux:   cat file.txt | grep error | wc -l
LCEL:    prompt       | model      | parser
```

Every component is a **Runnable** (a standard object that has `.invoke()`, `.stream()`, `.batch()` and async versions), so when you pipe them together, the whole chain automatically supports invoke, streaming, batching and async.

**LCEL example:**

```python
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()

prompt = ChatPromptTemplate.from_template("Explain {topic} to a beginner in 3 lines.")
model = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()        # turns AIMessage into a plain string

chain = prompt | model | parser

print(chain.invoke({"topic": "Apache Kafka"}))

for chunk in chain.stream({"topic": "Docker"}):   # streaming works automatically
    print(chunk, end="", flush=True)
```

Flow:

```text
{"topic": "Apache Kafka"}
        ↓  prompt  → fills template → messages
        ↓  model   → calls LLM      → AIMessage
        ↓  parser  → extracts text  → "Kafka is..."
```

### 7.3 Is LCEL legacy?

This is a common point of confusion:

- LCEL **replaced** legacy chains like `LLMChain` (so compared to them, LCEL is the "newer" way).
- LCEL **is still part of `langchain-core` in 1.x** and still works well for **simple, linear pipelines** (prompt → model → parser, simple RAG).
- But for **agents** and anything with loops, branching, memory or human approval, the recommended way is now **`create_agent`** (LangChain 1.x) or **LangGraph**, not big complex LCEL chains.

```text
Legacy chains (LLMChain, RetrievalQA)   → avoid for new code
LCEL pipes                              → fine for simple linear flows
create_agent / LangGraph                → recommended for agents and complex flows
```

### 7.4 Same task, three generations

```python
# 1) Legacy (2023)
chain = LLMChain(llm=llm, prompt=prompt)
chain.run(topic="RAG")

# 2) LCEL (2023–2024 onward)
chain = prompt | model | StrOutputParser()
chain.invoke({"topic": "RAG"})

# 3) LangChain 1.x agent (when tools/decisions are needed)
from langchain.agents import create_agent
agent = create_agent(model="openai:gpt-4o-mini", tools=[search_docs])
agent.invoke({"messages": [{"role": "user", "content": "Explain RAG using our docs"}]})
```

---

## 8. Quick Revision (Interview / Exam Ready)

- **LangChain** = open-source framework to build LLM apps/agents with standard interfaces and many integrations; avoids vendor lock-in.
- **Why it exists:** raw LLM APIs only do text in/out; LangChain adds prompts, memory, RAG, tools, agents, parsing, and tracing.
- **LangGraph** = low-level graph-based orchestration and runtime for stateful, controllable, long-running agents.
- **LangSmith** = LLMOps platform: tracing, evaluation, prompt management, monitoring, deployment.
- **LLMOps** = operating LLM apps in production: trace → evaluate → deploy → monitor → improve.
- **LangChain alternatives:** direct SDKs, LlamaIndex, Haystack, Semantic Kernel / Microsoft Agent Framework, DSPy, Pydantic AI.
- **LangGraph alternatives:** CrewAI, Microsoft Agent Framework/AutoGen, OpenAI Agents SDK, Google ADK, AWS Strands, Claude Agent SDK, LlamaIndex Workflows.
- **Choose LangGraph** for complex, stateful, production agents; **CrewAI** for fast role-based prototypes; **OpenAI Agents SDK** for simple OpenAI-only agents.
- **Evolution:** Chains (2022) → LCEL (2023) → LangGraph (2024) → LangChain/LangGraph 1.0 with `create_agent` + middleware (Oct 2025).
- **Completion model** = text in → text continuation out (legacy). **Chat model** = role-labeled messages in → AI message out (modern default).
- **Legacy LangChain** = `LLMChain`, `RetrievalQA`, `initialize_agent`, `AgentExecutor`, `.run()`; now in `langchain-classic`.
- **LCEL** = `prompt | model | parser` pipe syntax built on Runnables; still fine for simple linear flows, but agents should use `create_agent` / LangGraph.

---

## 9. What's Next in This Series

```text
01 Introduction to LangChain                         ← you are here
02 Environment Setup & Connecting to LLMs
03 Prompt Templates
04 Output Parsers & Structured Output
05 Chains with LCEL
06 Tools & Tool Calling
07 Agents with create_agent
08 Memory
09 RAG with LangChain
10 LangGraph basics
```
