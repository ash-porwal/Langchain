# 02 — LangChain: Talking to LLMs, Prompts, Chains & Environment Setup

> **Beginner → Pro guide in easy language.**
> Every new word is explained the first time it appears (look for 💡).
>
> **Roadmap of this README**
>
> | # | Topic |
> |---|---|
> | 1 | How to talk with an LLM using LangChain |
> | 2 | Connect with OpenAI (the most used provider) |
> | 3 | Connect with alternative LLMs (open-source APIs) |
> | 4 | Talk to an LLM in different languages |
> | 5 | Introduction to prompts (prompt templates) |
> | 6 | Chains and output parsers |
> | 7 | Local setup vs Google Colab |
> | 8 | Environment setup (conda / venv / uv) + API keys |
> | 9 | Project README: how to run, streaming, configuration, chat model examples, chat vs completion model |
> | ➕ | Extra: errors & fixes, revision notes, link to Agentic AI |

---

# 1. How to Talk with an LLM Using LangChain

## 1.1 What does "talking to an LLM" actually mean?

💡 **LLM (Large Language Model)** = an AI model that has read a huge amount of text and can now understand and write language. Examples: GPT (OpenAI), Llama (Meta), Gemini (Google), Claude (Anthropic).

The LLM does **not** live inside your laptop (usually). It lives on the provider's servers. "Talking" to it means:

```text
1. Your Python code prepares a question
2. The question is sent over the internet to the provider's API
3. The model generates an answer
4. The answer comes back to your code
```

💡 **Provider** = the company that runs the model (OpenAI, Groq, Google...).
💡 **API (Application Programming Interface)** = a door through which your code sends requests and receives replies.

## 1.2 Why use LangChain for this?

Every provider's API looks different. Without LangChain, switching from OpenAI to Groq means rewriting code.

LangChain gives you **one common remote control** for all of them:

```text
                    ┌─────────────► OpenAI
Your code ──► LangChain ───────────► Groq
                    ├─────────────► Google Gemini
                    └─────────────► Ollama (model on your own laptop)
```

**Analogy:** a universal TV remote. One set of buttons, works on Sony, Samsung, LG.

## 1.3 The smallest possible program

```python
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()                          # loads your secret API key from the .env file (Section 8)

llm = ChatOpenAI(model="gpt-4o-mini")  # 1. choose the model
reply = llm.invoke("Hi! What is Python?")  # 2. ask the question
print(reply.content)                   # 3. print only the answer text
```

What each line means:

| Line | Meaning in plain words |
|---|---|
| `load_dotenv()` | "Read my secret key from the `.env` file." |
| `ChatOpenAI(model=...)` | "I want to talk to this OpenAI model." |
| `llm.invoke(...)` | "Send this question and wait for the full answer." |
| `reply.content` | "Give me only the text part of the answer." |

💡 **`invoke`** = ask once and wait for the complete answer.

## 1.4 Three buttons on the remote: `invoke`, `stream`, `batch`

| Button | What happens | Real-life example |
|---|---|---|
| `invoke()` | Ask → wait → get the whole answer | Sending an email and waiting for the reply |
| `stream()` | Answer arrives word by word, live | Watching someone type in WhatsApp |
| `batch()` | Ask many questions at the same time | Giving 50 tickets to 50 workers at once |

```python
llm.invoke("What is SQL?")
llm.batch(["What is SQL?", "What is NoSQL?", "What is ETL?"])
for piece in llm.stream("What is Kafka?"):
    print(piece.content, end="", flush=True)
```

(Streaming is explained fully in Section 9.4.)

## 1.5 Three ways to send your question

**Way 1 – plain text** (quickest)

```python
llm.invoke("What is Docker?")
```

**Way 2 – role + text pairs** (adds instructions)

```python
llm.invoke([
    ("system", "You are a teacher for 10-year-old kids."),
    ("human",  "What is Docker?"),
])
```

**Way 3 – message objects** (clearest, most professional)

```python
from langchain_core.messages import SystemMessage, HumanMessage

llm.invoke([
    SystemMessage(content="You are a teacher for 10-year-old kids."),
    HumanMessage(content="What is Docker?"),
])
```

💡 **Message** = one line in a conversation, labeled with who said it:

| Label | Who | Purpose |
|---|---|---|
| `system` / `SystemMessage` | You, as the app builder | Rules and personality for the AI |
| `human` / `HumanMessage` | The user | The question |
| `ai` / `AIMessage` | The model | The answer |

## 1.6 What exactly comes back?

You don't get plain text. You get an **`AIMessage` object** — like a parcel with the answer inside plus a delivery slip.

```python
reply = llm.invoke("Say hello")
print(reply)
```

```text
content='Hello! How can I help you today?'
response_metadata={'model_name': 'gpt-4o-mini...', 'finish_reason': 'stop', ...}
usage_metadata={'input_tokens': 9, 'output_tokens': 9, 'total_tokens': 18}
id='run-...'
```

| Part | What it is | Why you care |
|---|---|---|
| `content` | The answer text | Show to the user |
| `usage_metadata` | Token counts | Cost tracking |
| `response_metadata` | Model name, finish reason | Debugging |
| `id` | Unique ID | Logs |

💡 **Token** = a small chunk of text (roughly ¾ of an English word). Models read and write tokens, and you pay per token.
💡 **Metadata** = information *about* the answer, not the answer itself.
💡 **finish_reason** = why the model stopped: `stop` (finished normally) or `length` (ran out of allowed tokens — answer got cut).

## 1.7 LLMs have no memory (very important)

Each `invoke()` is a fresh call. The model does **not** remember your last question.

```python
llm.invoke("My name is Ashish.")
llm.invoke("What is my name?")   # ❌ "I don't know your name."
```

To make it "remember", **send the previous messages again**:

```python
from langchain_core.messages import HumanMessage, AIMessage

llm.invoke([
    HumanMessage(content="My name is Ashish."),
    AIMessage(content="Nice to meet you, Ashish!"),
    HumanMessage(content="What is my name?"),
])   # ✅ "Your name is Ashish."
```

💡 **Stateless** = doesn't remember anything between calls. LLMs are stateless; **your app** keeps the history.

---

# 2. Connect with OpenAI (Most Used Provider)

## 2.1 Steps at a glance

```text
Create OpenAI account → add billing credits → create API key
      → put key in .env → pip install langchain-openai → write code
```

## 2.2 Get the API key

1. Sign up on the **OpenAI Platform** (developer site, not the ChatGPT app).
2. Add credits under **Billing** — the API is pay-per-use and separate from a ChatGPT Plus subscription.
3. Go to **API keys → Create new secret key** and copy it (it is shown only once).

💡 **API key** = a secret password for your code. Whoever has it can use the API **on your bill**, so keep it secret.

## 2.3 Save the key

In your project folder, create a file named `.env`:

```text
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxx
```

(Full details in Section 8.)

## 2.4 Install

```bash
pip install langchain langchain-openai python-dotenv
```

## 2.5 Code

```python
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",   # which model
    temperature=0.2,       # low = focused, consistent answers
    max_tokens=300,        # maximum answer length
)

reply = llm.invoke("Explain a data warehouse in 3 simple lines.")
print(reply.content)
print("Tokens used:", reply.usage_metadata["total_tokens"])
```

You don't pass the key in code — `ChatOpenAI` automatically looks for the environment variable named exactly `OPENAI_API_KEY`.

## 2.6 Which OpenAI model should I use?

| Your need | Pick | Why |
|---|---|---|
| Learning, practice, cheap tasks | **Mini/small model** (e.g., `gpt-4o-mini`) | Very cheap and fast |
| Best quality writing / analysis | **Flagship model** | Smarter, costs more |
| Hard logic, math, tricky code | **Reasoning model** | "Thinks" before answering; slower, costlier |

> Model names change over time. If a name stops working, check OpenAI's current model list — only the `model="..."` string changes, nothing else.

## 2.7 Same thing via Azure (common in companies)

Many enterprises use OpenAI models through **Azure OpenAI**:

```python
from langchain_openai import AzureChatOpenAI

llm = AzureChatOpenAI(
    azure_deployment="my-gpt-deployment",   # name your admin created in Azure
    api_version="2024-10-21",
)   # reads AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT from .env
```

---

# 3. Connect with Alternative LLMs (Open-Source APIs)

## 3.1 Why look beyond OpenAI?

| Reason | Example |
|---|---|
| 💰 Cheaper / free tier | Groq, Gemini free tiers |
| ⚡ Faster | Groq is known for very fast responses |
| 🔒 Privacy | Ollama runs on your laptop — data never leaves |
| 🔓 No lock-in | Switch any time prices or quality change |

💡 **Open-source (open-weight) model** = a model whose learned numbers ("weights") are publicly released, so anyone can download and run it. Examples: **Llama** (Meta), **Mistral**, **Qwen** (Alibaba), **Gemma** (Google), **DeepSeek**.

There are two ways to use open models:

```text
A) Someone else runs it for you (cloud API)  → Groq, Hugging Face, Together AI
B) You run it on your own machine            → Ollama
```

## 3.2 Groq — open models in the cloud, super fast

```bash
pip install langchain-groq
```

```text
# .env
GROQ_API_KEY=gsk_xxxxxxxx
```

```python
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2)
print(llm.invoke("What is Apache Airflow? 2 lines.").content)
```

Get the key from the **Groq Console** (free tier available — great for learning).

## 3.3 Ollama — open models on your own laptop

```text
1. Install Ollama (from its official website)
2. In terminal:   ollama pull llama3.2      ← downloads the model (~2 GB)
3. Ollama runs in the background on your PC
```

```bash
pip install langchain-ollama
```

```python
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.2", temperature=0.2)   # no API key, no internet needed
print(llm.invoke("What is Docker? 2 lines.").content)
```

💡 **Parameters (model size)** = how many learned numbers a model has. "8B" = 8 billion. Small models (1B–8B) run on normal laptops; big ones (70B+) need powerful GPUs.

## 3.4 Hugging Face — the "GitHub of AI models"

```bash
pip install langchain-huggingface
```

```text
# .env
HUGGINGFACEHUB_API_TOKEN=hf_xxxxxxxx
```

```python
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

load_dotenv()
endpoint = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    max_new_tokens=300,
    temperature=0.2,
)
llm = ChatHuggingFace(llm=endpoint)
print(llm.invoke("What is PySpark?").content)
```

> Some models (like Llama) need you to accept the license on the model's Hugging Face page first.

## 3.5 Google Gemini (not open-source, but a popular alternative)

```bash
pip install langchain-google-genai
```

```python
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()   # needs GOOGLE_API_KEY
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
print(llm.invoke("What is BigQuery?").content)
```

## 3.6 Comparison table

| Provider | Open-source models? | Runs where | Key needed | Cost | Best for |
|---|---|---|---|---|---|
| OpenAI | No | Cloud | `OPENAI_API_KEY` | Paid | Best all-round quality |
| Groq | Yes (Llama, Qwen…) | Cloud | `GROQ_API_KEY` | Free tier + paid | Speed, learning |
| Ollama | Yes | **Your laptop** | None | Free | Privacy, offline |
| Hugging Face | Yes (thousands) | Cloud | `HUGGINGFACEHUB_API_TOKEN` | Free tier + paid | Trying many models |
| Gemini | No | Cloud | `GOOGLE_API_KEY` | Free tier + paid | Long documents |
| Anthropic Claude | No | Cloud | `ANTHROPIC_API_KEY` | Paid | Writing, coding |

## 3.7 The real power: one line to switch

```python
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

model = init_chat_model("groq:llama-3.1-8b-instant")
# model = init_chat_model("openai:gpt-4o-mini")
# model = init_chat_model("ollama:llama3.2")

print(model.invoke("Hello!").content)
```

`init_chat_model("provider:model")` = LangChain's universal "plug" — the rest of your code never changes.

---

# 4. Talk to an LLM in Different Languages

## 4.1 Do LLMs understand other languages?

Yes. Modern LLMs are **multilingual** 💡 (trained on text in many languages). They can read and write Hindi, Telugu, Tamil, French, Spanish, German, Japanese and many more.

Quality tips:
- Best in English and widely-spoken languages.
- Smaller open models can be weaker in regional languages — try a bigger model if output looks poor.
- Non-English text usually uses **more tokens** → slightly higher cost.

## 4.2 Method 1 – just ask in that language

```python
print(llm.invoke("मशीन लर्निंग क्या है? आसान भाषा में दो लाइन में बताइए।").content)
```

## 4.3 Method 2 – fix the reply language using a system message

```python
from langchain_core.messages import SystemMessage, HumanMessage

reply = llm.invoke([
    SystemMessage(content="Always answer in simple Hindi, no matter which language the user uses."),
    HumanMessage(content="What is a database index?"),
])
print(reply.content)
```

## 4.4 Method 3 – reusable multi-language translator

```python
from langchain_core.prompts import ChatPromptTemplate

translator = ChatPromptTemplate.from_messages([
    ("system", "Translate the user's text into {language}. Return only the translated text."),
    ("human", "{text}"),
]) | llm

for lang in ["Hindi", "Telugu", "French", "Japanese"]:
    print(lang, "→", translator.invoke({"language": lang, "text": "The data pipeline finished successfully."}).content)
```

(`ChatPromptTemplate` and the `|` symbol are explained in Sections 5 and 6.)

## 4.5 Handy tricks

| Want | Write in system message |
|---|---|
| Hinglish | "Reply in Hinglish (Hindi words written in English letters)." |
| Bilingual | "Answer first in English, then the same in Hindi." |
| Keep tech words in English | "Translate to Hindi but keep technical terms like API, SQL in English." |
| Consistent translations | Use `temperature=0` |

---

# 5. Introduction to Prompts (Prompt Templates)

## 5.1 What is a prompt?

💡 **Prompt** = everything you send to the model: instruction + context + question.

**The quality of the answer depends heavily on the quality of the prompt.**

❌ Weak:
```text
Explain partitioning.
```

✅ Strong:
```text
You are a senior data engineer.                         ← ROLE
Explain table partitioning                              ← TASK
to a beginner who knows basic SQL.                      ← AUDIENCE / CONTEXT
Use one real example with a sales table.                ← EXAMPLE
Answer in 4 bullet points, under 80 words.              ← FORMAT + LIMITS
```

**5 ingredients of a good prompt:** Role · Task · Context · Format · Constraints.

## 5.2 Problem: hard-coded prompts don't scale

```python
topic = "Kafka"
prompt = f"Explain {topic} to a beginner."   # fine once, messy across a big app
```

## 5.3 Solution: prompt templates

💡 **Prompt template** = a prompt with blanks like `{topic}` that LangChain fills in at runtime. Like a **form letter**: "Dear {name}, your order {order_id} has shipped."

### `PromptTemplate` – for a single block of text

```python
from langchain_core.prompts import PromptTemplate

t = PromptTemplate.from_template("Explain {topic} to a {audience} in {n} lines.")
print(t.invoke({"topic": "Kafka", "audience": "school student", "n": 3}).to_string())
# Explain Kafka to a school student in 3 lines.
```

### `ChatPromptTemplate` – for role-based messages ✅ (use this most)

```python
from langchain_core.prompts import ChatPromptTemplate

t = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}. Keep answers short."),
    ("human",  "Explain {topic} with one real example."),
])

filled = t.invoke({"role": "cloud tutor", "topic": "auto scaling"})
print(filled.to_messages())          # see the final messages
print(llm.invoke(filled).content)    # send to the model
```

## 5.4 Putting chat history inside a template

💡 **`MessagesPlaceholder`** = a slot where a whole list of previous messages gets inserted.

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

t = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])

history = [HumanMessage(content="I work with AWS Glue."), AIMessage(content="Great, how can I help?")]
print((t | llm).invoke({"history": history, "question": "Which service did I mention?"}).content)
```

## 5.5 Few-shot prompting – teach by example

💡 **Zero-shot** = no examples, just the instruction.
💡 **Few-shot** = show a few input→output examples so the model copies the pattern.

```python
t = ChatPromptTemplate.from_messages([
    ("system", "Classify the log line as INFO, WARNING or ERROR. Reply with one word."),
    ("human", "Job completed in 4 minutes"),       ("ai", "INFO"),
    ("human", "Disk usage at 85%"),                ("ai", "WARNING"),
    ("human", "NullPointerException in stage 3"),  ("ai", "ERROR"),
    ("human", "{log}"),
])
print((t | llm).invoke({"log": "Retrying connection to database"}).content)   # WARNING
```

## 5.6 Prompt techniques cheat-sheet

| Technique | One-line idea | Example phrase |
|---|---|---|
| Role prompting | Give the AI a job title | "You are a senior SQL reviewer." |
| Zero-shot | Just ask | "Summarize this text." |
| Few-shot | Show examples | (see above) |
| Step-by-step | Ask it to reason in steps for hard problems | "Work through this step by step." |
| Output format | Ask for exact shape | "Return JSON with keys name, age." |
| Delimiters | Separate data from instructions | "Summarize the text between ### marks." |
| Constraints | Limit length/style | "Max 50 words, no jargon." |

---

# 6. Chains and Output Parsers

> ✏️ Spelling: it's **output parsers** — they *parse* (read and convert) the model's output.

## 6.1 What is a chain?

💡 **Chain** = connecting steps so the output of one step automatically becomes the input of the next.

```text
   Input dict              Step 1                Step 2              Step 3
{"topic": "Kafka"} ──► Prompt template ──► Chat model (LLM) ──► Output parser ──► Final result
                       fills blanks          generates answer      cleans/converts
```

In LangChain you connect steps with the **pipe symbol `|`**:

```python
chain = prompt | llm | parser
```

💡 **LCEL (LangChain Expression Language)** = the official name for this `|` style.
💡 **Runnable** = any piece that can be chained (prompt, model, parser, even a whole chain). Every Runnable has `.invoke()`, `.stream()`, `.batch()`.

**Analogy:** a factory conveyor belt. Raw material → cutting → painting → packing.
**Data engineer analogy:** like Linux pipes `cat logs | grep ERROR | wc -l`, or extract → transform → load.

## 6.2 What is an output parser and why do we need it?

The model returns an `AIMessage` object. Your program often needs something else:

| You need | Without parser | With parser |
|---|---|---|
| Plain text | `reply.content` every time | Chain returns a string directly |
| A Python list | Manually split text | Automatic list |
| JSON / dict | Manually `json.loads()` and pray | Automatic dict |
| Validated typed object | Lots of checks | Pydantic object |

## 6.3 Main output parsers

| Parser | Output type | Typical use |
|---|---|---|
| `StrOutputParser` | `str` | Chatbots, summaries |
| `CommaSeparatedListOutputParser` | `list` | "Give me 5 ideas" |
| `JsonOutputParser` | `dict` | Extract fields |
| `PydanticOutputParser` | Pydantic object | Strict typed data |
| `llm.with_structured_output(...)` | Pydantic object / dict | ✅ Modern & most reliable |

## 6.4 Example 1 — text chain

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("Give a 3-line beginner explanation of {topic}.")
chain = prompt | llm | StrOutputParser()

print(chain.invoke({"topic": "Delta Lake"}))   # plain string, no .content needed
```

## 6.5 Example 2 — list output

```python
from langchain_core.output_parsers import CommaSeparatedListOutputParser

parser = CommaSeparatedListOutputParser()
prompt = ChatPromptTemplate.from_template(
    "Name 5 {thing}.\n{format_instructions}"
).partial(format_instructions=parser.get_format_instructions())

chain = prompt | llm | parser
services = chain.invoke({"thing": "AWS streaming services"})
print(type(services), services)
# <class 'list'> ['Kinesis Data Streams', 'Firehose', 'MSK', ...]
```

💡 **`.partial(...)`** = fill some blanks in advance so you don't pass them every time.
💡 **Format instructions** = text the parser adds to the prompt telling the model exactly how to format its answer.

## 6.6 Example 3 — JSON output

```python
from langchain_core.output_parsers import JsonOutputParser

prompt = ChatPromptTemplate.from_template(
    "Return only JSON with keys 'service', 'category', 'serverless' (true/false) for: {name}"
)
chain = prompt | llm | JsonOutputParser()
print(chain.invoke({"name": "AWS Athena"}))
# {'service': 'AWS Athena', 'category': 'Analytics', 'serverless': True}
```

## 6.7 Example 4 — modern structured output (recommended)

💡 **Pydantic** = a Python library to define the exact shape and types of data, and validate it.

```python
from pydantic import BaseModel, Field

class Service(BaseModel):
    name: str = Field(description="Official service name")
    category: str = Field(description="e.g. Storage, Compute, Analytics")
    serverless: bool

structured_llm = llm.with_structured_output(Service)
s = structured_llm.invoke("Describe AWS Lambda")
print(s.name, s.category, s.serverless)   # AWS Lambda Compute True
```

Why better? It uses the model's built-in JSON/tool features, so the output is far less likely to break than "please return JSON" in plain text.

## 6.8 Example 5 — chaining two LLM steps (explain → translate)

```python
from langchain_core.output_parsers import StrOutputParser

explain   = ChatPromptTemplate.from_template("Explain {topic} in 4 lines.") | llm | StrOutputParser()
translate = ChatPromptTemplate.from_template("Translate to simple Hindi:\n{text}") | llm | StrOutputParser()

pipeline = explain | (lambda english: {"text": english}) | translate
print(pipeline.invoke({"topic": "data lakehouse"}))
```

💡 **`lambda`** = a tiny one-line function; here it reshapes the text into the `{"text": ...}` dict the next prompt expects.

## 6.9 Every chain gets superpowers for free

```python
chain.invoke({"topic": "Kafka"})                                 # one answer
chain.batch([{"topic": "Kafka"}, {"topic": "Flink"}])           # many in parallel
for c in chain.stream({"topic": "Spark"}): print(c, end="", flush=True)   # live
```

> **Old vs new:** tutorials using `LLMChain(...)` and `.run()` are **legacy** (old style). Use `prompt | llm | parser` with `.invoke()`.

---

# 7. Local Setup vs Google Colab

💡 **Google Colab** = a free online Jupyter notebook from Google that runs in your browser.
💡 **Local setup** = Python + editor (VS Code) installed on your own laptop.

| Point | Google Colab | Local setup |
|---|---|---|
| Install anything? | No | Python, VS Code, packages |
| Free GPU | Yes (limited time) | Only if your laptop has one |
| After you close it | Session resets, reinstall packages | Everything stays |
| Code format | Notebook cells | Real `.py` files and folders |
| Git / deployment / real app | Awkward | Natural |
| Secrets | Easy to accidentally share in notebook | Safe in `.env` |
| Ollama local models | Not practical | ✅ Yes |
| Best for | Quick tests, demos | Real projects, agents, team work |

**Analogy:** Colab = hotel kitchen (ready instantly, but you can't leave anything there). Local = your own kitchen (setup once, everything stays).

**Recommendation:** Use Colab to experiment, but build your LangChain/agent projects **locally**.

---

# 8. Environment Setup

## 8.1 Why a virtual environment?

💡 **Virtual environment** = a separate, isolated Python "box" for each project.

Without it, Project A needs `langchain 1.0`, Project B needs `langchain 0.3` → they break each other. With it, each project has its own box.

## 8.2 Which tool? conda vs venv vs uv

| Tool | What it is | Pros | Cons |
|---|---|---|---|
| **venv** | Comes with Python | Zero extra installs | Can't install a different Python version |
| **conda** (Anaconda / Miniconda) | Environment + package manager | Choose any Python version; data-science favorite | Heavy, slower; Anaconda may need a license in big companies |
| **uv** ⭐ | Modern, Rust-based tool | 10–100× faster, manages Python versions, lock files | One extra install |

**My suggestion:**
- ⭐ **uv** → best modern choice for new projects.
- **conda** → fine if your course or team already uses it (prefer **Miniconda** — lighter).
- **venv** → simplest if you can't install anything extra.

## 8.3 Step-by-step setup

### Step 1 – Create the project folder

```bash
mkdir langchain_latest
cd langchain_latest
code .            # opens VS Code here (optional)
```

### Step 2 – Create & activate the environment (pick ONE)

**🅰 conda**

```bash
conda create -p ./venv python=3.11 -y   # env stored inside project folder
conda activate ./venv
# later: conda deactivate
```

**🅱 venv**

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
# later: deactivate
```

**🅲 uv ⭐**

```bash
pip install uv
uv venv --python 3.11
# activate exactly like venv above
```

✅ **Check it's active:** your terminal starts with `(venv)` / `(.venv)`.
✅ **In VS Code:** `Ctrl+Shift+P` → **Python: Select Interpreter** → choose your env.

> Windows PowerShell says *"running scripts is disabled"*? Run once:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

### Step 3 – `requirements.txt`

💡 **requirements.txt** = shopping list of packages for your project.

```text
langchain
langchain-core
langchain-openai
langchain-groq
langchain-ollama
python-dotenv
pydantic
```

| Package | Job |
|---|---|
| `langchain` | Main framework (`init_chat_model`, agents) |
| `langchain-core` | Messages, prompts, parsers, Runnables |
| `langchain-openai` | Talk to OpenAI |
| `langchain-groq` | Talk to Groq |
| `langchain-ollama` | Talk to local Ollama |
| `python-dotenv` | Read `.env` file |
| `pydantic` | Structured output |

```bash
pip install -r requirements.txt        # conda / venv
uv pip install -r requirements.txt     # uv
```

Pro tip — lock exact versions once things work:

```bash
pip freeze > requirements-lock.txt
```

### Step 4 – API keys in `.env`

💡 **Environment variable** = a named value stored outside your code.
💡 **`.env` file** = a text file that holds these values for your project.
💡 **python-dotenv** = package that loads `.env` into environment variables. (Install name `python-dotenv`, import name `dotenv`, file name `.env`.)

❌ Never do this:

```python
llm = ChatOpenAI(api_key="sk-proj-abc123...")   # push to GitHub = key stolen = your bill
```

✅ Do this — create `.env`:

```text
OPENAI_API_KEY=sk-proj-xxxx
GROQ_API_KEY=gsk_xxxx
GOOGLE_API_KEY=xxxx
HUGGINGFACEHUB_API_TOKEN=hf_xxxx

# optional – LangSmith tracing (see every step of your app in a dashboard)
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_xxxx
```

Rules: one `KEY=value` per line · no spaces around `=` · no quotes needed.

```python
import os
from dotenv import load_dotenv

load_dotenv()
print("OpenAI key found?", bool(os.getenv("OPENAI_API_KEY")))   # never print the real key
```

LangChain auto-reads standard names: `ChatOpenAI` → `OPENAI_API_KEY`, `ChatGroq` → `GROQ_API_KEY`. A typo like `OPENAI_KEY` won't work.

### Step 5 – Protect secrets with `.gitignore`

```text
# .gitignore
.env
venv/
.venv/
__pycache__/
```

And share a safe template for teammates:

```text
# .env.example  (safe to commit – no real values)
OPENAI_API_KEY=
GROQ_API_KEY=
```

> If a key ever leaks → **revoke it immediately** in the provider dashboard and create a new one.
> In production, keys usually come from a secrets manager (AWS Secrets Manager, Azure Key Vault) instead of `.env`.

### Step 6 – Final folder structure

```text
langchain_latest/
├── .venv/  or  venv/        ← virtual environment (not committed)
├── .env                     ← real keys (NEVER commit)
├── .env.example             ← key names only (commit)
├── .gitignore
├── README.md                ← how to run (Section 9.1)
├── requirements.txt
├── config.py                ← model + settings in one place
├── 01_hello_llm.py
├── 02_streaming.py
├── 03_prompt_chain_parser.py
└── 04_chatbot_with_memory.py
```

---

# 9. Project README + Code: How to Run, Streaming, Configuration, Chat Model

## 9.1 `README.md` for your project (copy as-is)

````markdown
# langchain_latest

Hands-on LangChain practice: talking to LLMs (OpenAI / Groq / Ollama),
prompt templates, chains, output parsers, streaming and a chatbot with memory.

## Prerequisites
- Python 3.10 or newer (3.11 recommended)
- At least one of: OpenAI API key, Groq API key, or Ollama installed

## 1. Create and activate environment (choose one)

### uv (recommended)
```bash
pip install uv
uv venv --python 3.11
source .venv/bin/activate        # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

### conda
```bash
conda create -p ./venv python=3.11 -y
conda activate ./venv
pip install -r requirements.txt
```

### venv
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Add API keys
```bash
cp .env.example .env             # Windows: copy .env.example .env
```
Open `.env` and paste your keys.

## 3. Run the examples
| Command | What it shows |
|---|---|
| `python 01_hello_llm.py` | Basic chat model call |
| `python 02_streaming.py` | Live streaming output |
| `python 03_prompt_chain_parser.py` | Prompt template + chain + parsers |
| `python 04_chatbot_with_memory.py` | Terminal chatbot that remembers (type `exit`) |

## 4. Switch the model
Edit `MODEL` in `config.py`:
- `openai:gpt-4o-mini`
- `groq:llama-3.1-8b-instant`
- `ollama:llama3.2` (run `ollama pull llama3.2` first)

## 5. Deactivate
`deactivate` (venv/uv) or `conda deactivate` (conda)
````

## 9.2 `config.py` — all settings in one place

```python
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

MODEL = "openai:gpt-4o-mini"      # or "groq:llama-3.1-8b-instant" / "ollama:llama3.2"

def get_llm(temperature: float = 0.3, max_tokens: int = 500):
    """Return a ready-to-use chat model with common settings."""
    return init_chat_model(
        MODEL,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=30,
        max_retries=2,
    )
```

## 9.3 `01_hello_llm.py` — chat model example

```python
from langchain_core.messages import SystemMessage, HumanMessage
from config import get_llm

llm = get_llm(temperature=0)

messages = [
    SystemMessage(content="You are a friendly data engineering mentor. Use simple words."),
    HumanMessage(content="What is the difference between batch and streaming data processing?"),
]

reply = llm.invoke(messages)

print("ANSWER:\n", reply.content)
print("\nTOKENS:", reply.usage_metadata)
print("FINISH REASON:", reply.response_metadata.get("finish_reason"))
```

## 9.4 `02_streaming.py` — streaming output

### What is streaming?

💡 **Streaming** = receiving the answer in small pieces (**chunks**) as soon as the model produces them, instead of waiting for the whole answer.

```text
invoke():   You ask ─── ⏳ waiting 6 sec ─── "Full answer appears at once"
stream():   You ask ─ "A" ─ " data" ─ " pipeline" ─ " is" ... (appears live, like ChatGPT)
```

Why use it: users see output instantly, the app *feels* faster, and long answers are less boring to wait for.

### Code

```python
from config import get_llm

llm = get_llm()

print("AI: ", end="")
full_text = ""
for chunk in llm.stream("Explain what an ETL pipeline is in 6 short lines."):
    print(chunk.content, end="", flush=True)
    full_text += chunk.content          # keep the complete answer too

print("\n\n[Done] total characters:", len(full_text))
```

### The two magic settings in `print()`

| Setting | Meaning | What happens without it |
|---|---|---|
| `end=""` | Don't jump to a new line after printing | Each word on a separate line |
| `flush=True` | Show text **right now** | Python holds text in a **buffer** 💡 (temporary waiting area) and dumps it in bursts |

### Streaming a full chain

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

chain = ChatPromptTemplate.from_template("Give 5 tips on {topic}") | llm | StrOutputParser()
for piece in chain.stream({"topic": "writing fast SQL"}):
    print(piece, end="", flush=True)
```

### Pro notes

- Chunks can be **added together**: `final = chunk1 + chunk2 + ...` gives one complete `AIMessage`.
- In web apps (FastAPI, Streamlit), use the **async** version: `async for chunk in llm.astream(...)`.
  💡 **async** = lets your program handle other work while waiting for the network.

## 9.5 `03_prompt_chain_parser.py`

```python
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, CommaSeparatedListOutputParser
from config import get_llm

llm = get_llm(temperature=0.2)

# A) text chain in any language
explain = ChatPromptTemplate.from_messages([
    ("system", "You are a patient tutor. Reply in {language}."),
    ("human",  "Explain {topic} with one real-life example."),
]) | llm | StrOutputParser()

print(explain.invoke({"topic": "data partitioning", "language": "English"}))
print("\n---\n")
print(explain.invoke({"topic": "data partitioning", "language": "Hindi"}))

# B) list output
list_parser = CommaSeparatedListOutputParser()
lister = ChatPromptTemplate.from_template(
    "List 5 {thing}.\n{fi}"
).partial(fi=list_parser.get_format_instructions()) | llm | list_parser
print("\nLIST:", lister.invoke({"thing": "Python libraries for data engineering"}))

# C) structured output
class Tool(BaseModel):
    name: str
    purpose: str
    open_source: bool

print("\nSTRUCTURED:", llm.with_structured_output(Tool).invoke("Describe Apache Airflow"))
```

## 9.6 `04_chatbot_with_memory.py`

```python
from langchain_core.messages import SystemMessage, HumanMessage
from config import get_llm

llm = get_llm()
history = [SystemMessage(content="You are a helpful assistant. Keep replies under 80 words.")]

print("Chatbot ready! Type 'exit' to quit.\n")
while True:
    user = input("You: ").strip()
    if user.lower() in {"exit", "quit"}:
        print("Bye!")
        break

    history.append(HumanMessage(content=user))

    print("AI: ", end="")
    answer = None
    for chunk in llm.stream(history):                # streaming reply
        print(chunk.content, end="", flush=True)
        answer = chunk if answer is None else answer + chunk
    print()

    history.append(answer)                            # remember the AI reply
```

> 💡 **Context window** = the maximum amount of text (tokens) a model can read in one go. Since history grows every turn, real apps **trim or summarize** old messages to stay within it and to save cost.

## 9.7 Most used configuration settings

```python
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3,
    max_tokens=500,
    top_p=1.0,
    timeout=30,
    max_retries=2,
    stop=["END"],
    seed=42,
    frequency_penalty=0.0,
    presence_penalty=0.0,
)
```

| Setting | Plain-English meaning | Range | What value to use |
|---|---|---|---|
| `model` | Which brain to use | model name | Required |
| `temperature` | Creativity dial | `0` – `1` (some allow `2`) | **0** → facts, SQL, code, extraction, translation · **0.3–0.5** → chatbots, explanations · **0.7–1.0** → ideas, stories |
| `max_tokens` | Max answer length | e.g. `100`–`4000` | Short Q&A `200` · summary `500` · long report `1500+` |
| `top_p` | Only pick from the top-p most likely words | `0`–`1` | Keep `1.0`; tune temperature **or** top_p, not both |
| `timeout` | Seconds to wait before giving up | `10`–`120` | `30` normal · `60–120` long/reasoning answers |
| `max_retries` | Auto-retry on network/rate-limit errors | `0`–`5` | `2` |
| `stop` | Text that makes the model stop | list of strings | When you want output to end at a marker |
| `seed` | Makes output more repeatable (best effort) | any int | Testing, demos |
| `frequency_penalty` | Discourage repeating same words | `-2` – `2` | `0.3–0.5` if output repeats |
| `presence_penalty` | Encourage new topics | `-2` – `2` | `0.3–0.5` for more variety |
| `base_url` | Custom server address | URL | Proxies, OpenAI-compatible servers |

**Temperature in one picture:**

```text
0.0 ───────────── 0.5 ───────────── 1.0
"Robot"          "Teacher"         "Poet"
same answer      balanced          different every time
```

**Ready-made presets:**

```python
precise  = get_llm(temperature=0,   max_tokens=300)   # SQL, extraction, translation
friendly = get_llm(temperature=0.4, max_tokens=500)   # tutor, chatbot
creative = get_llm(temperature=0.9, max_tokens=800)   # brainstorming, content
```

> ⚠️ Not every model supports every setting (some reasoning models ignore `temperature`). If you get "unsupported parameter", remove it.

## 9.8 Chat model vs Completion model — which is better and when?

### Completion model (older style)

💡 **Completion model** = takes **one piece of text** and **continues** it. Like phone autocomplete.

```python
from langchain_openai import OpenAI          # note: OpenAI, NOT ChatOpenAI

completion_llm = OpenAI(model="gpt-3.5-turbo-instruct")
print(completion_llm.invoke("The three V's of big data are"))   # returns plain str
```

### Chat model (modern style)

💡 **Chat model** = takes a **list of messages with roles** and replies as the AI.

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

chat_llm = ChatOpenAI(model="gpt-4o-mini")
r = chat_llm.invoke([
    SystemMessage(content="You are a big-data tutor."),
    HumanMessage(content="What are the three V's of big data?"),
])
print(r.content)                              # returns AIMessage → .content
```

### Comparison

| Point | Completion model | Chat model |
|---|---|---|
| Input | One text string | Messages with roles |
| Output | Plain string | `AIMessage` (text + metadata + tool calls) |
| System instructions | Mixed into the text | Separate `SystemMessage` |
| Conversations | You format history manually | Built-in |
| Tool calling (needed for agents) | ❌ | ✅ |
| Structured output | Weak | ✅ Strong |
| Images/audio input | ❌ | ✅ many models |
| LangChain classes | `OpenAI`, `OllamaLLM` | `ChatOpenAI`, `ChatGroq`, `ChatOllama`, `init_chat_model` |
| New models released | Rare | Almost all |

### Verdict

✅ **Use chat models for almost everything** — chatbots, RAG, translation, summarization, structured output and especially **agents** (they need tool calling).

Use a completion model only for:
- maintaining **old code**,
- pure **autocomplete / text continuation**,
- a model that only offers a completion endpoint.

---

# ➕ Extra 1: Common Errors & Fixes

| Problem | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: dotenv` | Package missing / env not active | Activate env → `pip install python-dotenv` |
| `ModuleNotFoundError: langchain_openai` | Integration missing | `pip install langchain-openai` |
| `401 Unauthorized` | Wrong/missing key or name typo | Check `.env` names; call `load_dotenv()` |
| Key is `None` | Script run from another folder | Run from the project root |
| `429` / insufficient quota | No credits or too many requests | Add billing, wait, or use Groq/Ollama |
| Model not found | Retired or misspelled name | Check provider's model list |
| `Unable to infer model provider` | `init_chat_model` name lacks provider | Use `"provider:model"` |
| Ollama connection refused | Ollama not running / model not pulled | Start Ollama; `ollama pull llama3.2` |
| `conda` not recognized | Conda not on PATH | Use Anaconda Prompt or `conda init` |
| PowerShell blocks activation | Execution policy | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| Streaming one word per line | Missing `end=""` | Add it |
| Streaming comes in bursts | Buffering | Add `flush=True` |
| Long object printed instead of text | Printed the `AIMessage` | Use `.content` or `StrOutputParser()` |
| JSON parsing fails | Model added extra text | Use `with_structured_output()` and `temperature=0` |

---

# ➕ Extra 2: Quick Revision (Interview Ready)

1. LangChain = **one remote control** for all LLMs: `invoke`, `stream`, `batch`.
2. Replies are `AIMessage` objects → use `.content` for text, `usage_metadata` for tokens.
3. LLMs are **stateless** → memory = resending message history.
4. **OpenAI** = most used (paid); key `OPENAI_API_KEY`; class `ChatOpenAI`.
5. Alternatives: **Groq** (fast, open models), **Ollama** (local, private), **Hugging Face** (many models), **Gemini**, **Claude**.
6. `init_chat_model("provider:model")` switches providers in one line.
7. LLMs are **multilingual**; control reply language in the **system message**.
8. Good prompt = **Role + Task + Context + Format + Constraints**.
9. `ChatPromptTemplate` = reusable role-based prompt; `MessagesPlaceholder` = slot for history; few-shot = teach by examples.
10. **Chain** = `prompt | llm | parser` (LCEL).
11. Parsers: `StrOutputParser`, `CommaSeparatedListOutputParser`, `JsonOutputParser`; modern best = `with_structured_output()`.
12. Build real projects **locally**, not in Colab.
13. Envs: **uv** (fast, modern) / **conda** / **venv**. Keys in `.env`, never in code, never in Git.
14. Streaming: `for c in llm.stream(...)` + `print(c.content, end="", flush=True)`.
15. Key settings: `temperature`, `max_tokens`, `top_p`, `timeout`, `max_retries`.
16. **Chat models** are the default; **completion models** are legacy.

---

# ➕ Extra 3: How This Connects to Agentic AI

```text
What you learned here            →   Role inside an AI agent
────────────────────────────────────────────────────────────
Chat model                       →   The agent's brain
Messages + history               →   Short-term memory
System message / prompt template →   The agent's job description
Structured output                →   Turning decisions into actions your code can run
Streaming                        →   Showing live progress to the user
Multiple providers               →   Cheap model for easy steps, smart model for hard ones
```

**Next in this series:** `03` Tools & Tool Calling → `04` Agents with `create_agent` → `05` Memory → `06` RAG → `07` LangGraph.
