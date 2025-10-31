# Build Long-Context AI Apps with Jamba

This repository contains my personal notes, code exercises, and project work for the course **[Build Long-Context AI Apps with Jamba](https://www.deeplearning.ai/short-courses/build-long-context-ai-apps-with-jamba/)** by **[DeepLearning.AI](https://www.deeplearning.ai/)** and **[AI21 Labs](https://www.ai21.com/)**.

## 🚀 About This Course

The transformer architecture, while foundational to most modern Large Language Models (LLMs), faces significant computational expenses when handling very long input contexts. This limits its use in applications requiring the analysis of large documents, codebases, or extensive chat histories.

An alternative, Mamba (a selective state-space model), can process extremely long contexts with much lower computational cost. However, the pure Mamba architecture can struggle with context understanding, leading to lower-quality outputs.

**Jamba, by AI21 Labs, is the solution.** It's a novel, hybrid architecture that combines the computational efficiency of Mamba with the high-quality attention mechanism of a Transformer. The result is a model that achieves high-quality outputs, high throughput, and low memory usage, even on massive context windows.

In this course, we learn:
* The unique Jamba (Transformer-Mamba) hybrid architecture.
* How to use the AI21SDK to prompt Jamba with massive documents (e.g., a 200k-token annual report).
* Practical, hands-on application of Jamba for tool-calling.
* The training and evaluation techniques used to create long-context models.
* How to build a powerful, long-context Conversational RAG (Retrieval-Augmented Generation) app using Jamba and LangChain.

## 📚 Course Topics

Here is a detailed breakdown of the modules covered in this course.

<details>
<summary><strong>Module 1: Overview</strong></summary>

This foundational module sets the stage for the entire course, establishing the "why" behind the need for long-context models and introducing the "what" and "how" of Jamba. It moves beyond the buzzwords to define the specific, tangible problems that arise from the context window limitations of traditional transformer-based LLMs.

**The Problem: The Context Window Bottleneck**
We begin by exploring the current limitations of models like GPT-4, Claude 3, and Gemini. While their context windows have been expanding, they are still fundamentally constrained by the Transformer architecture's quadratic complexity ($O(n^2)$). This means that as the input context length ($n$) doubles, the computational cost and memory required to process it *quadruples*. This bottleneck makes it computationally and financially infeasible to process truly massive inputs, such as:
* An entire 500-page legal contract.
* A complete codebase for a large software project.
* Years of financial reports for a company.
* A patient's complete medical history.

**The Contenders: Transformers vs. Mamba**
The module introduces the two core architectures at play:
1.  **Transformers:** Celebrated for their "attention" mechanism, which allows them to build a deep, complex understanding of relationships between tokens, resulting in high-quality, coherent, and context-aware outputs. Their weakness is the $O(n^2)$ scaling.
2.  **Mamba (Selective State-Space Model - SSM):** A newer architecture that scales *linearly* with context length ($O(n)$). This makes it incredibly efficient for long sequences. However, as the course description notes, pure Mamba architectures have been found to underperform in tasks requiring deep contextual understanding, sometimes even failing at simple "pass-through" recall tasks.

**The Solution: Jamba, The Best of Both Worlds**
This is the core introduction to AI21 Labs' Jamba. The module explains, at a high level, how Jamba's unique hybrid architecture is designed to provide the "best of both worlds." It strategically blends Transformer and Mamba components to achieve:
* **High Quality:** Leveraging the Transformer's attention mechanism for robust context understanding and high-fidelity output.
* **High Throughput & Efficiency:** Leveraging Mamba's linear scaling to process vast amounts of text with significantly lower memory and compute requirements.

This introductory module provides the necessary mental model to understand *why* Jamba is a significant development and frames the practical, hands-on modules that follow. It lays out the course roadmap, from basic prompting with the SDK to building a full-scale, long-context RAG application.
</details>

<details>
<summary><strong>Module 2: Transformer-Mamba Hybrid LLM Architecture</strong></summary>

This is the central technical deep-dive of the course. Where the "Overview" explained the "what," this module explains the "how." It breaks open the black box of Jamba to reveal its novel internal structure, detailing the specific way AI21 Labs has fused two distinct model architectures into a single, cohesive, and performant system.

**A Deeper Look at the Components:**
1.  **The Transformer Block:** This section will likely review the self-attention mechanism, the "key-query-value" system, and the feed-forward networks that make up a standard Transformer block. The focus will be on *why* this is the gold standard for "quality" – its ability to compare every token to every other token, creating a rich contextual map. This section will also re-emphasize the $O(n^2)$ complexity as its critical flaw for long sequences.
2.  **The Mamba Block (Selective SSM):** This part delves into the Mamba architecture. Unlike Transformers, which are parallelizable but slow, traditional recurrent models (RNNs) are fast but struggle with long-term memory (the "vanishing gradient" problem). State-Space Models (SSMs) offer a middle ground, modeling the system as a continuous state that evolves over time. Mamba improves on "vanilla" SSMs by introducing a *selection* mechanism. This allows the model to "select" which information to keep in its "state" and which to forget, making it far more powerful at modeling long-range dependencies than an RNN, but still operating in a way that allows for linear $O(n)$ scaling.

**The Hybrid Architecture: Jamba's "Secret Sauce"**
This is the heart of the module. The course will explain how Jamba combines these blocks. The key questions to be answered are:
* **How are they mixed?** Is it a simple alternating structure (e.g., one Transformer layer, then one Mamba layer)?
* **Is it a "mixture-of-experts" (MoE) model?** Does Jamba *also* include MoE principles, using Mamba and Transformer layers as different "experts"?
* **What is the "Joint Mamba-Attention" (JMA) model?** Jamba's architecture is more than just a simple stack. This module will explain the specific "Jamba-block" which intelligently integrates Mamba and attention components *within* a single layer, alongside a feed-forward network. This allows the model to dynamically choose the best processing mechanism (SSM for sequence modeling, attention for deep context) for different parts of the input.

**Performance Implications:**
The module will tie this architecture back to the three pillars mentioned in the intro:
1.  **High-Quality Outputs:** Achieved by the inclusion of Transformer-style attention.
2.  **High Throughput:** Achieved by the Mamba components and an overall more efficient architecture.
3.  **Low Memory Usage:** The linear scaling of Mamba dramatically reduces the memory footprint for long contexts, making it possible to run large-context inference on more accessible hardware.

This module is essential for anyone who wants to move from being a user of Jamba to an *informed* developer who understands *why* it's designed the way it is.
</details>

<details>
<summary><strong>Module 3: Jamba Prompting and Documents</strong></summary>

This module marks the transition from theory to practice. It's the "Hello, World!" for long-context AI, focusing on the primary developer interface for Jamba: the AI21 SDK. The central theme is learning how to *actually* send a massive piece of text to the model and get a useful response.

**Introducing the AI21 SDK:**
The module will begin with the basics of setting up the developer environment:
* Installation: `pip install ai21`
* Authentication: How to obtain and set up your API key.
* Basic API Calls: Sending your first simple prompt to a Jamba model.

**The `document` Parameter: The Key to Long-Context**
The highlight of this module, as specified in the course description, is the `document` parameter. This is likely a specialized feature within the AI21 SDK designed to handle large inputs gracefully. This section will explore:
* **What is it?** Is it just a string parameter, or is it an abstraction that accepts file paths, URLs, or binary data?
* **Why use it?** How does using the `document` parameter differ from just concatenating a massive string into the `prompt` field? The course will likely explain that this parameter is the "correct" way to send large-scale data, allowing the SDK and the backend to optimize for streaming, chunking (if necessary), or other efficient data handling methods.
* **Practical Example:** The core of this module will be the 200k-token annual report example. This is a brilliant, real-world use case. You can't just "paste" a 200,000-token document into a text box. This hands-on lab will involve:
    1.  Loading the document (e.g., a large PDF or text file).
    2.  Using the `AI21SDK` to pass this document to Jamba.
    3.  Writing a prompt that *operates* on this document, for example:
        * "Summarize the 'Risk Factors' section from the attached annual report."
        * "What was the total revenue reported in this document? Quote the exact sentence."
        * "Find all mentions of 'forward-looking statements' and list them."

This module demonstrates Jamba's raw power in a tangible way. It moves beyond the abstract idea of a "large context window" and provides the direct commands to query a digital document that is *hundreds* of pages long in a single API call.
</details>

<details>
<summary><strong>Module 4: Tool Calling</strong></summary>

This module explores one of the most powerful capabilities of modern LLMs: making them *active* agents that can interact with external systems. "Tool calling" (also known as function calling) is the mechanism by which an LLM can understand a user's intent to run an external function, intelligently formulate the arguments needed for that function, and return the request in a structured format (like JSON).

**From "Text In, Text Out" to "Text In, Action Out"**
This section will define the core concepts of tool calling.
* **User Intent:** The model learns to recognize when a user's prompt cannot be answered by text generation alone (e.g., "What's the weather in London?" or "Add 'buy milk' to my to-do list").
* **Function Definitions:** You, the developer, provide the LLM with a "menu" of available tools, including:
    * The function's name (e.g., `get_current_weather`).
    * A description of what it does ("Retrieves the current weather for a given location").
    * The parameters it accepts (e.g., `location: string`, `unit: 'celsius' | 'fahrenheit'`).
* **Structured Output:** When the LLM detects an intent, it doesn't try to *guess* the weather. Instead, it stops and returns a special message, like: `{"tool_call": {"name": "get_current_weather", "arguments": {"location": "London", "unit": "celsius"}}}`.
* **Executing the Call:** Your application code then parses this JSON, runs your *actual* `get_current_weather("London", "celsius")` function, gets the real data (e.g., `{"temp": 15, "conditions": "Cloudy"}`), and then passes this information *back* to the LLM to formulate a natural language response.

**Hands-On Examples:**
The course provides two key examples to build practical skills:
1.  **Simple Arithmetic:** A "toy" example, like defining `add(a, b)` and `multiply(a, b)`. This is a perfect "hello world" for tool calling, allowing you to test the logic without relying on external APIs. The user asks, "What is 12 times 5?" and the model returns a call to `multiply(12, 5)`.
2.  **SEC 10-Q Report Function:** This is a sophisticated, real-world example. You'd define a function like `get_sec_filing(company_ticker, report_type, quarter, year)`. A user could then ask, "Can you get me the Q2 2024 10-Q report for NVDA?" The LLM would correctly parse this natural language request into the structured call: `{"tool_call": {"name": "get_sec_filing", "arguments": {"company_ticker": "NVDA", "report_type": "10-Q", "quarter": 2, "year": 2024}}}`. Your code would then fetch this document (which could be *very* large) from the SEC's EDGAR database.

**Why Jamba?**
The combination of long-context and tool calling is powerful. A user could provide a large document (using the skills from Module 3) and *then* ask a question that requires a tool. For example: "Please analyze this 100-page investment memo (long-context). Based on its risk analysis, call the `get_stock_price` tool for all companies mentioned." Jamba can hold the entire memo's context while *also* managing the tool-calling logic.
</details>

<details>
<summary><strong>Module 5: Expand the Context Window Size</strong></summary>

This module pivots back to the underlying theory, asking: "How is a long-context model *made*?" It's not as simple as just feeding a model longer texts. This section delves into the specialized techniques for *training* and *evaluating* models like Jamba, giving you an appreciation for the data science and engineering effort required.

**The Challenge of Training for Long Context**
Training a Transformer from scratch on a 200k+ token context is computationally (and financially) astronomical. This module will likely explore the common, more efficient strategies:
1.  **Continued Pre-training:** The most common approach. You take an existing model pre-trained on a shorter context (e.g., 4k or 8k) and "fine-tune" it on a new dataset composed exclusively of long-sequence documents. This "stretches" the model's effective context window.
2.  **Curriculum Learning:** A more gradual approach where the model is first trained on short documents, then medium, then long, and so on. This "eases" the model into handling longer contexts.
3.  **Architectural Support:** How does Jamba's hybrid architecture make this training more feasible? The Mamba component's linear scaling means that the "continued pre-training" step is *dramatically* faster and less memory-intensive than it would be for a pure Transformer, making it possible to train on 200k+ tokens where it would be impossible for others.

**How Do You Know It's Working? Evaluating Long Context**
Perhaps the most crucial part of this module is evaluation. A model claiming a 200k context window is useless if it "forgets" what was said in the first 10k tokens. This section will introduce the specialized metrics and benchmarks used to validate long-context performance:
* **Perplexity (PPL):** A classic NLP metric, but applied over long sequences. Does the model's "surprise" (perplexity) increase as the sequence gets longer?
* **"Needle in a Haystack" (NIAH) Test:** This is the *de facto* standard for long-context recall. The test involves:
    1.  Taking a long, irrelevant document (the "haystack," e.g., 100k tokens of essays).
    2.  Inserting a single, specific, unrelated fact (the "needle," e.g., "The best pizza topping is pineapple").
    3.  Placing this "needle" at various positions (e.g., at the 5% mark, 50% mark, 95% mark) within the haystack.
    4.  Asking the model a question it can *only* answer by finding the needle: "What is the best pizza topping?"
* **Long-Context Summarization & QA:** Evaluating the model's ability to perform tasks like summarizing an entire book or answering complex questions that require synthesizing information from the beginning, middle, *and* end of a massive document.

This module provides a critical look "under the hood," separating marketing claims ("We have a 1M token window!") from engineering reality (How well does it *actually work* at 1M tokens?).
</details>

<details>
<summary><strong>Module 6: Long Context Prompting</strong></summary>

This module is the "applied science" of prompt engineering, specifically for massive context windows. It builds on Module 3 ("Jamba Prompting and Documents"), which was about the *syntax* and *SDK*, and focuses instead on the *strategy* and *art* of writing prompts that get reliable, high-quality results from a 200k+ token input.

**The "Lost in the Middle" Problem**
A key challenge with *all* long-context models (not just Jamba) is a phenomenon known as "lost in the middle." Research has shown that many models exhibit a "U-shaped" performance curve: they are very good at recalling information from the very *beginning* and the very *end* of their context window, but their performance *dips* significantly for information placed in the "middle."

This module will teach strategies to mitigate this:
* **Prompt Structuring:** Does the placement of your instruction matter? Should your core instruction ("You are a legal assistant...") go at the beginning or the end of the prompt? (Hint: It's almost always better at the end, *after* the long document).
* **Instruction Placement:** If you provide a 200k-token document and then a 10-line prompt, how do you ensure the model pays attention to your prompt? The course will likely show techniques like: "Please analyze the *entire* document provided above and then answer the following question: ..."
* **Using Structural Cues:** Using XML tags (e.g., `<document>...</document>`, `<question>...</question>`) or Markdown (e.g., `# Document`, `# My Question`) to help the model differentiate between the massive context and the specific task you want it to perform.

**Advanced Prompting Techniques for Long-Context Tasks**
This module will move beyond simple Q&A to cover more complex, long-context-native tasks:
* **Comparative Analysis:** "Here are two 50k-token legal briefs (`<brief_A>...<brief_A>`, `<brief_B>...<brief_B>`). Please compare and contrast their main arguments."
* **Change Tracking:** "Here is the Q1 annual report (`<Q1>...</Q1>`) and the Q2 annual report (`<Q2>...</Q2>`). Please identify and summarize all *new* risk factors mentioned in Q2 that were not present in Q1."
* **Synthesis and Summarization:** Moving beyond "summarize this document" to "Summarize the *evolution* of the 'climate change' narrative across this entire 150k-token archive of news articles."

This module is critical for developers, as it teaches the practical skills to harness Jamba's power. A large context window is a powerful tool, but like any tool, it requires skill and technique to use effectively.
</details>

<details>
<summary><strong>Module 7: Conversational RAG</strong></summary>

This is the capstone module of the course, where all the preceding concepts—long-context, SDKs, and prompting—come together to build a sophisticated, real-world application: a **Conversational Retrieval-Augmented Generation (RAG)** pipeline.

**What is RAG?**
First, the module will define (or review) Retrieval-Augmented Generation (RAG). This is the standard pattern for building "chat with your docs" applications:
1.  **Retrieve:** A user asks a question ("What is our policy on remote work?").
2.  **Augment:** The system doesn't send this *directly* to the LLM. Instead, it first searches a knowledge base (e.g., a vector database of company HR documents) to find the most relevant "chunks" of text.
3.  **Generate:** The system then "augments" the original prompt, effectively saying to the LLM: "Using the following context: `[...relevant HR doc chunks...]`, please answer the user's question: 'What is our policy on remote work?'"

**Why Jamba & Long-Context are a "Game Changer" for RAG:**
The "dirty secret" of traditional RAG is that the LLM's small context window is the primary bottleneck. You might retrieve 50 relevant "chunks" from your database, but you can only *fit* the top 5 or 10 into an 8k context window. You are forced to throw away potentially useful information.

Jamba's 200k+ context window *fundamentally* changes this. This module will explore a "Next-Generation RAG" pipeline where:
* You can retrieve *all 50* relevant chunks and feed them *all* to Jamba.
* You can retrieve *entire documents* instead of just small chunks. Instead of "here are 10 paragraphs," you can say, "here is the *entire* 80-page HR manual, now answer the question."
* This gives the LLM a *much* more complete picture, reducing "hallucinations" and leading to higher-quality, more accurate, and more comprehensive answers.

**"Conversational" RAG:**
This adds another layer of complexity: memory. A "conversational" app needs to remember the chat history. For example:
* **User:** "What is our policy on remote work?"
* **App:** (Performs RAG) "You can work remotely up to 2 days a week..."
* **User:** "What about for international employees?"
The RAG system needs to understand that "What about" refers to the *remote work policy* and that the search query should be "remote work policy for international employees."

**Two Paths to Building:**
The course provides two hands-on labs for building this:
1.  **The Managed Way: AI21 Conversational RAG Tool:** This will likely be a high-level API or tool provided by AI21 that abstracts away the complexity. You'll learn how to "point" this tool at your data source and quickly deploy a powerful, conversational RAG bot.
2.  **The "From Scratch" Way: LangChain + Jamba:** This is for the developer who wants full control. You will build your own RAG pipeline from its core components:
    * **Orchestrator:** LangChain (or LlamaIndex)
    * **Vector Store:** A database like Chroma, Pinecone, or FAISS.
    * **Embedding Model:** To turn your documents into searchable vectors.
    * **LLM:** Jamba, as the generative "brain" with the massive context window.

This final module synthesizes everything you've learned to build a powerful, state-of-the-art AI application that would be impossible to create with traditional, short-context LLMs.
</details>

## 🙏 Acknowledgement

This repository is for my personal educational use and learning purposes only. All course materials, content, lectures, and associated licenses are the property of **DeepLearning.AI** and **AI21 Labs**. I do not claim any ownership of the original course content.