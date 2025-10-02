# 🚀 AI Projects Portfolio – by a 16-Year-Old Innovator

![GitHub Repo Size](https://img.shields.io/github/repo-size/FWKMultiverse/FWK-Multiverse)
![GitHub stars](https://img.shields.io/github/stars/FWKMultiverse/FWK-Multiverse)
![GitHub license](https://img.shields.io/github/license/FWKMultiverse/FWK-Multiverse)
![Python](https://img.shields.io/badge/python-3.11-blue)
![AI](https://img.shields.io/badge/AI-MultiAgent-green)
![RL](https://img.shields.io/badge/Reinforcement-Learning-orange)
![LLM](https://img.shields.io/badge/LLM-CodeGen-purple)
![GNN](https://img.shields.io/badge/Graph-NeuralNetwork-purple)
![RAG](https://img.shields.io/badge/RAG-VectorSearch-red)
![PyTorch](https://img.shields.io/badge/PyTorch-Framework-red)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Framework-orange)
![Keras](https://img.shields.io/badge/Keras-DeepLearning-red)
![LightGBM](https://img.shields.io/badge/LightGBM-ML-green)
![CatBoost](https://img.shields.io/badge/CatBoost-ML-blue)
![GaussianProcess](https://img.shields.io/badge/GaussianProcess-ML-lightgrey)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-yellow)
![Optuna](https://img.shields.io/badge/Optuna-Hyperparameter-lightgrey)
![AsyncIO](https://img.shields.io/badge/AsyncIO-Concurrent-lightblue)
![Joblib](https://img.shields.io/badge/Joblib-Parallel-lightgreen)
![Pandas](https://img.shields.io/badge/Pandas-Dataframe-blue)
![NumPy](https://img.shields.io/badge/NumPy-Array-lightblue)
![Shap](https://img.shields.io/badge/SHAP-ExplainableAI-orange)
![PyTorch Geometric](https://img.shields.io/badge/PyG-GraphNeuralNetwork-purple)
![Transformers](https://img.shields.io/badge/Transformers-NLP-blue)
![Datasets](https://img.shields.io/badge/Datasets-HuggingFace-lightblue)
![Accelerate](https://img.shields.io/badge/Accelerate-Training-purple)
![Docker](https://img.shields.io/badge/Docker-Sandbox-blue)
![MultiAgent](https://img.shields.io/badge/MultiAgent-System-green)
![Trading](https://img.shields.io/badge/Trading-Forex/Crypto-yellow)
![Numerai](https://img.shields.io/badge/Numerai-Challenge-darkblue)
![Private](https://img.shields.io/badge/Code-Private-red)
![Research](https://img.shields.io/badge/Research-Level-purple)

---

## 🌟 Introduction

Welcome to my portfolio. I am a 16-year-old self-taught AI developer with a formal education up to grade 9. My journey is driven by a singular passion: to build complex, intelligent systems that solve real-world problems. Instead of following a traditional learning path, I dive headfirst into creating full-scale, resilient, and research-level AI systems.

This document is not merely a list of projects; it is a testament to my architectural philosophy, my problem-solving approach, and my vision for the future of AI. The code for these projects remains private, as they represent my core intellectual property. However, the detailed descriptions here aim to provide a transparent and comprehensive look into the technical depth of my work.

⚠️ **Important Note:** All systems presented here are original concepts, architected and built independently by me. This is a high-level overview intended for technical evaluation, not an implementation guide.

---

## 💖 A Call for Support: Powering the Next Wave of AI Innovation

To push the boundaries of what is possible, especially in fields like Reinforcement Learning and Large Language Models, computational power is not just a facilitator—it is a fundamental requirement. Currently, my progress is constrained by the limitations of my existing hardware.

**I am seeking financial support specifically to upgrade my computer.**

A more powerful machine, equipped with a high-end GPU, will directly enable me to:
1.  **Train More Complex Models:** My current setup struggles with the VRAM and processing demands of larger models like those in Project 3 (Multi-Agent LLM). An upgrade would allow for training larger, more capable agents and exploring state-of-the-art architectures.
2.  **Accelerate Research & Development:** Iteration speed is critical in AI. Faster training cycles mean I can experiment with more hypotheses, fine-tune models more effectively, and ultimately innovate at a much faster pace.
3.  **Produce Demonstrable Results:** With adequate hardware, I can finally run my systems at their full potential and generate concrete, shareable results—such as live trading performance, Numerai competition rankings, and interactive demos of the game development AI. This moves my work from theoretical architecture to proven application.

Your support would be a direct investment in my potential and would be instrumental in unlocking the next stage of my projects. Every contribution, big or small, makes a significant difference.

[![Sponsor](https://img.shields.io/badge/Sponsor-My_AI_Research_on_GitHub-green)](https://github.com/sponsors/FWKMultiverse)
**Direct Link to Sponsor:** [https://github.com/sponsors/FWKMultiverse](https://github.com/sponsors/FWKMultiverse)

---

# 🏷️ Tags for Visibility & Areas of Interest

`AI` `Reinforcement Learning` `Multi-Agent Systems` `Large Language Models (LLM)` `Generative AI` `Graph Neural Networks (GNN)` `Retrieval-Augmented Generation (RAG)` `Quantitative Finance` `Algorithmic Trading` `AI for Games` `Explainable AI (XAI)` `AutoML` `Robust Systems` `High-Performance Computing` `AI Research`

**Companies & Research Labs:**
`OpenAI` `DeepMind` `Google Research` `Google Brain` `Jane Street` `NVIDIA` `Microsoft Research` `GitHub Copilot` `Unity` `Epic Games (Unreal Engine)` `High-Level AI/Tech Startups`

---

# 📂 Projects Overview

My work is concentrated into three core projects, each developed with a specific, ambitious goal in mind:

1.  **AI Trading System** – 28 days (A production-grade, multi-agent system for real-time financial markets.)
2.  **Numerai Competition System** – 6 days (An expert-level, ensemble-of-ensembles architecture for a leading data science tournament.)
3.  **Multi-Agent LLM for Game Development** – 8 days (A cutting-edge, research-frontier system emulating a full AI software development team.)

Each project is detailed extensively below.

---

## 1️⃣ Project 1: AI Trading System

**Timeline:** 28 Days
**Level:** Production-Grade / Professional
**Core Technologies:** Python, Async I/O, Multi-Agent Reinforcement Learning (MARL), Transformers, GNN, Flask, SHAP (XAI)

### 📝 In-Depth Project Overview
This system is an end-to-end, fully automated AI trading platform designed for the high-frequency, chaotic environments of Forex and Cryptocurrency markets. The core architectural philosophy is modularity and resilience, creating a system that can operate 24/7 with minimal human intervention. It is structured into a robust three-tier architecture:

-   **Tier 1: Fetcher (`fetcher.py`)**: The system's sensory organ. This asynchronous module is responsible for ingesting a wide array of data streams in real-time. It doesn't just pull price data; it gathers economic news from financial APIs, scrapes sentiment from social media, and even queries Google Search trends to gauge market psychology. Its design is hyper-focused on reliability, as it is the foundation upon which all decisions are made.
-   **Tier 2: AI Server (`AIServer.py`)**: The central nervous system. Built on Flask, this web server acts as the command and control center. It exposes a secure API for external interaction (e.g., from a trading terminal or a monitoring dashboard). It manages a sophisticated asynchronous task queue, ensuring that requests for trading signals and commands to retrain models are handled gracefully without blocking critical operations.
-   **Tier 3: AI Engine (`AIEngine.py`)**: The brain of the operation. This is where the core intelligence resides. It takes the processed data from the Fetcher, performs deep analysis, and leverages a society of specialized AI agents to generate a final, high-conviction trading signal.

### 🤖 Multi-Agent Architecture & Advanced Models
The system's intelligence is not monolithic. It is a collaborative **Multi-Agent Reinforcement Learning (MARL)** ecosystem where each agent possesses a unique expertise, mimicking a team of human analysts.

1.  **The Macro Agent (The Strategist)**: This agent analyzes the market from a bird's-eye view, operating on high timeframes (H1, H4, Daily). It uses a combination of classical indicators (RSI, MACD) and modern data representations. It consumes **News Embeddings** from a dedicated News Transformer to understand market narratives and **Graph Embeddings** from a GNN to understand inter-asset relationships. Its sole purpose is to determine the overall market regime: Bullish, Bearish, or Consolidation.

2.  **The Micro Agent (The Executioner)**: This is the primary decision-maker for market entry and exit, operating on low timeframes (M5, M15). It takes the strategic context from the Macro Agent as a critical input. Its core is a **Transformer Block**, which allows it to analyze time-series price data with an attention mechanism, capturing complex, non-linear patterns that traditional models would miss.

3.  **The Risk Agent (The Manager)**: This agent's function is purely risk management. It constantly monitors the entire portfolio's state (Profit/Loss, Equity, Max Drawdown). Based on market volatility and current exposure, it dynamically adjusts the position sizing (Lot Size) for trades suggested by the Micro Agent, ensuring catastrophic losses are avoided.

**Supporting Models:**
-   **Graph Neural Network (GNN) Analyzer**: This model creates a dynamic graph where assets are nodes and their real-time correlations are edges. By analyzing this graph, it generates a "Graph Embedding"—a rich, numerical summary of the entire market's interconnectedness, which is invaluable for the Macro Agent.
-   **News Transformer**: A fine-tuned NLP model that processes news headlines and articles, converting unstructured text into dense "News Embeddings" that capture sentiment and thematic shifts.

### 🛡️ Uncompromising Error Handling & Reliability
A trading system is only as good as its uptime. This system was built with a "fail-safe" mentality.
-   **Retry with Exponential Backoff**: A decorator `@retry_async` wraps all external API calls. If a request fails, it automatically retries with an increasing delay, gracefully handling temporary network issues.
-   **API Cooldown & Quota Management**: The system intelligently tracks API usage. If an API endpoint starts failing repeatedly (e.g., HTTP 429 Too Many Requests), it is placed in a temporary "cooldown" to prevent being blacklisted.
-   **Asynchronous Task Queues**: The AIServer uses `ThreadPoolExecutor` to offload long-running tasks like model training, ensuring the main server thread remains responsive to time-sensitive signal requests.
-   **Request Timeouts**: The `/get_signal` endpoint has a strict timeout. If the AI Engine cannot produce a signal within a set time, it returns an HTTP 504 error, preventing the client from hanging indefinitely.
-   **Automated Data Sanitization**: A startup function, `clean_corrupt_json_files`, scans the data directory for malformed files that could crash the training process and removes them.

### 🌟 Unique & Innovative Features
-   **Explainable AI (XAI) with SHAP**: To build trust in the system, I integrated the `shap` library. This allows us to visualize exactly which features (e.g., RSI, news sentiment, a specific asset correlation) contributed most to the Micro Agent's decision to buy or sell.
-   **Advanced Risk Management with Kelly Criterion**: Beyond the Risk Agent, the system includes a `calculate_optimal_lot_size` function based on the Kelly Criterion, a mathematical formula to determine optimal position size to maximize long-term growth.
-   **Multi-Timeframe Fusion**: The seamless integration of analysis from M5 up to Daily charts gives the AI a holistic market perspective, balancing short-term tactics with long-term strategy.

---

## 2️⃣ Project 2: Numerai Competition System

**Timeline:** 6 Days (Rapid Development Sprint)
**Level:** Expert / Research-Grade
**Core Technologies:** Python, LightGBM, CatBoost, GNN, Transformer, LSTM, Autoencoder, Gaussian Processes, Optuna

### 📝 In-Depth Project Overview
This project was an intensive, focused effort to build a top-tier system for the Numerai Tournament, a notoriously difficult challenge where participants must predict stock market returns from obfuscated data. Given the extreme time constraint of just six days, the strategy was to architect a massively complex and diverse **ensemble-of-ensembles** system to maximize predictive power and robustness.

-   **`data_converter.py`**: A highly efficient pre-processing script that transforms the raw Parquet data files provided by Numerai into a more manageable Era-grouped JSON format. This simple step drastically speeds up subsequent data loading and feature engineering.
-   **`AInumerai_Ultra.py`**: The monolithic core of the project. This script orchestrates the entire pipeline: advanced feature engineering, parallel training of over a dozen distinct model configurations, hyperparameter optimization, multi-layer ensembling, and final submission file generation.
-   **`CSVP.py`**: A custom validation tool that simulates Numerai's unique scoring and payout system locally. This allows for rapid iteration and testing of different models without having to wait for the official weekly results.

### 🤖 A Symphony of Models: The Ensemble-of-Ensembles Architecture
The system's power comes from its diversity, combining models that "think" about the problem in fundamentally different ways.

-   **Gradient Boosting Powerhouses (LightGBM & CatBoost)**: These are the workhorses, exceptionally skilled at finding patterns in large tabular datasets. Multiple versions of each are trained on different feature subsets and with different objectives.
-   **Graph Neural Network + Transformer Hybrid**: This innovative model treats all features as nodes in a fully connected graph. The GNN layer learns the relational structure between features, and the output is then fed into a Transformer to learn which features deserve the most "attention" for a given data point.
-   **Temporal Model (LSTM)**: This model processes the data era-by-era, treating it as a time series. It aims to capture temporal dependencies and regime changes that other models might miss.
-   **Unsupervised Feature Extractor (Autoencoder)**: A neural network trained to compress all input features into a small, dense latent space and then reconstruct them. The compressed representation (`autoencoder_feat_`) becomes a powerful new feature for the other models.
-   **Multi-Agent RL Decider**: This agent takes the predictions from the GNN, LGBM, and Temporal models as its "state" and learns an optimal policy (Action) for combining them into a more refined prediction.

**Meta-Models for Final Blending:**
-   **Ridge Regression for Neutralization**: A key technique in Numerai is to create predictions that are uncorrelated with known risk factors. A Ridge model is trained specifically to neutralize the main ensemble's predictions.
-   **Gaussian Process Regressor**: The final, ultimate aggregator. This powerful probabilistic model takes the outputs of *all* other models (including the neutralized one) and produces the final submission, complete with uncertainty estimates.

### 🛡️ Robustness Under Pressure
-   **Graceful Degradation**: The code checks for the availability of optional, heavy libraries like Dask or Spektral. If they aren't installed, the system automatically falls back to a simpler but still functional mode instead of crashing.
-   **Aggressive Memory Management**: A `reduce_memory_usage` function iterates through every column in the dataset, downcasting data types (e.g., float64 to float32) to drastically reduce RAM consumption, making it possible to run on consumer-grade hardware.
-   **Parallel Seed Training**: The entire training process is wrapped in `multiprocessing`. The system trains the same model architecture on multiple different random seeds simultaneously. This not only speeds up the process but also makes the final averaged result (bagging) much more stable.

### 🌟 Competition-Winning Features
-   **Automated Hyperparameter Tuning**: The system uses **Optuna** to intelligently search for the best hyperparameters for LightGBM and CatBoost, and **Keras Tuner** for the neural network models, automating one of the most time-consuming parts of machine learning.
-   **Adversarial Validation**: A clever technique to ensure the model will generalize well. A classifier is trained to distinguish between the training data and the validation data. Features that make this distinction easy are considered "unstable" and are down-weighted or removed.
-   **Purged Time-Series Cross-Validation**: The entire validation strategy is built around respecting the temporal nature of the data, using a strict walk-forward approach with "purging" to prevent any data from the future from leaking into the training of a model.

---

## 3️⃣ Project 3: Multi-Agent LLM for Game Development

**Timeline:** 8 Days
**Level:** Advanced R&D / Research Frontier
**Core Technologies:** Python, PPO-RL, LLMs (Salesforce/codegen-2B-mono), LoRA Fine-Tuning, GNN, RAG, Docker

### 📝 In-Depth Project Overview
This is my most ambitious and forward-looking project. It is a self-contained **AI Agent Swarm**, powered by a Large Language Model, designed to function as an autonomous game development assistant. The system can understand an entire codebase, write new code, identify and fix bugs, refactor existing code for better performance, and even contribute to game design. It integrates multiple state-of-the-art AI techniques into a single, cohesive system.

### 🤖 An Ecosystem of Specialized LLM Agents
The foundation is a single, powerful LLM (`Salesforce/codegen-2B-mono`), which is then specialized into a multitude of "expert" agents using efficient fine-tuning techniques.

-   **Core Model Technology**:
    -   **4-bit Quantization (BitsAndBytes)**: This technique drastically reduces the model's memory footprint, allowing a 2-billion-parameter model to run on consumer GPUs.
    -   **LoRA (Low-Rank Adaptation)**: An incredibly efficient fine-tuning method. Instead of retraining the entire model, we only train small "adapter" layers, saving immense amounts of time and computational resources.

-   **The Agent Swarm (>10 Specialized Agents)**:
    -   **Code Generation & Refinement**: `CodeGeneratorAgent`, `CodeRefinementAgent`, `AutoRefactoringAgent`.
    -   **Analysis & Critique**: `CodeCriticAgent`, `BugReportGeneratorAgent`, `CodeSummarizationAgent`.
    -   **Testing & Documentation**: `TestGenerationAgent`, `DocumentationAgent`.
    -   **Creative & Planning**: `AssetGeneratorAgent`, `GameDesignerAgent`.
    -   **Knowledge Retrieval**: `CodeQuestionAnsweringAgent`.

### 🛡️ Security-First Error Handling and Code Execution
Running AI-generated code is inherently risky. This project's most critical feature is its multi-layered security protocol.
-   **Sandboxed Code Execution with Docker**: The absolute cornerstone of safety. Every piece of code generated by an agent is executed inside a heavily restricted, ephemeral **Docker Container**. This container has no network access and is limited in CPU and RAM usage, completely isolating it from the host system and preventing any potential harm.
-   **Multi-layered Security Audits**: Before execution, code undergoes a rigorous automated audit:
    1.  **Static Analysis**: `luacheck` is used to find syntax errors.
    2.  **Pattern Matching**: Regular expressions scan for dangerous patterns like `os.execute` or `loadstring`.
    3.  **Vulnerability Scanning**: The code is checked for common vulnerability patterns (e.g., potential for SQL injection, buffer overflows).

### 🌟 The Research Frontier: Advanced AI Techniques
-   **Knowledge Graph-Augmented LLM**: This is the system's most profound innovation. The entire codebase of a game project is parsed into an **Abstract Syntax Tree (AST)** and then converted into a **Knowledge Graph**. This graph, where functions are nodes and calls are edges, is processed by a GNN (`ProjectGraphMemory`). The resulting graph embedding is injected directly into the LLM's context. This gives the LLM a holistic, structural understanding of the entire project, far beyond just reading text.
-   **Retrieval-Augmented Generation (RAG)**: The system maintains a `VectorizedMemory`—a database of high-quality code snippets from the project, embedded as vectors. When a new task arrives, the relevant agents first perform a similarity search to retrieve the most relevant, high-quality examples. These examples are then provided to the LLM as "inspiration," dramatically improving the quality of the generated code.
-   **Fine-tuning with Reinforcement Learning (PPO)**: The agents are not just trained on static data; they learn through trial and error. The `CodeEvaluator` (running in the secure Docker sandbox) acts as the RL "environment," providing a "reward" signal based on whether the generated code runs, passes tests, and is efficient. The agents are then fine-tuned using **Proximal Policy Optimization (PPO)** to maximize this reward.
-   **Prioritized Experience Replay (PER)**: A sophisticated learning buffer that allows the agents to learn more efficiently by replaying and focusing on the mistakes that were most surprising or led to the biggest errors.
-   **Sequential Fine-tuning Pipeline**: The `main` function is designed as a curriculum. The system can first master Roblox development, then sequentially learn Godot, Unity, and Unreal Engine, allowing a single core model to become a polyglot game development expert.

---

# 💡 Vision & Philosophy

My approach is to build systems that are not just academically interesting but are architected with the robustness required for real-world application. I believe in learning by building complex, end-to-end projects that force me to solve practical problems in reliability, scalability, and efficiency.

-   **Build First, Theorize Later**: I learn best by tackling ambitious, large-scale projects directly.
-   **Resilience is a Feature**: A system that cannot gracefully handle failure is incomplete.
-   **AI Must Be Transparent**: I strive to build systems that are explainable (XAI), not opaque black boxes.
-   **Compound Growth**: Each project serves as a foundation of knowledge and tools for the next, more ambitious endeavor.

---

# 📬 Contact

I am actively seeking opportunities and collaborations. Let's connect.
-   📧 **Email:** [yoglawm644@gmail.com](mailto:yoglawm644@gmail.com)
-   🌐 **Facebook Page:** [FWK Multiverse](https://www.facebook.com/FWKMultiverse/)
-   🐦 **Twitter/X:** [@FWK_Multiverse](https://x.com/FWK_Multiverse)

---

> ⚡ Thank you for taking the time to review my work. This portfolio is a living document that will evolve as I continue to build and innovate.
