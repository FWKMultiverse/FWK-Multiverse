# 🚀 AI Projects Portfolio – by a 16-Year-Old Enthusiast

![GitHub Repo Size](https://img.shields.io/github/repo-size/your-username/your-repo)
![GitHub stars](https://img.shields.io/github/stars/your-username/your-repo)
![GitHub license](https://img.shields.io/github/license/your-username/your-repo)
![Python](https://img.shields.io/badge/python-3.11-blue)
![AI](https://img.shields.io/badge/AI-MultiAgent-green)
![RL](https://img.shields.io/badge/Reinforcement-Learning-orange)
![Status](https://img.shields.io/badge/Code-Private-red)

---

Welcome!  
This repository is a **portfolio of personal AI projects** I’ve built independently.  
I am **16 years old** with formal education ending at grade 9, but I’ve followed my curiosity and passion to create AI systems that cover **trading, data science competitions, and multi-agent game development**.  

⚠️ **Important Notes**:  
- No source code or implementation details are included here.  
- All work has been created from scratch.  
- This document is a **high-level overview**, not a tutorial or technical deep-dive.  
- In the future, I may open-source smaller side projects, but **the core codebase will remain private**.  

---

# 📂 Projects Overview

I’ve developed 3 main projects so far:

1. **AI Trading System** – 28 days (Full-stack AI for Forex & Crypto)  
2. **Numerai Challenge System** – 6 days (Advanced ensemble learning)  
3. **Multi-Agent LLM for Game Development** – 8 days (Cutting-edge research-level system)  

Each project is described below in detail.

---

## 1️⃣ AI Trading System (28 Days)

**Level:** Production-Grade / Professional  
**Tech Stack:** Python, Async I/O, Multi-Agent RL, Transformers, GNN, Flask, XAI  

### 📝 Overview
This project is a **modular AI trading system** designed for real-time Forex and Crypto trading.  
It has three main layers:  

- **Fetcher (`fetcher.py`)** → Collects live market data (prices, news, sentiment, search trends).  
- **AI Server (`AIServer.py`)** → The command center (API interface, queue, routing).  
- **AI Engine (`AIEngine.py`)** → The core brain: processes signals and executes trades.  

The system focuses on **robustness, reliability, and real-time resilience**.

---

### 🤖 Agents & Models
- **Macro Agent** → Long-term market trends (H1, H4, Daily).  
- **Micro Agent** → Short-term trading (M5, M15), uses Transformer for time-series.  
- **Risk Agent** → Portfolio risk management (drawdown, PnL, lot sizing).  
- **GNN Analyzer** → Captures correlations between assets dynamically.  
- **News Transformer** → Embeds and interprets financial news for sentiment analysis.  

Together, these agents form a **Multi-Agent RL System** where each agent specializes in one aspect.

---

### 🛡️ Error Handling & Reliability
- **Retry with Exponential Backoff** → Automatic recovery for failed API calls.  
- **Rate Limiting & Cooldown** → Prevents API bans or throttling.  
- **Caching System** → Reduces unnecessary API calls, improves response time.  
- **Endpoint Guards** → Prevents invalid requests when AI isn’t ready.  
- **Timeouts** → Avoids system freeze when responses are delayed.  
- **Corrupt Data Cleaning** → Detects and removes broken data files before training.  

---

### 🌟 Special Features
- **Explainable AI (XAI with SHAP)** → Transparent reasoning behind trades.  
- **Kelly Criterion Risk Sizing** → Mathematically optimized lot sizing.  
- **Dynamic Graphs** → Real-time asset correlation analysis via GNN.  
- **Multi-Timeframe Fusion** → Combines micro + macro analysis for balance.  

---

## 2️⃣ Numerai Challenge System (6 Days)

**Level:** Expert / Research-Grade  
**Tech Stack:** Python, LightGBM, CatBoost, LSTM, Transformer, GNN, Gaussian Processes  

### 📝 Overview
Built in only 6 days, this project was made for the **Numerai Tournament**.  
It’s an **ensemble-of-ensembles system**, combining **10+ AI models** for structured tabular and temporal data.  

---

### 🔬 Pipeline
1. **Data Conversion** → Raw parquet → JSON (Era-grouped).  
2. **Feature Engineering** → Missing value handling, imputation, normalization.  
3. **Model Training** → Multiple ML and DL models run in parallel.  
4. **Ensembling** → Predictions blended via meta-models.  
5. **Validation** → Era-based cross-validation.  
6. **Submission Simulation** → Scoring predictions locally before submission.  

---

### 🤖 Models & Techniques
- **LightGBM & CatBoost** → Core tabular predictors.  
- **LSTM** → Learns temporal dependencies across Eras.  
- **GNN + Transformer Hybrid** → Captures relational + attention patterns.  
- **Autoencoder** → Compresses and extracts latent features.  
- **Multi-Agent RL** → Blends signals dynamically.  
- **Meta-Models** → Ridge Regression & Gaussian Processes refine predictions.  

---

### 🛡️ Error Handling & Robustness
- **Library Fallbacks** → System runs even if advanced libs are missing.  
- **Memory Reduction** → Downcasts data to run on limited RAM.  
- **Parallel Training** → Multiple seeds in parallel reduce overfitting.  
- **Adversarial Validation** → Removes unstable features before training.  
- **Purged CV** → Prevents data leakage across Eras.  

---

### 🌟 Highlights
- **Automated Hyperparameter Tuning** → Optuna + Keras Tuner.  
- **Bagging with Seeds** → Stability through ensemble averaging.  
- **MMC Neutralization** → Aligns with Numerai competition rules.  

---

## 3️⃣ Multi-Agent Game Development AI (8 Days)

**Level:** Advanced R&D / Research Frontier  
**Tech Stack:** Python, PPO-RL, LLMs (CodeGen-2B), LoRA Fine-Tuning, GNN, RAG  

---

### 📝 Overview
This is the **most ambitious project** so far.  
It’s not just one AI — but an **AI ecosystem** designed to replicate a **game development team**.  

Capabilities include:  
- Writing code across multiple game engines.  
- Reviewing & refining code.  
- Detecting and fixing bugs.  
- Designing new game features.  
- Generating tests & documentation.  

---

### 🤖 Agents
- **Code Agents** → Generator, Refiner, Auto-Refactorer.  
- **Analysis Agents** → Critic, Bug Reporter, Summarizer.  
- **Testing & Docs** → Test Generator, Documentation Agent.  
- **Creative Agents** → Asset Generator, Game Designer.  
- **Q&A Agent** → Explains and answers about codebase.  

Together, this mimics an **Agile dev team** loop (build → test → feedback → refine).

---

### 🧠 Models & Techniques
- **Foundation Model** → CodeGen-2B with LoRA fine-tuning + 4-bit quantization.  
- **Knowledge Graph + GNN** → AST-based project graph injected into LLM.  
- **RAG Simulation** → Vector memory of high-quality past code.  
- **PPO Reinforcement Learning** → LLM fine-tuned via rewards from evaluator.  
- **Prioritized Replay Buffer** → Faster learning from key mistakes.  
- **Sequential Fine-Tuning** → Learns multiple engines step by step.  

---

### 🛡️ Error Handling
- **Sandboxed Execution** → Docker containers for safe evaluation.  
- **Static + Dynamic Audits** → Regex, AST checks, vulnerability scanning.  
- **Retry + Logging** → All errors logged, retried, and analyzed.  

---

### 🌟 Future Plans
- **Full Chat Integration** → Conversational AI like GPT but code-focused.  
- **VSCode Integration** → AI directly embedded in dev tools.  
- **Multi-Engine Expansion** → Unity, Unreal, Godot, Roblox all supported.  
- **Cross-Domain Expansion** → From code to finance, data, and more.  
- **Project #3 Public Service** → Planned release for real-world usage.  

---

# 🔮 Final Notes

- These projects are a reflection of **my passion and need to create**.  
- I’m only **16 years old** but I believe in learning by building.  
- Code is **kept private** to protect originality and IP.  
- More projects will follow — and some may be released publicly in the future.  

---

# 📌 Summary

- **3 Projects** (Trading, Numerai, Game AI).  
- **10+ Agents per project** (multi-agent systems).  
- **Cutting-edge techniques** → RL, LLMs, GNNs, RAG, Ensemble ML.  
- **Focus** → Robustness, resilience, and creativity.  
- **Vision** → Building a complete AI development ecosystem.  

---

> Thanks for visiting! 🌟  
> This portfolio is a **living document** and will continue to grow with each project.  
> Stay tuned for updates on **Project 3 public release** and beyond.  

