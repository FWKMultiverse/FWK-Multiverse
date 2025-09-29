# 🚀 AI Systems Portfolio by a 16-Year-Old Developer

![Python Version](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Frameworks](https://img.shields.io/badge/Frameworks-TensorFlow%20%7C%20PyTorch%20%7C%20Scikit--Learn-orange.svg)
![License](https://img.shields.io/badge/Code-Proprietary-red.svg)
![Status](https://img.shields.io/badge/Status-Ongoing%20Development-green.svg)

Welcome to my portfolio! I'm a 16-year-old self-taught AI developer passionate about building complex, resilient, and intelligent systems from the ground up. This repository provides a high-level overview of my key projects in automated trading, competitive data science, and AI-driven game development.

> ⚠️ **Important Note:** This portfolio contains **high-level descriptions only**. The source code for these projects is proprietary and **will not be shared publicly**, as they are under continuous, independent development. All systems and models were built entirely from scratch.

---

##  प्रोजेक्ट 1: Production-Grade AI Trading System (28-Day Build)

A fully autonomous, multi-agent framework designed for live trading in Forex and Crypto markets. This system is engineered for high availability and resilience, capable of operating 24/7 with minimal human intervention.

**Core Components:**
* `fetcher.py`: A robust data ingestion module that aggregates market data from over 8 different APIs concurrently.
* `AIEngine.py`: The central intelligence core where multiple AI agents collaborate to analyze data, form strategies, and manage risk.
* `AIServer.py`: A production-ready Flask API server that handles asynchronous task management and exposes a clean interface for trade signal generation.

**AI Models & Agents:**
This system utilizes **8 distinct neural network models** orchestrated by three specialized agents:
1.  **🧠 Macro Agent:** Analyzes long-term market trends using news sentiment (News Transformer) and inter-asset correlations (Graph Neural Network) to establish a high-level market bias (e.g., bullish, bearish, neutral).
2.  **⚡ Micro Agent:** A Transformer-based agent that processes real-time price action and technical indicators to make precise, short-term decisions on entries, exits, and position management.
3.  **🛡️ Risk Agent:** A dedicated agent that dynamically adjusts trade parameters like lot size, stop-loss, and take-profit based on real-time account equity, drawdown, and overall market volatility.

---

## प्रोजेक्ट 2: Numerai Tournament AI System (6-Day Build)

A highly specialized and sophisticated modeling pipeline designed to compete in the Numerai data science tournament, a challenge known for its noisy, obfuscated financial data. This system was built for rapid experimentation and robust performance evaluation.

**System Highlights:**
* **Massive Ensemble Model:** The final prediction is a blend of over **10 diverse models**, including LightGBM, CatBoost, a temporal LSTM for time-series patterns, a GNN+Transformer hybrid, an Autoencoder for feature compression, and a Multi-Agent Reinforcement Learning (MARL) model to dynamically weight the ensemble components.
* **Advanced Feature Engineering:** The system automatically generates hundreds of new features by analyzing statistical properties (skew, kurtosis) of predefined feature groups and creating rolling-window statistics.
* **Adversarial Validation:** Before training, an adversarial model is used to identify and remove unstable features that differ between training and live data, drastically improving model generalization.
* **Hyperparameter Supremacy:** The pipeline integrates both **Optuna** and **Keras Tuner** to systematically find the optimal hyperparameters for each model, ensuring peak performance.

---

## प्रोजेक्ट 3: Multi-Agent Game Development AI (8-Day Build)

An experimental, cutting-edge AI swarm designed to act as an autonomous game development assistant. This system leverages a Large Language Model (LLM) as its core, augmented with specialized agents and memory systems to handle tasks across multiple game engines (Roblox, Godot, Unity, Unreal).

**AI Models & Agents:**
At its heart is a **`Salesforce/codegen-2B-mono` LLM**, fine-tuned using advanced techniques to run on consumer-grade hardware. This core is supported by a swarm of **10+ specialized agents**, including:
* **🧑‍🎨 Game Designer Agent:** Proposes new game mechanics and features.
* **✍️ Code Generator & Refinement Agents:** Write, critique, and improve code based on feedback.
* **🐛 Bug Report Agent:** Analyzes error logs to generate detailed bug reports.
* **🧪 Test Generation Agent:** Automatically writes unit and integration tests.
* **📚 Documentation Agent:** Generates clear, human-readable documentation for the code.
* **✨ Auto-Refactoring Agent:** Improves code quality based on principles like DRY and KISS.

**Groundbreaking Technologies Used:**
* **Resource-Efficient Training:** The LLM is trained using **4-bit quantization** and **LoRA (Low-Rank Adaptation)**, making it possible to fine-tune a 2-billion-parameter model on limited VRAM.
* **Knowledge Graph Memory (GNN):** The system builds a dynamic knowledge graph of the entire codebase to understand function calls, dependencies, and asset usage, providing deep context to the LLM.
* **Retrieval-Augmented Generation (RAG):** The AI maintains a vectorized memory of high-quality code snippets. When generating new code, it retrieves the most relevant examples to improve accuracy and quality, mimicking an expert developer's experience.
* **Intrinsic Curiosity (PPO Training):** The Reinforcement Learning loop includes a "curiosity module" that rewards the AI for exploring novel and effective solutions, preventing it from getting stuck in repetitive patterns.

---

### 🌟 Key Architectural Strengths Across All Projects

My development philosophy centers on building systems that are not just intelligent, but also incredibly robust and fail-safe.
* **Resilient Fetcher System:** API interactions are wrapped with automatic retries, exponential backoff, rate limiting, and intelligent cooldowns. The system can withstand API failures without crashing.
* **Server State Management:** The API server uses state flags and decorators to prevent requests from being processed before the AI models are fully initialized, eliminating race conditions.
* **Asynchronous Background Tasks:** Heavy computational tasks like model training and signal generation are offloaded to background worker threads and managed via queues, ensuring the main server remains responsive.
* **Data Sanitization & Validation:** Proactive functions scan for and automatically delete corrupted or empty data files (e.g., invalid JSON), preventing data-related runtime errors.
* **Extreme Resource Management:** A combination of hardware-level optimizations (like GPU offloading and model quantization) and software-level techniques (like memory reduction and data type optimization) are used to run complex models on limited hardware.
* **Explainable AI (XAI):** The trading system integrates **SHAP (SHapley Additive exPlanations)** to interpret and visualize why a model made a specific decision, which is crucial for debugging and building trust.
* **Robust Training Pipelines:** Training isn't just a script; it's a managed process with automated early stopping, learning rate reduction, and adversarial validation to ensure stable and effective model convergence.
* **Fallback & Failsafe Logic:** The code is filled with failsafe mechanisms. If a critical module or file is missing, the system gracefully falls back to a default or dummy implementation instead of crashing.

---

### 📞 Contact Me

I am always open to learning and discussing new ideas. Feel free to reach out!

* **Email:** [yoglawm644@gmail.com](mailto:yoglawm644@gmail.com)
* **Facebook Page:** [FWK Multiverse](https://www.facebook.com/FWKMultiverse/)
* **X (Twitter):** [@FWK_Multiverse](https://x.com/FWK_Multiverse)

Thank you for visiting!
