# 🅿️ ParkIQ: The Intelligent Parking Command Center
**Comprehensive Project & Architecture Report**

---

## 1. Introduction & Problem Statement

### 1.1 The Challenge of Urban Parking
In modern smart cities, urban parking congestion is a critical issue, contributing to approximately 30% of inner-city traffic as drivers circle for spaces. This leads to increased carbon emissions, wasted time, and lost municipal revenue. Current parking prediction models typically rely on simple deterministic machine learning algorithms. While these can provide point predictions (e.g., "The parking lot will be full"), they suffer from three major flaws:
1. **Lack of Uncertainty**: They cannot express when they are unsure. A prediction of "Full" might be a guess based on noisy data, but the model presents it as a fact.
2. **Black Box Nature**: They do not explain *why* they made a decision.
3. **Brittleness**: They fail silently when real-world conditions deviate from training data (e.g., an unprecedented storm).

### 1.2 The ParkIQ Solution
ParkIQ was built to solve these exact problems. It is a production-grade, highly interactive command center that leverages a **Bayesian Neural-Symbolic (BNS)** architecture. Instead of just predicting parking occupancy, ParkIQ is built around **MLOps principles of Trust, Explainability, and Reliability**. It tells city planners what will happen, *why* it will happen, and exactly *how confident* the AI is in its prediction.

---

## 2. Data & Feature Engineering

ParkIQ's predictive engine is fueled by a rich combination of temporal and environmental data.

### 2.1 The Features
The model consumes 13 distinct features for every prediction:
*   **Temporal Features**: Hour of Day, Day of Month, Day of Week, Month, and a Weekend boolean flag.
*   **Cyclical Encoding**: To ensure the neural network understands that Hour 23 is right next to Hour 0, temporal features (Hour, Day of Week, Month) are encoded using sine and cosine transformations.
*   **Weather Features**: Air Temperature (°C), Dew Point, Wind Speed (m/s), Atmospheric Pressure, and Rainfall (mm).

### 2.2 Preprocessing
All incoming data is passed through a cached `StandardScaler` to normalize the features into Z-scores. This ensures that the neural networks train smoothly and allows us to easily detect statistical anomalies in live data.

---

## 3. Core AI Architecture: The Model Zoo

ParkIQ does not rely on a single algorithm. Instead, it hosts a hot-swappable "Model Zoo" managed by a FastAPI backend. This allows stakeholders to choose the right tool for the job.

### 3.1 XGBoost (The Deterministic Baseline)
*   **What it is**: A highly optimized gradient boosted tree architecture.
*   **Role**: It serves as the high-accuracy baseline and provides the core logic for our SHAP explainability layer.

### 3.2 Bayesian Neural Networks (BNN)
*   **What it is**: Unlike standard neural networks that have fixed weights (e.g., a weight of `0.5`), a BNN learns a *probability distribution* for every single weight (e.g., a Gaussian distribution with a mean of `0.5` and a standard deviation of `0.1`).
*   **Variants Included**:
    *   `BNN-Small` (32→16 layer architecture)
    *   `BNN-Large` (128→64 layer architecture)
    *   `BNN-LowKL` & `BNN-HighKL`: These variants alter the Kullback-Leibler (KL) divergence penalty, changing how strictly the model adheres to its prior assumptions versus the training data.
*   **Role**: Because the weights are distributions, we can pass the same input through the network 50 times (Monte Carlo sampling) and get 50 slightly different answers. The spread of these answers gives us a mathematical quantification of "Epistemic Uncertainty" (what the model doesn't know).

### 3.3 Bayesian Neural-Symbolic (BNS) - The Flagship
*   **What it is**: The BNS model combines the deep probabilistic learning of the BNN-Large architecture with symbolic logic constraints.
*   **Target Similarity Constraint (TSC)**: During training, a custom loss function mathematically penalizes the network if its aggregated predictions violate known, macroscopic domain rules (like known empirical peak-hour congestion curves). 
*   **Role**: It provides the perfect balance of deep learning pattern recognition and strict adherence to real-world physics and city planning constraints.

---

## 4. Building Trust: The MLOps Layer

For an AI to be deployed in a city's infrastructure, it must be trusted. ParkIQ implements four major systems to guarantee transparency.

### 4.1 Model Arbitration Engine
In high-stakes environments, a single model failure can be costly. ParkIQ runs a dual-engine inference pipeline. 
*   Live data is fed simultaneously into the XGBoost baseline and the selected Bayesian model.
*   If both models output the same congestion level (e.g., Level 4 Critical), the system flags a "✅ Models Agree" verdict, signaling high trust.
*   If they disagree, the system flags a "⚠️ Model Conflict" and warns the operator that human judgment is required.

### 4.2 SHAP Explainability (Feature Attribution)
*   Using `shap.TreeExplainer`, the dashboard dynamically breaks down exactly *why* a specific prediction was made. 
*   It generates a localized bar chart showing how much impact Temperature, Rain, or Time of Day had on the current prediction, transforming the AI from a black box into an interpretable advisor.

### 4.3 Reliability Calibration
*   Does the model actually know when it is wrong? 
*   ParkIQ computes a Reliability Diagram (Calibration Curve) by binning the model's self-reported confidence against actual historical accuracy. A perfectly calibrated model should be exactly 80% accurate when it claims to be 80% confident. The dashboard visualizes this alignment to prove statistical honesty.

### 4.4 Data Drift & Anomaly Detection (Out-of-Distribution)
*   Models fail when real-world data looks nothing like historical training data (e.g., an unprecedented storm).
*   ParkIQ computes the Z-scores of incoming live features against the training scaler.
*   If the maximum deviation exceeds `2.0σ`, a **High Deviation Warning** is triggered, explicitly listing the anomalous features. If it exceeds `3.0σ`, it triggers a **Critical Out-of-Distribution** alert.

---

## 5. System Engineering & Deployment

### 5.1 FastAPI Backend
The computation is entirely decoupled from the user interface. A highly asynchronous `FastAPI` backend handles the heavy lifting. On startup, it synthesizes, trains, and caches the weights for the entire Bayesian model zoo. The `/predict` endpoint handles the complex Monte Carlo sampling, SHAP extraction, and drift Z-score calculations in milliseconds.

### 5.2 Streamlit Command Center (Frontend)
The user interface is designed for city operators. It features:
*   **Live Sensor Mode**: Automatically polls the backend for live (or simulated) data.
*   **Dynamic KPI Cards**: Glowing, responsive cards showing Congestion Level, Available Spots, AI Confidence, and Projected Revenue.
*   **Risk Radar**: A multi-dimensional polar chart assessing congestion, weather, time, and revenue risks simultaneously.
*   **PDF Reporting**: Generates professional, timestamped executive summaries of current conditions and model verdicts with a single click.

### 5.3 Docker Containerization
The entire stack is packaged into a dual-process Docker container, making it completely cloud-ready. A custom `start.sh` script launches `uvicorn` (backend) and `streamlit` (frontend) concurrently, while a `docker-compose.yml` file ensures the application can be seamlessly deployed to any environment (AWS, GCP, Azure) using a single command.

---

## 6. Conclusion
ParkIQ bridges the gap between theoretical probabilistic deep learning and actionable, MLOps-driven city intelligence. By combining Bayesian uncertainty, Symbolic constraints, and rigorous drift detection into a stunning, containerized dashboard, it represents the gold standard for modern, trustworthy AI deployments.
