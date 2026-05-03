# 🎛️ ParkIQ Dashboard: User Guide & Feature Breakdown

Welcome to the **ParkIQ Command Center**. This dashboard is the front-facing interface designed for city planners, traffic operators, and stakeholders to monitor parking congestion, understand AI reasoning, and take actionable steps to manage city infrastructure.

This document serves as a detailed walkthrough of the dashboard's capabilities, organized by the four main tabs.

---

## 1. 🎛️ Command Center (Main Tab)

The Command Center is your operational hub. It is designed to give you an immediate, high-level overview of the current or simulated situation.

### ⚙️ Environment Controls & Live Mode
*   **Live Sensor Feed**: Toggle this in the sidebar to simulate a real-time data stream. The dashboard will automatically pull data every 5 seconds.
*   **Manual Override**: When Live Mode is off, you can expand the "Environment Controls" panel to manually test scenarios (e.g., "What happens if there is heavy rain at 5:00 PM on a Friday?").

### 📊 Top KPIs (Key Performance Indicators)
*   **Congestion Level**: The AI's final prediction for parking density, rated from Level 0 (Empty) to Level 4 (Critical).
*   **Est. Available Spots**: A live calculation showing exactly how many physical parking spaces remain based on the predicted percentage of occupancy.
*   **AI Confidence**: Shows the model's certainty in its prediction. If confidence drops below 80%, the number turns yellow, signaling that the system is unsure (often due to highly unusual weather or time conditions).
*   **Projected Revenue**: Estimates the current revenue generation rate, which dynamically increases if Surge Pricing is activated at higher congestion levels.

### 🕸️ Risk Radar & Action Items
*   **Risk Radar**: A multi-dimensional chart that maps out 5 distinct risk factors: Congestion, Weather Risk, Time Risk, Capacity Risk, and Revenue Risk. A larger shape means higher overall operational risk.
*   **Action Items**: Based on the congestion level, the system automatically suggests standard operating procedures. For example, at Level 4, it will advise activating overflow facilities and alerting transport authorities.

---

## 2. 🔬 Model Intelligence (MLOps Tab)

The Model Intelligence tab is what separates ParkIQ from standard machine learning projects. It exists to build **Trust** by proving why the model makes its decisions.

### 🧩 SHAP Feature Importance (Explainability)
*   **What it does**: This bar chart breaks open the "black box" of the AI.
*   **How to read it**: It shows exactly which environmental factors (e.g., Temperature, Rainfall, Hour of Day) had the biggest impact on the current prediction. Longer bars mean higher impact.

### 📐 Calibration Diagram (Reliability)
*   **What it does**: It proves that the model isn't just confident, but *correctly* confident.
*   **How to read it**: The dashed line represents "perfect" confidence. If the blue bars align closely with the dashed line, it means when the AI says it is "90% confident," it is historically accurate 90% of the time.

### ⚖️ Model Arbitration
*   **What it does**: We don't rely on just one model. This section shows the prediction of the deterministic XGBoost model side-by-side with your chosen Bayesian Neural Network.
*   **How to read it**: If they agree, the Verdict shows "✅ Agree" for high trust. If they disagree, it flags a "⚠️ Conflict" and warns the operator that human judgement is required.

### 🌊 Data Drift & Anomaly Detection
*   **What it does**: ML models break when real-world data looks nothing like historical training data (e.g., an unprecedented storm).
*   **How to read it**: It calculates the Z-Score of the current weather/time data. If the deviation is in the "green," operations are normal. If it hits "red" (>3.0σ), it explicitly lists the anomalous features (e.g., Wind Speed) and warns that AI predictions may currently be unreliable.

---

## 3. 📊 Analytics

The Analytics tab provides a macroscopic view of historical and simulated data patterns, crucial for long-term city planning.

*   **Temporal Heatmap**: A massive grid showing average congestion levels across every hour of the day and every day of the week. Perfect for spotting recurring rush-hour bottlenecks.
*   **Weather Impact Scatter**: Plots Congestion Level against Rainfall and Temperature. This helps planners visualize how severe weather events correlate with parking scarcity.

---

## 4. 📄 Reports

For executive stakeholders who need summaries rather than live dashboards.

*   **One-Click PDF Generation**: Click "Generate Executive Summary" to instantly create a branded, formatted PDF.
*   **Contents**: The report captures a snapshot of the current timestamp, weather conditions, the AI's congestion prediction, uncertainty metrics, and the recommended operational actions.
*   **Use Case**: This is ideal for shift handovers, daily logging, or automated reporting to city council officials.

---

## 💡 Sidebar Options
Don't forget to utilize the left sidebar!
*   **Model Zoo**: Use the dropdown to switch the underlying brain of the dashboard. Want raw accuracy? Use XGBoost. Want deep uncertainty quantification? Switch to the BNS or BNN variants. The dashboard's "AI Confidence" and "Uncertainty Violin" charts will instantly update to reflect your chosen model's architecture.
