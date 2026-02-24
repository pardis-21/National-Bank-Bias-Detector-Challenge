# 🏦 National Bank Bias Detector (NBC)
**An AI-powered behavioral finance tool designed to identify and mitigate psychological trading biases.**

The NBC Bias Detector is a data-driven application that analyzes trading history (CSV) to identify common psychological pitfalls. It combines a custom mathematical "Bias Engine" with a Gemini-powered **AI Trading Coach** to provide traders with actionable feedback and personalized mentorship.

---

## 🚀 Quick Start Guide

### 1. Prerequisites
Ensure you have **Python 3.9** or higher installed. You can verify your version by running:
```bash
python --version
```

### 2. Installation & Setup
Clone this repository to your local machine and install the required dependencies
```bash
# Install the core stack
pip install streamlit pandas google-genai plotly
```

### 3. Configure Gemini AI (Required)
This application uses Google Gemini to power the AI Coaching feature
  1. Get a Free API Key: Visit Google AI Studio and generate a key
  2. Configure Secrets: * In the project root, create a folder named ```.streamlit```
       -Inside ```.streamlit```, create a file named ```secrets.toml```.
       -Add your key exactly as shown below:
```Ini, TOML
GEMINI_API_KEY = "YOUR_ACTUAL_API_KEY_HERE"
```

### 4. Running the Application
Launch the web interface through your terminal:
```bash
python -m streamlit run app.py
```

=========================================================================================================================================================================


# 📊 Bias Engine Trading Analyzer

## How to Use the App

### Step 1: Prepare Your Data

The Bias Engine analyzes your trade history. Ensure your CSV file includes these exact column headers:

| Column        | Description                                         |
| ------------- | --------------------------------------------------- |
| `timestamp`   | Date and time of the trade (e.g., 2025-10-01 14:30) |
| `asset`       | Ticker symbol or asset name (e.g., BTC, AAPL)       |
| `buy_sell`    | "Buy" or "Sell"                                     |
| `quantity`    | The amount traded                                   |
| `entry_price` | Price when the trade was opened                     |
| `exit_price`  | Price when the trade was closed                     |
| `profit_loss` | The net gain or loss for that trade                 |
| `balance`     | Account balance after the trade                     |

---

### Step 2: Analysis & Customization

* **Upload:** Use the sidebar to upload your trading CSVs.
* **Thresholds:** Adjust the detection sensitivities (e.g., defining how many trades per hour constitutes "Overtrading").
* **Review:** Explore the Dashboard metrics, equity curves, and specific trade flags.

---

### Step 3: Chat with the AI Coach

Interact with the AI Trading Coach at the bottom of the dashboard. Ask specific questions like:

* "Based on my Loss Aversion flags, what should my stop-loss strategy be?"
* "Is my overtrading linked to specific times of the day?"

---

## 🛠️ Project Architecture

```
app.py                     # Main entry point and UI layout
bias_engine.py             # Mathematical core that scans data for patterns
ai_coach.py                # Modular script handling Gemini API connection
.streamlit/secrets.toml    # Secure storage for API credentials (gitignored)
```

---

## 🧠 Biases Detected

**🔄 Overtrading**
Flags high-frequency trade bursts, rapid position flipping, and excessive volume vs. balance.

**😰 Loss Aversion**
Identifies the "Disposition Effect"—holding losers too long while cutting winners short.

**😤 Revenge Trading**
Detects oversized "makeup" trades opened immediately following a large loss.

---

## ❓ Troubleshooting

**404 Model Not Found**
Ensure you are using a supported model like `gemini-1.5-flash` in the sidebar.

**"bias_summary" Error**
Ensure you have uploaded a CSV file before attempting to chat with the AI.

**Secrets Missing**
Verify that your `.streamlit/secrets.toml` file is spelled correctly and contains the correct key.

---

## 🏛️ Project Info

Developed for the **National Bank Bias Detector Challenge (2025–2026)**
Created with passion for Behavioral Finance.
