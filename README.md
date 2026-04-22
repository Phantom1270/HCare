# MedVigilant 🛡️

A single-page medical safety web application with an integrated **AI Safety Evaluation Layer** that provides:
1. **Interaction Guard**: Drug-drug interaction alerts for a virtual medicine cabinet.
2. **DermAlert**: Skin rash triage using a pre-trained AI model to provide Green, Yellow, or Red risk assessments.
3. **Ask MedVigilant**: LLM-powered medication safety Q&A with multi-layer guardrails.
4. **Safety Evaluation Framework**: Adversarial robustness testing across all AI pipelines.

## Technologies Used
- **Frontend/Backend:** Python Streamlit
- **Database:** MongoDB Atlas
- **AI/ML:** PyTorch, DINOv2 (HuggingFace Transformers), EasyOCR
- **LLM:** Google Gemini API (free tier)
- **APIs:** RxNorm, OpenFDA
- **Safety:** Custom adversarial testing framework with runtime guardrails

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/Phantom1270/HCare.git
cd HCare
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Create a `.env` file in the project root
```env
MONGO_URI=your_mongodb_connection_string
GEMINI_API_KEY=your_gemini_api_key
```

**How to get these (both are free):**
- **MONGO_URI**: Create a free cluster at [mongodb.com/atlas](https://www.mongodb.com/cloud/atlas) → Connect → Drivers → Copy connection string
- **GEMINI_API_KEY**: Get a free key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)

### 4. Create `.streamlit/secrets.toml` (can be empty for local dev)
```bash
mkdir .streamlit
echo "# Secrets" > .streamlit/secrets.toml
```

### 5. Run the application
```bash
python -m streamlit run main.py
```

## Safety Evaluation Layer

Run the adversarial safety audit (no API keys needed):
```bash
python -X utf8 run_safety_audit.py --verbose
```

View the safety dashboard:
```bash
python -m streamlit run safety_dashboard.py
```

## Demo Instructions
- To test the **Interaction Guard**, add "Aspirin" and then "Warfarin" to your Virtual Medicine Cabinet. It should trigger a Red Alert due to the high-risk interaction.
- To test the **DermAlert**, upload any picture. The model will parse it and provide a risk assessment Level (Green, Yellow, or Red).
- To test the **Ask MedVigilant**, type a medication safety question like "Can I drink alcohol while on Warfarin?"
- To test **Safety Guardrails**, try typing `{"$gt": ""}` as a drug name — it will be blocked by the injection detector.

## 🌐 Features
- Displays top 3 AI predictions for better accuracy interpretation
- Runtime safety guardrails (input sanitization, output validation, confidence gating)
- 30 adversarial test cases across 4 pipelines
- LLM-powered Q&A with constrained system prompt