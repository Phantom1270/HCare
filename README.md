# MedVigilant

A single-page medical safety web application that provides:
1. **Interaction Guard**: Drug-drug interaction alerts for a virtual medicine cabinet.
2. **DermAlert**: Skin rash triage using a pre-trained AI model to provide Green, Yellow, or Red risk assessments.

## Technologies Used
- **Frontend/Backend:** Python Streamlit
- **Logic:** RxNorm API for chemical ID resolution
- **Machine Learning:** PyTorch & Torchvision (MobileNetV2)

## Setup Instructions

1. **Clone the repository or navigate to this directory.**
2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the application:**
   ```bash
   streamlit run main.py
   ```

## Demo Instructions
- To test the **Interaction Guard**, add "Aspirin" and then "Warfarin" to your Virtual Medicine Cabinet. It should trigger a Red Alert due to the high-risk interaction.
- To test the **DermAlert**, upload any picture. The model will parse it and provide a mock risk assessment Level (Green, Yellow, or Red).

## 🌐 Features Added
- Displays top 3 AI predictions for better accuracy interpretation