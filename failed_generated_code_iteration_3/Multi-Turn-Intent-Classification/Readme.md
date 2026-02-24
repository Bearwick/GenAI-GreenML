# Multi-Turn Intent Classification

## 🚀 Project Overview
This project predicts customer intent from WhatsApp-style multi-turn conversations using a zero-shot transformer model.

## 🤖 Model Choice
We use the `cross-encoder/nli-distilroberta-base` model from HuggingFace for zero-shot intent classification. This model is designed for natural language inference (NLI) tasks and can be adapted for intent classification by treating each intent as a possible label and selecting the most likely one for a given conversation.


## 📦 How to Run
1. Install dependencies:
    pip install -r requirements.txt
2. Put input in: `data/input.json`
3. Run:
    python run.py
4. Outputs saved to:
    - data/output/predictions.json
    - data/output/predictions.csv


## 📁 Project Structure
- `preprocess.py`: cleans and formats conversations
- `model.py`: runs classification
- `predictor.py`: handles input/output
- `run.py`: main entry script

## 📌 Notes
- No training needed — uses zero-shot classification.
- Can handle thousands of conversations efficiently.

## 📍 Example Prediction
Input: "Can we do a site visit this week?"  
Prediction: Book Appointment  
Rationale: The user requested a site visit after sharing preferences.

## ⚠️ Limitations & Edge Cases
- The `cross-encoder/nli-distilroberta-base` model is not officially supported for the HuggingFace zero-shot-classification pipeline.
- Rationale generation is generic and may not always reflect the specific context of the conversation.
- Only English conversations are supported.