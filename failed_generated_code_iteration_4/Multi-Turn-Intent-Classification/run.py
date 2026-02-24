import os
import json
import csv
import re
import emoji
from typing import List
from transformers import pipeline

class IntentDetector:

    #A class to determine customer intent from conversation text using zero-shot classification.
    def __init__(self):
        # Define possible customer intents
        self.intent_options = [
            "Book Appointment",
            "Product Inquiry",
            "Pricing Negotiation",
            "Support Request",
            "Follow-Up"
        ]
        # Initialize the zero-shot classification pipeline with a pre-trained model
        self.intent_pipeline = pipeline(
            task="zero-shot-classification",
            model="cross-encoder/nli-distilroberta-base",
            
            
        )

    def classify_intent(self, dialogue: str) -> dict:
        #Predicts the most likely intent from a given conversation.
        # Args: dialogue (str): The input conversation or message.
        # Returns: dict: Contains the predicted intent and a brief explanation.
       
        classification = self.intent_pipeline(dialogue, self.intent_options)
        top_intent = classification["labels"][0]
        explanation = (
            f"Based on the conversation, the customer is likely interested in '{top_intent.lower()}'."
        )
        return {
            "predicted_intent": top_intent,
            "rationale": explanation
        }


# Initialize the intent detector
intent_model = IntentDetector()

def clean_and_lowercase(text: str) -> str:
     # Remove emojis
    text = emoji.replace_emoji(text, replace='')
    cleaned = text.lower()  # make all characters lowercase
    cleaned = re.sub(r"[^\w\s]", "", cleaned)  # eliminate punctuation
    cleaned = re.sub(r"\s+", " ", cleaned)  # condense multiple spaces
    return cleaned.strip()  
# Formats a list of chat messages into a single conversation string
def create_conversation(messages: List[dict], max_messages: int = None) -> str:
    if max_messages is not None:
        messages = messages[-max_messages:]
    formatted_lines = [
        f"{m.get('sender', '').capitalize()}: {m.get('text', '')}" for m in messages
    ]
    return "\n".join(formatted_lines)

def predict_intents(input_file: str, json_output: str, csv_output: str):
    """
    Executes the prediction workflow:
    - Reads conversations from a JSON file
    - Predicts intent for each conversation
    - Saves results to both JSON and CSV formats
    """
    # Load conversation data
    with open(input_file, 'r') as infile:
        conversations = json.load(infile)

    output_data = []

    # Iterate through each conversation entry
    for entry in conversations:
        conv_id = entry.get('conversation_id')
        formatted_text = create_conversation(entry.get('messages', []))
        intent_result = intent_model.classify_intent(formatted_text)

        output_record = {
            "conversation_id": conv_id,
            "predicted_intent": intent_result["predicted_intent"],
            "rationale": intent_result["rationale"]
        }
        output_data.append(output_record)

    # Write results to a JSON file
    with open(json_output, 'w') as json_file:
        json.dump(output_data, json_file, indent=2)

    # Write results to a CSV file
    with open(csv_output, 'w', newline='') as csv_file:
        fieldnames = ["conversation_id", "predicted_intent", "rationale"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_data)


os.makedirs("data/output", exist_ok=True)
if __name__ == "__main__":
       predict_intents(
    input_file="data/input.json",
    json_output="data/output/predictions.json",
    csv_output="data/output/predictions.csv"
)
    
