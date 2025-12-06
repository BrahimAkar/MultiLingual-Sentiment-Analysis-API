from transformers import XLMRobertaTokenizer, AutoModelForSequenceClassification
import torch
import emoji
import re
import os
import google.generativeai as genai
from dotenv import load_dotenv
import unicodedata

MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
labels = ['Negative', 'Neutral', 'Positive']

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-1.5-pro")



def preprocess(text):
    text = ''.join(c for c in text if unicodedata.category(c)[0] != "C")
    text = emoji.demojize(text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def split_into_chunks(text, max_chunk_length=480):
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        if sum(len(w) for w in current_chunk) + len(word) + len(current_chunk) < max_chunk_length:
            current_chunk.append(word)
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def get_sentiment(chunk):
    tokens = tokenizer(chunk, truncation=True, padding=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        output = model(**tokens)
        probs = torch.nn.functional.softmax(output.logits, dim=1)[0]
        label_id = torch.argmax(probs).item()
        return labels[label_id]

def aggregate_with_gemini(predictions):
    prompt = (
        f"Given these sentiment predictions for parts of a long text:\n"
        f"{predictions}\n"
        f"Return the overall sentiment (one of: Positive, Negative, Neutral) "
        f"based on the pattern of chunk sentiments. Be logical and brief."
    )
    try:
        response = gemini_model.generate_content(prompt)
        final = response.text.strip().capitalize()
        if final not in labels:
            raise ValueError("Invalid Gemini response")
        return final
    except Exception as e:
        print("Gemini fallback due to error:", str(e))
        return majority_vote(predictions)

def majority_vote(predictions):
    return max(set(predictions), key=predictions.count)


def predict_sentiment(text):
    text = preprocess(text)
    chunks = split_into_chunks(text)
    chunk_sentiments = [get_sentiment(chunk) for chunk in chunks]

    if len(chunk_sentiments) == 1:
        final_sentiment = chunk_sentiments[0]
    elif GEMINI_API_KEY:
        final_sentiment = aggregate_with_gemini(chunk_sentiments)
    else:
        final_sentiment = majority_vote(chunk_sentiments)
        

    return {
        "sentiment": final_sentiment,
        "chunks": len(chunks),
        "chunk_sentiments": chunk_sentiments,
        "source": "Gemini" if len(chunk_sentiments) > 1 and GEMINI_API_KEY else "XLM-RoBERTa"
    }

