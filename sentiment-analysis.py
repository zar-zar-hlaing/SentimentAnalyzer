import sys
import re
import json
import torch
import argparse
from transformers import pipeline, DistilBertTokenizer, DistilBertForSequenceClassification
from polyglot.text import Text

def load_pipeline(language):
    language = language.lower()
    if language == "en":
        tokenizer = DistilBertTokenizer.from_pretrained(
            "distilbert-base-uncased-finetuned-sst-2-english"
        )
        model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased-finetuned-sst-2-english"
        )
        return pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            max_length=512,
            truncation=True,
        )

    models = {
        "ar": "CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment",
        "de": "oliverguhr/german-sentiment-bert",
        "el": "nlpaueb/bert-base-greek-uncased-v1",
        "es": "edumunozsala/roberta_bne_sentiment_analysis_es",
        "he": "avichr/heBERT_sentiment_analysis",
        "hu": "NYTK/sentiment-hts5-hubert-hungarian",
        "id": "mdhugol/indonesia-bert-sentiment-classification",
        "it": "MilaNLProc/feel-it-italian-sentiment",
        "nb": "NTCAL/norbert2_sentiment_norec",
        "pt": "neuralmind/bert-large-portuguese-cased",
        "ro": "racai/distilbert-base-romanian-cased",
        "ru": "blanchefort/rubert-base-cased-sentiment",
        "sk": "kinit/slovakbert-sentiment-twitter",
        "sv": "marma/bert-base-swedish-cased-sentiment",
        "th": "poom-sci/WangchanBERTa-finetuned-sentiment",
        "tr": "savasy/bert-base-turkish-sentiment-cased",
        "vi": "wonrax/phobert-base-vietnamese-sentiment",
        "zh": "IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment",
        "zh-cn": "IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment",
        "zht": "IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment",
    }

    if language in models:
        return pipeline(
            "sentiment-analysis", model=models[language], max_length=512, truncation=True
        )

    return None  # will fall back to polyglot


def run_huggingface(senti_pipeline, text, language, with_lineno=False):
    results = []
    lines = text if isinstance(text, list) else [text]

    for idx, content in enumerate(lines, 1):
        outputs = senti_pipeline(content)
        if not outputs:
            continue
        output = outputs[0]

        label = output["label"]
        sentiment_label = ""

        # Normalize label mapping
        if "LABEL_" in label:
            if language == "hu":
                sentiment_label = (
                    "negative"
                    if label in ["LABEL_0", "LABEL_1"]
                    else "neutral" if label == "LABEL_2" else "positive"
                )
            elif language == "id":
                mapping = {"LABEL_0": "positive", "LABEL_1": "neutral", "LABEL_2": "negative"}
                sentiment_label = mapping.get(label, label)
            else:
                mapping = {"LABEL_0": "negative", "LABEL_1": "positive"}
                sentiment_label = mapping.get(label, "neutral")
        else:
            mapping = {
                "Positivo": "positive",
                "Negativo": "negative",
                "POS": "positive",
                "NEG": "negative",
                "NEU": "neutral",
                "pos": "positive",
                "neg": "negative",
                "neu": "neutral",
            }
            sentiment_label = mapping.get(label, label)

        entry = {
            "text": content.strip(),
            "label": sentiment_label.lower(),
            "score": output["score"],
        }
        if with_lineno:
            entry["lineno"] = idx
        results.append(entry)
    return results


def run_polyglot(text, with_lineno=False):
    results = []
    lines = text if isinstance(text, list) else [text]

    for idx, content in enumerate(lines, 1):
        try:
            poly = Text(content)
            polarity = poly.polarity
        except ZeroDivisionError:
            polarity = 0

        if polarity > 0:
            sentiment_label = "positive"
        elif polarity < 0:
            sentiment_label = "negative"
        else:
            sentiment_label = "neutral"

        entry = {"text": content.strip(), "label": sentiment_label, "score": polarity}
        if with_lineno:
            entry["lineno"] = idx
        results.append(entry)
    return results


def main():
    parser = argparse.ArgumentParser(description="Multilingual Sentiment Analysis")
    parser.add_argument("language", help="Language code (e.g., en, de, es, zh)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", help="Text input for sentiment analysis")
    group.add_argument("--file", help="Path to text file for sentiment analysis")
    args = parser.parse_args()

    # Load pipeline or fallback
    senti_pipeline = load_pipeline(args.language)
    use_polyglot = senti_pipeline is None

    if args.text:
        input_text = args.text
        results = (
            run_polyglot(input_text)
            if use_polyglot
            else run_huggingface(senti_pipeline, input_text, args.language)
        )
        for r in results:
            print(json.dumps(r))

    elif args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        results = (
            run_polyglot(lines, with_lineno=True)
            if use_polyglot
            else run_huggingface(senti_pipeline, lines, args.language, with_lineno=True)
        )
        print(json.dumps({"result": results}))


if __name__ == "__main__":
    main()

