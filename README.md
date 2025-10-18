# Sentiment Analysis Tool

The **Sentiment Analysis Tool** processes multilingual text or files to return a **sentiment label** (`positive`, `negative`, or `neutral`) and **confidence score** for each input.  
It supports **40+ languages** using HuggingFace Transformers for major languages and **Polyglot** as a fallback for low-resource languages.  

This tool is designed for **large-scale text processing**, handles very large texts efficiently, and integrates easily with other NLP pipelines.

---

## Key Features

- **Multilingual Support**  
  Handles 40+ languages with HuggingFace models. Falls back to Polyglot for unsupported/low-resource languages.

- **Dynamic Pipeline Loading**  
  Automatically loads the correct model based on the language code argument.

- **Flexible Input**  
  Accepts:
  - Direct text via CLI argument  
  - File path for batch processing  

- **JSON Output**  
  Produces structured JSON mapping input text to sentiment label & score.

- **Batch Processing**  
  Supports line-by-line sentiment analysis for text files.

- **Label Normalization**  
  Maps different model outputs to uniform labels: `positive`, `negative`, `neutral`.

---

## Workflow Overview

1. **Load Language Model**  
   - Reads the language code argument.  
   - Loads HuggingFace or Polyglot model as needed.

2. **Input Handling**  
   - Text passed directly via `--text "..."`.  
   - File path passed via `--file /path/to/file.txt`.

3. **Sentiment Analysis**  
   - Classifies text into sentiment categories with scores.

4. **Label Normalization**  
   - Normalizes model-specific labels to `positive`, `negative`, `neutral`.

5. **Output Results**  
   - Returns JSON:  
     - **Text Mode:** `{"text": "...", "label": "...", "score": ...}`  
     - **File Mode:** JSON array with line numbers.

---

## Supported Languages

| No. | Code | Language              | Model |
|-----|------|----------------------|-------|
| 1   | en   | English              | distilbert-base-uncased-finetuned-sst-2-english |
| 2   | ar   | Arabic               | CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment |
| 3   | de   | German               | oliverguhr/german-sentiment-bert |
| 4   | el   | Greek                | nlpaueb/bert-base-greek-uncased-v1 |
| 5   | es   | Spanish              | edumunozsala/roberta_bne_sentiment_analysis_es |
| 6   | he   | Hebrew               | avichr/heBERT_sentiment_analysis |
| 7   | hu   | Hungarian            | NYTK/sentiment-hts5-hubert-hungarian |
| 8   | id   | Indonesian           | mdhugol/indonesia-bert-sentiment-classification |
| 9   | it   | Italian              | MilaNLProc/feel-it-italian-sentiment |
| 10  | nb   | Norwegian            | NTCAL/norbert2_sentiment_norec |
| 11  | pt   | Portuguese           | neuralmind/bert-large-portuguese-cased |
| 12  | ro   | Romanian             | racai/distilbert-base-romanian-cased |
| 13  | ru   | Russian              | blanchefort/rubert-base-cased-sentiment |
| 14  | sk   | Slovak               | kinit/slovakbert-sentiment-twitter |
| 15  | sv   | Swedish              | marma/bert-base-swedish-cased-sentiment |
| 16  | th   | Thai                 | poom-sci/WangchanBERTa-finetuned-sentiment |
| 17  | tr   | Turkish              | savasy/bert-base-turkish-sentiment-cased |
| 18  | vi   | Vietnamese           | wonrax/phobert-base-vietnamese-sentiment |
| 19  | zh   | Chinese (Simplified) | IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment |
| 20  | zht  | Chinese (Traditional)| ckiplab/bert-base-chinese |
| 21  | other| Low-resource         | Polyglot fallback |

---

## Requirements

- **Python:** 3.9+  
- **Install HuggingFace Transformers & Torch:**
  ```bash
  python3 -m pip install transformers torch torchvision torchaudio
  ```

- **Install Polyglot & Dependencies:**
  ```bash
  sudo apt-get install -y python3-numpy libicu-dev
  python3 -m pip install numpy six pyicu pycld2 morfessor polyglot
  ```

- **Verify Installation:**
  ```bash
  python3 -m pip show torch transformers polyglot
  ```


- **Download Polyglot Sentiment Models:**
  ```bash
  bash download-polyglot-sentiment2-models-for-each-language.sh
  ```

---

## Running the Tool

### Direct Text Input
```bash
python3 sentiment-analysis.py en --text "I love spending time with my family and friends."
```

**Sample Output:**
```json
{
  "text": "I love spending time with my family and friends.",
  "label": "positive",
  "score": 0.9997795224189758
}
```

---

### File Input
```bash
python3 sentiment-analysis.py en --file /path/to/test-input-sentiment.txt
```

**Sample Output:**
```json
{
  "result": [
    {
      "lineno": 1,
      "text": "I love spending time with my family and friends.",
      "label": "positive",
      "score": 0.9997795224189758
    },
    {
      "lineno": 2,
      "text": "I can't stand sitting in traffic for hours on end.",
      "label": "negative",
      "score": 0.9956425428390503
    }
  ]
}
```

---

## Use Cases

1. **Text Sentiment Analysis** – Classify user reviews, tweets, or comments.  
2. **Social Media Monitoring** – Track brand reputation in multiple languages.  
3. **Customer Feedback Analysis** – Analyze survey and ticket sentiment.  
4. **Preprocessing for NLP** – Use sentiment as a feature for downstream tasks.  
5. **Research & Analytics** – Process multilingual datasets for insights.  

---

## 📌 Notes

- Ensure HuggingFace models are downloaded for supported languages.  
- Polyglot handles low-resource languages.  
- Input can be provided as text or file path via CLI args.  
- Output is structured JSON for easy integration.  
- Labels are normalized to `positive`, `negative`, `neutral`.  

---

## References

- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [HuggingFace Model Hub](https://huggingface.co/models) 
- [PyTorch](https://pytorch.org/)  
- [Polyglot NLP](https://polyglot.readthedocs.io/en/latest/)  

---

## Author

Developed by **Zar Zar Hlaing**  

---

## License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.
