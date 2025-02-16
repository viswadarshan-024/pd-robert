# Fine-Tuned RoBERTa Model for Paraphrase Detection

### Model Description
This is a fine-tuned version of **RoBERTa-base** for **paraphrase detection**, trained on four benchmark datasets: **MRPC, QQP, PAWS-X, and PIT**. The model is designed for applications like **duplicate content detection, question answering, and semantic similarity analysis**. It demonstrates high performance across varied linguistic structures.

- **Developed by:** Viswadarshan R R  
- **Model Type:** Transformer-based Sentence Pair Classifier  
- **Language:** English  
- **Finetuned from:** `FacebookAI/roberta-base`

### Model Sources

- **Repository:** [Hugging Face Model Hub](https://huggingface.co/viswadarshan06/pd-bert/)  
- **Research Paper:** _Comparative Insights into Modern Architectures for Paraphrase Detection_ (Accepted at ICCIDS 2025)  
- **Demo:** (To be added upon deployment) 

## Uses

### Direct Use
- Identifying **duplicate questions** in FAQs and customer support.  
- Improving **semantic search** in information retrieval systems.  
- Enhancing **document deduplication** and content moderation.

### Downstream Use
The model can be further fine-tuned on domain-specific paraphrase datasets (e.g., medical, legal, or finance).

### Out-of-Scope Use
- The model is not designed for multilingual paraphrase detection since it is trained only on English datasets.
- May not perform well on low-resource languages without additional fine-tuning.

## Bias, Risks, and Limitations

### Known Limitations
- Struggles with idiomatic expressions: The model finds it difficult to detect paraphrases in figurative language.
- Contextual ambiguity: May fail when sentences require deep contextual reasoning.

### Recommendations
Users should fine-tune the model with additional cultural and idiomatic datasets for improved generalization in real-world applications.

## How to Get Started with the Model

To use the model, install **transformers** and load the fine-tuned model as follows:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the tokenizer and model
model_path = "viswadarshan06/pd-robert"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Encode sentence pairs
inputs = tokenizer("The car is fast.", "The vehicle moves quickly.", return_tensors="pt", padding=True, truncation=True)

# Get predictions
outputs = model(**inputs)
logits = outputs.logits
predicted_class = logits.argmax().item()

print("Paraphrase" if predicted_class == 1 else "Not a Paraphrase")
```

## Training Details
This model was trained using a combination of four datasets:

- **MRPC**: News-based paraphrases.
- **QQP**: Duplicate question detection.
- **PAWS-X**: Adversarial paraphrases for robustness testing.
- **PIT**: Short-text paraphrase dataset.

### Training Procedure

- **Tokenizer**: RobertaTokenizer
- **Batch Size**: 16
- **Optimizer**: AdamW
- **Loss Function**: Cross-entropy

#### Training Hyperparameters
- **Learning Rate**: 2e-5
- **Sequence Length**:
- MRPC: 256
- QQP: 336
- PIT: 64
- PAWS-X: 256

#### Speeds, Sizes, Times 

- **GPU Used**: NVIDIA A100
- **Total Training Time**: ~6 hours
- **Compute Units Used**: 80

### Testing Data, Factors & Metrics
#### Testing Data

The model was tested on combined test sets and evaluated on:
- Accuracy
- Precision
- Recall
- F1-Score
- Runtime

### Results

## **RoBERTa Model Evaluation Metrics**
| Model   | Dataset     | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | Runtime (sec) |
|---------|------------|-------------|--------------|------------|-------------|---------------|
| RoBERTa | MRPC Validation | 89.22 | 89.56 | 95.34 | 92.36 | 5.08 |
| RoBERTa | MRPC Test | 87.65 | 88.53 | 93.55 | 90.97 | 21.98 |
| RoBERTa | QQP Validation | 89.17 | 84.38 | 86.48 | 85.42 | 8.32 |
| RoBERTa | QQP Test | 89.36 | 85.14 | 86.56 | 85.84 | 19.44 |
| RoBERTa | PAWS-X Validation | 94.75 | 92.58 | 95.48 | 94.01 | 7.78 |
| RoBERTa | PAWS-X Test | 94.60 | 92.82 | 95.48 | 94.13 | 7.88 |
| RoBERTa | PIT Validation | 82.28 | 82.57 | 63.47 | 71.77 | 7.01 |
| RoBERTa | PIT Test | 90.45 | 84.67 | 66.29 | 74.35 | 1.47 |

### Summary
This RoBERTa-based Paraphrase Detection Model has been fine-tuned on four benchmark datasets: MRPC, QQP, PAWS-X, and PIT, enabling robust performance across diverse paraphrase structures. The model effectively identifies semantic similarity between sentence pairs, making it suitable for applications like semantic search, duplicate content detection, and question answering systems.

### **Citation**  

If you use this model, please cite:  

```bibtex
@inproceedings{viswadarshan2025paraphrase,
   title={Comparative Insights into Modern Architectures for Paraphrase Detection},
   author={Viswadarshan R R, Viswaa Selvam S, Felcia Lilian J, Mahalakshmi S},
   booktitle={International Conference on Computational Intelligence, Data Science, and Security (ICCIDS)},
   year={2025},
   publisher={IFIP AICT Series by Springer}
}
```

## Model Card Contact

📧 Email: viswadarshanrramiya@gmail.com

🔗 GitHub: [Viswadarshan R R](https://github.com/viswadarshan-024)
