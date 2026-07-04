# Video Presentation Transcript (English)
## Industrial Practice — Kazakh NER Pipeline
### Student: Tazdarbay Meirzhan | Astana IT University | SLV-group.kz

---

> **Instructions:** Open the `presentation/index.html` file in your browser and use the Arrow keys (→) to navigate the slides.
> Switch slides whenever you see **[→ SLIDE X]**.
> Total presentation duration: ~15 minutes.

---

## [→ SLIDE 1: Title]

Hello everyone! My name is Tazdarbay Meirzhan, and I am a second-year Computer Science student at Astana IT University.

The topic of my industrial practice is **Named Entity Recognition for the Kazakh Language**. The internship was completed at SLV-group.kz from January 6 to March 3, 2026.

In this video, I will walk you through the achievements of my 8-week practice, discussing the problem we solved, the technologies we utilized, and the final results.

---

## [→ SLIDE 2: Agenda]

Here is the agenda for today's presentation. We will go through the core chapters: beginning with the company and division overview, progressing through the technical implementation and training of the model, and concluding with the evaluation metrics, FastAPI deployment, and personal reflections. There are 16 slides in total.

---

## [→ SLIDE 3: Company Overview]

I completed my internship at **SLV-group.kz LLP**, a software engineering and IT consulting company based in Astana, Kazakhstan. The company specializes in custom software development, data engineering, and data science solutions for finance, government, and telecom industries.

I was placed in the IT Department, specifically the Software Development Division, as a Junior Data Engineer and ML Developer. I worked under the supervision of a Senior Software Developer, alongside a team of eight engineers working on NLP and data pipeline solutions. My academic advisor from AITU is Adilet Duman, a Senior Lecturer.

---

## [→ SLIDE 4: Internship Objectives]

The internship objectives were split into three categories:

**Primary Goal:** To design and implement an end-to-end Named Entity Recognition (NER) pipeline for the Kazakh language by fine-tuning a multilingual transformer model. 

**Technical Goals:** Getting hands-on experience with PyTorch and the Hugging Face ecosystem, managing raw NLP datasets, and writing a REST API for model inference.

**Learning Goals:** Understanding production development workflows, applying Computer Science theory to industry problems, and meeting deadlines in a team environment.

---

## [→ SLIDE 5: 8-Week Calendar Plan]

My 8-week calendar plan was structured as follows. The first week was dedicated to orientation, security policies, and standard procedures. In Week 2, we analyzed requirements and collected the KazNERD dataset. Weeks 3 and 4 were spent setting up the environment and designing the preprocessing pipeline. Weeks 5 and 6 focused on fine-tuning the model and performing evaluation. In Week 7, we built the FastAPI interface, and Week 8 was dedicated to writing documentation and consolidating the code.

---

## [→ SLIDE 6: KazNERD Dataset]

For this project, we used **KazNERD**, the largest open Kazakh NER corpus, created by the ISSAI lab at Nazarbayev University.

It contains over **112,000 sentences** annotated with **25 distinct entity types** such as PERSON, ORGANISATION, GPE, DATE, LAW, and others. The annotation follows the **IOB2** format (CoNLL-2002 standard). 

On the right side of the slide, you can see a sample format: "Еуразиялық" starts a LAW entity and gets a B-LAW tag, followed by continuation tokens with I-LAW. Common tokens get the "O" (Outside) tag.

Since each of the 25 tags has B- and I- variants plus the "O" tag, our model was configured to classify **51 distinct target classes**.

---

## [→ SLIDE 7: Technology Stack]

Our modern technology stack consists of:

- **Python and PyTorch** as the core deep learning frameworks.
- **Hugging Face Transformers** for downloading and training the model.
- **XLM-RoBERTa Base** as our backbone pretrained multilingual model.
- **FastAPI** and Uvicorn to serve the model as a lightweight REST API.
- **scikit-learn** to compute evaluation metrics.
- **Git** for version control.

---

## [→ SLIDE 8: System Architecture]

Here is the system architecture of our pipeline. The raw KazNERD dataset is processed by our custom IOB2 parser. Next, we generate tag mappings mapping the 51 classes. The text is processed using the XLM-RoBERTa Tokenizer. We resolve the subword-label mismatch using a custom label alignment script. Once the HF Dataset is ready, we run the Fine-Tuning process. Finally, we evaluate the model and export it to `./ner-kazakh-best` to be served by FastAPI.

---

## [→ SLIDE 9: Data Processing Pipeline]

Let's look at the implementation details.

**Step 1: The IOB2 Parser** reads files line-by-line, splitting words and labels. Empty lines indicate sentence boundaries.

**Step 2: Tag Mapping** extracts all unique labels from the training splits, sorts them, and creates two dictionaries: `tag2id` and `id2tag`. This maps textual annotations to numerical ids suitable for neural network training.

---

## [→ SLIDE 10: Tokenization & Label Alignment]

This is the most critical technical challenge of the pipeline.

Kazakh is an **agglutinative language** where suffixes are stacked. The XLM-RoBERTa SentencePiece tokenizer splits words into subwords. For instance, "Еуразиялық" becomes `["▁Еуразия", "лық"]`.

Because we have only one label per word, but multiple subword tokens, we must align them.

**Our solution:** We extract `word_ids` from the tokenizer output. The first subword gets the actual word label, while subsequent subwords get the corresponding "I-" tag (or the B-tag as a fallback). Special tokens (like CLS and SEP) are mapped to `-100`, which forces PyTorch to ignore them in loss calculation.

---

## [→ SLIDE 11: Model Fine-Tuning]

We selected **XLM-RoBERTa** because it is pretrained on 100 languages, including Kazakh, allowing excellent cross-lingual transfer. With 277 million parameters, it offers an optimal balance between accuracy and runtime resources.

We trained the model for **3 epochs** with a learning rate of **2e-5** using the **AdamW optimizer** and **fp16 mixed precision** on GPU. The best model checkpoint was saved based on validation F1-Score.

---

## [→ SLIDE 12: Evaluation Results]

Let's look at the final evaluation results on the test set.

Our model achieved:
- **Precision: 88.1%**
- **Recall: 86.5%**
- **Macro F1-Score: 87.3%**
- **Overall Accuracy: 94.1%**

Common categories like PERSON (92.4% F1), DATE (91.8% F1), and GPE (89.7% F1) achieved outstanding results. Law-related entities (LAW) were more challenging (79.3% F1) due to complex legal language patterns in Kazakh.

---

## [→ SLIDE 13: API Integration]

We wrapped our trained model in a **FastAPI** service.

The app initializes a Hugging Face token classification pipeline pointing to our saved `./ner-kazakh-best` directory. We exposed a POST endpoint `/predict` that accepts raw Kazakh text and returns a structured JSON payload with detected entities, their confidence scores, and character start/end spans.

For example, sending "Нұрсұлтан Назарбаев Астанада" returns "Нұрсұлтан Назарбаев" as a PERSON (96.4% confidence) and "Астанада" as a GPE (95.2% confidence).

---

## [→ SLIDE 14: Challenges & Solutions]

We successfully overcame several core challenges during development:

1. **Subword Mismatch:** Agglutinative structures caused tokens to split. Resolved by writing a custom token mapping utilizing the tokenizer's `word_ids` with a fallback mechanism.
2. **Class Imbalance:** 80% of data belongs to the "O" class. Resolved by prioritizing Macro F1-score during training and evaluations to ensure minority classes are not ignored.
3. **Hardware Resource Limits:** Training on CPU caused OOM crashes. Resolved by running training on a GPU instance with fp16 mixed-precision enabled.
4. **Transformers 5.x compatibility:** API signatures changed. We updated training configs to use `eval_strategy`, `warmup_steps`, and `processing_class`.

---

## [→ SLIDE 15: Personal Reflection]

This internship has been an invaluable learning experience.

On the technical side, I learned how to build and debug end-to-end NLP pipelines, fine-tune massive transformer architectures, and design lightweight microservices.

On the industry side, I learned that ML projects are **80% data engineering and 20% modeling**. Writing production-grade code requires strict documentation, reproducibility, and robust error handling.

Personally, contributing to **Kazakh language technologies** is highly motivating. Kazakh NLP tools are still growing, and this work provides a solid foundation for local AI applications.

---

## [→ SLIDE 16: Conclusion]

To conclude:

We successfully built, evaluated, and deployed a Kazakh Named Entity Recognition pipeline during this 8-week industrial practice at SLV-group.kz.

The model achieved an **87.3% Macro F1-score** across 51 token classes on the KazNERD dataset and is ready for production integration via our FastAPI service.

Thank you very much for your time and attention!

---

## ⏱ Expected Timing per Slide:

| Slide | Target Timestamp |
|---|---|
| 1. Title | 0:00 – 0:30 |
| 2. Agenda | 0:30 – 0:50 |
| 3. Company Overview | 0:50 – 1:40 |
| 4. Internship Objectives | 1:40 – 2:30 |
| 5. 8-Week Plan | 2:30 – 3:20 |
| 6. KazNERD Dataset | 3:20 – 4:30 |
| 7. Tech Stack | 4:30 – 5:10 |
| 8. System Architecture | 5:10 – 6:00 |
| 9. Data Pipeline Code | 6:00 – 7:00 |
| 10. Tokenization & Alignment | 7:00 – 8:20 |
| 11. Model Fine-tuning | 8:20 – 9:30 |
| 12. Evaluation Results | 9:30 – 11:00 |
| 13. API Integration | 11:00 – 12:00 |
| 14. Challenges & Solutions | 12:00 – 13:10 |
| 15. Personal Reflection | 13:10 – 14:15 |
| 16. Conclusion | 14:15 – 15:00 |
