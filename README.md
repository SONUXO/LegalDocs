# Prior Case Retrieval with BM25 and Event Extraction

## Overview
The task of **Prior Case Retrieval (PCR)** in the legal domain is about automatically citing relevant prior legal cases (based on facts and precedence) for a given query case.  

In this project, we implemented a **BM25 model**, which serves as a strong baseline for ranking the cited prior documents. Additionally, we explored the **role of events in legal case retrieval**, and our findings suggest that incorporating event extraction with BM25 makes it possible to effectively perform the court case retrieval task.

---

## File Structure
```
BM25withEventExtraction/
│--- data/
│--- evaluation/
│--- EventExtraction/
│--- results/
│--- BM25.ipynb
│--- BM25.py
│--- testing.ipynb
│--- extract_events.py
│--- legal_case_data.json
│--- top_10_cases.csv
```

---

## Tools & Techniques
1. **BM25 Model** – For ranking and matching court cases.
2. **spaCy** – For event extraction from legal case documents.

---

## Results
1. F1,RECALL AND PRECISION:
<img src="images/image1.png" alt="fw" width="500" height="500">

2. Top 10 cases:
<img src="images/image2.png" alt="Demo Screenshot" width="500" height="500">
