# **PII Detection & Sanitization Examples – Using Upasak PIISanitizer**

These examples demonstrate how to use **Upasak's PII Sanitization module** to:

1. Detect Personally Identifiable Information (PII) and Sensitive Information
2. Perform automatic sanitization (masking, redaction, pseudonymizing)
3. Run a batch sanitization pipeline on dataset list

Upasak supports both **rule-based** and **hybrid (rule-based + NER model)** PII detection.

---

## **Supported Actions**

* `redact` 
* `mask` 
* `pseudonymize` 

---

## **Example 1: detection_and_action.py**

This script shows single-input PII detection and sanitization in isolation.

### **Notes**

* `certain_detections` → high-confidence (>0.9)
* `uncertain_detections` → medium confidence (0.5–0.9)
* `is_nlp_based=True` enables NER-based detection

---

## **Example 2: pii_pipeline.py**

This script demonstrates **batch PII sanitization** for JSONL datasets.

### **Pipeline Features**

* Multi-record sanitization
* Outputs a new JSONL file
* Supports all PII actions (`redact`, `mask`, `pseudonymize`)
* Ignore **HITL (Human-In-The-Loop)** mode, it is for streamlit-app

---

## **When to Use Which**

| Example                     | Use Case                                            |
| --------------------------- | --------------------------------------------------- |
| **detection_and_action.py** | For single text inputs, testing, experimentation    |
| **pii_pipeline.py**         | For production-style batch sanitization of datasets |

---

## **Summary**

These examples help users:

* Detect PII in text using rule-based or hybrid methods
* Process both individual texts and large datasets
* Integrate PII safety into ML and LLM pipelines easily
