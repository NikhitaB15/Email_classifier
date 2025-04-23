# -*- coding: utf-8 -*-

import pandas as pd
import spacy
from spacy.matcher import PhraseMatcher
import re

# Initialize spaCy with improved NER
nlp = spacy.load("en_core_web_sm")

def enhance_pii_detection():
    # Add custom entity ruler
    ruler = nlp.add_pipe("entity_ruler", before="ner")

    patterns = [
    # Person names with titles
    {"label": "PERSON", "pattern": [{"LOWER": {"in": ["mr", "mrs", "ms", "dr"]}}, {"ENT_TYPE": "PERSON"}]},

    # Email addresses
    {"label": "EMAIL", "pattern": [{"TEXT": {"regex": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"}}]},

    # Phone numbers (both international and local)
    {"label": "PHONE", "pattern": [{"TEXT": {"regex": r"(\+\d{1,3}\s?)?(\d{3}|\(\d{3}\))[\s.-]?\d{3}[\s.-]?\d{4}"}}]},
    {"label": "PHONE", "pattern": [{"TEXT": {"regex": r"\b\d{10}\b"}}]},

    # Date of birth
    {"label": "DOB", "pattern": [{"TEXT": {"regex": r"\b\d{2}/\d{2}/\d{4}\b"}}]},

    # Aadhar numbers (Indian identification)
    {"label": "AADHAR", "pattern": [{"TEXT": {"regex": r"\b\d{4}\s\d{4}\s\d{4}\b"}}]},

    # Credit/Debit cards
    {"label": "CREDIT_DEBIT", "pattern": [{"TEXT": {"regex": r"\b(?:\d[ -]*?){13,16}\b"}}]},
    {"label": "CREDIT_DEBIT", "pattern": [{"TEXT": {"regex": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"}}]},

    # CVV numbers
    {"label": "CVV", "pattern": [{"TEXT": {"regex": r"\b\d{3,4}\b"}}]},

    # Expiry dates
    {"label": "EXPIRY", "pattern": [{"TEXT": {"regex": r"\b(0[1-9]|1[0-2])\/\d{2,4}\b"}}]}
]
    ruler.add_patterns(patterns)

    # Add title detection
    for title in ["Mr", "Mrs", "Ms", "Dr", "Professor"]:
        nlp.vocab[title].is_title = True


enhance_pii_detection()

def smart_mask(text):
    doc = nlp(text)
    masked = text
    replacements = []

    # Process entities in reverse order
    for ent in sorted(doc.ents, key=lambda x: x.start_char, reverse=True):
        if ent.label_ == "PERSON":
            # Additional verification for person names
            if any(title in ent.text for title in ["Mr", "Mrs", "Ms", "Dr"]):
                masked = masked[:ent.start_char] + "[NAME]" + masked[ent.end_char:]
            elif len(ent.text.split()) >= 2:  # At least first and last name
                masked = masked[:ent.start_char] + "[NAME]" + masked[ent.end_char:]
        elif ent.label_ == "ORG":
            # Verify if this might actually be a person's name
            if not any(x in ent.text.lower() for x in ["inc", "corp", "ltd"]):
                if len(ent.text.split()) >= 2:
                    masked = masked[:ent.start_char] + "[NAME]" + masked[ent.end_char:]
                else:
                    masked = masked[:ent.start_char] + "[ORG]" + masked[ent.end_char:]
        else:
            masked = masked[:ent.start_char] + f"[{ent.label_}]" + masked[ent.end_char:]

    # Additional regex patterns for context-based masking
    name_contexts = [
        r"(?i)(?:my name is|i am|contact)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)",
        r"(?i)(?:called|named)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)"
    ]

    for pattern in name_contexts:
        for match in re.finditer(pattern, masked):
            full_name = match.group(1)
            if len(full_name.split()) >= 2 and "[NAME]" not in masked[match.start(1):match.end(1)]:
                masked = masked[:match.start(1)] + "[NAME]" + masked[match.end(1):]

    return masked

def process_emails(df, text_column=0):
    # Clean text
    df['cleaned_text'] = df.iloc[:, text_column].astype(str).apply(
        lambda x: re.sub(r'\s+', ' ', x.replace('\n', ' ').replace('\t', ' ')).strip()
    )

    # Apply masking
    df['masked_text'] = df['cleaned_text'].apply(smart_mask)

    return df

# Example usage
if __name__ == "__main__":
    # Sample data
    data = {
        "emails": [
            "Contact Dr. Smith at john@example.com or +1 (555) 123-4567",
            "My name is Elena Ivanova, credit card 4111-1111-1111-1111",
            "Data Analytics request from Wei Liu (liuwei@business.cn)"
        ]
    }
    df = pd.DataFrame(data)

    # Process emails
    processed_df = process_emails(df)

    # Display results
    for idx, row in processed_df.iterrows():
        print(f"\nOriginal {idx+1}:")
        print(row['emails'])
        print(f"\nMasked {idx+1}:")
        print(row['masked_text'])
        print("\n" + "-"*50)

nlp = spacy.load("en_core_web_sm")
''' '''
# Add custom patterns for better PII detection
from spacy.pipeline import EntityRuler

ruler = nlp.add_pipe("entity_ruler", before="ner")

patterns = [

    # Email addresses
    {"label": "EMAIL", "pattern": [{"TEXT": {"regex": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"}}]},

    # Phone numbers (both international and local)
    {"label": "PHONE", "pattern": [{"TEXT": {"regex": r"(\+\d{1,3}\s?)?(\d{3}|\(\d{3}\))[\s.-]?\d{3}[\s.-]?\d{4}"}}]},
    {"label": "PHONE", "pattern": [{"TEXT": {"regex": r"\b\d{10}\b"}}]},

    # Date of birth
    {"label": "DOB", "pattern": [{"TEXT": {"regex": r"\b\d{2}/\d{2}/\d{4}\b"}}]},

    # Aadhar numbers (Indian identification)
    {"label": "AADHAR", "pattern": [{"TEXT": {"regex": r"\b\d{4}\s\d{4}\s\d{4}\b"}}]},

    # Credit/Debit cards
    {"label": "CREDIT_DEBIT", "pattern": [{"TEXT": {"regex": r"\b(?:\d[ -]*?){13,16}\b"}}]},
    {"label": "CREDIT_DEBIT", "pattern": [{"TEXT": {"regex": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"}}]},

    # CVV numbers
    {"label": "CVV", "pattern": [{"TEXT": {"regex": r"\b\d{3,4}\b"}}]},

    # Expiry dates
    {"label": "EXPIRY", "pattern": [{"TEXT": {"regex": r"\b(0[1-9]|1[0-2])\/\d{2,4}\b"}}]}
]
ruler.add_patterns(patterns)

df = pd.read_csv('/content/emails.csv')
processed_df = process_emails(df)

# Save results
processed_df.to_csv('masked_emails.csv', index=False)

from sklearn.model_selection import train_test_split
# 4. Prepare for Classification
df=pd.read_csv('masked_emails.csv')
emails = df['masked_text'].tolist()
if len(df.columns) > 1:  # If we have categories
    from sklearn.preprocessing import LabelEncoder
    from sklearn.feature_extraction.text import TfidfVectorizer

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(df.iloc[:, 1])  # Assuming 2nd column has categories

    # Vectorize text
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X = vectorizer.fit_transform(df['masked_text'])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Data prepared: {X_train.shape[0]} training, {X_test.shape[0]} test samples")

# Show sample processed data
print("\nSample processed emails:")
for i, (orig, masked) in enumerate(zip(emails[:3], df['masked_text'][:3])):
    print(f"\nOriginal {i+1}:\n{orig}")

    print(f"\nMasked {i+1}:\n{masked}")

