import spacy
import re
from spacy.matcher import PhraseMatcher

nlp = spacy.load("en_core_web_sm")

def init_pii_matcher():
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

init_pii_matcher()

def mask_pii(text):
    doc = nlp(text)
    masked = text
    entities = []
    replacements = []
    
    # Process in reverse to maintain positions
    for ent in sorted(doc.ents, key=lambda x: x.start_char, reverse=True):
        entity_type = map_label(ent.label_)
        if entity_type:
            original = text[ent.start_char:ent.end_char]
            entities.append({
                "position": [ent.start_char, ent.end_char],
                "classification": entity_type,
                "entity": original
            })
            masked = masked[:ent.start_char] + f"[{entity_type}]" + masked[ent.end_char:]
    
    # Additional regex patterns
    name_contexts = [
        r"(?i)(?:my name is|i am|contact)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)",
        r"(?i)(?:called|named)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)"
    ]
    
    for pattern in name_contexts:
        for match in re.finditer(pattern, masked):
            full_name = match.group(1)
            if len(full_name.split()) >= 2:
                entities.append({
                    "position": [match.start(1), match.end(1)],
                    "classification": "full_name",
                    "entity": full_name
                })
                masked = masked[:match.start(1)] + "[full_name]" + masked[match.end(1):]
    
    return {
        "original_text": text,
        "masked_text": masked,
        "entities": entities
    }

def map_label(spacy_label):
    label_map = {
        "PERSON": "full_name",
        "EMAIL": "email",
        "PHONE": "phone_number",
        "DOB": "dob",
        "AADHAR": "aadhar_num",
        "CREDIT_DEBIT": "credit_debit_no",
        "CVV": "cvv_no",
        "EXPIRY": "expiry_no"
    }
    return label_map.get(spacy_label)

def demask_email(masked_text, entities):
    text = masked_text
    for entity in sorted(entities, key=lambda x: x['position'][0], reverse=True):
        start, end = entity['position']
        placeholder = f"[{entity['classification']}]"
        if text[start:end] == placeholder:
            text = text[:start] + entity['entity'] + text[end:]
    return text