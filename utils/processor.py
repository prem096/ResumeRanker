from transformers import BertTokenizer, BertModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np

# Scoring weights (must sum to 1.0)
WEIGHT_BERT = 0.50
WEIGHT_TFIDF = 0.25
WEIGHT_SKILLS = 0.25

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


def _normalize_score(value):
    """Clamp score into [0, 1] range."""
    return float(np.clip(value, 0.0, 1.0))


def get_bert_embeddings(texts):
    """Generate BERT [CLS] embeddings for a list of texts in one batch."""
    if not texts:
        return []

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
    )
    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    return outputs.last_hidden_state[:, 0, :].cpu().numpy()


def compute_tfidf_similarity(jd_text, resume_text):
    """TF-IDF cosine similarity between JD and resume."""
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    matrix = vectorizer.fit_transform([jd_text, resume_text])
    return float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0])


def compute_skill_score(jd_skills, resume_skills):
    """
    Weighted skill overlap: matched JD skills / total JD skills.
    Returns 0 when the JD has no extractable skills.
    """
    jd_set = set(jd_skills)
    resume_set = set(resume_skills)
    if not jd_set:
        return 0.0
    return len(jd_set & resume_set) / len(jd_set)


def compute_hybrid_score(jd_text, resume_text, jd_skills, resume_skills):
    """
    Hybrid match score combining semantic, lexical, and skill signals.
    Returns (final_score, bert_score, tfidf_score, skill_score).
    """
    bert_vectors = get_bert_embeddings([jd_text, resume_text])
    bert_score = float(cosine_similarity([bert_vectors[0]], [bert_vectors[1]])[0][0])
    tfidf_score = compute_tfidf_similarity(jd_text, resume_text)
    skill_score = compute_skill_score(jd_skills, resume_skills)

    final_score = (
        WEIGHT_BERT * bert_score
        + WEIGHT_TFIDF * tfidf_score
        + WEIGHT_SKILLS * skill_score
    )
    return (
        _normalize_score(final_score),
        _normalize_score(bert_score),
        _normalize_score(tfidf_score),
        _normalize_score(skill_score),
    )


def compute_similarity(jd_text, resumes_texts, jd_skills=None, resume_skills_map=None):
    """
    Compare JD with each resume using hybrid scoring.
    Returns sorted list of (name, final_score).
    """
    if not resumes_texts:
        return []

    jd_skills = jd_skills or []
    resume_skills_map = resume_skills_map or {}
    result = []

    for name, resume_text in resumes_texts:
        resume_skills = resume_skills_map.get(name, [])
        score, _, _, _ = compute_hybrid_score(jd_text, resume_text, jd_skills, resume_skills)
        result.append((name, score))

    return sorted(result, key=lambda x: x[1], reverse=True)
