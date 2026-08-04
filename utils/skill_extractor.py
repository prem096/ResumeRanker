import spacy
import json
import os
import re

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise RuntimeError("Run `python -m spacy download en_core_web_sm` to install the model.")

_skills_cache = None
_phrase_to_canonical = {}
_ruler_skill_count = 0


def load_skills(filepath="utils/skills.json"):
    global _skills_cache, _phrase_to_canonical
    if _skills_cache is not None:
        return _skills_cache

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Skill file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        _skills_cache = json.load(f)

    _phrase_to_canonical = {}
    for canonical, synonyms in _skills_cache.items():
        canonical_key = canonical.lower()
        _phrase_to_canonical[canonical_key] = canonical_key
        for phrase in synonyms:
            _phrase_to_canonical[phrase.lower()] = canonical_key

    return _skills_cache


def _canonicalize(skill_phrase):
    return _phrase_to_canonical.get(skill_phrase.lower(), skill_phrase.lower())


def _ensure_skill_ruler(skill_dict):
    """Add skill patterns after NER so they are not overwritten."""
    global _ruler_skill_count

    pattern_count = sum(len(synonyms) for synonyms in skill_dict.values())
    if _ruler_skill_count == pattern_count and "entity_ruler" in nlp.pipe_names:
        return

    if "entity_ruler" in nlp.pipe_names:
        nlp.remove_pipe("entity_ruler")

    # Place after NER so skill labels take precedence over GPE/PERSON/ORG
    ruler = nlp.add_pipe("entity_ruler", last=True, config={"overwrite_ents": True})
    patterns = [
        {"label": "SKILL", "pattern": phrase}
        for synonyms in skill_dict.values()
        for phrase in synonyms
    ]
    ruler.add_patterns(patterns)
    _ruler_skill_count = pattern_count


def _keyword_scan(text, skill_dict):
    """Reliable dictionary scan with word-boundary matching."""
    text_lower = text.lower()
    detected = set()

    for synonyms in skill_dict.values():
        for phrase in synonyms:
            phrase_lower = phrase.lower()
            if len(phrase_lower) <= 3:
                if re.search(rf"\b{re.escape(phrase_lower)}\b", text_lower):
                    detected.add(_canonicalize(phrase_lower))
            elif phrase_lower in text_lower:
                detected.add(_canonicalize(phrase_lower))

    return detected


def extract_skills(text, skill_dict=None):
    skill_dict = skill_dict or load_skills()
    _ensure_skill_ruler(skill_dict)
    doc = nlp(text)
    detected = set()

    for ent in doc.ents:
        if ent.label_ == "SKILL":
            detected.add(_canonicalize(ent.text.strip()))

    detected.update(_keyword_scan(text, skill_dict))
    return sorted(detected)


def compare_skills(jd_text, resume_text, skill_dict=None):
    skill_dict = skill_dict or load_skills()
    jd_skills = set(extract_skills(jd_text, skill_dict))
    resume_skills = set(extract_skills(resume_text, skill_dict))

    matched = sorted(resume_skills & jd_skills)
    missing = sorted(jd_skills - resume_skills)
    extra = sorted(resume_skills - jd_skills)
    return matched, missing, extra


def generate_suggestions(missing_skills, descriptions=None):
    suggestions = []
    descriptions = descriptions or {}
    for skill in missing_skills:
        line = f"Consider learning {skill}"
        if skill in descriptions:
            line += f" ({descriptions[skill]})"
        suggestions.append(line)
    return suggestions


def infer_job_role(resume_skills, role_skill_map):
    scores = {
        role: len(set(resume_skills) & {s.lower() for s in skills})
        for role, skills in role_skill_map.items()
    }
    best_role = max(scores, key=scores.get) if scores else "Unknown"
    return best_role if scores.get(best_role, 0) > 0 else "Unknown"
