"""Smoke tests for all ResumeRanker modules."""
import json
import os
import sys
import tempfile
import traceback
from contextlib import contextmanager

PASS = 0
FAIL = 0
WARN = 0


def ok(name, detail=""):
    global PASS
    PASS += 1
    print(f"  [PASS] {name}" + (f" — {detail}" if detail else ""))


def fail(name, detail=""):
    global FAIL
    FAIL += 1
    print(f"  [FAIL] {name}" + (f" — {detail}" if detail else ""))


def warn(name, detail=""):
    global WARN
    WARN += 1
    print(f"  [WARN] {name}" + (f" — {detail}" if detail else ""))


def section(title):
    print(f"\n=== {title} ===")


@contextmanager
def client_session(app, user):
    with app.test_client() as client:
        with client.session_transaction() as sess:
            sess["_user_id"] = str(user.id)
            sess["_fresh"] = True
        yield client


def test_json_configs():
    section("Config JSON files")
    for path in ["utils/skills.json", "utils/role_skills.json", "utils/jd_categories.json"]:
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            ok(path, f"{len(data)} entries")
        except Exception as e:
            fail(path, str(e))


def test_parser():
    section("utils/parser.py")
    try:
        import fitz
        from utils.parser import extract_text_from_pdf

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            path = tmp.name

        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Python developer with Flask and machine learning experience.")
        doc.save(path)
        doc.close()

        text = extract_text_from_pdf(path)
        os.unlink(path)

        if "Python" in text and "Flask" in text:
            ok("extract_text_from_pdf", f"got {len(text)} chars")
        else:
            fail("extract_text_from_pdf", f"unexpected text: {text!r}")
    except Exception as e:
        fail("parser", traceback.format_exc().splitlines()[-1])


def test_skill_extractor():
    section("utils/skill_extractor.py")
    try:
        from utils.skill_extractor import (
            load_skills, extract_skills, compare_skills,
            generate_suggestions, infer_job_role
        )

        skills = load_skills()
        if isinstance(skills, dict) and len(skills) > 0:
            ok("load_skills", f"{len(skills)} skill groups")
        else:
            fail("load_skills", "empty or invalid")

        text = "Experienced Python developer skilled in Flask, machine learning, and SQL."
        found = extract_skills(text, skills)
        if any("python" in s for s in found):
            ok("extract_skills", f"found: {found[:6]}")
        else:
            fail("extract_skills", f"expected python, got {found}")

        jd = "Looking for Python, Flask, Docker, and Kubernetes expertise."
        resume = "Python and Flask developer with Git experience."
        matched, missing, extra = compare_skills(jd, resume, skills)
        if "python" in matched and "flask" in matched and "docker" in missing:
            ok("compare_skills", f"matched={matched}, missing={missing}")
        else:
            fail("compare_skills", f"matched={matched}, missing={missing}, extra={extra}")

        suggestions = generate_suggestions(missing)
        if suggestions and "Consider learning" in suggestions[0]:
            ok("generate_suggestions")
        else:
            fail("generate_suggestions", str(suggestions))

        with open("utils/role_skills.json", encoding="utf-8") as f:
            role_map = json.load(f)
        role = infer_job_role(found, role_map)
        ok("infer_job_role", f"role={role}")
    except Exception as e:
        fail("skill_extractor", traceback.format_exc().splitlines()[-1])


def test_processor():
    section("utils/processor.py")
    try:
        from utils.processor import (
            compute_hybrid_score, compute_similarity,
            compute_tfidf_similarity, compute_skill_score
        )

        jd = "Senior Python developer with Flask, Docker, and machine learning."
        resume = "Python developer experienced in Flask, SQL, and Git."

        tfidf = compute_tfidf_similarity(jd, resume)
        if 0 <= tfidf <= 1:
            ok("compute_tfidf_similarity", f"{tfidf:.3f}")
        else:
            fail("compute_tfidf_similarity", str(tfidf))

        skill = compute_skill_score(["python", "flask", "docker"], ["python", "flask", "git"])
        if abs(skill - 2 / 3) < 0.01:
            ok("compute_skill_score", f"{skill:.3f}")
        else:
            fail("compute_skill_score", str(skill))

        final, bert, tfidf_s, skill_s = compute_hybrid_score(
            jd, resume, ["python", "flask", "docker"], ["python", "flask", "git"]
        )
        if 0 <= final <= 1 and final > 0:
            ok("compute_hybrid_score", f"final={final:.3f} bert={bert:.3f} tfidf={tfidf_s:.3f} skill={skill_s:.3f}")
        else:
            fail("compute_hybrid_score", f"final={final}")

        ranked = compute_similarity(
            jd,
            [("resume_a.pdf", resume), ("resume_b.pdf", "Java Spring developer.")],
            jd_skills=["python", "flask", "docker"],
            resume_skills_map={
                "resume_a.pdf": ["python", "flask", "git"],
                "resume_b.pdf": ["java", "spring"],
            },
        )
        if ranked[0][0] == "resume_a.pdf" and ranked[0][1] >= ranked[1][1]:
            ok("compute_similarity ranking", f"{ranked}")
        else:
            fail("compute_similarity ranking", str(ranked))
    except Exception as e:
        fail("processor", traceback.format_exc().splitlines()[-1])


def test_pdf_generator():
    section("utils/pdf_generator.py")
    try:
        from utils.pdf_generator import generate_pdf_summary

        with tempfile.TemporaryDirectory() as tmpdir:
            path = generate_pdf_summary(
                "test_resume.pdf",
                "test_jd.pdf",
                0.85,
                ["python", "flask"],
                ["docker"],
                ["git"],
                ["Consider learning docker"],
                role="Backend Developer",
                output_dir=tmpdir,
            )
            if os.path.exists(path) and os.path.getsize(path) > 500:
                ok("generate_pdf_summary", f"{os.path.getsize(path)} bytes")
            else:
                fail("generate_pdf_summary", f"missing or tiny file: {path}")
    except Exception as e:
        err = str(e)
        if any(x in err.lower() for x in ("libgobject", "gtk", "pango")):
            warn("generate_pdf_summary", "WeasyPrint system libs missing on Windows")
        else:
            fail("pdf_generator", traceback.format_exc().splitlines()[-1])


def test_history_tracker():
    section("utils/history_tracker.py + auth/models.py")
    try:
        from app import app
        from extensions import db
        from auth.models import MatchHistory, User
        from utils.history_tracker import save_history, load_user_history
        from werkzeug.security import generate_password_hash

        with app.app_context():
            db.create_all()
            user = User.query.filter_by(username="__test_user__").first()
            if not user:
                user = User(username="__test_user__", password=generate_password_hash("test"))
                db.session.add(user)
                db.session.commit()

            save_history(
                user.id, "recruiter", "r.pdf", "jd.pdf", 0.82,
                ["python"], ["docker"], ["git"], "Developer", "summary.pdf"
            )
            history = load_user_history(user.id)
            if history and history[0]["score"] == 82.0:
                ok("save_history + load_user_history", f"{len(history)} entries for test user")
            else:
                fail("history_tracker", f"unexpected history: {history[:1]}")

            MatchHistory.query.filter_by(user_id=user.id, summary_file="summary.pdf").delete()
            User.query.filter_by(username="__test_user__").delete()
            db.session.commit()
    except Exception as e:
        fail("history_tracker", traceback.format_exc().splitlines()[-1])


def test_auth():
    section("auth/ (routes, forms, models)")
    try:
        from app import app
        from auth.forms import LoginForm, RegisterForm

        with app.app_context():
            client = app.test_client()

            r = client.get("/login")
            ok("GET /login", f"status {r.status_code}") if r.status_code == 200 else fail("GET /login", f"status {r.status_code}")

            r = client.get("/register")
            ok("GET /register", f"status {r.status_code}") if r.status_code == 200 else fail("GET /register", f"status {r.status_code}")

            r = client.get("/")
            if r.status_code in (302, 401):
                ok("GET / (auth protected)", f"status {r.status_code}")
            else:
                fail("GET /", f"unexpected status {r.status_code}")

            ok("LoginForm import")
            ok("RegisterForm import")
    except Exception as e:
        fail("auth", traceback.format_exc().splitlines()[-1])


def test_app_routes():
    section("app.py routes (with auth)")
    try:
        from app import app
        from extensions import db
        from auth.models import User
        from werkzeug.security import generate_password_hash

        with app.app_context():
            db.create_all()
            user = User.query.filter_by(username="__route_test__").first()
            if not user:
                user = User(username="__route_test__", password=generate_password_hash("pass123"))
                db.session.add(user)
                db.session.commit()

            with client_session(app, user) as client:
                r = client.get("/")
                if r.status_code == 200 and b"ResumeRanker" in r.data:
                    ok("GET / (authenticated)")
                else:
                    fail("GET / (authenticated)", f"status {r.status_code}")

                r = client.get("/history")
                ok("GET /history", f"status {r.status_code}") if r.status_code == 200 else fail("GET /history", f"status {r.status_code}")

                r = client.get("/download")
                ok("GET /download", f"status {r.status_code}") if r.status_code == 200 else fail("GET /download", f"status {r.status_code}")

            User.query.filter_by(username="__route_test__").delete()
            db.session.commit()
    except Exception as e:
        fail("app routes", traceback.format_exc().splitlines()[-1])


if __name__ == "__main__":
    print("ResumeRanker module verification\n")
    test_json_configs()
    test_parser()
    test_skill_extractor()
    test_processor()
    test_pdf_generator()
    test_history_tracker()
    test_auth()
    test_app_routes()

    print(f"\n{'=' * 40}")
    print(f"Results: {PASS} passed, {FAIL} failed, {WARN} warnings")
    sys.exit(1 if FAIL else 0)
