from datetime import datetime
import csv
import os
from extensions import db
from auth.models import MatchHistory

HISTORY_FILE = "user_data/history.csv"


def _migrate_csv_history():
    """Import legacy CSV history into the database once."""
    if not os.path.exists(HISTORY_FILE):
        return

    if MatchHistory.query.first() is not None:
        return

    with open(HISTORY_FILE, mode="r", encoding="utf-8") as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) < 11:
                continue
            entry = MatchHistory(
                user_id=int(row[0]),
                mode=row[1],
                file1=row[2],
                file2=row[3],
                score=float(row[4]),
                matched_skills=row[5],
                missing_skills=row[6],
                extra_skills=row[7],
                inferred_role=row[8],
                summary_file=row[9],
                timestamp=datetime.fromisoformat(row[10]),
            )
            db.session.add(entry)
    db.session.commit()


def save_history(user_id, mode, resume_name, jd_name, score, matched, missing, extra, inferred_role, summary_file):
    entry = MatchHistory(
        user_id=user_id,
        mode=mode,
        file1=resume_name,
        file2=jd_name,
        score=round(score * 100, 2),
        matched_skills=", ".join(matched),
        missing_skills=", ".join(missing),
        extra_skills=", ".join(extra),
        inferred_role=inferred_role,
        summary_file=summary_file,
        timestamp=datetime.utcnow(),
    )
    db.session.add(entry)
    db.session.commit()


def load_user_history(user_id):
    entries = (
        MatchHistory.query
        .filter_by(user_id=user_id)
        .order_by(MatchHistory.timestamp.desc())
        .all()
    )
    return [
        {
            "mode": entry.mode,
            "file1": entry.file1,
            "file2": entry.file2,
            "score": entry.score,
            "matched": entry.matched_skills.split(", ") if entry.matched_skills else [],
            "missing": entry.missing_skills.split(", ") if entry.missing_skills else [],
            "extra": entry.extra_skills.split(", ") if entry.extra_skills else [],
            "inferred_role": entry.inferred_role,
            "summary_file": entry.summary_file,
            "timestamp": entry.timestamp,
        }
        for entry in entries
    ]
