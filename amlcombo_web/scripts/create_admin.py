"""Create an initial admin/researcher user from the command line.

Run via `docker compose exec web python scripts/create_admin.py`.
"""

from __future__ import annotations

import getpass
import sys

from app.db import SessionLocal, Base, engine
from app.models import User
from app.security import hash_password


def main():
    Base.metadata.create_all(bind=engine)  # idempotent

    email = input("Email: ").strip().lower()
    full_name = input("Full name (optional): ").strip() or None
    institution = input("Institution (optional): ").strip() or None
    password = getpass.getpass("Password (≥ 8 chars): ")
    if len(password) < 8:
        print("Password too short", file=sys.stderr)
        sys.exit(1)

    with SessionLocal() as db:
        existing = db.query(User).filter(User.email == email).first()
        if existing is not None:
            print(f"User {email} already exists (id={existing.id})")
            sys.exit(0)
        user = User(
            email=email, password_hash=hash_password(password),
            full_name=full_name, institution=institution,
            is_verified=True,
        )
        db.add(user)
        db.commit()
        print(f"Created user {email} (id={user.id})")


if __name__ == "__main__":
    main()
