"""Pipeline stages.

Deliberately empty: importing ``triage.pipeline.dedup`` should not drag in
SQLAlchemy. The parse and fingerprint tests run with no services and no
database driver installed, and that stays true only if this file imports
nothing.
"""
