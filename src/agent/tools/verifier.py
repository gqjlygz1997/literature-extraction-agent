"""Verify extracted records against schema and evidence."""


class Verifier:
    """Lightweight validation for raw and clean extraction outputs."""

    def verify_raw_record(self, record, extraction_config) -> list[str]:
        """Return validation problems for one raw extraction record."""

        problems = []
        if not getattr(record, "paper_id", ""):
            problems.append("missing paper_id")
        if not getattr(record, "data", None):
            problems.append("empty extracted data")
        return problems

