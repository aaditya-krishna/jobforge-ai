from pathlib import Path

from jobforge.resume import load_resume, pdf_to_text


def make_pdf(lines: list[str]) -> bytes:
    """A minimal one-page PDF with Helvetica text, so the test needs no PDF-writing dependency."""
    ops = " ".join(f"({line}) Tj 0 -16 Td" for line in lines)
    stream = f"BT /F1 12 Tf 72 720 Td {ops} ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out, offsets = bytearray(b"%PDF-1.4\n"), []
    for i, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % o for o in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref)
    return bytes(out)


def test_pdf_to_text_extracts_lines(tmp_path: Path):
    path = tmp_path / "resume.pdf"
    path.write_bytes(make_pdf(["Jane Example", "Python and SQL", "Kubernetes on AWS"]))
    text = pdf_to_text(path)
    assert "Python and SQL" in text and "Kubernetes on AWS" in text


def test_pdf_to_text_accepts_file_objects(tmp_path: Path):
    path = tmp_path / "resume.pdf"
    path.write_bytes(make_pdf(["Go and Rust"]))
    with path.open("rb") as f:  # what a Streamlit upload looks like
        assert "Go and Rust" in pdf_to_text(f)


def test_load_resume_reads_text_files(tmp_path: Path):
    path = tmp_path / "resume.md"
    path.write_text("  Data engineer: Spark, dbt  \n", encoding="utf-8")
    assert load_resume(path) == "Data engineer: Spark, dbt"
