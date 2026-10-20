"""Mutation test for verify_claims.py: it must pass on the manuscript as written and fail,
for the right reason, on deliberately broken copies.

  1. unchanged manuscript                         -> exit 0
  2. one number altered (11.3 -> 11.4)            -> exit 1, one MISMATCH
  3. one amber number added with no claim         -> exit 1, one UNCOVERED
  4. a sign flipped in a signed number            -> exit 1, MISMATCH
  5. an anchor sentence reworded                  -> exit 1, ANCHOR

Usage: python -I test_verify_claims.py
"""
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(HERE, "..", "paper", "manuscript.tex")
SCRIPT = os.path.join(HERE, "verify_claims.py")


def run(text):
    with tempfile.NamedTemporaryFile("w", suffix=".tex", delete=False, encoding="utf-8") as f:
        f.write(text)
        path = f.name
    env = dict(os.environ, VERIFY_TEX=path)
    p = subprocess.run([sys.executable, "-I", SCRIPT], capture_output=True, text=True, env=env)
    os.unlink(path)
    return p.returncode, p.stdout


def first(pattern, text):
    """Position and text of the first match in the body (after the abstract), so the test
    adapts when numbers change."""
    import re
    start = text.index(r"\section{Introduction}")
    m = re.compile(pattern).search(text, start)
    return m.start(), m.group(0)


def main():
    base = open(TEX, encoding="utf-8").read()
    # a plain decimal amber number, nudged by one in its last digit
    pos, tok = first(r"\\pn\{[0-9]+\.[0-9]\}", base)
    val = tok[4:-1]
    bumped = "%.1f" % (float(val) + 0.3)
    altered = base[:pos] + r"\pn{%s}" % bumped + base[pos + len(tok):]
    # a signed amber number with its sign flipped
    pos2, tok2 = first(r"\\pn\{\$\+[0-9.]+\$\}", base)
    flipped = base[:pos2] + tok2.replace("$+", "$-") + base[pos2 + len(tok2):]
    cases = [
        ("unchanged", base, 0, None),
        ("number altered", altered, 1, "MISMATCH"),
        ("uncovered number", base.replace(r"\section{Discussion}", r"An extra \pn{42} here. \section{Discussion}", 1), 1, "UNCOVERED"),
        ("sign flipped", flipped, 1, "MISMATCH"),
        ("anchor reworded", base.replace("In index points, a Friday changes", "In index points, Fridays change", 1), 1, "ANCHOR"),
    ]
    bad = 0
    for name, text, want, marker in cases:
        if text == base and name != "unchanged":
            print("SETUP FAIL  %-18s replacement did not apply" % name)
            bad += 1
            continue
        code, out = run(text)
        ok = code == want and (marker is None or marker in out)
        print("%s  %-18s exit %d (want %d)%s" % ("PASS" if ok else "FAIL", name, code, want,
                                                  "" if marker is None else ", reports %s" % marker))
        bad += 0 if ok else 1
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
