"""Static checks for the conference manuscript, standing in for a compiler that is not installed.

Checks: environment nesting, brace depth, cite keys against the bib in both directions,
ref/label resolution, figure files, tabular column counts, ASCII hygiene, em dashes,
placeholders, and a word count with a page estimate.
"""
import io
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TEX = os.path.join("paper1_conf", "main.tex")
BIB = os.path.join("paper1_conf", "references.bib")
FIGDIR = os.path.join("paper1_conf", "figures")
WORDS_PER_PAGE = 900          # two-column IEEE conference page, incl. captions
WORDS_PER_FLOAT = 260         # a figure or table displaces roughly this much text

tex = io.open(TEX, encoding="utf-8").read()
bib = io.open(BIB, encoding="utf-8").read()
fail = []

print("=== STRUCTURE ===")
print("lines            : %d" % len(tex.splitlines()))
print("begin{document}  : %d" % tex.count("\\begin{document}"))
print("end{document}    : %d" % tex.count("\\end{document}"))
if tex.count("\\begin{document}") != 1 or tex.count("\\end{document}") != 1:
    fail.append("document environment not exactly once")

body = tex.split("\\begin{document}")[1].split("\\end{document}")[0]
preamble = tex.split("\\begin{document}")[0]

begin_re = re.compile(r"\\begin\{([^}]+)\}")
end_re = re.compile(r"\\end\{([^}]+)\}")
stack, env_problems = [], []
for m in re.finditer(r"\\(begin|end)\{([^}]+)\}", tex):
    kind, name = m.group(1), m.group(2)
    if kind == "begin":
        stack.append(name)
    else:
        if not stack:
            env_problems.append("close of %s with nothing open" % name)
        elif stack[-1] != name:
            env_problems.append("close of %s while %s open" % (name, stack[-1]))
            stack.pop()
        else:
            stack.pop()
print("environments     : %s" % sorted({n for n in begin_re.findall(tex)}))
print("unclosed         : %s" % (stack or "none"))
print("mismatches       : %s" % (env_problems or "none"))
if stack or env_problems:
    fail.append("environment nesting")

print("\n=== BRACES ===")
depth, worst = 0, 0
i = 0
while i < len(tex):
    ch = tex[i]
    if ch == "\\":
        i += 2
        continue
    if ch == "{":
        depth += 1
        worst = max(worst, depth)
    elif ch == "}":
        depth -= 1
        if depth < 0:
            fail.append("brace closed below zero")
            break
    i += 1
print("final depth      : %d (max %d)" % (depth, worst))
if depth != 0:
    fail.append("brace imbalance")

print("\n=== CITATIONS ===")
cited = set()
for group in re.findall(r"\\cite\{([^}]+)\}", body):
    cited.update(k.strip() for k in group.split(","))
bibkeys = re.findall(r"@\w+\{([^,]+),", bib)
print("bib entries      : %d" % len(bibkeys))
print("cited keys       : %d" % len(cited))
missing = sorted(cited - set(bibkeys))
unused = sorted(set(bibkeys) - cited)
print("MISSING          : %s" % (missing or "none"))
print("UNUSED           : %s" % (unused or "none"))
if missing:
    fail.append("cites without bib entries")
if unused:
    fail.append("bib entries never cited")

print("\n=== CROSS REFERENCES ===")
labels = set(re.findall(r"\\label\{([^}]+)\}", body))
refs = set(re.findall(r"\\ref\{([^}]+)\}", body))
print("labels           : %d" % len(labels))
print("refs             : %d" % len(refs))
print("BROKEN           : %s" % (sorted(refs - labels) or "none"))
print("UNUSED labels    : %s" % (sorted(labels - refs) or "none"))
if refs - labels:
    fail.append("refs without labels")
if labels - refs:
    fail.append("labels never referenced")

print("\n=== FIGURES ===")
for path in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", body):
    full = os.path.join("paper1_conf", path)
    ok = os.path.exists(full)
    size = "%.1f KB" % (os.path.getsize(full) / 1024.0) if ok else "MISSING"
    print("  %-34s %s" % (path, size))
    if not ok:
        fail.append("missing figure %s" % path)

print("\n=== TABULAR WIDTHS ===")
for block in re.findall(r"\\begin\{tabular\}\{([^}]*)\}(.*?)\\end\{tabular\}", body, re.S):
    spec, rows = block
    ncols = len(re.findall(r"[lcrp]", spec))
    for row in rows.split("\\\\"):
        row = row.strip()
        if not row or row.startswith("\\hline") and row.count("&") == 0:
            continue
        cells = row.count("&") + 1
        if cells != ncols:
            print("  %d-cell row in a %d-column table: %s" % (cells, ncols, row[:60]))
            fail.append("tabular column mismatch")

print("\n=== HYGIENE ===")
nonascii = sorted({ch for ch in tex if ord(ch) > 127})
print("non-ASCII        : %s" % (nonascii or "none"))
if nonascii:
    fail.append("non-ASCII characters")
cr = tex.count("\r")
print("stray CR         : %d" % cr)
if cr:
    fail.append("carriage returns")
dashes = tex.count("---")
print("em dashes        : %d" % dashes)
if dashes:
    fail.append("em dash present")
for token in ("TODO", "FIXME", "XXX", "??", "VERIFY", "[insert"):
    if token in body:
        print("placeholder      : %s" % token)
        fail.append("placeholder %s" % token)
print("placeholders     : none" if not any(t in body for t in ("TODO", "FIXME", "XXX", "??")) else "")

print("\n=== LENGTH ===")
words_body = len(re.sub(r"\\[a-zA-Z]+(\[[^\]]*\])?(\{[^}]*\})?", " ", body).split())
floats = body.count("\\begin{figure}") + body.count("\\begin{table}")
i = tex.index("\\begin{abstract}") + len("\\begin{abstract}")
abstract = tex[i:tex.index("\\end{abstract}", i)]
words_abs = len([w for w in re.sub(r"\\[a-zA-Z]+(\[[^\]]*\])?(\{[^{}]*\})?", " ", abstract)
                 .replace("{", " ").replace("}", " ").split() if any(c.isalnum() for c in w)])
print("abstract words   : %d (IEEE asks for 150-250)" % words_abs)
if not 150 <= words_abs <= 250:
    fail.append("abstract outside 150-250 words")
print("body words       : %d" % words_body)
print("floats           : %d figures+tables" % floats)
estimate = words_body / WORDS_PER_PAGE + floats * WORDS_PER_FLOAT / WORDS_PER_PAGE
print("estimated pages  : %.1f (+ references, about 0.9 for %d entries)" % (estimate, len(bibkeys)))
print("target           : 6 pages including references")

print("\n=== RESULT ===")
if fail:
    print("PROBLEMS: %s" % "; ".join(sorted(set(fail))))
else:
    print("clean")
