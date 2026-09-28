"""A6 related-work table: fetch source abstracts / full-text snapshots used to verify each cell.

Snapshots are written to a scratch directory (NOT to this results folder), so no
third-party full text is stored with the paper's analysis files. Only short
supporting quotes are copied into results.json / results.md by build_table.py.

Usage: python fetch_sources.py <snapshot_dir>
"""
import html
import json
import re
import sys
import urllib.request
from pathlib import Path

SNAP = Path(sys.argv[1])
SNAP.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "pg26038-reanalysis/1.0 (related-work verification)"}


def get(url, timeout=60):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


def strip_jats(s):
    s = re.sub(r"<[^>]+>", " ", s or "")
    return re.sub(r"\s+", " ", html.unescape(s)).strip()


log = {}

# Crossref abstracts (publisher-deposited)
for key, doi in {"yang2023ecasegformer": "10.3390/agriculture13081513",
                 "ref29": "10.1111/ppa.70191"}.items():
    url = f"https://api.crossref.org/works/{doi}"
    m = json.loads(get(url))["message"]
    txt = strip_jats(m.get("abstract", ""))
    (SNAP / f"{key}_crossref_abstract.txt").write_text(txt, encoding="utf-8")
    log[key] = {"url": url, "chars": len(txt)}

# Europe PMC (AGRICOLA records) abstracts
for key, doi in {"ref26": "10.1016/j.compag.2021.106191",
                 "ref28": "10.1016/j.biosystemseng.2021.08.011"}.items():
    url = ("https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:%22"
           + doi + "%22&resultType=core&format=json")
    d = json.loads(get(url))
    r = d["resultList"]["result"][0]
    txt = strip_jats(r.get("abstractText", ""))
    (SNAP / f"{key}_europepmc_abstract.txt").write_text(txt, encoding="utf-8")
    log[key] = {"url": url, "record": f"{r.get('source')}:{r.get('id')}", "chars": len(txt)}

# OpenAlex abstract (inverted index) for the app paper
url = "https://api.openalex.org/works/doi:10.1016/j.inpa.2021.01.004"
m = json.loads(get(url))
inv = m.get("abstract_inverted_index") or {}
pos = {p: w for w, ps in inv.items() for p in ps}
txt = " ".join(pos[i] for i in sorted(pos))
(SNAP / "ref27_openalex_abstract.txt").write_text(txt, encoding="utf-8")
log["ref27_abstract"] = {"url": url, "chars": len(txt)}

# Embrapa Alice record (abstract meta tag) for Goncalves et al. 2021
url = "https://www.alice.cnptia.embrapa.br/alice/handle/doc/1134326"
t = get(url)
mm = re.search(r'<meta name="DC.description" content="([^"]*)"', t)
txt = html.unescape(mm.group(1)) if mm else ""
(SNAP / "ref28_alice_abstract.txt").write_text(txt, encoding="utf-8")
log["ref28_alice"] = {"url": url, "chars": len(txt)}

(SNAP / "fetch_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
print(json.dumps(log, indent=2))
