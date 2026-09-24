"""Zero-framework web demo for Market RabbitHole."""

from __future__ import annotations

import html
import sys
from email.parser import BytesParser
from email.policy import default
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from rabbithole.market_agent import build_market_agent  # noqa: E402
from rabbithole.semantic_rag import load_uploaded_chunks  # noqa: E402

_agent = None

CSS = """
:root{--ink:#191714;--paper:#f3eddf;--orange:#ef5b3f;--lime:#c7ed52;--blue:#6579ff;--line:#29251f;--soft:#ded6c5}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace}main{max-width:1220px;margin:auto;padding:32px}
.mast{display:grid;grid-template-columns:1.4fr .6fr;align-items:end;border-bottom:3px solid;padding-bottom:24px}.eyebrow,.badge{display:inline-block;background:var(--lime);border:2px solid;padding:4px 9px;border-radius:99px;font-weight:800;font-size:12px}
h1{font:900 clamp(52px,8vw,104px)/.78 Arial,sans-serif;letter-spacing:-.07em;margin:12px 0 0}.tag{font:800 19px/1.35 Arial,sans-serif;max-width:340px}.sub{font-size:12px;opacity:.65}
.research-form{display:grid;grid-template-columns:1fr auto;gap:12px;margin:30px 0}.question{padding:20px;border:3px solid;background:#fff;font:700 17px inherit;box-shadow:6px 6px 0 var(--line)}button{background:var(--orange);border:3px solid;padding:0 28px;font:900 17px inherit;cursor:pointer;box-shadow:6px 6px 0 var(--line)}.upload{grid-column:1/-1;border:2px dashed;padding:13px;background:#fff}.upload input{margin-left:10px}
.grid{display:grid;grid-template-columns:1.4fr .6fr;gap:20px}.card{border:2px solid;background:#fff;padding:20px;box-shadow:5px 5px 0 var(--line);margin-bottom:20px}.dark{background:var(--ink);color:#fff}.blue{background:var(--blue);color:#fff}.orange{background:var(--orange)}
h2,h3{font-family:Arial,sans-serif;margin:0 0 14px}.strength{font:900 48px Arial}.angle{border-left:7px solid var(--lime);padding:8px 14px;margin:12px 0}.evidence{border-top:1px solid var(--soft);padding:18px 0}.meta{display:flex;gap:7px;flex-wrap:wrap;margin-bottom:9px}.label{border:1px solid;padding:2px 6px;font-size:11px;font-weight:800}.support{background:var(--lime)}.contradict{background:var(--orange)}.neutral{background:var(--soft)}
blockquote{margin:10px 0;padding-left:13px;border-left:3px solid var(--soft);font-family:Georgia,serif}.source{color:inherit;font-weight:800}.conflict{border-top:2px solid;padding-top:12px;margin-top:12px}.signal{padding:8px;margin:7px 0;background:#fff;color:var(--ink)}.driver{font:800 18px Arial;margin:9px 0}.trace{display:flex;gap:8px;overflow:auto}.step{min-width:190px;border:1px solid;padding:10px;font-size:12px}.error{border:3px solid var(--orange);background:#fff;padding:22px;box-shadow:6px 6px 0 var(--orange)}
@media(max-width:780px){.mast,.grid,.research-form{grid-template-columns:1fr}.tag{margin-top:25px}.question,button{width:100%;min-height:60px}.upload{grid-column:auto}.upload input{display:block;margin:8px 0 0}}
"""


def esc(value: object) -> str:
    return html.escape(str(value))


def page(result=None, question: str = "", error: str = "") -> str:
    if result:
        body = render_result(result)
    elif error:
        body = f"<div class='error'><h2>SETUP / RESEARCH ERROR</h2><p>{esc(error)}</p><p class='sub'>Check .env, network access, and installed dependencies, then retry.</p></div>"
    else:
        body = "<div class='card'><h2>FLAGSHIP RESEARCH QUESTION</h2><p><b>Why are European natural gas prices rising despite high storage levels?</b></p><p class='sub'>Research across your local knowledge base and live market sources. Follow the evidence. Surface conflicting signals.</p></div>"
    return f"""<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'><title>Market RabbitHole</title><style>{CSS}</style></head><body><main>
    <header class='mast'><div><span class='eyebrow'>LOCAL RAG × LIVE WEB × ZHIPU</span><h1>MARKET<br>RABBIT HOLE</h1></div><div><p class='tag'>Don't just read the market.<br>Understand what's moving it.</p><p class='sub'>Evidence-backed market research, one rabbit hole at a time.</p></div></header>
    <form class='research-form' method='post' enctype='multipart/form-data'><input class='question' name='question' value='{esc(question)}' placeholder='Why are European gas prices rising?' required><button>RESEARCH →</button><label class='upload'><b>UPLOAD PRIVATE RESEARCH (OPTIONAL)</b><input type='file' name='sources' accept='.pdf,.md,.txt' multiple><span class='sub'> PDF, MD or TXT · used for this run only</span></label></form>{body}</main></body></html>"""


def render_result(result) -> str:
    angles = "".join(f"<div class='angle'><b>{esc(a.name)}</b><br>{esc(a.question)}<div class='sub'>{esc(a.rationale)}</div></div>" for a in result.angles)
    local_evidence = "".join(render_evidence(item) for item in result.evidence if item.source_type == "LOCAL") or "<p class='sub'>No relevant local evidence was retrieved.</p>"
    web_evidence = "".join(render_evidence(item) for item in result.evidence if item.source_type == "WEB") or "<p class='sub'>Live web evidence is unavailable for this run.</p>"
    conflicts = "".join(render_conflict(item) for item in result.conflicts) or "<p class='sub'>No angle contains both meaningful supporting and contradicting evidence.</p>"
    drivers = "".join(f"<div class='driver'>{esc(item)}</div>" for item in result.brief.key_drivers)
    caveats = [*result.brief.conflicting_signals, *result.warnings]
    contrary = "".join(f"<li>{esc(item)}</li>" for item in caveats) or "<li>No material caveat was identified.</li>"
    monitor = "".join(f"<li>{esc(item)}</li>" for item in result.brief.monitor_next)
    trace = "".join(f"<div class='step'><b>{esc(t.step)}</b><br><span class='sub'>{esc(t.detail)}</span></div>" for t in result.trace)
    sources = "".join(render_source(item) for item in result.evidence)
    return f"""<div class='grid'><section>
      <div class='card dark'><h2>EXECUTIVE SUMMARY</h2><p>{esc(result.brief.current_view)}</p></div>
      <div class='card'><h2>RESEARCH MAP</h2>{angles}</div>
      <div class='card'><h2>EVIDENCE FROM UPLOADED / LOCAL DOCUMENTS</h2>{local_evidence}</div>
      <div class='card'><h2>LIVE WEB EVIDENCE</h2>{web_evidence}</div>
      <div class='card'><h2>SOURCES</h2><ol>{sources}</ol></div>
    </section><aside>
      <div class='card blue'><div class='strength'>{esc(result.brief.evidence_strength)}</div><b>EVIDENCE STRENGTH</b></div>
      <div class='card'><h3>KEY FINDINGS</h3>{drivers}</div>
      <div class='card orange'><h3>CONFLICT RADAR</h3>{conflicts}</div>
      <div class='card'><h3>RISKS / CAVEATS</h3><ul>{contrary}</ul></div>
      <div class='card'><h3>WHAT TO MONITOR NEXT</h3><ul>{monitor}</ul></div>
    </aside></div><div class='card'><h3>RESEARCH TRACE</h3><div class='trace'>{trace}</div></div>"""


def render_evidence(item) -> str:
    stance_class = item.stance.lower()
    source_link = f"<a class='source' href='{esc(item.locator)}' target='_blank' rel='noopener'>{esc(item.source)}</a>" if item.source_type == "WEB" and item.locator else f"<span class='source'>{esc(item.source)}</span>"
    date = f"<span class='label'>{esc(item.date)}</span>" if item.date else ""
    relevance = f" · local similarity {item.relevance:.2f}" if item.relevance is not None else ""
    return f"""<article class='evidence'><div class='meta'><span class='label'>{esc(item.citation_id)}</span><span class='label'>{esc(item.source_type)}</span><span class='label'>{esc(item.research_angle)}</span><span class='label {stance_class}'>{esc(item.stance)}</span>{date}</div><h3>{esc(item.key_claim or item.title)}</h3><div>{source_link}{esc(relevance)}</div><blockquote>{esc(item.excerpt)}</blockquote><p class='sub'>{esc(item.explanation)}</p></article>"""


def render_source(item) -> str:
    label = f"[{esc(item.citation_id)}] {esc(item.title)} — {esc(item.source)}"
    if item.source_type == "WEB" and item.locator:
        return f"<li><a class='source' href='{esc(item.locator)}' target='_blank' rel='noopener'>{label}</a>{' · ' + esc(item.date) if item.date else ''}</li>"
    return f"<li>{label}{' · ' + esc(item.date) if item.date else ''}</li>"


def render_conflict(item) -> str:
    support = "".join(f"<div class='signal'><b>SUPPORT</b><br>{esc(claim)}</div>" for claim in item.support)
    contradict = "".join(f"<div class='signal'><b>CONTRADICT</b><br>{esc(claim)}</div>" for claim in item.contradict)
    return f"<div class='conflict'><span class='badge'>CONFLICT DETECTED</span><h3>{esc(item.angle)}</h3>{support}{contradict}</div>"


def get_agent():
    global _agent
    if _agent is None:
        _agent = build_market_agent(ROOT / "sample_corpus")
    return _agent


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.respond(page())

    def do_POST(self):
        size = int(self.headers.get("Content-Length", 0))
        if size > 15 * 1024 * 1024:
            self.respond(page(error="Upload request exceeds the 15 MB limit."), status=413)
            return
        body = self.rfile.read(size)
        question, files = parse_submission(self.headers.get("Content-Type", ""), body)
        try:
            uploaded_chunks = load_uploaded_chunks(files)
            result = get_agent().research(question, uploaded_chunks)
            self.respond(page(result, question))
        except Exception as exc:
            self.respond(page(question=question, error=str(exc)), status=500)

    def respond(self, body: str, status: int = 200):
        payload = body.encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        return


def parse_submission(content_type: str, body: bytes) -> tuple[str, list[tuple[str, bytes]]]:
    if not content_type.startswith("multipart/form-data"):
        question = parse_qs(body.decode()).get("question", [""])[0]
        return question, []
    message = BytesParser(policy=default).parsebytes(
        f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode() + body
    )
    question = ""
    files: list[tuple[str, bytes]] = []
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        filename = part.get_filename()
        payload = part.get_payload(decode=True) or b""
        if name == "question":
            question = payload.decode(part.get_content_charset() or "utf-8", errors="replace")
        elif name == "sources" and filename and Path(filename).suffix.lower() in {".pdf", ".md", ".txt"}:
            files.append((Path(filename).name, payload))
    return question, files


if __name__ == "__main__":
    print("Market RabbitHole → http://127.0.0.1:8000")
    HTTPServer(("127.0.0.1", 8000), Handler).serve_forever()
