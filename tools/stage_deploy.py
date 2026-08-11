#!/usr/bin/env python3
"""Stage soar as a deploy-ready static tree for thomasddewitt.com.

The site is a fully static Cloudflare R2 bucket, deployed with

    rclone sync . r2-website:personal-website

run from the root of the separate ``personal-website`` repo. The URL
convention there is stable — ``thomasddewitt.com/thought-cloud/<slug>/`` — and
soar's canonical URL is already baked into ``web/soar/index.html``. So this
tool emits exactly one folder, ready to be dropped into that repo's working
tree:

    <out>/thought-cloud/soar/        contents of web/soar/, fingerprinted

A thought-cloud folder is one self-contained artifact, so the baked demo set
lives inside the app at web/soar/demos/ and is staged with it, verbatim.

Why fingerprinting: the Cloudflare edge plus a 4h browser cache TTL have
served stale JavaScript against fresh HTML before. The site convention is
fingerprinted filenames for sub-site assets. Soar is ~30 ES modules that
import each other by plain relative name, so fingerprinting means rewriting
the import graph, not just renaming files.

Only the app's own mutable source is fingerprinted. ``index.html`` keeps its
name (it *is* the canonical URL), and ``vendor/``, ``ocean/`` and the demo
assets are left alone — they are immutable blobs or runtime-resolved data.

The hash is transitive: a file's name changes when the file changes *or* when
anything in its transitive dependency set changes, so a fresh ``constants.js``
renames every module downstream of it and no cached mix of old and new can
survive. The scheme is cycle-tolerant by construction (it hashes an unordered
dependency *set*, not a topological order), which matters because an ES module
graph is allowed to have cycles.

Usage:

    uv run python tools/stage_deploy.py [--out DIR] [--clean]
"""

from __future__ import annotations

import argparse
import hashlib
import posixpath
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SOAR_SRC = REPO_ROOT / "web" / "soar"

SITE_SUBDIR = "thought-cloud"
APP_DIRNAME = "soar"

HASH_CHARS = 8

# Files whose names carry a content hash. Everything else keeps its name.
# NOT fingerprinted, deliberately: index.html (the canonical URL), vendor/**
# (immutable third-party blobs), ocean/** (runtime-resolved tile data).
FINGERPRINT_PATTERNS = (
    "*.js",
    "ingest/*.js",
    "raymarch.wgsl",
    "hud.wgsl",
    "bird.wgsl",
    "style.css",
    "viewer.css",
)

# Directories under web/soar/ that are copied through untouched: no renaming,
# no rewriting, and excluded from the reference gate. Vendor code is minified
# third-party output that never names an app file (verified by grep), and the
# ocean tile is binary.
# Copied through untouched: never imported, fetched by constructed paths,
# and immutable in practice. `demos/` is baked output, hundreds of MB.
OPAQUE_DIRS = ("vendor/", "ocean/", "demos/")

# rclone sets each object's Content-Type from its file extension via Go's mime
# table. On macOS — where the deploy runs — that falls back to a builtin list
# that does not reliably include `.mjs`. An `.mjs` served as
# application/octet-stream is rejected outright by Chrome's strict MIME check
# for ES modules, which would kill video export silently, since it hangs off a
# dynamic import("./video.js"). Staging the file as plain `.js` removes the
# dependence on the deploy host's mime table entirely. Source tree keeps .mjs.
EXTENSION_FIXUPS = {
    "vendor/mediabunny/mediabunny.min.mjs": "vendor/mediabunny/mediabunny.min.js",
}

# Extensions we are willing to hand to a static host, i.e. ones whose
# Content-Type we can predict. Anything else is a staging error, not a warning.
ALLOWED_EXTENSIONS = {
    ".html", ".js", ".css", ".wgsl", ".json",
    ".bin", ".gz", ".webp", ".jpg", ".jpeg", ".png", ".mp4", ".webm",
    ".so", ".txt",
}
# License/notice files that legitimately carry no extension.
ALLOWED_EXTENSIONLESS = {"LICENSE", "COPYING", "NOTICE"}

EXPECTED_PLUGIN_COUNT = 11
PLUGIN_DIR = "vendor/h5wasm-plugins/plugins"

TEXT_SUFFIXES = {".js", ".mjs", ".wgsl", ".css", ".html", ".json"}

REMOTE_SCHEMES = ("http://", "https://", "//", "data:", "blob:", "about:")


class StagingError(RuntimeError):
    """Anything that would produce a tree we cannot vouch for."""


# --------------------------------------------------------------------------
# Lexing: we need to know which byte ranges of a file are string literals and
# which are comments. References only ever live in strings, and the gate has to
# tell a real leftover reference from prose in a comment that happens to name a
# module ("see filters.js" and friends are all over these files).
# --------------------------------------------------------------------------


@dataclass
class Span:
    start: int          # index of the first character of the literal content
    end: int            # index one past the last character of the content
    has_interp: bool = False


@dataclass
class Lexed:
    strings: list[Span] = field(default_factory=list)
    comments: list[Span] = field(default_factory=list)

    def classify(self, index: int) -> str:
        for span in self.comments:
            if span.start <= index < span.end:
                return "comment"
        for span in self.strings:
            if span.start <= index < span.end:
                return "string"
        return "code"


# What a `/` may legally follow and still begin a regex literal rather than be
# division. Deliberately NOT newline: `prev` tracks the last non-whitespace
# character, so a division continued onto the next line (spectral.js wraps one
# across four) keeps the operand's `)` as `prev` and reads as division. Every
# real regex literal in this codebase follows `(`, `=` or `!` on its own line.
_REGEX_PRECEDERS = set("(,=:[!&|?{};+-*%~^<>")


def lex_js(text: str, where: str) -> Lexed:
    """Split JavaScript into string and comment spans.

    Deliberately strict: an unterminated literal raises rather than being
    skipped, because a file we cannot parse is a file whose references we
    cannot claim to have rewritten.
    """
    out = Lexed()
    n = len(text)
    i = 0
    prev = "\n"  # last significant code character, for regex-vs-divide
    while i < n:
        c = text[i]
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            start = i
            j = text.find("\n", i)
            i = n if j < 0 else j
            out.comments.append(Span(start, i))
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "*":
            j = text.find("*/", i + 2)
            if j < 0:
                raise StagingError(f"{where}: unterminated block comment at offset {i}")
            out.comments.append(Span(i, j + 2))
            i = j + 2
            continue
        if c == "/" and prev in _REGEX_PRECEDERS:
            i = _consume_regex(text, i, where)
            prev = "/"
            continue
        if c in "\"'":
            i, span = _consume_quoted(text, i, where)
            out.strings.append(span)
            prev = c
            continue
        if c == "`":
            i, span = _consume_template(text, i, where)
            out.strings.append(span)
            prev = "`"
            continue
        if not c.isspace():
            prev = c
        i += 1
    return out


def _consume_quoted(text: str, i: int, where: str) -> tuple[int, Span]:
    quote = text[i]
    j = i + 1
    n = len(text)
    while j < n:
        c = text[j]
        if c == "\\":
            j += 2
            continue
        if c == "\n":
            raise StagingError(
                f"{where}: unterminated {quote}-string starting at offset {i}")
        if c == quote:
            return j + 1, Span(i + 1, j)
        j += 1
    raise StagingError(f"{where}: unterminated {quote}-string starting at offset {i}")


def _consume_template(text: str, i: int, where: str) -> tuple[int, Span]:
    """Consume a template literal, brace-counting through ``${...}``.

    The whole literal, interpolations included, is reported as one string span
    and flagged, so reference extraction skips it (a path built at runtime is
    not ours to rewrite) while the gate still treats its contents as string
    context.
    """
    j = i + 1
    n = len(text)
    has_interp = False
    while j < n:
        c = text[j]
        if c == "\\":
            j += 2
            continue
        if c == "`":
            return j + 1, Span(i + 1, j, has_interp)
        if c == "$" and j + 1 < n and text[j + 1] == "{":
            has_interp = True
            j = _consume_interpolation(text, j + 2, where)
            continue
        j += 1
    raise StagingError(f"{where}: unterminated template literal at offset {i}")


def _consume_interpolation(text: str, i: int, where: str) -> int:
    """Consume a ``${...}`` body, returning the index just past its ``}``.

    An interpolation is arbitrary expression territory: it can hold quoted
    strings and further template literals, each of which may itself contain
    braces. Counting braces without lexing those first is what breaks on
    ``ingest/netcdf.js``, whose error message nests a template inside an
    interpolation inside a template.
    """
    j = i
    n = len(text)
    depth = 1
    while j < n:
        c = text[j]
        if c == "\\":
            j += 2
            continue
        if c in "\"'":
            j, _ = _consume_quoted(text, j, where)
            continue
        if c == "`":
            j, _ = _consume_template(text, j, where)
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return j + 1
        j += 1
    raise StagingError(
        f"{where}: unterminated interpolation starting at offset {i}")


def _consume_regex(text: str, i: int, where: str) -> int:
    j = i + 1
    n = len(text)
    in_class = False
    while j < n:
        c = text[j]
        if c == "\\":
            j += 2
            continue
        if c == "\n":
            raise StagingError(f"{where}: unterminated regex literal at offset {i}")
        if c == "[":
            in_class = True
        elif c == "]":
            in_class = False
        elif c == "/" and not in_class:
            j += 1
            while j < n and text[j].isalpha():
                j += 1
            return j
        j += 1
    raise StagingError(f"{where}: unterminated regex literal at offset {i}")


def lex_css(text: str, where: str) -> Lexed:
    out = Lexed()
    n = len(text)
    i = 0
    while i < n:
        c = text[i]
        if c == "/" and i + 1 < n and text[i + 1] == "*":
            j = text.find("*/", i + 2)
            if j < 0:
                raise StagingError(f"{where}: unterminated CSS comment at offset {i}")
            out.comments.append(Span(i, j + 2))
            i = j + 2
            continue
        if c in "\"'":
            i, span = _consume_quoted(text, i, where)
            out.strings.append(span)
            continue
        if text.startswith("url(", i):
            j = text.find(")", i)
            if j < 0:
                raise StagingError(f"{where}: unterminated url() at offset {i}")
            inner = text[i + 4:j]
            if inner[:1] not in {"\"", "'"}:
                out.strings.append(Span(i + 4, j))
            i += 4
            continue
        i += 1
    return out


def lex_wgsl(text: str, where: str) -> Lexed:
    out = Lexed()
    n = len(text)
    i = 0
    while i < n:
        if text.startswith("//", i):
            j = text.find("\n", i)
            end = n if j < 0 else j
            out.comments.append(Span(i, end))
            i = end
            continue
        if text.startswith("/*", i):
            j = text.find("*/", i + 2)
            if j < 0:
                raise StagingError(f"{where}: unterminated WGSL comment at offset {i}")
            out.comments.append(Span(i, j + 2))
            i = j + 2
            continue
        i += 1
    return out


_HTML_ATTR_RE = re.compile(r"""([A-Za-z_:][-A-Za-z0-9_:.]*)\s*=\s*("([^"]*)"|'([^']*)')""")


def lex_html(text: str, where: str) -> Lexed:
    out = Lexed()
    n = len(text)
    i = 0
    while i < n:
        if text.startswith("<!--", i):
            j = text.find("-->", i + 4)
            if j < 0:
                raise StagingError(f"{where}: unterminated HTML comment at offset {i}")
            out.comments.append(Span(i, j + 3))
            i = j + 3
            continue
        if text[i] == "<":
            j = text.find(">", i)
            if j < 0:
                raise StagingError(f"{where}: unterminated tag at offset {i}")
            tag = text[i:j]
            for m in _HTML_ATTR_RE.finditer(tag):
                inner = m.start(2) + 1 + i
                out.strings.append(Span(inner, inner + len(m.group(3) or m.group(4) or "")))
            i = j + 1
            continue
        i += 1
    return out


LEXERS = {
    ".js": lex_js,
    ".mjs": lex_js,
    ".css": lex_css,
    ".wgsl": lex_wgsl,
    ".html": lex_html,
}


def lex(relpath: str, text: str) -> Lexed:
    suffix = Path(relpath).suffix
    lexer = LEXERS.get(suffix)
    if lexer is None:
        raise StagingError(f"{relpath}: no lexer for '{suffix}'; refusing to guess "
                           "at its references")
    return lexer(text, relpath)


# --------------------------------------------------------------------------
# Reference discovery
# --------------------------------------------------------------------------


@dataclass
class Reference:
    span: Span
    raw: str
    target: str   # relpath under soar/


def resolve(from_relpath: str, raw: str) -> str | None:
    """Resolve a literal to a soar-relative path, or None if it is not ours.

    Two resolution bases are in play. ``./x`` and ``../x`` are module- (or
    stylesheet-) relative, matching how the browser resolves import specifiers
    and ``new URL(..., import.meta.url)``. A bare ``x`` is *document*-relative:
    viewer.js does ``fetch("raymarch.wgsl")``, which the browser resolves
    against the page, not the module — the same base index.html's own
    href/src use.
    """
    if not raw or raw.startswith(REMOTE_SCHEMES) or raw.startswith("/"):
        return None
    if "\n" in raw or " " in raw.strip():
        return None
    base = posixpath.dirname(from_relpath) if raw.startswith((".", "..")) else ""
    joined = posixpath.normpath(posixpath.join(base, raw))
    if joined.startswith("..") or joined in {".", ""}:
        return None
    return joined


def find_references(relpath: str, text: str, renames: dict[str, str]) -> list[Reference]:
    refs = []
    for span in lex(relpath, text).strings:
        if span.has_interp:
            continue
        raw = text[span.start:span.end]
        target = resolve(relpath, raw)
        if target is not None and target in renames:
            refs.append(Reference(span, raw, target))
    return refs


def rewrite(text: str, refs: list[Reference], renames: dict[str, str]) -> str:
    """Swap each reference's final path component for its staged name."""
    out = text
    for ref in sorted(refs, key=lambda r: r.span.start, reverse=True):
        new_base = posixpath.basename(renames[ref.target])
        prefix = ref.raw[:ref.raw.rfind("/") + 1] if "/" in ref.raw else ""
        out = out[:ref.span.start] + prefix + new_base + out[ref.span.end:]
    return out


# --------------------------------------------------------------------------
# Hashing
# --------------------------------------------------------------------------


def transitive_closure(deps: dict[str, set[str]]) -> dict[str, set[str]]:
    """Fixpoint closure. Cycle-tolerant: no ordering is ever required."""
    closure = {k: set(v) for k, v in deps.items()}
    changed = True
    while changed:
        changed = False
        for node, reach in closure.items():
            grown = set(reach)
            for d in reach:
                grown |= closure.get(d, set())
            if grown != reach:
                closure[node] = grown
                changed = True
    return closure


def find_cycles(deps: dict[str, set[str]]) -> list[list[str]]:
    """Report every elementary-ish cycle we can spot with a colored DFS."""
    cycles: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    color: dict[str, int] = {}
    stack: list[str] = []

    def visit(node: str) -> None:
        color[node] = 1
        stack.append(node)
        for d in sorted(deps.get(node, ())):
            if color.get(d, 0) == 0:
                visit(d)
            elif color.get(d) == 1:
                cycle = stack[stack.index(d):] + [d]
                key = tuple(cycle)
                if key not in seen:
                    seen.add(key)
                    cycles.append(cycle)
        stack.pop()
        color[node] = 2

    for node in sorted(deps):
        if color.get(node, 0) == 0:
            visit(node)
    return cycles


# --------------------------------------------------------------------------
# Staging
# --------------------------------------------------------------------------


def is_opaque(relpath: str) -> bool:
    return relpath.startswith(OPAQUE_DIRS)


def collect_soar_files() -> list[str]:
    files = []
    for path in sorted(SOAR_SRC.rglob("*")):
        if path.is_file():
            files.append(path.relative_to(SOAR_SRC).as_posix())
    return files


def fingerprint_targets(all_files: list[str]) -> list[str]:
    chosen = []
    for relpath in all_files:
        if is_opaque(relpath):
            continue
        for pattern in FINGERPRINT_PATTERNS:
            if "/" in pattern:
                matched = Path(relpath).match(pattern) and relpath.count("/") == pattern.count("/")
            else:
                matched = "/" not in relpath and Path(relpath).match(pattern)
            if matched:
                chosen.append(relpath)
                break
    return chosen


def read_text(path: Path, relpath: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise StagingError(f"{relpath}: not valid UTF-8, cannot inspect its "
                           f"references ({exc})") from exc


def stage(out_root: Path, clean: bool) -> dict:
    if not SOAR_SRC.is_dir():
        raise StagingError(f"missing source tree: {SOAR_SRC}")
    if not (SOAR_SRC / "demos" / "index.json").is_file():
        raise StagingError(
            f"missing demo index: {SOAR_SRC / 'demos' / 'index.json'}. The "
            "demo set is gitignored — bake it with tools/prebake_demos.py, or "
            "regenerate just the index with --index-only.")

    site_root = out_root / SITE_SUBDIR
    app_out = site_root / APP_DIRNAME

    if site_root.exists():
        if not clean:
            raise StagingError(
                f"{site_root} already exists. Staging into it would mix this "
                "run's fingerprinted names with a previous run's leftovers, "
                "which is exactly the stale-asset failure this tool exists to "
                "prevent. Re-run with --clean.")
        shutil.rmtree(site_root)

    all_files = collect_soar_files()
    targets = fingerprint_targets(all_files)
    if not targets:
        raise StagingError("found no files to fingerprint; the source layout "
                           "must have changed")

    # Pass 1: read every text file we might rewrite, and map its references.
    sources: dict[str, str] = {}
    for relpath in all_files:
        if is_opaque(relpath):
            continue
        if Path(relpath).suffix in TEXT_SUFFIXES:
            sources[relpath] = read_text(SOAR_SRC / relpath, relpath)

    # A provisional rename map (names unknown yet) is enough to discover the
    # dependency edges, since resolution only needs to know *which* files are
    # renamed, not what to.
    provisional = {t: t for t in targets}
    provisional.update(EXTENSION_FIXUPS)

    refs_by_file: dict[str, list[Reference]] = {}
    for relpath, text in sources.items():
        refs_by_file[relpath] = find_references(relpath, text, provisional)

    deps = {t: {r.target for r in refs_by_file.get(t, []) if r.target in set(targets)}
            for t in targets}
    cycles = find_cycles(deps)
    closure = transitive_closure(deps)

    own_hash = {t: hashlib.sha256((SOAR_SRC / t).read_bytes()).hexdigest()
                for t in targets}

    renames: dict[str, str] = {}
    for t in targets:
        digest = hashlib.sha256()
        digest.update((SOAR_SRC / t).read_bytes())
        for dep in sorted(own_hash[d] for d in closure[t] - {t}):
            digest.update(dep.encode("ascii"))
        short = digest.hexdigest()[:HASH_CHARS]
        p = Path(t)
        renames[t] = (p.parent / f"{p.stem}.{short}{p.suffix}").as_posix()
    renames.update(EXTENSION_FIXUPS)

    # Pass 2: emit.
    for relpath in all_files:
        src = SOAR_SRC / relpath
        dest = app_out / renames.get(relpath, relpath)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if relpath in sources:
            refs = find_references(relpath, sources[relpath], renames)
            dest.write_text(rewrite(sources[relpath], refs, renames), encoding="utf-8")
        else:
            shutil.copy2(src, dest)

    verify(app_out, all_files, renames, sources)

    return {
        "app_out": app_out,
        "renames": renames,
        "cycles": cycles,
        "deps": deps,
    }


# --------------------------------------------------------------------------
# Correctness gate
# --------------------------------------------------------------------------


def _is_whole_string(lexed, text: str, m: re.Match) -> bool:
    """Is this match the entire contents of its string literal?

    A module specifier or a fetch target is always the whole literal
    (``fetch("hud.wgsl")``). A shader name inside a compile-error message is
    not, and rewriting it would be rewriting English.
    """
    for span in lexed.strings:
        if span.start <= m.start() and m.end() <= span.end:
            return text[span.start:span.end].strip() == m.group(0)
    return False


def verify(app_out: Path, all_files: list[str],
           renames: dict[str, str], sources: dict[str, str]) -> None:
    problems: list[str] = []
    prose_mentions: list[str] = []

    renamed_originals = {r: renames[r] for r in renames if renames[r] != r}
    original_basenames = {posixpath.basename(r): r for r in renamed_originals}

    emitted = {p.relative_to(app_out).as_posix()
               for p in app_out.rglob("*") if p.is_file()}

    # 1. Every rewritten reference must land on a file that exists.
    for relpath in sources:
        staged = renames.get(relpath, relpath)
        text = (app_out / staged).read_text(encoding="utf-8")
        for span in lex(staged, text).strings:
            if span.has_interp:
                continue
            raw = text[span.start:span.end]
            resolved = resolve(staged, raw)
            if resolved is None:
                continue
            if resolved in emitted:
                continue
            # Only complain about things we actually manage. A name that never
            # existed under web/soar/ is built at run time against some other
            # base — scene.js's "volume.bin" default is a demo asset fetched
            # from demos/<id>/, not a file of ours that went missing.
            if resolved not in all_files:
                continue
            if Path(resolved).suffix in TEXT_SUFFIXES | {".so", ".bin"}:
                problems.append(
                    f"{staged}: reference {raw!r} resolves to {resolved!r}, "
                    "which does not exist in the staged tree")

    # 2. No emitted file may still name a file that got renamed. Comments are
    #    exempt: these sources refer to modules in prose all over the place
    #    ("see filters.js"), and rewriting English would be vandalism.
    for relpath in sorted(emitted):
        if is_opaque(relpath) or Path(relpath).suffix not in TEXT_SUFFIXES:
            continue
        text = (app_out / relpath).read_text(encoding="utf-8")
        lexed = lex(relpath, text)
        for basename, original in original_basenames.items():
            for m in re.finditer(re.escape(basename), text):
                before = text[m.start() - 1] if m.start() else " "
                if before.isalnum() or before in "_-":
                    continue
                # Trailing boundary too, or "index.js" matches inside
                # "index.json" and every fetch of the demo index looks broken.
                after = text[m.end()] if m.end() < len(text) else " "
                if after.isalnum() or after in "_-":
                    continue
                kind = lexed.classify(m.start())
                if kind == "comment":
                    prose_mentions.append(f"{relpath}: {basename} (comment)")
                elif kind == "string" and not _is_whole_string(lexed, text, m):
                    # The name sits inside a longer string: an error message
                    # naming the shader that failed to compile, not a URL.
                    # A real specifier is the entire string literal.
                    prose_mentions.append(f"{relpath}: {basename} (message)")
                else:
                    problems.append(
                        f"{relpath}: still names {original!r} (staged as "
                        f"{renames[original]!r}) in {kind} context at offset {m.start()}")

    # 3. No .mjs anywhere: the deploy host's mime table cannot be trusted to
    #    give it a module-safe Content-Type.
    for relpath in sorted(emitted):
        if relpath.endswith(".mjs"):
            problems.append(f"{relpath}: .mjs will not get a reliable "
                            "Content-Type from rclone on macOS")
    for relpath in sorted(emitted):
        if is_opaque(relpath) or Path(relpath).suffix not in TEXT_SUFFIXES:
            continue
        text = (app_out / relpath).read_text(encoding="utf-8")
        for m in re.finditer(r"\.mjs\b", text):
            problems.append(f"{relpath}: still references a .mjs path at "
                            f"offset {m.start()}")

    # 3b. Every emitted module must actually parse. Rewriting is text surgery
    #     on real JavaScript, and a rewrite that lands one bracket wrong
    #     produces a file that serves with a cheerful 200 and dies in the
    #     browser as a bare SyntaxError with no clue which file. `node --check`
    #     is the cheapest real parser to hand. Absence of node is an error
    #     rather than a skipped check: a staging run that silently stops
    #     verifying is worse than one that refuses to finish.
    if shutil.which("node") is None:
        raise StagingError(
            "node is required to syntax-check the staged modules; install it "
            "(dnf install nodejs) or the staged tree cannot be trusted")
    for relpath in sorted(emitted):
        if not relpath.endswith(".js") or is_opaque(relpath):
            continue
        done = subprocess.run(["node", "--check", str(app_out / relpath)],
                              capture_output=True, text=True)
        if done.returncode != 0:
            first = (done.stderr.strip().splitlines() or ["(no detail)"])
            problems.append(f"{relpath}: does not parse — "
                            + " / ".join(first[:3]))

    # 4. Extension allowlist across both emitted folders.
    surprises = []
    for root in (app_out,):
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix in ALLOWED_EXTENSIONS:
                continue
            if not suffix and path.name in ALLOWED_EXTENSIONLESS:
                continue
            surprises.append(str(path.relative_to(root.parent)))
    if surprises:
        problems.append("unexpected extensions, Content-Type unpredictable: "
                        + ", ".join(sorted(surprises)))

    # 5. The h5wasm decompression plugins. A .gitignore rule has dropped these
    #    before; a tree without them loads until someone opens a compressed
    #    netCDF, then fails at the worst moment.
    plugins = sorted((app_out / PLUGIN_DIR).glob("*.so")) if (app_out / PLUGIN_DIR).is_dir() else []
    if len(plugins) != EXPECTED_PLUGIN_COUNT:
        problems.append(
            f"expected {EXPECTED_PLUGIN_COUNT} .so plugins in {PLUGIN_DIR}, "
            f"found {len(plugins)}; refusing to stage a tree that cannot read "
            "compressed netCDF")

    # 6. index.html must keep its name — it is the canonical URL.
    if not (app_out / "index.html").is_file():
        problems.append("index.html is missing from the staged app")

    if problems:
        raise StagingError("staged tree failed verification:\n  - "
                           + "\n  - ".join(problems))

    verify.prose_mentions = prose_mentions  # type: ignore[attr-defined]


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def folder_totals(root: Path) -> tuple[dict[str, tuple[int, int]], int, dict[str, int]]:
    per: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    exts: dict[str, int] = defaultdict(int)
    total = 0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        folder = posixpath.dirname(rel) or "."
        size = path.stat().st_size
        per[folder][0] += 1
        per[folder][1] += size
        exts[path.suffix.lower() or "(none)"] += 1
        total += size
    return {k: (v[0], v[1]) for k, v in per.items()}, total, dict(exts)


def human(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n:,} B" if unit == "B" else f"{n / 1024 ** (('B', 'KiB', 'MiB', 'GiB').index(unit)):.1f} {unit}"
        n_next = n
    return f"{n} B"


def report(result: dict) -> None:
    app_out: Path = result["app_out"]
    renames: dict[str, str] = result["renames"]

    print(f"staged app   : {app_out}")
    print()

    cycles = result["cycles"]
    if cycles:
        print("module graph: CYCLES PRESENT (transitive hashing tolerates them)")
        for cycle in cycles:
            print("  " + " -> ".join(cycle))
    else:
        print("module graph: acyclic (checked; transitive hashing used anyway)")
    print()

    fingerprinted = {k: v for k, v in renames.items() if k not in EXTENSION_FIXUPS}
    print(f"fingerprinted files ({len(fingerprinted)}):")
    width = max(len(k) for k in fingerprinted)
    for old in sorted(fingerprinted):
        print(f"  {old:<{width}}  ->  {fingerprinted[old]}")
    print()
    print("staged-time renames (not fingerprinted):")
    for old, new in EXTENSION_FIXUPS.items():
        print(f"  {old}  ->  {new}   (rclone/macOS mime table has no .mjs)")
    print()

    grand = 0
    all_exts: dict[str, int] = defaultdict(int)
    for label, root in ((f"{SITE_SUBDIR}/{APP_DIRNAME}", app_out),):
        per, total, exts = folder_totals(root)
        print(f"{label}/")
        for folder in sorted(per):
            count, size = per[folder]
            shown = "(root)" if folder == "." else folder + "/"
            print(f"  {shown:<40} {count:>4} files  {size:>12,} B")
        print(f"  {'subtotal':<40} {'':>4}         {total:>12,} B")
        print()
        grand += total
        for k, v in exts.items():
            all_exts[k] += v

    print(f"grand total: {grand:,} B  ({grand / 1024 / 1024:.1f} MiB)")
    print()
    print("extensions in the staged tree (spot-check the Content-Type story):")
    for ext in sorted(all_exts):
        print(f"  {ext:<10} {all_exts[ext]:>4}")

    mentions = getattr(verify, "prose_mentions", [])
    if mentions:
        print()
        print(f"prose mentions of renamed files left alone ({len(mentions)}, all "
              "in comments):")
        for m in mentions:
            print(f"  {m}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stage soar + its demos as a deploy-ready static tree.")
    parser.add_argument("--out", default=str(REPO_ROOT / "dist"),
                        help="output directory (default: <repo>/dist)")
    parser.add_argument("--clean", action="store_true",
                        help="remove an existing staged tree first")
    args = parser.parse_args(argv)

    result = stage(Path(args.out).resolve(), args.clean)
    report(result)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except StagingError as exc:
        print(f"stage_deploy: {exc}", file=sys.stderr)
        sys.exit(1)
