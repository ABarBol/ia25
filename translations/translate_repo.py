import os
import re
import time
from pathlib import Path
import requests
import nbformat

# Configuration
TARGET_LANG = os.getenv("TARGET_LANG", "es")
ROOT_DIR = Path(".")
OUTPUT_DIR = ROOT_DIR / f"translated_{TARGET_LANG}"  # kept for compatibility / future use


def safe_translate(translator, text, retries=3, delay=1.5):
    """
    Translate text with retry logic.

    Preserves trailing newline if the source text ends with one.
    Returns the original text on repeated failures.
    """
    if not text.strip():
        return text

    for attempt in range(retries):
        try:
            translated = translator.translate(text)
            if translated is not None:
                if text.endswith("\n") and not translated.endswith("\n"):
                    translated += "\n"
                return translated
        except Exception as exc:
            print(f"[WARNING] Translation error: '{text[:40]}...' ({exc}) - Attempt {attempt+1}/{retries}")
            time.sleep(delay)

    return text


def _protect_placeholders(text):
    """
    Replace regions that must not be altered by machine translation with placeholders.

    The function protects:
      - inline code fragments enclosed in backticks
      - HTML comments
      - HTML tags (to avoid altering attributes)
      - Markdown images
      - Markdown links (only the URL part)
      - plain URLs

    Returns a tuple (protected_text, placeholders_dict).
    """
    placeholders = {}
    idx = 0

    def new_ph(kind):
        nonlocal idx
        ph = f"___PLH_{kind}_{idx}___"
        idx += 1
        return ph

    # Inline code `...`
    def repl_code(m):
        ph = new_ph("CODE")
        placeholders[ph] = m.group(0)
        return ph

    text = re.sub(r'`[^`]+`', repl_code, text)

    # HTML comments <!-- ... -->
    def repl_html_comment(m):
        ph = new_ph("HTMLC")
        placeholders[ph] = m.group(0)
        return ph

    text = re.sub(r'<!--[\s\S]*?-->', repl_html_comment, text)

    # Protect HTML tags <...> (including attributes)
    def repl_tag(m):
        ph = new_ph("TAG")
        placeholders[ph] = m.group(0)
        return ph

    text = re.sub(r'<[^>]+>', repl_tag, text)

    # Images ![alt](url)
    def repl_img(m):
        ph = new_ph("IMG")
        placeholders[ph] = m.group(0)
        return ph

    text = re.sub(r'!\[.*?\]\(.*?\)', repl_img, text)

    # Links [text](url) -> protect only the URL (keep visible text)
    def repl_link(m):
        ph = new_ph("LINKURL")
        placeholders[ph] = m.group(2)
        return f'[{m.group(1)}]({ph})'

    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', repl_link, text)

    # Plain URLs (autolinks like <http://...> already protected by TAG)
    def repl_url(m):
        ph = new_ph("URL")
        placeholders[ph] = m.group(0)
        return ph

    text = re.sub(r'https?://\S+', repl_url, text)

    return text, placeholders


def _restore_placeholders(text, placeholders):
    """
    Restore previously stored placeholders back to their original content.
    """
    for ph, orig in placeholders.items():
        text = text.replace(ph, orig)
    return text


def translate_paragraph_lines(lines, translator):
    """
    Translate a block of related lines (a paragraph or a group of list items).

    The function:
      - preserves each line's prefix (indentation, list marker, blockquote markers, checkboxes)
      - sends the core content of the block in a single request to preserve coherence
      - restores placeholders after translation

    Returns a list of translated lines, each ending with a newline.
    """
    prefixes = []
    cores = []

    # Support blockquote markers, list markers, ordered lists and task checkboxes.
    prefix_re = re.compile(r'^(\s*(?:>+\s*)?(?:\d+\.\s+|[-*+]\s+|\[[ xX]\]\s*)?)')

    for line in lines:
        m = prefix_re.match(line)
        prefix = m.group(1) if m else ""
        core = line[len(prefix):].rstrip("\n")
        prefixes.append(prefix)
        cores.append(core)

    core_text = "\n".join(cores)
    protected_text, placeholders = _protect_placeholders(core_text)
    translated_protected = safe_translate(translator, protected_text)
    translated_core = _restore_placeholders(translated_protected, placeholders)

    translated_lines = translated_core.split("\n")

    out_lines = []
    # Reattach prefixes; if count differs, attach remaining translated lines without prefix
    for i, tline in enumerate(translated_lines):
        prefix = prefixes[i] if i < len(prefixes) else ""
        out_lines.append(prefix + tline + "\n")

    return out_lines


def _is_fence_line(line):
    """
    Detect code fence lines for both backticks and tildes (e.g. ``` or ~~~).
    """
    return bool(re.match(r'^\s*(`{3,}|~{3,})', line))


def translate_file(filepath, translator):
    """
    Translate a Markdown file in-place by paragraphs, preserving fenced code blocks,
    file permissions and timestamps.
    """
    rel_path = filepath.relative_to(ROOT_DIR)

    with open(filepath, "r", encoding="utf-8", newline="") as fh:
        lines = fh.readlines()

    translated_lines = []
    in_code_block = False
    buffer = []

    def flush_buffer():
        nonlocal buffer, translated_lines
        if not buffer:
            return
        translated_lines.extend(translate_paragraph_lines(buffer, translator))
        buffer = []

    for line in lines:
        stripped = line.lstrip()
        if _is_fence_line(line):
            # Flush pending paragraph before entering/exiting a fenced block
            flush_buffer()
            in_code_block = not in_code_block
            translated_lines.append(line)
            continue

        if in_code_block:
            translated_lines.append(line)
            continue

        if line.strip() == "":
            # Blank line -> flush paragraph and keep blank
            flush_buffer()
            translated_lines.append(line)
            continue

        # Accumulate paragraph lines
        buffer.append(line)

    flush_buffer()

    # If nothing changed, skip writing
    if translated_lines == lines:
        print(f"[SKIP] No changes: {filepath}")
        return

    # Preserve permissions and timestamps if available
    try:
        st = filepath.stat()
    except Exception:
        st = None

    # Write back in-place
    with open(filepath, "w", encoding="utf-8", newline="") as fh:
        fh.writelines(translated_lines)

    if st:
        try:
            os.chmod(filepath, st.st_mode)
        except Exception:
            pass
        try:
            os.utime(filepath, (st.st_atime, st.st_mtime))
        except Exception:
            pass

    print(f"[SUCCESS] Rewrote in-place: {filepath}")


def translate_notebook(nb_path, translator):
    """
    Translate markdown cells in a Jupyter notebook (.ipynb) in-place.

    The translation logic mirrors translate_file: it processes paragraphs,
    preserves fenced code blocks inside markdown cells, and keeps outputs/metadata.
    """
    nb = nbformat.read(nb_path, as_version=4)
    changed = False
    for cell in nb.cells:
        if cell.cell_type != "markdown":
            continue
        src = cell.source.splitlines(keepends=True)
        translated = []
        in_code_block = False
        buffer = []

        def flush_buffer_cell():
            nonlocal buffer, translated, changed
            if not buffer:
                return
            out = translate_paragraph_lines(buffer, translator)
            if out != buffer:
                changed = True
            translated.extend(out)
            buffer = []

        for line in src:
            if _is_fence_line(line):
                flush_buffer_cell()
                in_code_block = not in_code_block
                translated.append(line)
                continue
            if in_code_block:
                translated.append(line)
                continue
            if line.strip() == "":
                flush_buffer_cell()
                translated.append(line)
                continue
            buffer.append(line)

        flush_buffer_cell()
        new_source = "".join(translated)
        if new_source != cell.source:
            cell.source = new_source
            changed = True

    if changed:
        nbformat.write(nb, nb_path)
        print(f"[SUCCESS] Rewrote notebook in-place: {nb_path}")
    else:
        print(f"[SKIP] No changes in notebook: {nb_path}")


def get_local_translator():
    """
    Initialize a translator that proxies requests to a local LibreTranslate instance.

    The returned object exposes a .translate(text) method compatible with
    deep_translator.LibreTranslator behavior.
    """
    print("[INFO] Connecting to local LibreTranslate (http://localhost:5000/translate)")

    class LocalTranslator:
        def __init__(self, target_lang=TARGET_LANG):
            self.url = "http://localhost:5000/translate"
            self.target = target_lang
            self.api_key = "dummy_key"

        def translate(self, text):
            if not text.strip():
                return text
            try:
                resp = requests.post(
                    self.url,
                    headers={"Content-Type": "application/json"},
                    json={
                        "q": text,
                        "source": "auto",
                        "target": self.target,
                        "format": "text",
                        "alternatives": 3,
                        "api_key": self.api_key,
                    },
                    timeout=15,
                )
                resp.raise_for_status()
                data = resp.json()
                # LibreTranslate returns translatedText; fall back to original text if missing
                return data.get("translatedText", text)
            except Exception as exc:
                print(f"[WARNING] Local LibreTranslate error: {exc}")
                return text

    translator = LocalTranslator(TARGET_LANG)

    # Connection smoke test
    test = translator.translate("test")
    if not test or test == "test":
        raise ConnectionError(f"Failed to connect to local LibreTranslate at {translator.url}")

    print("[INFO] LibreTranslate local instance is ready.")
    return translator


def translate_repo(root=ROOT_DIR):
    """
    Translates all Markdown (.md) and Jupyter notebooks (.ipynb) in the repository
    in-place using the local LibreTranslate instance.
    """
    translator = get_local_translator()

    md_files = [
        f for f in Path(root).rglob("*.md")
        if not any(part.startswith("translated_") or part == "venv" for part in f.parts)
    ]
    ipynb_files = [
        f for f in Path(root).rglob("*.ipynb")
        if not any(part.startswith("translated_") or part == "venv" for part in f.parts)
    ]

    print(f"[INFO] Markdown files discovered: {len(md_files)}")
    print(f"[INFO] Notebook files discovered: {len(ipynb_files)}")

    for file in md_files:
        translate_file(file, translator)

    for nb in ipynb_files:
        translate_notebook(nb, translator)

    print("\n[SUCCESS] Translation process completed. (In-place modification)")


if __name__ == "__main__":
    translate_repo()