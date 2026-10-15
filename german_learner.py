"""
German Learning Assistant — PyQt6 Desktop App

Requirements:
    pip install PyQt6 deep-translator google-genai python-dotenv gTTS
    (optional offline TTS fallback)  pip install pyttsx3
"""

import sys, re, sqlite3, json, csv, os, tempfile, hashlib, time
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QTextEdit, QTextBrowser, QPushButton, QLabel,
    QTableWidget, QTableWidgetItem, QHeaderView, QFrame,
    QLineEdit, QMessageBox, QScrollArea, QSizePolicy, QFileDialog
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize, QUrl
from PyQt6.QtGui import (
    QFont, QColor, QTextCursor, QTextCharFormat, QPalette, QCursor
)

from deep_translator import GoogleTranslator
# NOTE: google.genai is imported lazily inside LookupWorker._try_gemini, so the
# ~30 MB it costs is only paid when Gemini is actually used (never in Google-only mode).
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DB_PATH = Path.home() / ".german_learner_vocab.db"
MEMORY_PATH = Path.home() / ".german_learner_memory.db"   # local cache of Gemini answers
TTS_CACHE_DIR = Path(tempfile.gettempdir()) / "german_learner_tts"
TTS_CACHE_DIR.mkdir(exist_ok=True)

GEMINI_TIMEOUT_SECONDS_WORD = 15
GEMINI_TIMEOUT_SECONDS_PHRASE = 30

GEMINI_PROMPT = """You are an expert German language tutor.

Return ONLY a valid raw JSON object — no markdown, no backticks, no explanation.

Required fields (use null for fields that don't apply):
{
  "word": "",
  "lemma": "",
  "part_of_speech": "",
  "context_meaning": "",
  "other_meanings": [],
  "gender": "",
  "plural": "",
  "conjugation": {"ich":"","du":"","er":"","wir":"","ihr":"","sie":"","past":"","perfect":"","separable":false},
  "comparison": {"comparative":"","superlative":""},
  "cefr_level": "",
  "pronunciation": "",
  "word_family": [],
  "collocations": [],
  "common_phrases": [],
  "synonyms": [],
  "antonyms": [],
  "grammar_notes": "",
  "common_mistakes": [],
  "memory_tip": "",
  "example_de": "",
  "example_en": ""
}

Rules:
- context_meaning: meaning IN the given sentence context
- other_meanings: list of other common meanings (strings)
- conjugation: only for verbs, null otherwise
- comparison: only for adjectives, null otherwise
- gender: only for nouns (der/die/das), null otherwise
- cefr_level: A1/A2/B1/B2/C1/C2
- pronunciation: IPA notation
- word_family: list of related words with brief meaning e.g. ["lieben (to love)", "lieblich (lovely)"]
- collocations: list of common collocations e.g. ["aus Liebe (out of love)"]
- common_phrases: list of common phrases using this word
- common_mistakes: list of strings describing typical learner errors
- memory_tip: a short memorable tip to remember the word

Phrase / sentence mode:
- If the input has multiple words or is a full sentence, still fill the same JSON fields.
- Put the natural English translation in context_meaning.
- Set word to the exact phrase/sentence being analyzed.
- Use grammar_notes to explain the grammar like a teacher: word order, verb position, cases, articles, prepositions, tense, separable verbs, adjective endings, or any other useful structure.
- Keep grammar_notes concise, practical, and learner-friendly.
- If helpful, use example_de and example_en to echo the selected German sentence and its translation.
"""

# Canonical column order used by the DB, the full-row reader and the CSV export.
FULL_COLUMNS = [
    "word", "lemma", "part_of_speech", "gender", "plural",
    "context_meaning", "other_meanings", "cefr_level", "pronunciation",
    "conjugation", "comparison",
    "synonyms", "antonyms", "word_family", "collocations", "common_phrases",
    "grammar_notes", "common_mistakes", "memory_tip",
    "example_de", "example_en", "source", "added_at",
]

# Export order, tuned for reading on a phone: the first few columns carry
# everything you need to learn the word; reference material follows.
# (database column, CSV header)
EXPORT_SPEC = [
    ("gender",          "Gender"),
    ("word",            "Word"),
    ("context_meaning", "Meaning (in context)"),
    ("part_of_speech",  "Part of Speech"),
    ("plural",          "Plural"),
    ("memory_tip",      "Memory Tip"),
    ("example_de",      "Example (DE)"),
    ("example_en",      "Example (EN)"),
    ("cefr_level",      "CEFR"),
    ("pronunciation",   "Pronunciation (IPA)"),
    ("other_meanings",  "Other Meanings"),
    ("collocations",    "Collocations"),
    ("common_phrases",  "Common Phrases"),
    ("synonyms",        "Synonyms"),
    ("antonyms",        "Antonyms"),
    ("word_family",     "Word Family"),
    ("conjugation",     "Conjugation"),
    ("comparison",      "Comparison"),
    ("grammar_notes",   "Grammar Notes"),
    ("common_mistakes", "Common Mistakes"),
    ("lemma",           "Lemma"),
    ("source",          "Source"),
    ("added_at",        "Saved At"),
]
CSV_HEADERS = [h for _, h in EXPORT_SPEC]


# ─────────────────────────────────────────────
#  Serialisation helpers
# ─────────────────────────────────────────────
def to_db(value):
    """Store lists/dicts as JSON so nothing from Gemini is lost."""
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def to_csv_cell(raw):
    """Turn a stored value into something readable in a spreadsheet cell."""
    if raw in (None, ""):
        return ""
    text = str(raw)
    # Try to unpack JSON we stored earlier
    if text.startswith("[") or text.startswith("{"):
        try:
            parsed = json.loads(text)
        except Exception:
            return text
        if isinstance(parsed, list):
            return " | ".join(str(x) for x in parsed if x not in (None, ""))
        if isinstance(parsed, dict):
            parts = []
            for k, v in parsed.items():
                if v in (None, "", False) and v is not False:
                    continue
                if isinstance(v, bool):
                    v = "yes" if v else "no"
                parts.append(f"{k}: {v}")
            return " | ".join(parts)
        return str(parsed)
    return text


# ─────────────────────────────────────────────
#  Database
# ─────────────────────────────────────────────
class VocabDB:
    def __init__(self):
        self.conn = sqlite3.connect(str(DB_PATH))
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS vocab (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                word            TEXT UNIQUE,
                lemma           TEXT,
                part_of_speech  TEXT,
                gender          TEXT,
                plural          TEXT,
                context_meaning TEXT,
                other_meanings  TEXT,
                cefr_level      TEXT,
                pronunciation   TEXT,
                conjugation     TEXT,
                comparison      TEXT,
                synonyms        TEXT,
                antonyms        TEXT,
                word_family     TEXT,
                collocations    TEXT,
                common_phrases  TEXT,
                grammar_notes   TEXT,
                common_mistakes TEXT,
                memory_tip      TEXT,
                example_de      TEXT,
                example_en      TEXT,
                source          TEXT,
                added_at        DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()
        self._migrate()

    def _migrate(self):
        """Add any column missing from an older database file."""
        existing = {r[1] for r in self.conn.execute("PRAGMA table_info(vocab)")}
        for col in FULL_COLUMNS:
            if col == "added_at":
                continue
            if col not in existing:
                self.conn.execute(f"ALTER TABLE vocab ADD COLUMN {col} TEXT")
        self.conn.commit()

    def save(self, d):
        """Persist EVERY field Gemini returned, not just the displayed ones."""
        cols = [c for c in FULL_COLUMNS if c != "added_at"]
        placeholders = ",".join("?" for _ in cols)
        values = [to_db(d.get(c)) for c in cols]
        try:
            self.conn.execute(
                f"INSERT OR REPLACE INTO vocab ({','.join(cols)}) VALUES ({placeholders})",
                values,
            )
            self.conn.commit()
            return True
        except Exception as e:
            print("DB save error:", e)
            return False

    def all_words(self):
        """Light rows for the on-screen table."""
        cur = self.conn.execute(
            "SELECT id,word,part_of_speech,context_meaning,cefr_level,example_de,example_en "
            "FROM vocab ORDER BY added_at DESC"
        )
        return cur.fetchall()

    def all_full(self):
        """Every stored column, for CSV export."""
        cur = self.conn.execute(
            f"SELECT {','.join(FULL_COLUMNS)} FROM vocab ORDER BY added_at DESC"
        )
        return cur.fetchall()

    def delete(self, word_id):
        self.conn.execute("DELETE FROM vocab WHERE id=?", (word_id,))
        self.conn.commit()

    def clear_all(self):
        self.conn.execute("DELETE FROM vocab")
        self.conn.commit()

    def exists(self, word):
        cur = self.conn.execute("SELECT 1 FROM vocab WHERE LOWER(word)=LOWER(?)", (word,))
        return cur.fetchone() is not None


# ─────────────────────────────────────────────
#  Local memory (Gemini answers only)
# ─────────────────────────────────────────────
class LookupMemory:
    """Tiny local store of past Gemini answers (plain sqlite3, no new library).

    Lookup order in the app: this memory -> Gemini. Only results whose source
    is "Gemini" are ever stored; Google Translate output never is.
    A fresh connection is opened per call, so it is safe from any worker thread.
    """

    @staticmethod
    def _norm(s):
        return " ".join((s or "").split()).casefold()

    @classmethod
    def _key(cls, text, context, is_phrase):
        # A word's meaning depends on its sentence, so the sentence is part of the key.
        return f"{'P' if is_phrase else 'W'}|{cls._norm(text)}|{cls._norm(context)}"

    @staticmethod
    def _connect():
        conn = sqlite3.connect(str(MEMORY_PATH), timeout=5)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memory (
                key        TEXT PRIMARY KEY,
                payload    TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        return conn

    @classmethod
    def get(cls, text, context, is_phrase):
        try:
            conn = cls._connect()
            try:
                row = conn.execute(
                    "SELECT payload FROM memory WHERE key=?",
                    (cls._key(text, context, is_phrase),),
                ).fetchone()
            finally:
                conn.close()
            return json.loads(row[0]) if row else None
        except Exception as e:
            print("Memory read error:", e)
            return None

    @classmethod
    def put(cls, text, context, is_phrase, data):
        if data.get("source") != "Gemini":      # never store anything but Gemini output
            return False
        try:
            conn = cls._connect()
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO memory (key, payload) VALUES (?, ?)",
                    (cls._key(text, context, is_phrase), json.dumps(data, ensure_ascii=False)),
                )
                conn.commit()
            finally:
                conn.close()
            return True
        except Exception as e:
            print("Memory write error:", e)
            return False


# ─────────────────────────────────────────────
#  Text-to-speech worker
# ─────────────────────────────────────────────
class SpeakWorker(QThread):
    """Produces an mp3 of German text (gTTS), or speaks offline via pyttsx3."""
    audio_ready = pyqtSignal(str)   # path to mp3
    spoken      = pyqtSignal()      # offline engine already finished speaking
    error       = pyqtSignal(str)

    def __init__(self, text):
        super().__init__()
        self.text = text

    def run(self):
        # ── Preferred: gTTS (accurate German voice) ─────────────────────
        try:
            from gtts import gTTS
            key = hashlib.md5(self.text.encode("utf-8")).hexdigest()
            path = TTS_CACHE_DIR / f"{key}.mp3"
            if not path.exists():
                gTTS(text=self.text, lang="de", slow=False).save(str(path))
            self.audio_ready.emit(str(path))
            return
        except Exception as e:
            gtts_err = str(e)

        # ── Fallback: pyttsx3 (offline, uses system German voice) ───────
        try:
            import pyttsx3
            engine = pyttsx3.init()
            for v in engine.getProperty("voices"):
                blob = f"{getattr(v,'id','')} {getattr(v,'name','')} {getattr(v,'languages','')}".lower()
                if "de" in blob or "german" in blob or "deutsch" in blob:
                    engine.setProperty("voice", v.id)
                    break
            engine.setProperty("rate", 150)
            engine.say(self.text)
            engine.runAndWait()
            self.spoken.emit()
            return
        except Exception as e2:
            self.error.emit(f"Speech failed (gTTS: {gtts_err}; offline: {e2})")


# ─────────────────────────────────────────────
#  Lookup worker
# ─────────────────────────────────────────────
class _Cancelled(Exception):
    """Raised inside a worker when a newer click superseded it."""


class LookupWorker(QThread):
    result_ready = pyqtSignal(dict)
    error        = pyqtSignal(str)

    def __init__(self, text, context_sentence, is_phrase, google_only=False):
        super().__init__()
        self.text             = text
        self.context_sentence = context_sentence
        self.is_phrase        = is_phrase
        self.google_only      = google_only
        self._gemini_error    = ""

    def run(self):
        try:
            if self.google_only:
                self._google_translate_only()
            else:
                self._lookup_memory_then_gemini()
        except _Cancelled:
            pass                      # superseded by a newer click: stay silent
        except Exception as e:
            self.error.emit(f"Unexpected error: {e}")

    def _google_translate_only(self):
        try:
            translator = GoogleTranslator(source="de", target="en")
            translation = translator.translate(self.text)
            payload = {
                "word": self.text,
                "context_meaning": translation,
                "source": "Google Translate",
            }
            if self.is_phrase:
                payload["is_phrase"]  = True
                payload["example_de"] = self.text
                payload["example_en"] = translation
            else:
                payload["is_phrase"]      = False
                payload["part_of_speech"] = "translation"
                payload["example_de"]     = self.context_sentence or ""
                payload["example_en"]     = (
                    translator.translate(self.context_sentence)
                    if self.context_sentence else ""
                )
            self.result_ready.emit(payload)
        except Exception as e:
            self.error.emit(f"Translation failed: {e}")

    def _lookup_memory_then_gemini(self):
        """1) local memory  ->  2) Gemini  ->  3) on failure: show error only.
        Google Translate is NEVER used here; that is the "Google Only" button's job."""
        hint = "  Try the 'Google Only' button for a simple translation."

        cached = LookupMemory.get(self.text, self.context_sentence, self.is_phrase)
        if cached:
            cached["from_memory"] = True
            self.result_ready.emit(cached)
            return

        if not GEMINI_API_KEY:
            self.error.emit("No GEMINI_API_KEY found in .env." + hint)
            return

        result = self._try_gemini(is_phrase=self.is_phrase)
        if result:
            LookupMemory.put(self.text, self.context_sentence, self.is_phrase, result)
            self.result_ready.emit(result)
        else:
            reason = self._gemini_error or "Gemini did not respond."
            self.error.emit(f"{reason}{hint}")

    def _build_gemini_prompt(self, is_phrase=False):
        ctx = ""
        if is_phrase:
            if self.context_sentence and self.context_sentence.strip() != self.text.strip():
                ctx = (
                    f'\nThe selected text appears in this sentence: "{self.context_sentence}"\n'
                    f'Use that context for context_meaning and grammar_notes.\n'
                )
            ctx += (
                "\nThe selected text is a phrase or full sentence.\n"
                "Explain the grammar like a teacher for a German learner.\n"
                "Focus on sentence structure, verb position, cases, articles, prepositions, tense,\n"
                "and any special points that help the learner understand why it is built that way.\n"
            )
            return GEMINI_PROMPT + ctx + f"\nGerman phrase/sentence: {self.text}"

        if self.context_sentence and self.context_sentence.strip() != self.text.strip():
            ctx = (
                f'\nContext sentence (for meaning only, do not copy into the word field): "{self.context_sentence}"\n'
                f'IMPORTANT: Analyze ONLY the exact selected word below. Do not expand it into a phrase or sentence.\n'
                f'The word field must be exactly the selected word, with no surrounding text.\n'
            )
        return GEMINI_PROMPT + ctx + f"\nExact selected word: {self.text}"

    def _try_gemini(self, is_phrase=False):
        """Ask Gemini. Returns a dict, or None on timeout/any API failure.

        The wait is polled in 0.2 s slices so that (a) the timeout really is a
        timeout and (b) a newer click can cancel this lookup immediately.
        """
        import concurrent.futures
        timeout = GEMINI_TIMEOUT_SECONDS_PHRASE if is_phrase else GEMINI_TIMEOUT_SECONDS_WORD

        def _call():
            from google import genai            # lazy import
            from google.genai import types
            client = genai.Client(
                api_key=GEMINI_API_KEY,
                # also cap the HTTP call itself (milliseconds) so no thread lingers
                http_options=types.HttpOptions(timeout=int(timeout * 1000)),
            )
            prompt = self._build_gemini_prompt(is_phrase=is_phrase)
            response = client.models.generate_content(
                model="gemma-4-26b-a4b-it",
                contents=prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(
                        thinking_level="minimal", include_thoughts=False
                    )
                )
            )
            parts = [p.text for p in response.candidates[0].content.parts
                     if not getattr(p, "thought", False) and getattr(p, "text", None)]
            return "\n".join(parts).strip()

        # NOT a `with` block: its exit waits for the call to finish, which silently
        # defeats the timeout.
        ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            future = ex.submit(_call)
            deadline = time.monotonic() + timeout
            while not future.done():
                concurrent.futures.wait([future], timeout=0.2)
                if self.isInterruptionRequested():
                    raise _Cancelled()
                if time.monotonic() >= deadline:
                    print(f"Gemini timed out after {timeout}s")
                    self._gemini_error = f"Gemini timed out after {timeout}s."
                    return None
            raw = future.result()

            raw = re.sub(r"^```[a-z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if not m:
                self._gemini_error = "Gemini returned an unreadable answer."
                return None
            data = json.loads(m.group(0))
            data["is_phrase"] = is_phrase
            data["source"]    = "Gemini"
            if not is_phrase:
                data["word"] = self.text   # never let the model widen the selection
            return data
        except _Cancelled:
            raise
        except Exception as e:
            print("Gemini error:", e)
            self._gemini_error = f"Gemini error: {e}"
            return None
        finally:
            ex.shutdown(wait=False, cancel_futures=True)


# ─────────────────────────────────────────────
#  Display helpers
# ─────────────────────────────────────────────
SECTION_STYLE = """
    QFrame#section {
        border: 1px solid #E0E0EC;
        border-radius: 6px;
        background: #FAFAFA;
    }
"""

def make_section(title, content_widget, accent="#7F77DD"):
    frame = QFrame()
    frame.setObjectName("section")
    frame.setStyleSheet(SECTION_STYLE)
    vlay = QVBoxLayout(frame)
    vlay.setContentsMargins(0, 0, 0, 0)
    vlay.setSpacing(0)
    title_bar = QLabel(f"  {title}")
    title_bar.setStyleSheet(
        f"background: {accent}22; color: {accent}; font-size: 10px; font-weight: bold; "
        f"padding: 3px 6px; border-radius: 5px 5px 0 0;"
    )
    vlay.addWidget(title_bar)
    vlay.addWidget(content_widget)
    return frame

def pill_label(text, bg="#7F77DD", fg="#FFFFFF"):
    lbl = QLabel(text)
    lbl.setStyleSheet(
        f"background: {bg}; color: {fg}; border-radius: 8px; "
        f"padding: 2px 10px; font-size: 11px;"
    )
    lbl.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    return lbl

def body_label(text, color="#1A1A1A", size=11, bold=False):
    lbl = QLabel(text)
    lbl.setWordWrap(True)
    w = QFont.Weight.Bold if bold else QFont.Weight.Normal
    lbl.setFont(QFont("", size, w))
    lbl.setStyleSheet(f"color: {color}; padding: 2px 8px;")
    return lbl

def list_widget(items, bullet="•", color="#333333"):
    w = QWidget()
    lay = QVBoxLayout(w)
    lay.setContentsMargins(8, 4, 8, 4)
    lay.setSpacing(2)
    for item in items:
        lbl = QLabel(f"{bullet} {item}")
        lbl.setWordWrap(True)
        lbl.setStyleSheet(f"color: {color}; font-size: 11px;")
        lay.addWidget(lbl)
    return w


# ─────────────────────────────────────────────
#  Lookup Panel
# ─────────────────────────────────────────────
class LookupPanel(QFrame):
    save_requested  = pyqtSignal(dict)
    speak_requested = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.current_data = None
        self._setup_ui()

    def _setup_ui(self):
        self.setFrameShape(QFrame.Shape.StyledPanel)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        top = QFrame()
        top.setStyleSheet("background: #F0F0F8; border-bottom: 1px solid #E0E0EC;")
        top_lay = QHBoxLayout(top)
        top_lay.setContentsMargins(10, 8, 10, 8)
        lbl = QLabel("Word Lookup")
        lbl.setFont(QFont("", 10, QFont.Weight.Bold))
        lbl.setStyleSheet("color: #1A1A1A; background: transparent;")
        top_lay.addWidget(lbl)
        top_lay.addStretch()
        self.speak_btn = QPushButton("🔊")
        self.speak_btn.setFixedSize(QSize(28, 22))
        self.speak_btn.setToolTip("Hear this word / phrase in German")
        self.speak_btn.setVisible(False)
        self.speak_btn.clicked.connect(self._on_speak)
        top_lay.addWidget(self.speak_btn)
        self.source_badge = QLabel("")
        self.source_badge.setVisible(False)
        top_lay.addWidget(self.source_badge)
        outer.addWidget(top)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.content = QWidget()
        self.content_lay = QVBoxLayout(self.content)
        self.content_lay.setContentsMargins(8, 8, 8, 8)
        self.content_lay.setSpacing(6)
        self.content_lay.addStretch()
        scroll.setWidget(self.content)
        outer.addWidget(scroll, 1)

        bottom = QFrame()
        bottom.setStyleSheet("border-top: 1px solid #E0E0EC;")
        bot_lay = QHBoxLayout(bottom)
        bot_lay.setContentsMargins(10, 6, 10, 6)
        self.save_btn = QPushButton("+ Save to Vocabulary")
        self.save_btn.setEnabled(False)
        self.save_btn.setFixedHeight(30)
        self.save_btn.clicked.connect(self._on_save)
        bot_lay.addWidget(self.save_btn)
        outer.addWidget(bottom)

        self._set_placeholder()

    def _clear_content(self):
        while self.content_lay.count():
            item = self.content_lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _set_placeholder(self):
        self._clear_content()
        ph = QLabel("Click any word in the text\nto see a full breakdown")
        ph.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ph.setStyleSheet("color: #AAAAAA; font-size: 12px;")
        self.content_lay.addStretch()
        self.content_lay.addWidget(ph)
        self.content_lay.addStretch()

    def show_loading(self, text, is_phrase=False):
        self.current_data = None
        self._clear_content()
        self.source_badge.setVisible(False)
        self.speak_btn.setVisible(False)
        self.save_btn.setEnabled(False)
        self.save_btn.setText("+ Save to Vocabulary")
        verb = "Translating" if is_phrase else "Looking up"
        ph = QLabel(f'{verb}  "{text}"…')
        ph.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ph.setStyleSheet("color: #888888; font-size: 12px;")
        self.content_lay.addStretch()
        self.content_lay.addWidget(ph)
        self.content_lay.addStretch()

    def show_result(self, data, already_saved):
        self.current_data = data
        self._clear_content()
        is_phrase = data.get("is_phrase", False)
        source    = data.get("source", "")

        if source:
            color = "#5B8DD9" if source == "Google Translate" else "#7F77DD"
            self.source_badge.setStyleSheet(
                f"color:#FFF;background:{color};border-radius:8px;padding:2px 10px;font-size:10px;"
            )
            self.source_badge.setText(
                f"via {source} (memory)" if data.get("from_memory") else f"via {source}"
            )
            self.source_badge.setVisible(True)

        self.speak_btn.setVisible(bool(data.get("word")))

        if is_phrase:
            self._render_phrase(data)
        else:
            self._render_word(data)

        self.content_lay.addStretch()

        if already_saved:
            self.save_btn.setText("✓ Already saved")
            self.save_btn.setEnabled(False)
        else:
            self.save_btn.setText("+ Save phrase" if is_phrase else "+ Save to Vocabulary")
            self.save_btn.setEnabled(True)

    def _render_phrase(self, d):
        lay    = self.content_lay
        word   = d.get("word", "")
        trans  = d.get("context_meaning", "")
        gnotes = d.get("grammar_notes", "")
        ex_de  = d.get("example_de", "")
        ex_en  = d.get("example_en", "")

        title = QLabel(word[:60] + ("…" if len(word) > 60 else ""))
        title.setFont(QFont("", 14, QFont.Weight.Bold))
        title.setWordWrap(True)
        title.setStyleSheet("color:#1A1A1A;")
        lay.addWidget(title)

        sub = QLabel("Phrase / sentence translation")
        sub.setStyleSheet("color:#888;font-size:10px;")
        lay.addWidget(sub)

        box = QWidget(); bl = QVBoxLayout(box)
        bl.setContentsMargins(8, 6, 8, 6)
        bl.addWidget(body_label(trans, color="#1A1A1A", size=12))
        lay.addWidget(make_section("Translation", box, "#5B8DD9"))

        if gnotes:
            box = QWidget(); bl = QVBoxLayout(box)
            bl.setContentsMargins(8, 6, 8, 6)
            bl.addWidget(body_label(gnotes, color="#333", size=11))
            lay.addWidget(make_section("Grammar breakdown", box, "#C0652B"))

        if ex_de or ex_en:
            box = QWidget(); bl = QVBoxLayout(box)
            bl.setContentsMargins(8, 6, 8, 6); bl.setSpacing(2)
            if ex_de: bl.addWidget(body_label(ex_de, color="#1A1A1A", size=11, bold=True))
            if ex_en: bl.addWidget(body_label(ex_en, color="#555555", size=10))
            lay.addWidget(make_section("Selected sentence", box, "#3A9A5C"))

    def _render_word(self, d):
        lay    = self.content_lay
        word   = d.get("word", "")
        lemma  = d.get("lemma", "")
        pos    = d.get("part_of_speech", "")
        cefr   = d.get("cefr_level", "")
        pron   = d.get("pronunciation", "")
        gender = d.get("gender", "")
        plural = d.get("plural", "")

        title = QLabel(word)
        title.setFont(QFont("", 17, QFont.Weight.Bold))
        title.setStyleSheet("color:#1A1A1A;")
        lay.addWidget(title)

        if lemma and lemma.lower() != word.lower():
            sub = QLabel(f"Base form: {lemma}")
            sub.setStyleSheet("color:#888;font-size:10px;")
            lay.addWidget(sub)

        pills = QHBoxLayout()
        pills.setSpacing(4); pills.setContentsMargins(0, 0, 0, 0)
        if pos:    pills.addWidget(pill_label(pos,    "#555577"))
        if cefr:   pills.addWidget(pill_label(cefr,   "#3A9A5C"))
        if gender: pills.addWidget(pill_label(gender, "#C0652B"))
        if pron:   pills.addWidget(pill_label(pron,   "#888888"))
        pills.addStretch()
        pw = QWidget(); pw.setLayout(pills)
        lay.addWidget(pw)

        if plural:
            pl = QLabel(f"Plural: {plural}")
            pl.setStyleSheet("color:#555;font-size:11px;")
            lay.addWidget(pl)

        ctx = d.get("context_meaning", "")
        if ctx:
            box = QWidget(); bl = QVBoxLayout(box)
            bl.setContentsMargins(8, 6, 8, 6)
            bl.addWidget(body_label(ctx, color="#1A1A1A", size=13, bold=True))
            lay.addWidget(make_section("Meaning in context", box, "#3A9A5C"))

        others = d.get("other_meanings", []) or []
        if isinstance(others, list) and others:
            lay.addWidget(make_section("Other meanings", list_widget(others), "#7F77DD"))

        ex_de = d.get("example_de", "")
        ex_en = d.get("example_en", "")
        if ex_de:
            box = QWidget(); bl = QVBoxLayout(box)
            bl.setContentsMargins(8, 6, 8, 6); bl.setSpacing(2)
            bl.addWidget(body_label(ex_de, color="#1A1A1A", size=11, bold=True))
            if ex_en: bl.addWidget(body_label(ex_en, color="#555555", size=10))
            lay.addWidget(make_section("Example / Context sentence", box, "#5B8DD9"))

        conj   = d.get("conjugation") or {}
        comp   = d.get("comparison") or {}
        gnotes = d.get("grammar_notes", "")

        if isinstance(conj, dict) and any(conj.get(k) for k in ["ich", "du", "er"]):
            box = QWidget(); bl = QVBoxLayout(box)
            bl.setContentsMargins(8, 6, 8, 6); bl.setSpacing(2)
            pairs = [("ich", conj.get("ich", "")), ("du", conj.get("du", "")),
                     ("er/sie/es", conj.get("er", "")), ("wir", conj.get("wir", "")),
                     ("ihr", conj.get("ihr", "")), ("sie", conj.get("sie", ""))]
            for pronoun, form in pairs:
                if form:
                    row = QHBoxLayout(); row.setContentsMargins(0, 0, 0, 0)
                    pl = QLabel(pronoun); pl.setFixedWidth(70)
                    pl.setStyleSheet("color:#888;font-size:11px;")
                    fl = QLabel(form)
                    fl.setStyleSheet("color:#1A1A1A;font-size:11px;font-weight:bold;")
                    row.addWidget(pl); row.addWidget(fl); row.addStretch()
                    rw = QWidget(); rw.setLayout(row)
                    bl.addWidget(rw)
            for k, label in [("past", "Past (Präteritum)"), ("perfect", "Perfect (Perfekt)")]:
                if conj.get(k):
                    bl.addWidget(body_label(f"{label}: {conj[k]}", color="#555", size=10))
            sep = conj.get("separable")
            if sep is not None:
                bl.addWidget(body_label(f"Separable: {'Yes' if sep else 'No'}", color="#888", size=10))
            lay.addWidget(make_section("Conjugation", box, "#C0652B"))

        if isinstance(comp, dict) and any(comp.values()):
            box = QWidget(); bl = QVBoxLayout(box)
            bl.setContentsMargins(8, 6, 8, 6); bl.setSpacing(2)
            if comp.get("comparative"):
                bl.addWidget(body_label(f"Comparative: {comp['comparative']}", size=11))
            if comp.get("superlative"):
                bl.addWidget(body_label(f"Superlative: {comp['superlative']}", size=11))
            lay.addWidget(make_section("Comparison", box, "#C0652B"))

        if gnotes:
            box = QWidget(); bl = QVBoxLayout(box)
            bl.setContentsMargins(8, 6, 8, 6)
            bl.addWidget(body_label(gnotes, color="#333", size=11))
            lay.addWidget(make_section("Grammar notes", box, "#C0652B"))

        syns = d.get("synonyms", []) or []
        ants = d.get("antonyms", []) or []
        if syns or ants:
            box = QWidget(); bl = QVBoxLayout(box)
            bl.setContentsMargins(8, 4, 8, 4); bl.setSpacing(2)
            if syns: bl.addWidget(body_label("Synonyms: " + ", ".join(syns), size=11))
            if ants: bl.addWidget(body_label("Antonyms: " + ", ".join(ants), size=11, color="#C0652B"))
            lay.addWidget(make_section("Synonyms & Antonyms", box, "#888888"))

        wf = d.get("word_family", []) or []
        if wf:
            lay.addWidget(make_section("Word family", list_widget(wf, "🔗", "#333"), "#3A9A5C"))

        coll = d.get("collocations", []) or []
        if coll:
            lay.addWidget(make_section("Common collocations", list_widget(coll, "❤️", "#333"), "#E05588"))

        phrases = d.get("common_phrases", []) or []
        if phrases:
            lay.addWidget(make_section("Common phrases", list_widget(phrases, "💬", "#333"), "#5B8DD9"))

        mistakes = d.get("common_mistakes", []) or []
        if mistakes:
            lay.addWidget(make_section("Common mistakes", list_widget(mistakes, "⚠️", "#C0652B"), "#C0652B"))

        tip = d.get("memory_tip", "")
        if tip:
            box = QWidget(); bl = QVBoxLayout(box)
            bl.setContentsMargins(8, 6, 8, 6)
            bl.addWidget(body_label(f"💡 {tip}", color="#5B4A00", size=11))
            box.setStyleSheet("background:#FFFBEA;")
            lay.addWidget(make_section("Memory tip", box, "#D4A000"))

    def show_error(self, msg):
        self._clear_content()
        self.source_badge.setVisible(False)
        self.speak_btn.setVisible(False)
        self.save_btn.setEnabled(False)
        err = QLabel(f"Error: {msg}")
        err.setStyleSheet("color:#CC3333;padding:10px;")
        err.setWordWrap(True)
        self.content_lay.addWidget(err)
        self.content_lay.addStretch()

    def mark_saved(self):
        self.save_btn.setText("✓ Saved!")
        self.save_btn.setEnabled(False)

    def _on_save(self):
        if self.current_data:
            self.save_requested.emit(self.current_data)

    def _on_speak(self):
        if self.current_data and self.current_data.get("word"):
            self.speak_requested.emit(self.current_data["word"])


# ─────────────────────────────────────────────
#  Clickable text browser
# ─────────────────────────────────────────────
class GermanTextBrowser(QTextBrowser):
    text_selected = pyqtSignal(str, str, bool)

    def __init__(self):
        super().__init__()
        self.setReadOnly(True)
        self.setOpenLinks(False)
        self.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse |
            Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        self.viewport().setCursor(QCursor(Qt.CursorShape.IBeamCursor))
        self.setStyleSheet("QTextBrowser{color:#1A1A1A;background:#FFFFFF;}")
        self._press_pos = None
        self._last_selection = ""

    def mousePressEvent(self, e):
        self._press_pos = e.position().toPoint()
        super().mousePressEvent(e)

    def mouseReleaseEvent(self, e):
        super().mouseReleaseEvent(e)
        if e.button() != Qt.MouseButton.LeftButton:
            return

        release_pos    = e.position().toPoint()
        release_cursor = self.textCursor()
        click_cursor   = self.cursorForPosition(e.pos())

        selected  = None
        is_phrase = False

        dragged = False
        if self._press_pos is not None:
            dragged = (abs(self._press_pos.x() - release_pos.x()) +
                       abs(self._press_pos.y() - release_pos.y())) >= 10

        if dragged and release_cursor.hasSelection():
            selected = self._normalize_selected_text(release_cursor.selectedText())
            if selected:
                is_phrase = bool(re.search(r"\s", selected))

        if selected:
            self._last_selection = selected
            highlight_cursor = QTextCursor(release_cursor)
            self._apply_highlight(highlight_cursor)
            context = selected if is_phrase else (self._extract_sentence(click_cursor) or selected)
            self.text_selected.emit(selected, context, is_phrase)
            self._press_pos = None
            return

        wc = click_cursor
        wc.select(QTextCursor.SelectionType.WordUnderCursor)
        word = re.sub(r"[^\w\-äöüÄÖÜß]", "", wc.selectedText().strip(), flags=re.UNICODE)
        if word and re.search(r"[a-zA-ZäöüÄÖÜß]", word):
            self._last_selection = word
            context = self._extract_sentence(wc)
            self._apply_highlight(wc)
            self.text_selected.emit(word, context, False)

        self._press_pos = None

    def current_speech_text(self):
        """What the read-aloud button should say: live selection, last pick, or all."""
        cursor = self.textCursor()
        if cursor.hasSelection():
            sel = self._normalize_selected_text(cursor.selectedText())
            if sel:
                return sel
        if self._last_selection:
            return self._last_selection
        return self.toPlainText().strip()

    def _normalize_selected_text(self, text):
        if not text:
            return ""
        text = text.replace(" ", " ").replace(" ", " ")
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _extract_sentence(self, cursor):
        block_text = cursor.block().text().strip()
        if not block_text:
            return ""
        pos = cursor.positionInBlock()
        sentences = re.split(r"(?<=[.!?])\s+", block_text)
        count = 0
        for s in sentences:
            count += len(s) + 1
            if count >= pos:
                return s.strip()
        return block_text

    def _apply_highlight(self, cursor):
        full = QTextCursor(self.document())
        full.select(QTextCursor.SelectionType.Document)
        full.setCharFormat(QTextCharFormat())
        fmt = QTextCharFormat()
        fmt.setBackground(QColor("#BFD7F5"))
        fmt.setForeground(QColor("#0A3060"))
        cursor.setCharFormat(fmt)

    def load_german_text(self, text):
        self.setPlainText(text)
        self._last_selection = ""


# ─────────────────────────────────────────────
#  Vocabulary Panel
# ─────────────────────────────────────────────
class VocabPanel(QFrame):
    def __init__(self, db: VocabDB):
        super().__init__()
        self.db = db
        self._all_rows = []
        self._setup_ui()
        self.refresh()

    def _setup_ui(self):
        self.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        hdr = QHBoxLayout()
        lbl = QLabel("Saved Vocabulary")
        lbl.setFont(QFont("", 10, QFont.Weight.Bold))
        lbl.setStyleSheet("color:#1A1A1A;")
        hdr.addWidget(lbl)
        hdr.addStretch()
        self.count_label = QLabel("0 words")
        self.count_label.setStyleSheet("color:#666;font-size:11px;")
        hdr.addWidget(self.count_label)
        layout.addLayout(hdr)

        self.search = QLineEdit()
        self.search.setPlaceholderText("Filter…")
        self.search.setFixedHeight(28)
        self.search.setStyleSheet("color:#1A1A1A;background:#FFFFFF;")
        self.search.textChanged.connect(self.filter_words)
        layout.addWidget(self.search)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Word", "POS", "Meaning", ""])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(3, 28)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(False)
        self.table.setFont(QFont("", 11))
        self.table.setStyleSheet("""
            QTableWidget { color:#1A1A1A; background:#FFFFFF; }
            QTableWidget::item { color:#1A1A1A; }
            QTableWidget::item:alternate { background:#F3F3F8; color:#1A1A1A; }
            QTableWidget::item:selected  { background:#BFD7F5; color:#0A3060; }
            QHeaderView::section { color:#1A1A1A; background:#ECECEC; }
        """)
        layout.addWidget(self.table)

        btn_row = QHBoxLayout()
        export_btn = QPushButton("Export CSV")
        export_btn.setFixedHeight(28)
        export_btn.setToolTip("Export every stored field, UTF-8 with BOM so umlauts survive Excel")
        export_btn.clicked.connect(self.export_csv)
        btn_row.addWidget(export_btn)

        self.clear_btn = QPushButton("Clear Vocab")
        self.clear_btn.setFixedHeight(28)
        self.clear_btn.setStyleSheet("color:#FFFFFF;background:#C65A5A;")
        self.clear_btn.clicked.connect(self.clear_vocab)
        btn_row.addWidget(self.clear_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

    def refresh(self):
        self._all_rows = self.db.all_words()
        self._render(self._all_rows)

    def _render(self, rows):
        self.table.setRowCount(0)
        for row_id, word, pos, meaning, cefr, ex_de, ex_en in rows:
            r = self.table.rowCount()
            self.table.insertRow(r)

            w_item = QTableWidgetItem(word)
            w_item.setFont(QFont("", 11, QFont.Weight.Bold))
            w_item.setForeground(QColor("#1A1A1A"))
            w_item.setData(Qt.ItemDataRole.UserRole, row_id)
            self.table.setItem(r, 0, w_item)

            p_item = QTableWidgetItem(pos or "")
            p_item.setForeground(QColor("#666666"))
            p_item.setFont(QFont("", 10))
            self.table.setItem(r, 1, p_item)

            short = (meaning or "").split(".")[0]
            if short:
                short += "."
            m_item = QTableWidgetItem(short)
            m_item.setForeground(QColor("#333333"))
            m_item.setToolTip(f"{meaning}\n\n{ex_de}\n{ex_en}")
            self.table.setItem(r, 2, m_item)

            del_btn = QPushButton("×")
            del_btn.setFixedSize(QSize(24, 24))
            del_btn.setStyleSheet("color:#cc3333;font-weight:bold;border:none;background:transparent;")
            del_btn.clicked.connect(lambda _, rid=row_id: self._delete(rid))
            self.table.setCellWidget(r, 3, del_btn)

        self.count_label.setText(f"{len(rows)} word{'s' if len(rows) != 1 else ''}")

    def filter_words(self, text):
        t = text.lower()
        filtered = [r for r in self._all_rows
                    if t in (r[1] or "").lower() or t in (r[3] or "").lower()]
        self._render(filtered)

    def _delete(self, row_id):
        self.db.delete(row_id)
        self.refresh()

    def clear_vocab(self):
        reply = QMessageBox.question(
            self, "Clear Vocabulary", "Delete all saved vocabulary entries?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self.db.clear_all()
        self.refresh()
        QMessageBox.information(self, "Cleared", "Saved vocabulary has been cleared.")

    def export_csv(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Vocabulary", "vocab.csv", "CSV files (*.csv)"
        )
        if not path:
            return
        try:
            # utf-8-sig writes a BOM: Excel then reads ä ö ü ß correctly.
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(CSV_HEADERS)
                idx = {c: i for i, c in enumerate(FULL_COLUMNS)}
                for row in self.db.all_full():
                    w.writerow([to_csv_cell(row[idx[col]]) for col, _ in EXPORT_SPEC])
            QMessageBox.information(
                self, "Exported",
                f"Saved to {path}\n\nAll {len(CSV_HEADERS)} fields exported, UTF-8 with BOM."
            )
        except Exception as e:
            QMessageBox.warning(self, "Export failed", str(e))


# ─────────────────────────────────────────────
#  Main Window
# ─────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.db            = VocabDB()
        self._workers      = set()     # running lookups (old ones are cancelled, not dropped)
        self._req_id       = 0         # newest request wins; older answers are ignored
        self._last_request = None      # (text, context, is_phrase) of the latest click
        self.speak_worker  = None
        self.player        = None
        self.audio_out     = None
        self.setWindowTitle("German Learning Assistant")
        self.resize(1200, 720)
        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 8, 10, 0)
        root.setSpacing(4)

        main_split = QSplitter(Qt.Orientation.Horizontal)
        left_split = QSplitter(Qt.Orientation.Vertical)

        input_frame = QFrame()
        input_frame.setFrameShape(QFrame.Shape.StyledPanel)
        in_layout = QVBoxLayout(input_frame)
        in_layout.setContentsMargins(10, 10, 10, 8)
        in_layout.setSpacing(6)

        hdr = QHBoxLayout()
        lbl = QLabel("German Text")
        lbl.setFont(QFont("", 10, QFont.Weight.Bold))
        lbl.setStyleSheet("color:#1A1A1A;")
        hdr.addWidget(lbl)
        hdr.addStretch()

        self.speak_text_btn = QPushButton("🔊 Read Aloud")
        self.speak_text_btn.setFixedHeight(26)
        self.speak_text_btn.setToolTip(
            "Speak the selected German text (or the whole text if nothing is selected)"
        )
        self.speak_text_btn.setVisible(False)
        self.speak_text_btn.clicked.connect(self._on_read_aloud)
        hdr.addWidget(self.speak_text_btn)

        self.load_btn = QPushButton("Load →")
        self.load_btn.setFixedHeight(26)
        self.load_btn.clicked.connect(self._switch_to_reader)
        hdr.addWidget(self.load_btn)

        self.edit_btn = QPushButton("Edit text")
        self.edit_btn.setFixedHeight(26)
        self.edit_btn.setVisible(False)
        self.edit_btn.clicked.connect(self._switch_to_editor)
        hdr.addWidget(self.edit_btn)

        self.google_only_btn = QPushButton("Google Only")
        self.google_only_btn.setCheckable(True)
        self.google_only_btn.setFixedHeight(26)
        self.google_only_btn.setMinimumWidth(104)
        self.google_only_btn.setStyleSheet(
            "QPushButton:checked { background:#5B8DD9; color:#FFFFFF; font-weight:bold; "
            "border:1px solid #3F6FB8; border-radius:3px; }"
        )
        self.google_only_btn.setToolTip("Skip memory and Gemini; use basic Google Translate (needs internet). Results are not saved to memory.")
        self.google_only_btn.toggled.connect(self._on_google_only_toggled)
        hdr.addWidget(self.google_only_btn)
        in_layout.addLayout(hdr)

        self.editor = QTextEdit()
        self.editor.setPlaceholderText(
            "Paste your German text here, then click 'Load →'\n\n"
            "Example:\nHallo meine liebe Kleine. Das Buch liegt auf dem Tisch."
        )
        self.editor.setFont(QFont("", 13))
        self.editor.setStyleSheet("QTextEdit{color:#1A1A1A;background:#FFFFFF;}")
        in_layout.addWidget(self.editor)

        self.reader = GermanTextBrowser()
        self.reader.setFont(QFont("", 13))
        self.reader.setVisible(False)
        self.reader.text_selected.connect(self._on_text_selected)
        in_layout.addWidget(self.reader)

        left_split.addWidget(input_frame)

        self.lookup = LookupPanel()
        self.lookup.save_requested.connect(self._on_save)
        self.lookup.speak_requested.connect(self._speak)
        left_split.addWidget(self.lookup)
        left_split.setSizes([300, 380])

        main_split.addWidget(left_split)
        self.vocab_panel = VocabPanel(self.db)
        main_split.addWidget(self.vocab_panel)
        main_split.setSizes([750, 420])
        root.addWidget(main_split, 1)

        self.status = QLabel("Ready — paste German text and click 'Load →'")
        self.status.setFixedHeight(20)
        self.status.setStyleSheet(
            "font-size:11px; color:#555; padding:0 4px; "
            "border-top: 1px solid #E0E0EC; background: #F7F7FA;"
        )
        root.addWidget(self.status, 0)

    # ── view switching ───────────────────────────────────────────────
    def _switch_to_reader(self):
        text = self.editor.toPlainText().strip()
        if not text:
            return
        self.reader.load_german_text(text)
        self.editor.setVisible(False)
        self.reader.setVisible(True)
        self.load_btn.setVisible(False)
        self.edit_btn.setVisible(True)
        self.speak_text_btn.setVisible(True)
        self.status.setText(
            "Click a word for full breakdown  |  Drag to select a phrase  |  🔊 reads the selection"
        )

    def _switch_to_editor(self):
        self.editor.setVisible(True)
        self.reader.setVisible(False)
        self.load_btn.setVisible(True)
        self.edit_btn.setVisible(False)
        self.speak_text_btn.setVisible(False)

    def _on_google_only_toggled(self, checked):
        self.google_only_btn.setText("✓ Google Only" if checked else "Google Only")
        if self._last_request:
            # re-run the word/phrase you are looking at, in the new mode
            self._start_lookup(*self._last_request)
        else:
            self.status.setText(
                "Google Translate only: ON" if checked else "Google Translate only: OFF (Gemini)"
            )

    # ── lookup ───────────────────────────────────────────────────────
    def _on_text_selected(self, text, context, is_phrase):
        self._start_lookup(text, context, is_phrase)

    def _start_lookup(self, text, context, is_phrase):
        for old in self._workers:
            old.requestInterruption()          # superseded: stop it quietly
        self._req_id += 1
        rid = self._req_id
        self._last_request = (text, context, is_phrase)

        google_only = self.google_only_btn.isChecked()
        via   = "Google Translate" if google_only else "memory/Gemini"
        verb  = "Translating" if is_phrase else "Looking up"
        short = text if len(text) <= 40 else text[:38] + "…"
        self.status.setText(f"{verb} '{short}' via {via}…")
        self.lookup.show_loading(text, is_phrase)

        w = LookupWorker(text, context, is_phrase, google_only=google_only)
        w.result_ready.connect(lambda data, rid=rid, text=text: self._on_lookup_result(data, rid, text))
        w.error.connect(lambda msg, rid=rid: self._on_lookup_error(msg, rid))
        w.finished.connect(lambda w=w: self._workers.discard(w))
        w.finished.connect(w.deleteLater)
        self._workers.add(w)
        w.start()

    def _on_lookup_result(self, data, rid, text):
        if rid != self._req_id:
            return                              # stale answer from an older click
        if not data.get("is_phrase", False):
            data["word"] = text
        already = self.db.exists(data.get("word", ""))
        self.lookup.show_result(data, already)
        src = data.get("source", "")
        self.status.setText(
            f"Done: '{data.get('word', '')}'"
            + (f"  [{src}{' · memory' if data.get('from_memory') else ''}]" if src else "")
        )

    def _on_lookup_error(self, msg, rid):
        if rid != self._req_id:
            return
        self.lookup.show_error(msg)
        self.status.setText(f"Error: {msg}")

    def _on_save(self, data):
        ok = self.db.save(data)
        self.lookup.mark_saved()
        self.vocab_panel.refresh()
        self.status.setText(
            f"Saved (all fields): {data.get('word', '')}" if ok else "Save failed"
        )

    def closeEvent(self, event):
        for w in list(self._workers):
            w.requestInterruption()
        for w in list(self._workers):
            w.wait(1500)
        super().closeEvent(event)

    # ── read aloud ───────────────────────────────────────────────────
    def _on_read_aloud(self):
        text = self.reader.current_speech_text()
        if not text:
            self.status.setText("Nothing to read")
            return
        self._speak(text)

    def _speak(self, text):
        text = (text or "").strip()
        if not text:
            return
        if self.speak_worker and self.speak_worker.isRunning():
            self.status.setText("Already speaking…")
            return
        preview = text if len(text) <= 40 else text[:38] + "…"
        self.status.setText(f"🔊 Speaking: {preview}")
        self.speak_worker = SpeakWorker(text)
        self.speak_worker.audio_ready.connect(self._play_audio)
        self.speak_worker.spoken.connect(lambda: self.status.setText("🔊 Done"))
        self.speak_worker.error.connect(lambda m: self.status.setText(m))
        self.speak_worker.start()

    def _play_audio(self, path):
        try:
            from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
            # keep references on self so Qt does not garbage-collect mid-playback
            self.player    = QMediaPlayer()
            self.audio_out = QAudioOutput()
            self.audio_out.setVolume(1.0)
            self.player.setAudioOutput(self.audio_out)
            self.player.setSource(QUrl.fromLocalFile(path))
            self.player.play()
            self.status.setText("🔊 Playing…")
        except Exception as e:
            # No QtMultimedia — try the OS default player instead
            try:
                if sys.platform == "darwin":
                    os.system(f'afplay "{path}" &')
                elif sys.platform.startswith("win"):
                    os.startfile(path)
                else:
                    os.system(f'(mpg123 -q "{path}" || ffplay -nodisp -autoexit -loglevel quiet "{path}") &')
                self.status.setText("🔊 Playing…")
            except Exception as e2:
                self.status.setText(f"Playback failed: {e} / {e2}")


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window,          QColor("#F9F9F9"))
    palette.setColor(QPalette.ColorRole.WindowText,      QColor("#1A1A1A"))
    palette.setColor(QPalette.ColorRole.Base,            QColor("#FFFFFF"))
    palette.setColor(QPalette.ColorRole.AlternateBase,   QColor("#F3F3F8"))
    palette.setColor(QPalette.ColorRole.Text,            QColor("#1A1A1A"))
    palette.setColor(QPalette.ColorRole.Button,          QColor("#ECECEC"))
    palette.setColor(QPalette.ColorRole.ButtonText,      QColor("#1A1A1A"))
    palette.setColor(QPalette.ColorRole.Highlight,       QColor("#4A7DC8"))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#FFFFFF"))
    app.setPalette(palette)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())