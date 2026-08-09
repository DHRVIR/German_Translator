
"""
German Learning Assistant — PyQt6 Desktop App
Requirements: pip install PyQt6 deep-translator google-genai python-dotenv
"""

import sys, re, sqlite3, json, csv
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QTextEdit, QTextBrowser, QPushButton, QLabel,
    QTableWidget, QTableWidgetItem, QHeaderView, QFrame,
    QLineEdit, QMessageBox, QScrollArea, QSizePolicy
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import (
    QFont, QColor, QTextCursor, QTextCharFormat, QPalette, QCursor
)
from PyQt6.QtWidgets import QFileDialog
from google.genai import types
from google import genai
from deep_translator import GoogleTranslator
from dotenv import load_dotenv
import os

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DB_PATH = Path.home() / ".german_learner_vocab.db"
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
                added_at        DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()
        self._migrate()

    def _migrate(self):
        """Add new columns if upgrading from old schema."""
        existing = {r[1] for r in self.conn.execute("PRAGMA table_info(vocab)")}
        new_cols = {
            "lemma":"TEXT","part_of_speech":"TEXT","gender":"TEXT","plural":"TEXT",
            "context_meaning":"TEXT","other_meanings":"TEXT","cefr_level":"TEXT",
            "pronunciation":"TEXT","synonyms":"TEXT","antonyms":"TEXT",
            "word_family":"TEXT","collocations":"TEXT","common_phrases":"TEXT",
            "grammar_notes":"TEXT","common_mistakes":"TEXT","memory_tip":"TEXT",
        }
        for col, typ in new_cols.items():
            if col not in existing:
                self.conn.execute(f"ALTER TABLE vocab ADD COLUMN {col} {typ}")
        self.conn.commit()

    def save(self, d):
        def j(v): return json.dumps(v, ensure_ascii=False) if isinstance(v, (list,dict)) else (v or "")
        try:
            self.conn.execute("""
                INSERT OR IGNORE INTO vocab
                (word,lemma,part_of_speech,gender,plural,context_meaning,other_meanings,
                 cefr_level,pronunciation,synonyms,antonyms,word_family,collocations,
                 common_phrases,grammar_notes,common_mistakes,memory_tip,example_de,example_en)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                d.get("word",""), d.get("lemma",""), d.get("part_of_speech",""),
                d.get("gender",""), d.get("plural",""), d.get("context_meaning",""),
                j(d.get("other_meanings",[])), d.get("cefr_level",""),
                d.get("pronunciation",""), j(d.get("synonyms",[])),
                j(d.get("antonyms",[])), j(d.get("word_family",[])),
                j(d.get("collocations",[])), j(d.get("common_phrases",[])),
                d.get("grammar_notes",""), j(d.get("common_mistakes",[])),
                d.get("memory_tip",""), d.get("example_de",""), d.get("example_en",""),
            ))
            self.conn.commit()
            return True
        except Exception as e:
            print("DB save error:", e)
            return False

    def all_words(self):
        cur = self.conn.execute(
            "SELECT id,word,part_of_speech,context_meaning,cefr_level,example_de,example_en "
            "FROM vocab ORDER BY added_at DESC"
        )
        return cur.fetchall()

    def delete(self, word_id):
        self.conn.execute("DELETE FROM vocab WHERE id=?", (word_id,))
        self.conn.commit()

    def exists(self, word):
        cur = self.conn.execute("SELECT 1 FROM vocab WHERE LOWER(word)=LOWER(?)", (word,))
        return cur.fetchone() is not None


# ─────────────────────────────────────────────
#  Worker
# ─────────────────────────────────────────────
class LookupWorker(QThread):
    result_ready = pyqtSignal(dict)
    error        = pyqtSignal(str)

    def __init__(self, text, context_sentence, is_phrase):
        super().__init__()
        self.text             = text
        self.context_sentence = context_sentence
        self.is_phrase        = is_phrase

    def run(self):
        try:
            if self.is_phrase:
                self._translate_phrase()
            else:
                self._lookup_word()
        except Exception as e:
            self.error.emit(f"Unexpected error: {e}")

    def _translate_phrase(self):
        if GEMINI_API_KEY:
            result = self._try_gemini(is_phrase=True)
            if result:
                self.result_ready.emit(result)
                return
        try:
            translation = GoogleTranslator(source="de", target="en").translate(self.text)
            self.result_ready.emit({
                "word": self.text, "is_phrase": True,
                "context_meaning": translation, "source": "Google Translate",
            })
        except Exception as e:
            self.error.emit(f"Translation failed: {e}")

    def _lookup_word(self):
        if GEMINI_API_KEY:
            result = self._try_gemini(is_phrase=False)
            if result:
                self.result_ready.emit(result)
                return
        self._fallback_translate()

    def _build_gemini_prompt(self, is_phrase=False):
        ctx = ""
        if self.context_sentence and self.context_sentence.strip() != self.text.strip():
            ctx = (
                f'\nThe selected text appears in this sentence: "{self.context_sentence}"\n'
                f'Use that context for context_meaning and grammar_notes.\n'
            )

        if is_phrase:
            ctx += (
                "\nThe selected text is a phrase or full sentence.\n"
                "Explain the grammar like a teacher for a German learner.\n"
                "Focus on sentence structure, verb position, cases, articles, prepositions, tense,\n"
                "and any special points that help the learner understand why it is built that way.\n"
            )

        label = "German phrase/sentence" if is_phrase else "German word"
        return GEMINI_PROMPT + ctx + f"\n{label}: {self.text}"

    def _try_gemini(self, is_phrase=False):
        import concurrent.futures
        def _call():
            client = genai.Client(api_key=GEMINI_API_KEY)
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

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(_call)
                try:
                    timeout = GEMINI_TIMEOUT_SECONDS_PHRASE if is_phrase else GEMINI_TIMEOUT_SECONDS_WORD
                    raw = future.result(timeout=timeout)
                except concurrent.futures.TimeoutError:
                    return None

            raw = re.sub(r"^```[a-z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if not m:
                return None
            data = json.loads(m.group(0))
            data["is_phrase"] = is_phrase
            data["source"] = "Gemini"
            return data
        except Exception as e:
            print("Gemini error:", e)
            return None

    def _fallback_translate(self):
        try:
            translation = GoogleTranslator(source="de", target="en").translate(self.text)
            self.result_ready.emit({
                "word": self.text, "is_phrase": False,
                "part_of_speech": "translation",
                "context_meaning": translation,
                "example_de": self.context_sentence or "",
                "source": "Google Translate",
            })
        except Exception as e:
            self.error.emit(f"Lookup failed: {e}")


# ─────────────────────────────────────────────
#  Helpers for rich display
# ─────────────────────────────────────────────
SECTION_STYLE = """
    QFrame#section {
        border: 1px solid #E0E0EC;
        border-radius: 6px;
        background: #FAFAFA;
    }
"""

def make_section(title, content_widget, accent="#7F77DD"):
    """Return a framed section widget with a title bar."""
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
    """Widget showing a bulleted list of strings."""
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
#  Lookup Panel (rich, scrollable)
# ─────────────────────────────────────────────
class LookupPanel(QFrame):
    save_requested = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.current_data = None
        self._setup_ui()

    def _setup_ui(self):
        self.setFrameShape(QFrame.Shape.StyledPanel)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── Top bar ──────────────────────────────
        top = QFrame()
        top.setStyleSheet("background: #F0F0F8; border-bottom: 1px solid #E0E0EC;")
        top_lay = QHBoxLayout(top)
        top_lay.setContentsMargins(10, 8, 10, 8)

        lbl = QLabel("Word Lookup")
        lbl.setFont(QFont("", 10, QFont.Weight.Bold))
        lbl.setStyleSheet("color: #1A1A1A; background: transparent;")
        top_lay.addWidget(lbl)
        top_lay.addStretch()

        self.source_badge = QLabel("")
        self.source_badge.setVisible(False)
        top_lay.addWidget(self.source_badge)
        outer.addWidget(top)

        # ── Scroll area for rich content ─────────
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

        # ── Save button ───────────────────────────
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

        # placeholder
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
        word      = data.get("word", "")

        # Source badge
        if source:
            color = "#5B8DD9" if source == "Google Translate" else "#7F77DD"
            self.source_badge.setStyleSheet(
                f"color:#FFF;background:{color};border-radius:8px;padding:2px 10px;font-size:10px;"
            )
            self.source_badge.setText(f"via {source}")
            self.source_badge.setVisible(True)

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
        lay = self.content_lay
        word  = d.get("word", "")
        trans = d.get("context_meaning", "")
        gnotes = d.get("grammar_notes", "")
        ex_de = d.get("example_de", "")
        ex_en = d.get("example_en", "")

        title = QLabel(word[:60] + ("…" if len(word) > 60 else ""))
        title.setFont(QFont("", 14, QFont.Weight.Bold))
        title.setWordWrap(True)
        title.setStyleSheet("color:#1A1A1A;")
        lay.addWidget(title)

        sub = QLabel("Phrase / sentence translation")
        sub.setStyleSheet("color:#888;font-size:10px;")
        lay.addWidget(sub)

        box = QWidget()
        bl = QVBoxLayout(box)
        bl.setContentsMargins(8,6,8,6)
        bl.addWidget(body_label(trans, color="#1A1A1A", size=12))
        lay.addWidget(make_section("Translation", box, "#5B8DD9"))

        if gnotes:
            box = QWidget()
            bl = QVBoxLayout(box)
            bl.setContentsMargins(8,6,8,6)
            bl.addWidget(body_label(gnotes, color="#333", size=11))
            lay.addWidget(make_section("Grammar breakdown", box, "#C0652B"))

        if ex_de or ex_en:
            box = QWidget()
            bl = QVBoxLayout(box)
            bl.setContentsMargins(8,6,8,6)
            bl.setSpacing(2)
            if ex_de:
                bl.addWidget(body_label(ex_de, color="#1A1A1A", size=11, bold=True))
            if ex_en:
                bl.addWidget(body_label(ex_en, color="#555555", size=10))
            lay.addWidget(make_section("Selected sentence", box, "#3A9A5C"))

    def _render_word(self, d):
        lay = self.content_lay

        # ── Word + meta row ──────────────────────
        word = d.get("word", "")
        lemma = d.get("lemma", "")
        pos   = d.get("part_of_speech", "")
        cefr  = d.get("cefr_level", "")
        pron  = d.get("pronunciation", "")
        gender= d.get("gender","")
        plural= d.get("plural","")

        title = QLabel(word)
        title.setFont(QFont("", 17, QFont.Weight.Bold))
        title.setStyleSheet("color:#1A1A1A;")
        lay.addWidget(title)

        if lemma and lemma.lower() != word.lower():
            sub = QLabel(f"Base form: {lemma}")
            sub.setStyleSheet("color:#888;font-size:10px;")
            lay.addWidget(sub)

        # Pills row
        pills = QHBoxLayout()
        pills.setSpacing(4)
        pills.setContentsMargins(0,0,0,0)
        if pos:   pills.addWidget(pill_label(pos, "#555577", "#FFFFFF"))
        if cefr:  pills.addWidget(pill_label(cefr, "#3A9A5C", "#FFFFFF"))
        if gender:pills.addWidget(pill_label(gender, "#C0652B", "#FFFFFF"))
        if pron:  pills.addWidget(pill_label(pron, "#888888", "#FFFFFF"))
        pills.addStretch()
        pw = QWidget(); pw.setLayout(pills)
        lay.addWidget(pw)

        if plural:
            pl = QLabel(f"Plural: {plural}")
            pl.setStyleSheet("color:#555;font-size:11px;")
            lay.addWidget(pl)

        # ── Context meaning ──────────────────────
        ctx = d.get("context_meaning","")
        if ctx:
            box = QWidget(); bl = QVBoxLayout(box)
            bl.setContentsMargins(8,6,8,6)
            bl.addWidget(body_label(ctx, color="#1A1A1A", size=13, bold=True))
            lay.addWidget(make_section("Meaning in context", box, "#3A9A5C"))

        # ── Other meanings ───────────────────────
        others = d.get("other_meanings", [])
        if others and isinstance(others, list) and len(others) > 0:
            box = list_widget(others)
            lay.addWidget(make_section("Other meanings", box, "#7F77DD"))

        # ── Example ──────────────────────────────
        ex_de = d.get("example_de","")
        ex_en = d.get("example_en","")
        if ex_de:
            box = QWidget(); bl = QVBoxLayout(box)
            bl.setContentsMargins(8,6,8,6); bl.setSpacing(2)
            bl.addWidget(body_label(ex_de, color="#1A1A1A", size=11, bold=True))
            if ex_en: bl.addWidget(body_label(ex_en, color="#555555", size=10))
            lay.addWidget(make_section("Example / Context sentence", box, "#5B8DD9"))

        # ── Grammar ──────────────────────────────
        conj   = d.get("conjugation") or {}
        comp   = d.get("comparison") or {}
        gnotes = d.get("grammar_notes","")

        if conj and isinstance(conj, dict) and any(conj.get(k) for k in ["ich","du","er"]):
            box = QWidget(); bl = QVBoxLayout(box)
            bl.setContentsMargins(8,6,8,6); bl.setSpacing(2)
            pairs = [("ich",conj.get("ich","")),("du",conj.get("du","")),
                     ("er/sie/es",conj.get("er","")),("wir",conj.get("wir","")),
                     ("ihr",conj.get("ihr","")),("sie",conj.get("sie",""))]
            for pronoun, form in pairs:
                if form:
                    row = QHBoxLayout()
                    row.setContentsMargins(0,0,0,0)
                    pl = QLabel(pronoun); pl.setFixedWidth(70)
                    pl.setStyleSheet("color:#888;font-size:11px;")
                    fl = QLabel(form)
                    fl.setStyleSheet("color:#1A1A1A;font-size:11px;font-weight:bold;")
                    row.addWidget(pl); row.addWidget(fl); row.addStretch()
                    rw = QWidget(); rw.setLayout(row)
                    bl.addWidget(rw)
            for k,label in [("past","Past (Präteritum)"),("perfect","Perfect (Perfekt)")]:
                if conj.get(k):
                    bl.addWidget(body_label(f"{label}: {conj[k]}", color="#555", size=10))
            sep = conj.get("separable")
            if sep is not None:
                bl.addWidget(body_label(f"Separable: {'Yes' if sep else 'No'}", color="#888", size=10))
            lay.addWidget(make_section("Conjugation", box, "#C0652B"))

        if comp and isinstance(comp, dict) and any(comp.values()):
            box = QWidget(); bl = QVBoxLayout(box)
            bl.setContentsMargins(8,6,8,6); bl.setSpacing(2)
            if comp.get("comparative"):
                bl.addWidget(body_label(f"Comparative: {comp['comparative']}", size=11))
            if comp.get("superlative"):
                bl.addWidget(body_label(f"Superlative: {comp['superlative']}", size=11))
            lay.addWidget(make_section("Comparison", box, "#C0652B"))

        if gnotes:
            box = QWidget(); bl = QVBoxLayout(box)
            bl.setContentsMargins(8,6,8,6)
            bl.addWidget(body_label(gnotes, color="#333", size=11))
            lay.addWidget(make_section("Grammar notes", box, "#C0652B"))

        # ── Synonyms / Antonyms ──────────────────
        syns = d.get("synonyms",[]) or []
        ants = d.get("antonyms",[]) or []
        if syns or ants:
            box = QWidget(); bl = QVBoxLayout(box)
            bl.setContentsMargins(8,4,8,4); bl.setSpacing(2)
            if syns: bl.addWidget(body_label("Synonyms: " + ", ".join(syns), size=11))
            if ants: bl.addWidget(body_label("Antonyms: " + ", ".join(ants), size=11, color="#C0652B"))
            lay.addWidget(make_section("Synonyms & Antonyms", box, "#888888"))

        # ── Word family ──────────────────────────
        wf = d.get("word_family",[]) or []
        if wf:
            lay.addWidget(make_section("Word family", list_widget(wf, "🔗", "#333"), "#3A9A5C"))

        # ── Collocations ─────────────────────────
        coll = d.get("collocations",[]) or []
        if coll:
            lay.addWidget(make_section("Common collocations", list_widget(coll, "❤️", "#333"), "#E05588"))

        # ── Common phrases ───────────────────────
        phrases = d.get("common_phrases",[]) or []
        if phrases:
            lay.addWidget(make_section("Common phrases", list_widget(phrases, "💬", "#333"), "#5B8DD9"))

        # ── Common mistakes ──────────────────────
        mistakes = d.get("common_mistakes",[]) or []
        if mistakes:
            lay.addWidget(make_section("Common mistakes", list_widget(mistakes, "⚠️", "#C0652B"), "#C0652B"))

        # ── Memory tip ───────────────────────────
        tip = d.get("memory_tip","")
        if tip:
            box = QWidget(); bl = QVBoxLayout(box)
            bl.setContentsMargins(8,6,8,6)
            bl.addWidget(body_label(f"💡 {tip}", color="#5B4A00", size=11))
            box.setStyleSheet("background:#FFFBEA;")
            lay.addWidget(make_section("Memory tip", box, "#D4A000"))

    def show_error(self, msg):
        self._clear_content()
        self.source_badge.setVisible(False)
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

    def mouseReleaseEvent(self, e):
        super().mouseReleaseEvent(e)
        if e.button() != Qt.MouseButton.LeftButton:
            return
        cursor = self.textCursor()
        selected = cursor.selectedText().strip()
        if selected and len(selected.split()) > 1:
            selected = re.sub(r"[\u2029\u2028]", " ", selected)
            selected = re.sub(r"\s+", " ", selected).strip()
            if re.search(r"[a-zA-ZäöüÄÖÜß]", selected):
                self._apply_highlight(cursor)
                context = self._extract_sentence(cursor) or selected
                self.text_selected.emit(selected, context, True)
            return
        wc = self.cursorForPosition(e.pos())
        wc.select(QTextCursor.SelectionType.WordUnderCursor)
        word = re.sub(r"[^\w\-äöüÄÖÜß]", "", wc.selectedText().strip(), flags=re.UNICODE)
        if word and re.search(r"[a-zA-ZäöüÄÖÜß]", word):
            context = self._extract_sentence(wc)
            self._apply_highlight(wc)
            self.text_selected.emit(word, context, False)

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
        layout.setContentsMargins(10,10,10,10)
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
        export_btn.clicked.connect(self.export_csv)
        btn_row.addWidget(export_btn)
        layout.addLayout(btn_row)

    def refresh(self):
        self._all_rows = self.db.all_words()
        self._render(self._all_rows)

    def _render(self, rows):
        self.table.setRowCount(0)
        # rows: id, word, part_of_speech, context_meaning, cefr_level, example_de, example_en
        for row in rows:
            row_id, word, pos, meaning, cefr, ex_de, ex_en = row
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

            short = (meaning or "").split(".")[0] + "."
            m_item = QTableWidgetItem(short)
            m_item.setForeground(QColor("#333333"))
            m_item.setToolTip(f"{meaning}\n\n{ex_de}\n{ex_en}")
            self.table.setItem(r, 2, m_item)

            if cefr:
                cefr_item = QTableWidgetItem(cefr)
                cefr_item.setForeground(QColor("#3A9A5C"))
                cefr_item.setFont(QFont("", 10, QFont.Weight.Bold))
                self.table.setItem(r, 2, m_item)   # keep meaning in col 2

            del_btn = QPushButton("×")
            del_btn.setFixedSize(QSize(24, 24))
            del_btn.setStyleSheet("color:#cc3333;font-weight:bold;border:none;background:transparent;")
            del_btn.clicked.connect(lambda _, rid=row_id: self._delete(rid))
            self.table.setCellWidget(r, 3, del_btn)

        self.count_label.setText(f"{len(rows)} word{'s' if len(rows)!=1 else ''}")

    def filter_words(self, text):
        t = text.lower()
        filtered = [r for r in self._all_rows
                    if t in (r[1] or "").lower() or t in (r[3] or "").lower()]
        self._render(filtered)

    def _delete(self, row_id):
        self.db.delete(row_id)
        self.refresh()

    def export_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export Vocabulary", "vocab.csv", "CSV files (*.csv)")
        if path:
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["Word","POS","Meaning","CEFR","Example (DE)","Example (EN)"])
                for row in self.db.all_words():
                    w.writerow(row[1:])
            QMessageBox.information(self, "Exported", f"Saved to {path}")


# ─────────────────────────────────────────────
#  Main Window
# ─────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.db     = VocabDB()
        self.worker = None
        self.setWindowTitle("German Learning Assistant")
        self.resize(1200, 720)
        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(10,8,10,0)
        root.setSpacing(4)

        main_split = QSplitter(Qt.Orientation.Horizontal)
        left_split = QSplitter(Qt.Orientation.Vertical)

        # ── Text input / reader ──────────────────
        input_frame = QFrame()
        input_frame.setFrameShape(QFrame.Shape.StyledPanel)
        in_layout = QVBoxLayout(input_frame)
        in_layout.setContentsMargins(10,10,10,8)
        in_layout.setSpacing(6)

        hdr = QHBoxLayout()
        lbl = QLabel("German Text")
        lbl.setFont(QFont("", 10, QFont.Weight.Bold))
        lbl.setStyleSheet("color:#1A1A1A;")
        hdr.addWidget(lbl)
        hdr.addStretch()
        self.load_btn = QPushButton("Load →")
        self.load_btn.setFixedHeight(26)
        self.load_btn.clicked.connect(self._switch_to_reader)
        hdr.addWidget(self.load_btn)
        self.edit_btn = QPushButton("Edit text")
        self.edit_btn.setFixedHeight(26)
        self.edit_btn.setVisible(False)
        self.edit_btn.clicked.connect(self._switch_to_editor)
        hdr.addWidget(self.edit_btn)
        in_layout.addLayout(hdr)

        self.editor = QTextEdit()
        self.editor.setPlaceholderText(
            "Paste your German text here, then click 'Load →'\n\n"
            "Example:\nHallo meine liebe kleiner."
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
        left_split.addWidget(self.lookup)
        left_split.setSizes([300, 380])

        main_split.addWidget(left_split)
        self.vocab_panel = VocabPanel(self.db)
        main_split.addWidget(self.vocab_panel)
        main_split.setSizes([750, 420])
        root.addWidget(main_split, 1)   # stretch=1 so it fills all available space

        self.status = QLabel("Ready — paste German text and click 'Load →'")
        self.status.setFixedHeight(20)
        self.status.setStyleSheet(
            "font-size:11px; color:#555; padding:0 4px; "
            "border-top: 1px solid #E0E0EC; background: #F7F7FA;"
        )
        root.addWidget(self.status, 0)  # stretch=0 → only takes what it needs

    def _switch_to_reader(self):
        text = self.editor.toPlainText().strip()
        if not text: return
        self.reader.load_german_text(text)
        self.editor.setVisible(False)
        self.reader.setVisible(True)
        self.load_btn.setVisible(False)
        self.edit_btn.setVisible(True)
        self.status.setText("Click a word for full breakdown  |  Drag to select a phrase")

    def _switch_to_editor(self):
        self.editor.setVisible(True)
        self.reader.setVisible(False)
        self.load_btn.setVisible(True)
        self.edit_btn.setVisible(False)

    def _on_text_selected(self, text, context, is_phrase):
        self.status.setText("Translating…" if is_phrase else f"Looking up '{text}'…")
        self.lookup.show_loading(text, is_phrase)
        if self.worker and self.worker.isRunning():
            print("Worker busy, skipping")
            return
        self.worker = LookupWorker(text, context, is_phrase)
        self.worker.result_ready.connect(self._on_lookup_result)
        self.worker.error.connect(self._on_lookup_error)
        self.worker.start()

    def _on_lookup_result(self, data):
        already = self.db.exists(data.get("word",""))
        self.lookup.show_result(data, already)
        src = data.get("source","")
        self.status.setText(f"Done: '{data.get('word','')}'" + (f"  [{src}]" if src else ""))

    def _on_lookup_error(self, msg):
        self.lookup.show_error(msg)
        self.status.setText(f"Error: {msg}")

    def _on_save(self, data):
        self.db.save(data)
        self.lookup.mark_saved()
        self.vocab_panel.refresh()
        self.status.setText(f"Saved: {data.get('word','')}")


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
german_learning_assistant_phrase_timeout_increased.py
"""
German Learning Assistant — PyQt6 Desktop App
Requirements: pip install PyQt6 deep-translator google-genai python-dotenv
"""

import sys, re, sqlite3, json, csv
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QTextEdit, QTextBrowser, QPushButton, QLabel,
    QTableWidget, QTableWidgetItem, QHeaderView, QFrame,
    QLineEdit, QMessageBox, QScrollArea, QSizePolicy
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import (
    QFont, QColor, QTextCursor, QTextCharFormat, QPalette, QCursor
)
from PyQt6.QtWidgets import QFileDialog
from google.genai import types
from google import genai
from deep_translator import GoogleTranslator
from dotenv import load_dotenv
import os

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DB_PATH = Path.home() / ".german_learner_vocab.db"
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
                added_at        DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()
        self._migrate()

    def _migrate(self):
        """Add new columns if upgrading from old schema."""
        existing = {r[1] for r in self.conn.execute("PRAGMA table_info(vocab)")}
        new_cols = {
            "lemma":"TEXT","part_of_speech":"TEXT","gender":"TEXT","plural":"TEXT",
            "context_meaning":"TEXT","other_meanings":"TEXT","cefr_level":"TEXT",
            "pronunciation":"TEXT","synonyms":"TEXT","antonyms":"TEXT",
            "word_family":"TEXT","collocations":"TEXT","common_phrases":"TEXT",
            "grammar_notes":"TEXT","common_mistakes":"TEXT","memory_tip":"TEXT",
        }
        for col, typ in new_cols.items():
            if col not in existing:
                self.conn.execute(f"ALTER TABLE vocab ADD COLUMN {col} {typ}")
        self.conn.commit()

    def save(self, d):
        def j(v): return json.dumps(v, ensure_ascii=False) if isinstance(v, (list,dict)) else (v or "")
        try:
            self.conn.execute("""
                INSERT OR IGNORE INTO vocab
                (word,lemma,part_of_speech,gender,plural,context_meaning,other_meanings,
                 cefr_level,pronunciation,synonyms,antonyms,word_family,collocations,
                 common_phrases,grammar_notes,common_mistakes,memory_tip,example_de,example_en)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                d.get("word",""), d.get("lemma",""), d.get("part_of_speech",""),
                d.get("gender",""), d.get("plural",""), d.get("context_meaning",""),
                j(d.get("other_meanings",[])), d.get("cefr_level",""),
                d.get("pronunciation",""), j(d.get("synonyms",[])),
                j(d.get("antonyms",[])), j(d.get("word_family",[])),
                j(d.get("collocations",[])), j(d.get("common_phrases",[])),
                d.get("grammar_notes",""), j(d.get("common_mistakes",[])),
                d.get("memory_tip",""), d.get("example_de",""), d.get("example_en",""),
            ))
            self.conn.commit()
            return True
        except Exception as e:
            print("DB save error:", e)
            return False

    def all_words(self):
        cur = self.conn.execute(
            "SELECT id,word,part_of_speech,context_meaning,cefr_level,example_de,example_en "
            "FROM vocab ORDER BY added_at DESC"
        )
        return cur.fetchall()

    def delete(self, word_id):
        self.conn.execute("DELETE FROM vocab WHERE id=?", (word_id,))
        self.conn.commit()

    def exists(self, word):
        cur = self.conn.execute("SELECT 1 FROM vocab WHERE LOWER(word)=LOWER(?)", (word,))
        return cur.fetchone() is not None


# ─────────────────────────────────────────────
#  Worker
# ─────────────────────────────────────────────
class LookupWorker(QThread):
    result_ready = pyqtSignal(dict)
    error        = pyqtSignal(str)

    def __init__(self, text, context_sentence, is_phrase):
        super().__init__()
        self.text             = text
        self.context_sentence = context_sentence
        self.is_phrase        = is_phrase

    def run(self):
        try:
            if self.is_phrase:
                self._translate_phrase()
            else:
                self._lookup_word()
        except Exception as e:
            self.error.emit(f"Unexpected error: {e}")

    def _translate_phrase(self):
        if GEMINI_API_KEY:
            result = self._try_gemini(is_phrase=True)
            if result:
                self.result_ready.emit(result)
                return
        try:
            translation = GoogleTranslator(source="de", target="en").translate(self.text)
            self.result_ready.emit({
                "word": self.text, "is_phrase": True,
                "context_meaning": translation, "source": "Google Translate",
            })
        except Exception as e:
            self.error.emit(f"Translation failed: {e}")

    def _lookup_word(self):
        if GEMINI_API_KEY:
            result = self._try_gemini(is_phrase=False)
            if result:
                self.result_ready.emit(result)
                return
        self._fallback_translate()

    def _build_gemini_prompt(self, is_phrase=False):
        ctx = ""
        if self.context_sentence and self.context_sentence.strip() != self.text.strip():
            ctx = (
                f'\nThe selected text appears in this sentence: "{self.context_sentence}"\n'
                f'Use that context for context_meaning and grammar_notes.\n'
            )

        if is_phrase:
            ctx += (
                "\nThe selected text is a phrase or full sentence.\n"
                "Explain the grammar like a teacher for a German learner.\n"
                "Focus on sentence structure, verb position, cases, articles, prepositions, tense,\n"
                "and any special points that help the learner understand why it is built that way.\n"
            )

        label = "German phrase/sentence" if is_phrase else "German word"
        return GEMINI_PROMPT + ctx + f"\n{label}: {self.text}"

    def _try_gemini(self, is_phrase=False):
        import concurrent.futures
        def _call():
            client = genai.Client(api_key=GEMINI_API_KEY)
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

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(_call)
                try:
                    timeout = GEMINI_TIMEOUT_SECONDS_PHRASE if is_phrase else GEMINI_TIMEOUT_SECONDS_WORD
                    raw = future.result(timeout=timeout)
                except concurrent.futures.TimeoutError:
                    return None

            raw = re.sub(r"^```[a-z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if not m:
                return None
            data = json.loads(m.group(0))
            data["is_phrase"] = is_phrase
            data["source"] = "Gemini"
            return data
        except Exception as e:
            print("Gemini error:", e)
            return None

    def _fallback_translate(self):
        try:
            translation = GoogleTranslator(source="de", target="en").translate(self.text)
            self.result_ready.emit({
                "word": self.text, "is_phrase": False,
                "part_of_speech": "translation",
                "context_meaning": translation,
                "example_de": self.context_sentence or "",
                "source": "Google Translate",
            })
        except Exception as e:
            self.error.emit(f"Lookup failed: {e}")


# ─────────────────────────────────────────────
#  Helpers for rich display
# ─────────────────────────────────────────────
SECTION_STYLE = """
    QFrame#section {
        border: 1px solid #E0E0EC;
        border-radius: 6px;
        background: #FAFAFA;
    }
"""

def make_section(title, content_widget, accent="#7F77DD"):
    """Return a framed section widget with a title bar."""
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
    """Widget showing a bulleted list of strings."""
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
#  Lookup Panel (rich, scrollable)
# ─────────────────────────────────────────────
class LookupPanel(QFrame):
    save_requested = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.current_data = None
        self._setup_ui()

    def _setup_ui(self):
        self.setFrameShape(QFrame.Shape.StyledPanel)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── Top bar ──────────────────────────────
        top = QFrame()
        top.setStyleSheet("background: #F0F0F8; border-bottom: 1px solid #E0E0EC;")
        top_lay = QHBoxLayout(top)
        top_lay.setContentsMargins(10, 8, 10, 8)

        lbl = QLabel("Word Lookup")
        lbl.setFont(QFont("", 10, QFont.Weight.Bold))
        lbl.setStyleSheet("color: #1A1A1A; background: transparent;")
        top_lay.addWidget(lbl)
        top_lay.addStretch()

        self.source_badge = QLabel("")
        self.source_badge.setVisible(False)
        top_lay.addWidget(self.source_badge)
        outer.addWidget(top)

        # ── Scroll area for rich content ─────────
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

        # ── Save button ───────────────────────────
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

        # placeholder
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
        word      = data.get("word", "")

        # Source badge
        if source:
            color = "#5B8DD9" if source == "Google Translate" else "#7F77DD"
            self.source_badge.setStyleSheet(
                f"color:#FFF;background:{color};border-radius:8px;padding:2px 10px;font-size:10px;"
            )
            self.source_badge.setText(f"via {source}")
            self.source_badge.setVisible(True)

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
        lay = self.content_lay
        word  = d.get("word", "")
        trans = d.get("context_meaning", "")
        gnotes = d.get("grammar_notes", "")
        ex_de = d.get("example_de", "")
        ex_en = d.get("example_en", "")

        title = QLabel(word[:60] + ("…" if len(word) > 60 else ""))
        title.setFont(QFont("", 14, QFont.Weight.Bold))
        title.setWordWrap(True)
        title.setStyleSheet("color:#1A1A1A;")
        lay.addWidget(title)

        sub = QLabel("Phrase / sentence translation")
        sub.setStyleSheet("color:#888;font-size:10px;")
        lay.addWidget(sub)

        box = QWidget()
        bl = QVBoxLayout(box)
        bl.setContentsMargins(8,6,8,6)
        bl.addWidget(body_label(trans, color="#1A1A1A", size=12))
        lay.addWidget(make_section("Translation", box, "#5B8DD9"))

        if gnotes:
            box = QWidget()
            bl = QVBoxLayout(box)
            bl.setContentsMargins(8,6,8,6)
            bl.addWidget(body_label(gnotes, color="#333", size=11))
            lay.addWidget(make_section("Grammar breakdown", box, "#C0652B"))

        if ex_de or ex_en:
            box = QWidget()
            bl = QVBoxLayout(box)
            bl.setContentsMargins(8,6,8,6)
            bl.setSpacing(2)
            if ex_de:
                bl.addWidget(body_label(ex_de, color="#1A1A1A", size=11, bold=True))
            if ex_en:
                bl.addWidget(body_label(ex_en, color="#555555", size=10))
            lay.addWidget(make_section("Selected sentence", box, "#3A9A5C"))

    def _render_word(self, d):
        lay = self.content_lay

        # ── Word + meta row ──────────────────────
        word = d.get("word", "")
        lemma = d.get("lemma", "")
        pos   = d.get("part_of_speech", "")
        cefr  = d.get("cefr_level", "")
        pron  = d.get("pronunciation", "")
        gender= d.get("gender","")
        plural= d.get("plural","")

        title = QLabel(word)
        title.setFont(QFont("", 17, QFont.Weight.Bold))
        title.setStyleSheet("color:#1A1A1A;")
        lay.addWidget(title)

        if lemma and lemma.lower() != word.lower():
            sub = QLabel(f"Base form: {lemma}")
            sub.setStyleSheet("color:#888;font-size:10px;")
            lay.addWidget(sub)

        # Pills row
        pills = QHBoxLayout()
        pills.setSpacing(4)
        pills.setContentsMargins(0,0,0,0)
        if pos:   pills.addWidget(pill_label(pos, "#555577", "#FFFFFF"))
        if cefr:  pills.addWidget(pill_label(cefr, "#3A9A5C", "#FFFFFF"))
        if gender:pills.addWidget(pill_label(gender, "#C0652B", "#FFFFFF"))
        if pron:  pills.addWidget(pill_label(pron, "#888888", "#FFFFFF"))
        pills.addStretch()
        pw = QWidget(); pw.setLayout(pills)
        lay.addWidget(pw)

        if plural:
            pl = QLabel(f"Plural: {plural}")
            pl.setStyleSheet("color:#555;font-size:11px;")
            lay.addWidget(pl)

        # ── Context meaning ──────────────────────
        ctx = d.get("context_meaning","")
        if ctx:
            box = QWidget(); bl = QVBoxLayout(box)
            bl.setContentsMargins(8,6,8,6)
            bl.addWidget(body_label(ctx, color="#1A1A1A", size=13, bold=True))
            lay.addWidget(make_section("Meaning in context", box, "#3A9A5C"))

        # ── Other meanings ───────────────────────
        others = d.get("other_meanings", [])
        if others and isinstance(others, list) and len(others) > 0:
            box = list_widget(others)
            lay.addWidget(make_section("Other meanings", box, "#7F77DD"))

        # ── Example ──────────────────────────────
        ex_de = d.get("example_de","")
        ex_en = d.get("example_en","")
        if ex_de:
            box = QWidget(); bl = QVBoxLayout(box)
            bl.setContentsMargins(8,6,8,6); bl.setSpacing(2)
            bl.addWidget(body_label(ex_de, color="#1A1A1A", size=11, bold=True))
            if ex_en: bl.addWidget(body_label(ex_en, color="#555555", size=10))
            lay.addWidget(make_section("Example / Context sentence", box, "#5B8DD9"))

        # ── Grammar ──────────────────────────────
        conj   = d.get("conjugation") or {}
        comp   = d.get("comparison") or {}
        gnotes = d.get("grammar_notes","")

        if conj and isinstance(conj, dict) and any(conj.get(k) for k in ["ich","du","er"]):
            box = QWidget(); bl = QVBoxLayout(box)
            bl.setContentsMargins(8,6,8,6); bl.setSpacing(2)
            pairs = [("ich",conj.get("ich","")),("du",conj.get("du","")),
                     ("er/sie/es",conj.get("er","")),("wir",conj.get("wir","")),
                     ("ihr",conj.get("ihr","")),("sie",conj.get("sie",""))]
            for pronoun, form in pairs:
                if form:
                    row = QHBoxLayout()
                    row.setContentsMargins(0,0,0,0)
                    pl = QLabel(pronoun); pl.setFixedWidth(70)
                    pl.setStyleSheet("color:#888;font-size:11px;")
                    fl = QLabel(form)
                    fl.setStyleSheet("color:#1A1A1A;font-size:11px;font-weight:bold;")
                    row.addWidget(pl); row.addWidget(fl); row.addStretch()
                    rw = QWidget(); rw.setLayout(row)
                    bl.addWidget(rw)
            for k,label in [("past","Past (Präteritum)"),("perfect","Perfect (Perfekt)")]:
                if conj.get(k):
                    bl.addWidget(body_label(f"{label}: {conj[k]}", color="#555", size=10))
            sep = conj.get("separable")
            if sep is not None:
                bl.addWidget(body_label(f"Separable: {'Yes' if sep else 'No'}", color="#888", size=10))
            lay.addWidget(make_section("Conjugation", box, "#C0652B"))

        if comp and isinstance(comp, dict) and any(comp.values()):
            box = QWidget(); bl = QVBoxLayout(box)
            bl.setContentsMargins(8,6,8,6); bl.setSpacing(2)
            if comp.get("comparative"):
                bl.addWidget(body_label(f"Comparative: {comp['comparative']}", size=11))
            if comp.get("superlative"):
                bl.addWidget(body_label(f"Superlative: {comp['superlative']}", size=11))
            lay.addWidget(make_section("Comparison", box, "#C0652B"))

        if gnotes:
            box = QWidget(); bl = QVBoxLayout(box)
            bl.setContentsMargins(8,6,8,6)
            bl.addWidget(body_label(gnotes, color="#333", size=11))
            lay.addWidget(make_section("Grammar notes", box, "#C0652B"))

        # ── Synonyms / Antonyms ──────────────────
        syns = d.get("synonyms",[]) or []
        ants = d.get("antonyms",[]) or []
        if syns or ants:
            box = QWidget(); bl = QVBoxLayout(box)
            bl.setContentsMargins(8,4,8,4); bl.setSpacing(2)
            if syns: bl.addWidget(body_label("Synonyms: " + ", ".join(syns), size=11))
            if ants: bl.addWidget(body_label("Antonyms: " + ", ".join(ants), size=11, color="#C0652B"))
            lay.addWidget(make_section("Synonyms & Antonyms", box, "#888888"))

        # ── Word family ──────────────────────────
        wf = d.get("word_family",[]) or []
        if wf:
            lay.addWidget(make_section("Word family", list_widget(wf, "🔗", "#333"), "#3A9A5C"))

        # ── Collocations ─────────────────────────
        coll = d.get("collocations",[]) or []
        if coll:
            lay.addWidget(make_section("Common collocations", list_widget(coll, "❤️", "#333"), "#E05588"))

        # ── Common phrases ───────────────────────
        phrases = d.get("common_phrases",[]) or []
        if phrases:
            lay.addWidget(make_section("Common phrases", list_widget(phrases, "💬", "#333"), "#5B8DD9"))

        # ── Common mistakes ──────────────────────
        mistakes = d.get("common_mistakes",[]) or []
        if mistakes:
            lay.addWidget(make_section("Common mistakes", list_widget(mistakes, "⚠️", "#C0652B"), "#C0652B"))

        # ── Memory tip ───────────────────────────
        tip = d.get("memory_tip","")
        if tip:
            box = QWidget(); bl = QVBoxLayout(box)
            bl.setContentsMargins(8,6,8,6)
            bl.addWidget(body_label(f"💡 {tip}", color="#5B4A00", size=11))
            box.setStyleSheet("background:#FFFBEA;")
            lay.addWidget(make_section("Memory tip", box, "#D4A000"))

    def show_error(self, msg):
        self._clear_content()
        self.source_badge.setVisible(False)
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

    def mouseReleaseEvent(self, e):
        super().mouseReleaseEvent(e)
        if e.button() != Qt.MouseButton.LeftButton:
            return
        cursor = self.textCursor()
        selected = cursor.selectedText().strip()
        if selected and len(selected.split()) > 1:
            selected = re.sub(r"[\u2029\u2028]", " ", selected)
            selected = re.sub(r"\s+", " ", selected).strip()
            if re.search(r"[a-zA-ZäöüÄÖÜß]", selected):
                self._apply_highlight(cursor)
                context = self._extract_sentence(cursor) or selected
                self.text_selected.emit(selected, context, True)
            return
        wc = self.cursorForPosition(e.pos())
        wc.select(QTextCursor.SelectionType.WordUnderCursor)
        word = re.sub(r"[^\w\-äöüÄÖÜß]", "", wc.selectedText().strip(), flags=re.UNICODE)
        if word and re.search(r"[a-zA-ZäöüÄÖÜß]", word):
            context = self._extract_sentence(wc)
            self._apply_highlight(wc)
            self.text_selected.emit(word, context, False)

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
        layout.setContentsMargins(10,10,10,10)
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
        export_btn.clicked.connect(self.export_csv)
        btn_row.addWidget(export_btn)
        layout.addLayout(btn_row)

    def refresh(self):
        self._all_rows = self.db.all_words()
        self._render(self._all_rows)

    def _render(self, rows):
        self.table.setRowCount(0)
        # rows: id, word, part_of_speech, context_meaning, cefr_level, example_de, example_en
        for row in rows:
            row_id, word, pos, meaning, cefr, ex_de, ex_en = row
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

            short = (meaning or "").split(".")[0] + "."
            m_item = QTableWidgetItem(short)
            m_item.setForeground(QColor("#333333"))
            m_item.setToolTip(f"{meaning}\n\n{ex_de}\n{ex_en}")
            self.table.setItem(r, 2, m_item)

            if cefr:
                cefr_item = QTableWidgetItem(cefr)
                cefr_item.setForeground(QColor("#3A9A5C"))
                cefr_item.setFont(QFont("", 10, QFont.Weight.Bold))
                self.table.setItem(r, 2, m_item)   # keep meaning in col 2

            del_btn = QPushButton("×")
            del_btn.setFixedSize(QSize(24, 24))
            del_btn.setStyleSheet("color:#cc3333;font-weight:bold;border:none;background:transparent;")
            del_btn.clicked.connect(lambda _, rid=row_id: self._delete(rid))
            self.table.setCellWidget(r, 3, del_btn)

        self.count_label.setText(f"{len(rows)} word{'s' if len(rows)!=1 else ''}")

    def filter_words(self, text):
        t = text.lower()
        filtered = [r for r in self._all_rows
                    if t in (r[1] or "").lower() or t in (r[3] or "").lower()]
        self._render(filtered)

    def _delete(self, row_id):
        self.db.delete(row_id)
        self.refresh()

    def export_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export Vocabulary", "vocab.csv", "CSV files (*.csv)")
        if path:
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["Word","POS","Meaning","CEFR","Example (DE)","Example (EN)"])
                for row in self.db.all_words():
                    w.writerow(row[1:])
            QMessageBox.information(self, "Exported", f"Saved to {path}")


# ─────────────────────────────────────────────
#  Main Window
# ─────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.db     = VocabDB()
        self.worker = None
        self.setWindowTitle("German Learning Assistant")
        self.resize(1200, 720)
        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(10,8,10,0)
        root.setSpacing(4)

        main_split = QSplitter(Qt.Orientation.Horizontal)
        left_split = QSplitter(Qt.Orientation.Vertical)

        # ── Text input / reader ──────────────────
        input_frame = QFrame()
        input_frame.setFrameShape(QFrame.Shape.StyledPanel)
        in_layout = QVBoxLayout(input_frame)
        in_layout.setContentsMargins(10,10,10,8)
        in_layout.setSpacing(6)

        hdr = QHBoxLayout()
        lbl = QLabel("German Text")
        lbl.setFont(QFont("", 10, QFont.Weight.Bold))
        lbl.setStyleSheet("color:#1A1A1A;")
        hdr.addWidget(lbl)
        hdr.addStretch()
        self.load_btn = QPushButton("Load →")
        self.load_btn.setFixedHeight(26)
        self.load_btn.clicked.connect(self._switch_to_reader)
        hdr.addWidget(self.load_btn)
        self.edit_btn = QPushButton("Edit text")
        self.edit_btn.setFixedHeight(26)
        self.edit_btn.setVisible(False)
        self.edit_btn.clicked.connect(self._switch_to_editor)
        hdr.addWidget(self.edit_btn)
        in_layout.addLayout(hdr)

        self.editor = QTextEdit()
        self.editor.setPlaceholderText(
            "Paste your German text here, then click 'Load →'\n\n"
            "Example:\nHallo meine liebe kleiner."
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
        left_split.addWidget(self.lookup)
        left_split.setSizes([300, 380])

        main_split.addWidget(left_split)
        self.vocab_panel = VocabPanel(self.db)
        main_split.addWidget(self.vocab_panel)
        main_split.setSizes([750, 420])
        root.addWidget(main_split, 1)   # stretch=1 so it fills all available space

        self.status = QLabel("Ready — paste German text and click 'Load →'")
        self.status.setFixedHeight(20)
        self.status.setStyleSheet(
            "font-size:11px; color:#555; padding:0 4px; "
            "border-top: 1px solid #E0E0EC; background: #F7F7FA;"
        )
        root.addWidget(self.status, 0)  # stretch=0 → only takes what it needs

    def _switch_to_reader(self):
        text = self.editor.toPlainText().strip()
        if not text: return
        self.reader.load_german_text(text)
        self.editor.setVisible(False)
        self.reader.setVisible(True)
        self.load_btn.setVisible(False)
        self.edit_btn.setVisible(True)
        self.status.setText("Click a word for full breakdown  |  Drag to select a phrase")

    def _switch_to_editor(self):
        self.editor.setVisible(True)
        self.reader.setVisible(False)
        self.load_btn.setVisible(True)
        self.edit_btn.setVisible(False)

    def _on_text_selected(self, text, context, is_phrase):
        self.status.setText("Translating…" if is_phrase else f"Looking up '{text}'…")
        self.lookup.show_loading(text, is_phrase)
        if self.worker and self.worker.isRunning():
            print("Worker busy, skipping")
            return
        self.worker = LookupWorker(text, context, is_phrase)
        self.worker.result_ready.connect(self._on_lookup_result)
        self.worker.error.connect(self._on_lookup_error)
        self.worker.start()

    def _on_lookup_result(self, data):
        already = self.db.exists(data.get("word",""))
        self.lookup.show_result(data, already)
        src = data.get("source","")
        self.status.setText(f"Done: '{data.get('word','')}'" + (f"  [{src}]" if src else ""))

    def _on_lookup_error(self, msg):
        self.lookup.show_error(msg)
        self.status.setText(f"Error: {msg}")

    def _on_save(self, data):
        self.db.save(data)
        self.lookup.mark_saved()
        self.vocab_panel.refresh()
        self.status.setText(f"Saved: {data.get('word','')}")


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