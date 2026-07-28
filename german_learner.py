"""
German Learning Assistant — PyQt6 Desktop App
Requirements: pip install PyQt6 deep-translator google-genai python-dotenv
"""

import sys
import re
import sqlite3
from pathlib import Path
from google.genai import types
from deep_translator import GoogleTranslator
import json
from google import genai
from deep_translator import GoogleTranslator
from PyQt6.QtWidgets import QFileDialog
import csv

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QTextEdit, QTextBrowser, QPushButton, QLabel,
    QTableWidget, QTableWidgetItem, QHeaderView, QFrame,
    QLineEdit, QMessageBox, QSizePolicy
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import (
    QFont, QColor, QTextCursor, QTextCharFormat, QPalette, QCursor
)

from dotenv import load_dotenv
import os

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

DB_PATH = Path.home() / ".german_learner_vocab.db"


# ─────────────────────────────────────────────
#  Database
# ─────────────────────────────────────────────
class VocabDB:
    def __init__(self):
        self.conn = sqlite3.connect(str(DB_PATH))
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS vocab (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                word       TEXT UNIQUE,
                pos        TEXT,
                meaning    TEXT,
                example_de TEXT,
                example_en TEXT,
                added_at   DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    def save(self, word, pos, meaning, example_de, example_en):
        try:
            self.conn.execute(
                "INSERT OR IGNORE INTO vocab "
                "(word, pos, meaning, example_de, example_en) VALUES (?,?,?,?,?)",
                (word, pos, meaning, example_de, example_en)
            )
            self.conn.commit()
            return True
        except Exception:
            return False

    def all_words(self):
        cur = self.conn.execute(
            "SELECT id, word, pos, meaning, example_de, example_en "
            "FROM vocab ORDER BY added_at DESC"
        )
        return cur.fetchall()

    def delete(self, word_id):
        self.conn.execute("DELETE FROM vocab WHERE id=?", (word_id,))
        self.conn.commit()

    def exists(self, word):
        cur = self.conn.execute(
            "SELECT 1 FROM vocab WHERE LOWER(word)=LOWER(?)", (word,)
        )
        return cur.fetchone() is not None


# ─────────────────────────────────────────────
#  Background worker
# ─────────────────────────────────────────────
class LookupWorker(QThread):
    result_ready = pyqtSignal(dict)
    error        = pyqtSignal(str)

    def __init__(self, text, context_sentence, is_phrase):
        super().__init__()
        self.text             = text
        self.context_sentence = context_sentence   # full sentence the word sits in
        self.is_phrase        = is_phrase

    def run(self):
        try:
            if self.is_phrase:
                self._translate_phrase()
            else:
                self._lookup_word()
        except Exception as e:
            self.error.emit(f"Unexpected error: {e}")

    # ── Phrase: deep-translator only (fast, no API needed) ────────────────
    def _translate_phrase(self):
        try:
            translation = GoogleTranslator(source="de", target="en").translate(self.text)
            self.result_ready.emit({
                "word":       self.text,
                "pos":        "phrase",
                "meaning":    translation,
                "example_de": "",
                "example_en": "",
                "is_phrase":  True,
                "source":     "Google Translate",
            })
        except Exception as e:
            self.error.emit(f"Translation failed: {e}")

    # ── Single word: try Gemini (with context), fall back to deep-translator ──
    def _lookup_word(self):
        if GEMINI_API_KEY:
            result = self._try_gemini()
            if result:
                self.result_ready.emit(result)
                return

        self._fallback_translate()

    def _try_gemini(self):
        """
        Call Gemini with a hard 8-second timeout using concurrent.futures.
        Returns a result dict on success, None on timeout or any failure.
        """


        def _call():
            client = genai.Client(api_key=GEMINI_API_KEY)

            if self.context_sentence and self.context_sentence.strip() != self.text.strip():
                context_block = (
                    f'The word appears in this German sentence:\n'
                    f'"{self.context_sentence}"\n\n'
                    f'Use the sentence to determine the correct meaning in context.\n\n'
                )
            else:
                context_block = ""

            prompt = (
                "You are a German language tutor.\n"
                "Return ONLY a valid raw JSON object — no markdown, no backticks.\n\n"
                "Fields:\n"
                "  word        — the word as given\n"
                "  pos         — grammar label e.g. 'noun (der)', 'verb', 'adjective'\n"
                "  meaning     — concise English explanation in context; include der/die/das for nouns\n"
                "  example_de  — the context sentence if provided, otherwise a natural German example\n"
                "  example_en  — English translation of example_de\n\n"
                + context_block +
                f"German word to look up: {self.text}"
            )

            response = client.models.generate_content(
                model="gemma-4-26b-a4b-it",
                contents=prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(
                        thinking_level="minimal",
                        include_thoughts=False
                    )
                )
            )
            # return response.text.strip()

            parts = []
            for part in response.candidates[0].content.parts:
                if getattr(part, "thought", False):
                    continue

                if getattr(part, "text", None):
                    parts.append(part.text)

            raw = "\n".join(parts).strip()
            
            return raw

        try:
            # with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            #     future = ex.submit(_call)
            #     try:
            #         raw = future.result(timeout=8)   # give up after 8 s
            #     except concurrent.futures.TimeoutError:
            #         return None   # → fallback to deep-translator

            # executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

            # future = executor.submit(_call)

            # try:
            #     raw = future.result(timeout=8)
            # except concurrent.futures.TimeoutError:
            #     future.cancel()
            #     executor.shutdown(wait=False)
            #     return None
            # finally:
            #     executor.shutdown(wait=False)

            raw = _call()

            raw = re.sub(r"^```[a-z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)

            match = re.search(r"\{.*\}", raw, re.DOTALL)

            if not match:
                raise Exception("Gemini returned no JSON")

            data = json.loads(match.group(0))
            data["is_phrase"] = False
            data["source"]    = "Gemini"
            return data

        except Exception as e:
            print("Gemini error:", e)
            return None

    def _fallback_translate(self):
        try:
            translation = GoogleTranslator(source="de", target="en").translate(self.text)
            self.result_ready.emit({
                "word":       self.text,
                "pos":        "translation",
                "meaning":    translation,
                "example_de": self.context_sentence or "",
                "example_en": "",
                "is_phrase":  False,
                "source":     "Google Translate",
            })
        except Exception as e:
            self.error.emit(f"Lookup failed: {e}")


# ─────────────────────────────────────────────
#  Clickable text browser
# ─────────────────────────────────────────────
class GermanTextBrowser(QTextBrowser):
    # word/phrase text, context sentence, is_phrase
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
        self.setStyleSheet(
            "QTextBrowser { color: #1A1A1A; background-color: #FFFFFF; }"
        )

    def mouseReleaseEvent(self, e):
        super().mouseReleaseEvent(e)
        if e.button() != Qt.MouseButton.LeftButton:
            return

        cursor = self.textCursor()
        selected = cursor.selectedText().strip()

        # ── Multi-word selection → phrase translation ──────────────────────
        if selected and len(selected.split()) > 1:
            selected = selected.replace("\u2029", " ").replace("\u2028", " ")
            selected = re.sub(r"\s+", " ", selected).strip()
            if re.search(r"[a-zA-ZäöüÄÖÜß]", selected):
                self._apply_highlight(cursor)
                # For phrases the phrase itself is the context
                self.text_selected.emit(selected, selected, True)
            return

        # ── Single word click → contextual lookup ─────────────────────────
        wc = self.cursorForPosition(e.pos())
        wc.select(QTextCursor.SelectionType.WordUnderCursor)
        word = wc.selectedText().strip()
        word = re.sub(r"[^\w\-äöüÄÖÜß]", "", word, flags=re.UNICODE)
        if not word or not re.search(r"[a-zA-ZäöüÄÖÜß]", word):
            return

        # Extract the sentence the word lives in
        context = self._extract_sentence(wc)
        self._apply_highlight(wc)
        self.text_selected.emit(word, context, False)

    def _extract_sentence(self, cursor):
        """Return the sentence (or paragraph line) containing the cursor."""
        # Get the block (paragraph) text — fast and reliable
        block_text = cursor.block().text().strip()
        if not block_text:
            return ""

        # Split block into sentences and return the one containing the word
        pos_in_block = cursor.positionInBlock()
        sentences = re.split(r"(?<=[.!?])\s+", block_text)
        char_count = 0
        for sent in sentences:
            char_count += len(sent) + 1   # +1 for the space/punctuation consumed
            if char_count >= pos_in_block:
                return sent.strip()
        return block_text   # fallback: whole block

    def _apply_highlight(self, cursor):
        full = QTextCursor(self.document())
        full.select(QTextCursor.SelectionType.Document)
        full.setCharFormat(QTextCharFormat())        # clear all
        fmt = QTextCharFormat()
        fmt.setBackground(QColor("#BFD7F5"))
        fmt.setForeground(QColor("#0A3060"))
        cursor.setCharFormat(fmt)

    def load_german_text(self, text):
        self.setPlainText(text)


# ─────────────────────────────────────────────
#  Lookup Panel
# ─────────────────────────────────────────────
class LookupPanel(QFrame):
    save_requested = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.current_data = None
        self._setup_ui()

    def _setup_ui(self):
        self.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(5)

        # Header row: "Word Lookup" + source badge
        header_row = QHBoxLayout()
        lbl = QLabel("Word Lookup")
        lbl.setFont(QFont("", 10, QFont.Weight.Bold))
        lbl.setStyleSheet("color: #1A1A1A;")
        header_row.addWidget(lbl)
        header_row.addStretch()
        self.source_badge = QLabel("")
        self.source_badge.setStyleSheet(
            "color: #FFFFFF; background: #7F77DD; border-radius: 8px; "
            "padding: 1px 8px; font-size: 10px;"
        )
        self.source_badge.setVisible(False)
        header_row.addWidget(self.source_badge)
        layout.addLayout(header_row)

        self.word_label = QLabel("Click any word in the text above")
        self.word_label.setFont(QFont("", 15, QFont.Weight.Bold))
        self.word_label.setWordWrap(True)
        self.word_label.setStyleSheet("color: #1A1A1A;")
        layout.addWidget(self.word_label)

        self.pos_label = QLabel("")
        self.pos_label.setStyleSheet("color: #666666; font-size: 11px;")
        layout.addWidget(self.pos_label)

        self.meaning_label = QLabel("")
        self.meaning_label.setWordWrap(True)
        self.meaning_label.setFont(QFont("", 12))
        self.meaning_label.setStyleSheet("color: #1A1A1A;")
        layout.addWidget(self.meaning_label)

        # Example / context box
        self.example_frame = QFrame()
        self.example_frame.setStyleSheet(
            "QFrame { border-left: 3px solid #7F77DD; background: #F4F4FB; "
            "border-radius: 0 6px 6px 0; }"
        )
        ex_layout = QVBoxLayout(self.example_frame)
        ex_layout.setContentsMargins(10, 6, 8, 6)
        ex_layout.setSpacing(2)
        self.ex_header = QLabel("Example")
        self.ex_header.setStyleSheet("color: #888888; font-size: 10px;")
        self.example_de = QLabel("")
        self.example_de.setWordWrap(True)
        self.example_de.setFont(QFont("", 11, QFont.Weight.Bold))
        self.example_de.setStyleSheet("color: #1A1A1A;")
        self.example_en = QLabel("")
        self.example_en.setWordWrap(True)
        self.example_en.setStyleSheet("color: #444444; font-size: 11px;")
        ex_layout.addWidget(self.ex_header)
        ex_layout.addWidget(self.example_de)
        ex_layout.addWidget(self.example_en)
        layout.addWidget(self.example_frame)
        self.example_frame.setVisible(False)

        self.save_btn = QPushButton("+ Save to Vocabulary")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self._on_save)
        self.save_btn.setFixedHeight(30)
        layout.addWidget(self.save_btn)
        layout.addStretch()

    def show_loading(self, text, is_phrase=False):
        self.current_data = None
        display = text if len(text) <= 44 else text[:42] + "…"
        self.word_label.setText(display)
        self.pos_label.setText("translating…" if is_phrase else "looking up…")
        self.meaning_label.setText("")
        self.example_frame.setVisible(False)
        self.source_badge.setVisible(False)
        self.save_btn.setEnabled(False)
        self.save_btn.setText("+ Save to Vocabulary")

    def show_result(self, data, already_saved):
        self.current_data = data
        is_phrase = data.get("is_phrase", False)
        source    = data.get("source", "")
        word      = data.get("word", "")

        display = word if len(word) <= 44 else word[:42] + "…"
        self.word_label.setText(display)

        # Source badge
        if source:
            color = "#5B8DD9" if source == "Google Translate" else "#7F77DD"
            self.source_badge.setStyleSheet(
                f"color: #FFFFFF; background: {color}; border-radius: 8px; "
                "padding: 1px 8px; font-size: 10px;"
            )
            self.source_badge.setText(f"via {source}")
            self.source_badge.setVisible(True)
        else:
            self.source_badge.setVisible(False)

        if is_phrase:
            self.pos_label.setText("phrase translation")
            self.meaning_label.setText(data.get("meaning", ""))
            self.ex_header.setText("Selected text")
            self.example_de.setText(word)
            self.example_en.setText(data.get("meaning", ""))
            self.example_frame.setVisible(True)
        else:
            self.pos_label.setText(data.get("pos", ""))
            self.meaning_label.setText(data.get("meaning", ""))
            ex_de = data.get("example_de", "")
            ex_en = data.get("example_en", "")
            if ex_de:
                self.ex_header.setText("Context / example")
                self.example_de.setText(ex_de)
                self.example_en.setText(ex_en)
                self.example_frame.setVisible(True)
            else:
                self.example_frame.setVisible(False)

        if already_saved:
            self.save_btn.setText("✓ Already saved")
            self.save_btn.setEnabled(False)
        else:
            self.save_btn.setText("+ Save phrase" if is_phrase else "+ Save to Vocabulary")
            self.save_btn.setEnabled(True)

    def show_error(self, msg):
        self.word_label.setText("Error")
        self.pos_label.setText("")
        self.meaning_label.setText(msg)
        self.example_frame.setVisible(False)
        self.source_badge.setVisible(False)
        self.save_btn.setEnabled(False)

    def mark_saved(self):
        self.save_btn.setText("✓ Saved!")
        self.save_btn.setEnabled(False)

    def _on_save(self):
        if self.current_data:
            self.save_requested.emit(self.current_data)


# ─────────────────────────────────────────────
#  Vocabulary Panel
# ─────────────────────────────────────────────
class VocabPanel(QFrame):
    def __init__(self, db: VocabDB):
        super().__init__()
        self.db       = db
        self._all_rows = []
        self._setup_ui()
        self.refresh()

    def _setup_ui(self):
        self.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        header = QHBoxLayout()
        lbl = QLabel("Saved Vocabulary")
        lbl.setFont(QFont("", 10, QFont.Weight.Bold))
        lbl.setStyleSheet("color: #1A1A1A;")
        header.addWidget(lbl)
        header.addStretch()
        self.count_label = QLabel("0 words")
        self.count_label.setStyleSheet("color: #666666; font-size: 11px;")
        header.addWidget(self.count_label)
        layout.addLayout(header)

        self.search = QLineEdit()
        self.search.setPlaceholderText("Filter…")
        self.search.setFixedHeight(28)
        self.search.setStyleSheet("color: #1A1A1A; background: #FFFFFF;")
        self.search.textChanged.connect(self.filter_words)
        layout.addWidget(self.search)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Word", "Meaning", ""])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(2, 28)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(False)
        self.table.setFont(QFont("", 11))
        self.table.setStyleSheet("""
            QTableWidget            { color: #1A1A1A; background: #FFFFFF; }
            QTableWidget::item      { color: #1A1A1A; }
            QTableWidget::item:alternate { background: #F3F3F8; color: #1A1A1A; }
            QTableWidget::item:selected  { background: #BFD7F5; color: #0A3060; }
            QHeaderView::section    { color: #1A1A1A; background: #ECECEC; }
        """)
        layout.addWidget(self.table)

        export_btn = QPushButton("Export to CSV")
        export_btn.setFixedHeight(28)
        export_btn.clicked.connect(self.export_csv)
        layout.addWidget(export_btn)

    def refresh(self):
        self._all_rows = self.db.all_words()
        self._render(self._all_rows)

    def _render(self, rows):
        self.table.setRowCount(0)
        for row_id, word, pos, meaning, ex_de, ex_en in rows:
            r = self.table.rowCount()
            self.table.insertRow(r)

            w_item = QTableWidgetItem(word)
            w_item.setFont(QFont("", 11, QFont.Weight.Bold))
            w_item.setForeground(QColor("#1A1A1A"))
            w_item.setData(Qt.ItemDataRole.UserRole, row_id)
            self.table.setItem(r, 0, w_item)

            short  = (meaning or "").split(".")[0] + "."
            m_item = QTableWidgetItem(short)
            m_item.setForeground(QColor("#333333"))
            m_item.setToolTip(f"{meaning}\n\n{ex_de}\n{ex_en}")
            self.table.setItem(r, 1, m_item)

            del_btn = QPushButton("×")
            del_btn.setFixedSize(QSize(24, 24))
            del_btn.setStyleSheet(
                "color: #cc3333; font-weight: bold; border: none; background: transparent;"
            )
            del_btn.clicked.connect(lambda _, rid=row_id: self._delete(rid))
            self.table.setCellWidget(r, 2, del_btn)

        self.count_label.setText(f"{len(rows)} word{'s' if len(rows) != 1 else ''}")

    def filter_words(self, text):
        filtered = [
            r for r in self._all_rows
            if text.lower() in (r[1] or "").lower()
            or text.lower() in (r[3] or "").lower()
        ]
        self._render(filtered)

    def _delete(self, row_id):
        self.db.delete(row_id)
        self.refresh()

    def export_csv(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Vocabulary", "vocab.csv", "CSV files (*.csv)"
        )
        if path:
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["Word", "Part of Speech", "Meaning", "Example (DE)", "Example (EN)"])
                for row_id, word, pos, meaning, ex_de, ex_en in self.db.all_words():
                    w.writerow([word, pos, meaning, ex_de, ex_en])
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
        self.resize(1100, 680)
        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 8, 10, 8)
        root.setSpacing(6)

        main_split = QSplitter(Qt.Orientation.Horizontal)
        left_split = QSplitter(Qt.Orientation.Vertical)

        # ── Text input / reader ───────────────────────────────────────────
        input_frame = QFrame()
        input_frame.setFrameShape(QFrame.Shape.StyledPanel)
        in_layout = QVBoxLayout(input_frame)
        in_layout.setContentsMargins(10, 10, 10, 8)
        in_layout.setSpacing(6)

        hdr = QHBoxLayout()
        lbl = QLabel("German Text")
        lbl.setFont(QFont("", 10, QFont.Weight.Bold))
        lbl.setStyleSheet("color: #1A1A1A;")
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
            "Example:\nDas Buch liegt auf dem Tisch. Die Sonne scheint hell."
        )
        self.editor.setFont(QFont("", 13))
        self.editor.setStyleSheet(
            "QTextEdit { color: #1A1A1A; background-color: #FFFFFF; }"
        )
        in_layout.addWidget(self.editor)

        self.reader = GermanTextBrowser()
        self.reader.setFont(QFont("", 13))
        self.reader.setVisible(False)
        self.reader.text_selected.connect(self._on_text_selected)
        in_layout.addWidget(self.reader)

        left_split.addWidget(input_frame)

        # ── Lookup panel ─────────────────────────────────────────────────
        self.lookup = LookupPanel()
        self.lookup.save_requested.connect(self._on_save)
        left_split.addWidget(self.lookup)

        left_split.setSizes([320, 220])
        main_split.addWidget(left_split)

        # ── Vocab panel ──────────────────────────────────────────────────
        self.vocab_panel = VocabPanel(self.db)
        main_split.addWidget(self.vocab_panel)

        main_split.setSizes([700, 380])
        root.addWidget(main_split)

        # Status bar
        self.status = QLabel("Ready — paste German text and click 'Load →'")
        self.status.setStyleSheet("font-size: 11px; color: #555555; padding: 2px 4px;")
        root.addWidget(self.status)

    def _switch_to_reader(self):
        text = self.editor.toPlainText().strip()
        if not text:
            return
        self.reader.load_german_text(text)
        self.editor.setVisible(False)
        self.reader.setVisible(True)
        self.load_btn.setVisible(False)
        self.edit_btn.setVisible(True)
        self.status.setText(
            "Click a word to look it up in context  |  "
            "Drag to select multiple words for phrase translation"
        )

    def _switch_to_editor(self):
        self.editor.setVisible(True)
        self.reader.setVisible(False)
        self.load_btn.setVisible(True)
        self.edit_btn.setVisible(False)

    def _on_text_selected(self, text, context, is_phrase):
        if is_phrase:
            self.status.setText(f"Translating phrase…")
        else:
            self.status.setText(f"Looking up '{text}' in context…")
        self.lookup.show_loading(text, is_phrase)

        if self.worker and self.worker.isRunning():
            print("Lookup already running")
            return

        self.worker = LookupWorker(text, context, is_phrase)
        self.worker.result_ready.connect(self._on_lookup_result)
        self.worker.error.connect(self._on_lookup_error)
        self.worker.start()

    def _on_lookup_result(self, data):
        already = self.db.exists(data.get("word", ""))
        self.lookup.show_result(data, already)
        src = data.get("source", "")
        self.status.setText(
            f"Done: '{data.get('word', '')}'"
            + (f"  [{src}]" if src else "")
        )

    def _on_lookup_error(self, msg):
        self.lookup.show_error(msg)
        self.status.setText(f"Error: {msg}")

    def _on_save(self, data):
        self.db.save(
            data.get("word", ""),
            data.get("pos", ""),
            data.get("meaning", ""),
            data.get("example_de", ""),
            data.get("example_en", ""),
        )
        self.lookup.mark_saved()
        self.vocab_panel.refresh()
        self.status.setText(f"Saved: {data.get('word', '')}")


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