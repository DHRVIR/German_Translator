"""
German Learning Assistant — PyQt6 Desktop App
Requirements: pip install PyQt6 deep-translator anthropic
"""

import sys
import re
import sqlite3
import threading
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QTextEdit, QTextBrowser, QPushButton, QLabel,
    QTableWidget, QTableWidgetItem, QHeaderView, QFrame,
    QLineEdit, QMessageBox, QSizePolicy, QScrollArea
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import (
    QFont, QColor, QTextCursor, QTextCharFormat, QPalette,
    QTextDocument, QCursor
)

# ── optional: set your Anthropic API key here or via env var ANTHROPIC_API_KEY ──
ANTHROPIC_API_KEY = ""   # leave empty to use env var

DB_PATH = Path.home() / ".german_learner_vocab.db"

# ─────────────────────────────────────────────
#  Database
# ─────────────────────────────────────────────
class VocabDB:
    def __init__(self):
        self.conn = sqlite3.connect(str(DB_PATH))
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS vocab (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                word     TEXT UNIQUE,
                pos      TEXT,
                meaning  TEXT,
                example_de TEXT,
                example_en TEXT,
                added_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    def save(self, word, pos, meaning, example_de, example_en):
        try:
            self.conn.execute(
                "INSERT OR IGNORE INTO vocab (word, pos, meaning, example_de, example_en) VALUES (?,?,?,?,?)",
                (word, pos, meaning, example_de, example_en)
            )
            self.conn.commit()
            return True
        except Exception:
            return False

    def all_words(self):
        cur = self.conn.execute(
            "SELECT id, word, pos, meaning, example_de, example_en FROM vocab ORDER BY added_at DESC"
        )
        return cur.fetchall()

    def delete(self, word_id):
        self.conn.execute("DELETE FROM vocab WHERE id=?", (word_id,))
        self.conn.commit()

    def exists(self, word):
        cur = self.conn.execute("SELECT 1 FROM vocab WHERE LOWER(word)=LOWER(?)", (word,))
        return cur.fetchone() is not None


# ─────────────────────────────────────────────
#  Background worker for API lookup
# ─────────────────────────────────────────────
class LookupWorker(QThread):
    result_ready = pyqtSignal(dict)
    error        = pyqtSignal(str)

    def __init__(self, text, is_phrase, api_key):
        super().__init__()
        self.text      = text
        self.is_phrase = is_phrase
        self.api_key   = api_key

    def run(self):
        if self.is_phrase:
            self._translate_phrase()
        else:
            self._lookup_word()

    def _translate_phrase(self):
        try:
            from deep_translator import GoogleTranslator
            translation = GoogleTranslator(source="de", target="en").translate(self.text)
            data = {
                "word":      self.text,
                "pos":       "phrase",
                "meaning":   translation,
                "example_de": "",
                "example_en": "",
                "is_phrase":  True,
            }
            self.result_ready.emit(data)
        except Exception as e:
            self.error.emit(f"Translation failed: {e}")

    def _lookup_word(self):
        try:
            import anthropic, json
            key = self.api_key or None
            client = anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()
            msg = client.messages.create(
                model="claude-opus-4-5",
                max_tokens=500,
                system=(
                    "You are a German language tutor. Respond ONLY with a raw JSON object "
                    "(no markdown, no backticks) with these fields: "
                    "word, pos (grammar label e.g. 'noun (neuter)', 'verb', 'adjective'), "
                    "meaning (1-2 sentence English explanation; include der/die/das for nouns), "
                    "example_de (one natural German example sentence), "
                    "example_en (English translation of the example)."
                ),
                messages=[{"role": "user", "content": self.text}]
            )
            text = "".join(b.text for b in msg.content if hasattr(b, "text"))
            data = json.loads(text.strip())
            data["is_phrase"] = False
            self.result_ready.emit(data)
            return
        except Exception:
            pass

        try:
            from deep_translator import GoogleTranslator
            translation = GoogleTranslator(source="de", target="en").translate(self.text)
            data = {
                "word":       self.text,
                "pos":        "—",
                "meaning":    translation,
                "example_de": "",
                "example_en": "",
                "is_phrase":  False,
            }
            self.result_ready.emit(data)
        except Exception as e2:
            self.error.emit(f"Lookup failed: {e2}")


# ─────────────────────────────────────────────
#  Clickable text area
# ─────────────────────────────────────────────
class GermanTextBrowser(QTextBrowser):
    text_selected = pyqtSignal(str, bool)   # (text, is_phrase)

    def __init__(self):
        super().__init__()
        self.setReadOnly(True)
        self.setOpenLinks(False)
        self.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse |
            Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        self.viewport().setCursor(QCursor(Qt.CursorShape.IBeamCursor))

    def mouseReleaseEvent(self, e):
        super().mouseReleaseEvent(e)
        if e.button() != Qt.MouseButton.LeftButton:
            return

        cursor = self.textCursor()
        selected = cursor.selectedText().strip()

        # Multi-word selection
        if selected and len(selected.split()) > 1:
            selected = selected.replace("\u2029", " ").replace("\u2028", " ")
            selected = re.sub(r"\s+", " ", selected).strip()
            if re.search(r"[a-zA-ZäöüÄÖÜß]", selected):
                self._highlight_cursor(cursor)
                self.text_selected.emit(selected, True)
            return

        # Single word — use word-under-cursor for accuracy
        wc = self.cursorForPosition(e.pos())
        wc.select(QTextCursor.SelectionType.WordUnderCursor)
        word = wc.selectedText().strip()
        word = re.sub(r"[^\w\-äöüÄÖÜß]", "", word, flags=re.UNICODE)
        if word and re.search(r"[a-zA-ZäöüÄÖÜß]", word):
            self._highlight_cursor(wc)
            self.text_selected.emit(word, False)

    def _highlight_cursor(self, cursor):
        doc = self.document()
        clear_fmt = QTextCharFormat()
        full = QTextCursor(doc)
        full.select(QTextCursor.SelectionType.Document)
        full.setCharFormat(clear_fmt)
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
        layout.setSpacing(6)

        lbl = QLabel("Word Lookup")
        lbl.setFont(QFont("", 10, QFont.Weight.Bold))
        layout.addWidget(lbl)

        self.word_label = QLabel("Click any word in the text above")
        self.word_label.setFont(QFont("", 16, QFont.Weight.Bold))
        layout.addWidget(self.word_label)

        self.pos_label = QLabel("")
        self.pos_label.setStyleSheet("color: grey; font-size: 11px;")
        layout.addWidget(self.pos_label)

        self.meaning_label = QLabel("")
        self.meaning_label.setWordWrap(True)
        self.meaning_label.setFont(QFont("", 12))
        layout.addWidget(self.meaning_label)

        # Example box
        self.example_frame = QFrame()
        self.example_frame.setStyleSheet(
            "QFrame { border-left: 3px solid #7F77DD; background: #F4F4FB; "
            "border-radius: 0 6px 6px 0; padding: 4px; }"
        )
        ex_layout = QVBoxLayout(self.example_frame)
        ex_layout.setContentsMargins(10, 6, 8, 6)
        ex_layout.setSpacing(2)
        self.example_de = QLabel("")
        self.example_de.setWordWrap(True)
        self.example_de.setFont(QFont("", 11, QFont.Weight.Bold))
        self.example_en = QLabel("")
        self.example_en.setWordWrap(True)
        self.example_en.setStyleSheet("color: #555; font-size: 11px;")
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
        display = text if len(text) <= 40 else text[:38] + "…"
        self.word_label.setText(display)
        self.pos_label.setText("translating…" if is_phrase else "looking up…")
        self.meaning_label.setText("")
        self.example_frame.setVisible(False)
        self.save_btn.setEnabled(False)
        self.save_btn.setText("+ Save to Vocabulary")

    def show_result(self, data, already_saved):
        self.current_data = data
        is_phrase = data.get("is_phrase", False)
        word = data.get("word", "")
        display = word if len(word) <= 40 else word[:38] + "…"
        self.word_label.setText(display)
        if is_phrase:
            # Show original German in example box, translation as meaning
            self.pos_label.setText("phrase translation")
            self.meaning_label.setText(data.get("meaning", ""))
            self.example_de.setText(word)
            self.example_en.setText(data.get("meaning", ""))
            self.example_frame.setVisible(True)
        else:
            self.pos_label.setText(data.get("pos", ""))
            self.meaning_label.setText(data.get("meaning", ""))
            ex_de = data.get("example_de", "")
            ex_en = data.get("example_en", "")
            if ex_de:
                self.example_de.setText(ex_de)
                self.example_en.setText(ex_en)
                self.example_frame.setVisible(True)
            else:
                self.example_frame.setVisible(False)
        if already_saved:
            self.save_btn.setText("✓ Already saved")
            self.save_btn.setEnabled(False)
        else:
            btn_text = "+ Save phrase" if is_phrase else "+ Save to Vocabulary"
            self.save_btn.setText(btn_text)
            self.save_btn.setEnabled(True)

    def show_error(self, msg):
        self.word_label.setText("Error")
        self.pos_label.setText("")
        self.meaning_label.setText(msg)
        self.example_frame.setVisible(False)
        self.save_btn.setEnabled(False)

    def mark_saved(self):
        self.save_btn.setText("✓ Saved!")
        self.save_btn.setEnabled(False)

    def _on_save(self):
        if self.current_data:
            self.save_requested.emit(self.current_data)


# ─────────────────────────────────────────────
#  Vocabulary Table
# ─────────────────────────────────────────────
class VocabPanel(QFrame):
    def __init__(self, db: VocabDB):
        super().__init__()
        self.db = db
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
        header.addWidget(lbl)
        header.addStretch()
        self.count_label = QLabel("0 words")
        self.count_label.setStyleSheet("color: grey; font-size: 11px;")
        header.addWidget(self.count_label)
        layout.addLayout(header)

        self.search = QLineEdit()
        self.search.setPlaceholderText("Filter…")
        self.search.setFixedHeight(28)
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
            w_item.setData(Qt.ItemDataRole.UserRole, row_id)
            self.table.setItem(r, 0, w_item)
            short = (meaning or "").split(".")[0] + "."
            m_item = QTableWidgetItem(short)
            m_item.setToolTip(f"{meaning}\n\n{ex_de}\n{ex_en}")
            self.table.setItem(r, 1, m_item)
            del_btn = QPushButton("×")
            del_btn.setFixedSize(QSize(24, 24))
            del_btn.setStyleSheet("color: #cc3333; font-weight: bold; border: none;")
            del_btn.clicked.connect(lambda _, rid=row_id: self._delete(rid))
            self.table.setCellWidget(r, 2, del_btn)
        self.count_label.setText(f"{len(rows)} word{'s' if len(rows)!=1 else ''}")

    def filter_words(self, text):
        filtered = [
            r for r in self._all_rows
            if text.lower() in (r[1] or "").lower() or text.lower() in (r[3] or "").lower()
        ]
        self._render(filtered)

    def _delete(self, row_id):
        self.db.delete(row_id)
        self.refresh()

    def export_csv(self):
        from PyQt6.QtWidgets import QFileDialog
        import csv
        path, _ = QFileDialog.getSaveFileName(self, "Export Vocabulary", "vocab.csv", "CSV files (*.csv)")
        if path:
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["Word", "Part of Speech", "Meaning", "Example (DE)", "Example (EN)"])
                for row_id, word, pos, meaning, ex_de, ex_en in self.db.all_words():
                    w.writerow([word, pos, meaning, ex_de, ex_en])
            QMessageBox.information(self, "Exported", f"Saved to {path}")


# ─────────────────────────────────────────────
#  API Key Dialog
# ─────────────────────────────────────────────
class ApiKeyBar(QFrame):
    def __init__(self, initial_key=""):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)
        lbl = QLabel("Anthropic API key:")
        lbl.setStyleSheet("font-size: 11px; color: grey;")
        layout.addWidget(lbl)
        self.key_input = QLineEdit(initial_key)
        self.key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.key_input.setPlaceholderText("sk-ant-… (optional if set via env var)")
        self.key_input.setFixedHeight(26)
        self.key_input.setStyleSheet("font-size: 11px;")
        layout.addWidget(self.key_input)

    def get_key(self):
        return self.key_input.text().strip()


# ─────────────────────────────────────────────
#  Main Window
# ─────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.db = VocabDB()
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

        # API key bar
        self.api_bar = ApiKeyBar(ANTHROPIC_API_KEY)
        root.addWidget(self.api_bar)

        # Main splitter: left | right
        main_split = QSplitter(Qt.Orientation.Horizontal)

        # Left: top text area | bottom lookup
        left_split = QSplitter(Qt.Orientation.Vertical)

        # ── Input / text display ──────────────────
        input_frame = QFrame()
        input_frame.setFrameShape(QFrame.Shape.StyledPanel)
        in_layout = QVBoxLayout(input_frame)
        in_layout.setContentsMargins(10, 10, 10, 8)
        in_layout.setSpacing(6)
        hdr = QHBoxLayout()
        lbl = QLabel("German Text")
        lbl.setFont(QFont("", 10, QFont.Weight.Bold))
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
        in_layout.addWidget(self.editor)

        self.reader = GermanTextBrowser()
        self.reader.setFont(QFont("", 13))
        self.reader.setStyleSheet("QTextBrowser { color: #1A1A1A; background-color: #FFFFFF; }")
        self.reader.setVisible(False)
        self.reader.text_selected.connect(self._on_text_selected)
        in_layout.addWidget(self.reader)

        left_split.addWidget(input_frame)

        # ── Lookup panel ──────────────────────────
        self.lookup = LookupPanel()
        self.lookup.save_requested.connect(self._on_save)
        left_split.addWidget(self.lookup)

        left_split.setSizes([320, 220])
        main_split.addWidget(left_split)

        # ── Vocabulary panel ──────────────────────
        self.vocab_panel = VocabPanel(self.db)
        main_split.addWidget(self.vocab_panel)

        main_split.setSizes([700, 380])
        root.addWidget(main_split)

        # Status bar
        self.status = QLabel("Ready. Paste German text and click 'Load →'")
        self.status.setStyleSheet("font-size: 11px; color: grey; padding: 2px 4px;")
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
        self.status.setText("Click a word to look it up  |  Select multiple words to translate a phrase")

    def _switch_to_editor(self):
        self.editor.setVisible(True)
        self.reader.setVisible(False)
        self.load_btn.setVisible(True)
        self.edit_btn.setVisible(False)

    def _on_text_selected(self, text, is_phrase):
        label = "Translating phrase…" if is_phrase else f"Looking up: {text}…"
        self.status.setText(label)
        self.lookup.show_loading(text, is_phrase)
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
        self.worker = LookupWorker(text, is_phrase, self.api_bar.get_key())
        self.worker.result_ready.connect(self._on_lookup_result)
        self.worker.error.connect(self._on_lookup_error)
        self.worker.start()

    def _on_lookup_result(self, data):
        already = self.db.exists(data.get("word", ""))
        self.lookup.show_result(data, already)
        self.status.setText(f"Looked up: {data.get('word','')}")

    def _on_lookup_error(self, msg):
        self.lookup.show_error(msg)
        self.status.setText("Lookup failed")

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
    palette.setColor(QPalette.ColorRole.Button,          QColor("#ECECEC"))
    palette.setColor(QPalette.ColorRole.ButtonText,      QColor("#1A1A1A"))
    palette.setColor(QPalette.ColorRole.Highlight,       QColor("#4A7DC8"))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#FFFFFF"))
    app.setPalette(palette)

    win = MainWindow()
    win.show()
    sys.exit(app.exec())