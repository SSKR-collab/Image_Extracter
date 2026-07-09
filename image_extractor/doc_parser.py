import re
from image_extractor.base_analyzer import BaseAnalyzer


class DocParser(BaseAnalyzer):
    """
    Groups OCR words into lines and paragraphs using spatial bounding boxes.
    Performs sentence segmentation, typography heuristic audits,
    page structure margins calculations, and document/content classifications.
    """
    VERSION = "1.2.0"

    # Common English stop words to assist language heuristic
    ENGLISH_STOP_WORDS = {"the", "of", "and", "to", "a", "in", "is", "it", "you", "that", 
                          "he", "was", "for", "on", "are", "as", "with", "his", "they", "i"}

    def analyze(self, file_path: str, img, context: dict) -> dict:
        results = {
            "facts": {
                "paragraphs": [],
                "sentences": [],
                "typography": {},
                "page_structure": {},
                "statistics": {
                    "word_count": 0,
                    "line_count": 0,
                    "paragraph_count": 0,
                    "sentence_count": 0
                }
            },
            "indicators": [],
            "assessments": {},
            "errors": []
        }

        # Retrieve words from OCR plugin
        ocr_out = context.get("ocr_engine", {})
        words = ocr_out.get("facts", {}).get("words", [])
        raw_text = ocr_out.get("facts", {}).get("raw_text", "").strip()

        if not raw_text:
            # No text, document parsing is not applicable
            results["assessments"]["document_classification"] = {
                "document_type": "Non-Text Image (e.g., Photograph/Graphic)",
                "document_confidence": 0.90,
                "content_type": "Visual Only",
                "content_confidence": 0.90
            }
            return results

        # 1. Spatial Layout Reconstruction (Paragraph & Line grouping)
        paragraphs_reconstructed = []
        if words and any(w.get("bbox") for w in words):
            paragraphs_reconstructed = self._reconstruct_layout_spatially(words)
        else:
            # Fallback to plain text split if no bounding boxes are present
            paragraphs_reconstructed = self._reconstruct_layout_textually(raw_text)

        results["facts"]["paragraphs"] = paragraphs_reconstructed

        # 2. Sentence Segmentation
        sentences = self._segment_sentences(raw_text)
        results["facts"]["sentences"] = sentences

        # 3. Typography & Page Structure Heuristics
        results["facts"]["typography"] = self._extract_typography(words, paragraphs_reconstructed)
        results["facts"]["page_structure"] = self._extract_page_structure(words, raw_text)

        # Calculate counts
        word_count = len(raw_text.split())
        line_count = sum(len(p["lines"]) for p in paragraphs_reconstructed)
        paragraph_count = len(paragraphs_reconstructed)
        sentence_count = len(sentences)

        results["facts"]["statistics"] = {
            "word_count": word_count,
            "line_count": line_count,
            "paragraph_count": paragraph_count,
            "sentence_count": sentence_count
        }

        # 4. Language Detection Heuristic
        lang_res, lang_conf = self._detect_language(raw_text)
        results["assessments"]["language_detection"] = {
            "language": lang_res,
            "confidence": round(lang_conf, 3)
        }

        # 5. Document Type & Content Classification
        results["assessments"]["document_classification"] = self._classify_document(
            raw_text, line_count, word_count, context
        )

        return results

    def _reconstruct_layout_spatially(self, words: list) -> list:
        """
        Group words into lines and paragraphs using bounding boxes [x, y, w, h].
        """
        # Remove words without valid bounding box
        valid_words = [w for w in words if w.get("bbox") and len(w["bbox"]) == 4]
        if not valid_words:
            return []

        # Sort words primarily by y_top and then by x_left
        valid_words.sort(key=lambda w: (w["bbox"][1], w["bbox"][0]))

        # Group words into lines
        lines = []
        for word in valid_words:
            w_x, w_y, w_w, w_h = word["bbox"]
            w_y_center = w_y + w_h / 2.0
            
            placed = False
            for line in lines:
                line_y_centers = [w["bbox"][1] + w["bbox"][3]/2.0 for w in line]
                line_avg_h = sum(w["bbox"][3] for w in line) / len(line)
                line_avg_y_center = sum(line_y_centers) / len(line)
                
                # Threshold: vertical difference less than 50% of average line height
                if abs(w_y_center - line_avg_y_center) < (line_avg_h * 0.5):
                    line.append(word)
                    placed = True
                    break
            
            if not placed:
                lines.append([word])

        # Sort words inside each line horizontally by x
        for line in lines:
            line.sort(key=lambda w: w["bbox"][0])

        # Sort all lines by their average y_top
        lines.sort(key=lambda line: sum(w["bbox"][1] for w in line) / len(line))

        # Group lines into paragraphs
        paragraphs = []
        current_paragraph_lines = []
        
        last_line_y_bottom = -1
        last_line_height = 20
        
        for line in lines:
            line_y_top = sum(w["bbox"][1] for w in line) / len(line)
            line_height = sum(w["bbox"][3] for w in line) / len(line)
            line_y_bottom = line_y_top + line_height
            line_text = " ".join(w["text"] for w in line)
            
            if not current_paragraph_lines:
                current_paragraph_lines.append(line_text)
            else:
                vertical_gap = line_y_top - last_line_y_bottom
                is_break = False
                
                # 1. Large vertical gap
                if vertical_gap > (last_line_height * 1.8):
                    is_break = True
                
                # 2. Horizontal indent of first word in line
                first_word_x = line[0]["bbox"][0]
                if len(line) > 1 and first_word_x > 80:
                    is_break = True

                if is_break:
                    paragraphs.append({
                        "lines": current_paragraph_lines,
                        "text": " ".join(current_paragraph_lines)
                    })
                    current_paragraph_lines = [line_text]
                else:
                    current_paragraph_lines.append(line_text)

            last_line_y_bottom = line_y_bottom
            last_line_height = line_height

        if current_paragraph_lines:
            paragraphs.append({
                "lines": current_paragraph_lines,
                "text": " ".join(current_paragraph_lines)
            })

        return paragraphs

    def _reconstruct_layout_textually(self, text: str) -> list:
        """
        Split raw text into paragraphs based on empty lines.
        """
        paragraphs = []
        raw_paragraphs = text.split("\n\n")
        for rp in raw_paragraphs:
            rp_clean = rp.strip()
            if not rp_clean:
                continue
            lines = [line.strip() for line in rp_clean.split("\n") if line.strip()]
            paragraphs.append({
                "lines": lines,
                "text": " ".join(lines)
            })
        return paragraphs

    def _segment_sentences(self, text: str) -> list:
        """
        Split text into sentences using standard punctuation heuristics.
        """
        # Split text into sentences, avoiding splitting on abbreviations
        sentence_end = re.compile(
            r'(?<!\bMr)(?<!\bMrs)(?<!\bMs)(?<!\bDr)(?<!\bJan)(?<!\bFeb)(?<!\bMar)(?<!\bApr)(?<!\bAug)(?<!\bSept)(?<!\bOct)(?<!\bNov)(?<!\bDec)(?<=[.?!])\s+(?=[A-Z"“])'
        )
        sentences = sentence_end.split(text)
        return [s.strip() for s in sentences if s.strip()]

    def _extract_typography(self, words: list, paragraphs: list) -> dict:
        """
        Extracts typographical hints (indentation counts, uppercase lines, etc.)
        """
        typo = {
            "indented_paragraphs_count": 0,
            "uppercase_lines": []
        }
        
        # 1. Check Indentations of paragraph starting words
        for p in paragraphs:
            lines = p.get("lines", [])
            if lines:
                p_text_words = p["text"].split()
                if p_text_words:
                    first_w = p_text_words[0].strip(".,?!\"'();:")
                    # Find coordinates for this word
                    matching_words = [w for w in words if w.get("text", "").strip(".,?!\"'();:") == first_w and w.get("bbox")]
                    if matching_words:
                        x_coord = matching_words[0]["bbox"][0]
                        # Indented if offset is large (typical margin is < 40px)
                        if x_coord > 60:
                            typo["indented_paragraphs_count"] += 1
                            
        # 2. Check for Uppercase lines
        for p in paragraphs:
            for line in p.get("lines", []):
                letters = "".join(c for c in line if c.isalpha())
                if letters and letters.isupper() and len(letters) > 3:
                    typo["uppercase_lines"].append(line)
                    
        return typo

    def _extract_page_structure(self, words: list, raw_text: str) -> dict:
        """
        Calculates content margins and scans for page continuation hyphenations.
        """
        structure = {
            "content_boundaries": {
                "top": None,
                "bottom": None,
                "left": None,
                "right": None
            },
            "page_continuation_hyphenated": False
        }
        
        # Compute margins from word boxes
        valid_boxes = [w["bbox"] for w in words if w.get("bbox") and len(w["bbox"]) == 4]
        if valid_boxes:
            structure["content_boundaries"]["left"] = min(b[0] for b in valid_boxes)
            structure["content_boundaries"]["top"] = min(b[1] for b in valid_boxes)
            structure["content_boundaries"]["right"] = max(b[0] + b[2] for b in valid_boxes)
            structure["content_boundaries"]["bottom"] = max(b[1] + b[3] for b in valid_boxes)
            
        # Check if the page ends with a hyphenated word (like 'Sab-')
        clean_text = raw_text.strip()
        if clean_text and clean_text[-1] in ("-", "–", "—"):
            structure["page_continuation_hyphenated"] = True
            
        return structure

    def _detect_language(self, text: str) -> tuple:
        """
        Heuristic language detection based on common English words.
        """
        words = [w.lower().strip(".,?!\"'();:") for w in text.split()]
        if not words:
            return "unknown", 0.0
            
        english_word_count = sum(1 for w in words if w in self.ENGLISH_STOP_WORDS)
        english_ratio = english_word_count / len(words)
        
        if english_ratio > 0.08:
            conf = min(0.5 + english_ratio * 3, 0.99)
            return "English (en)", conf
            
        return "Unknown", 0.3

    def _classify_document(self, text: str, line_count: int, word_count: int, context: dict) -> dict:
        """
        Refined classification combining text context and NLP proverb scores.
        """
        text_lower = text.lower()
        
        # Check proverbs from EntityNlp context
        proverb_count = context.get("entity_nlp", {}).get("facts", {}).get("entities_summary", {}).get("proverb_count", 0)
        
        # Heuristics
        invoice_keywords = {"invoice", "receipt", "total due", "billing", "amount due", "payment"}
        book_keywords = {"chapter", "said", "cried", "shook", "she", "he", "replied"}
        code_keywords = {"import ", "def ", "class ", "function", "const ", "let ", "public class"}

        inv_matches = sum(1 for kw in invoice_keywords if kw in text_lower)
        book_matches = sum(1 for kw in book_keywords if kw in text_lower)
        code_matches = sum(1 for kw in code_keywords if kw in text_lower)

        doc_type = "Standard Printed Document Page"
        content_type = "General Text Content"
        doc_conf = 0.70
        content_conf = 0.50

        if proverb_count >= 3:
            doc_type = "Book Page"
            content_type = "Collection of English Proverbs/Idioms"
            doc_conf = 0.95
            content_conf = 0.95
        elif code_matches >= 3 or ("def " in text_lower and ":" in text_lower):
            doc_type = "Text Dump"
            content_type = "Source Code"
            doc_conf = 0.85
            content_conf = 0.90
        elif inv_matches >= 3 or ("total" in text_lower and "invoice" in text_lower):
            doc_type = "Business Document"
            content_type = "Invoice / Receipt"
            doc_conf = 0.90
            content_conf = 0.95
        elif book_matches >= 3 and line_count >= 5:
            doc_type = "Scanned Printed Book Page"
            content_type = "Narrative Text / Prose"
            doc_conf = 0.85
            content_conf = 0.80
        elif line_count <= 3:
            doc_type = "General Graphic / Image"
            content_type = "Short Caption / Non-document"
            doc_conf = 0.60
            content_conf = 0.50

        return {
            "document_type": doc_type,
            "document_confidence": round(doc_conf, 2),
            "content_type": content_type,
            "content_confidence": round(content_conf, 2)
        }
