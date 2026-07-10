import re
from image_extractor.base_analyzer import BaseAnalyzer


class DocParser(BaseAnalyzer):
    """
    Groups OCR words into lines and paragraphs using spatial bounding boxes.
    Detects vertical columns/gutters, preserves reading order, maps margins,
    performs sentence segmentation, and runs typography and content classifications.
    """
    VERSION = "1.3.0"

    # Common English stop words to assist language heuristic
    ENGLISH_STOP_WORDS = {"the", "of", "and", "to", "a", "in", "is", "it", "you", "that", 
                          "he", "was", "for", "on", "are", "as", "with", "his", "they", "i"}

    # Common proverbs for document classification heuristic
    COMMON_PROVERBS = [
        "the customer is always right",
        "east, west, home's best",
        "when in rome, do as the romans do",
        "action speaks louder than words",
        "all that glitters is not gold",
        "a picture is worth a thousand words",
        "birds of a feather flock together",
        "early bird catches the worm",
        "look before you leap",
        "practice makes perfect",
        "better late than never",
        "don't judge a book by its cover",
        "no pain no gain",
        "two wrongs don't make a right",
        "out of sight, out of mind"
    ]

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
            results["assessments"]["document_classification"] = {
                "document_type": "Non-Text Image (e.g., Photograph/Graphic)",
                "document_confidence": 0.90,
                "content_type": "Visual Only",
                "content_confidence": 0.90
            }
            return results

        # 1. Column-aware Spatial Layout Reconstruction (Paragraph & Line grouping)
        paragraphs_reconstructed = []
        if words and any(w.get("bbox") for w in words):
            paragraphs_reconstructed = self._detect_columns_and_group_words(words)
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
            raw_text, line_count, word_count
        )

        return results

    def _detect_columns_and_group_words(self, words: list) -> list:
        """
        Detects vertical column gutters using horizontal projection profiles,
        ensuring multi-column documents preserve reading order.
        """
        valid_words = [w for w in words if w.get("bbox") and len(w["bbox"]) == 4]
        if not valid_words:
            return []
            
        # Get overall boundaries
        xs = [w["bbox"][0] for w in valid_words]
        xe = [w["bbox"][0] + w["bbox"][2] for w in valid_words]
        min_x, max_x = min(xs), max(xe)
        page_width = max_x - min_x
        
        # Formulate horizontal occupancy histogram
        bin_size = 8
        num_bins = int(page_width / bin_size) + 2
        occupancy = [0] * num_bins
        
        for w in valid_words:
            left = w["bbox"][0] - min_x
            right = left + w["bbox"][2]
            
            start_bin = max(0, int(left / bin_size))
            end_bin = min(num_bins - 1, int(right / bin_size))
            for b in range(start_bin, end_bin + 1):
                occupancy[b] += 1
                
        # Locate vertical gutters (continuous bins of zero or very low occupancy)
        min_gutter_bins = 3
        gutters = []
        current_gutter_start = -1
        
        for b in range(1, num_bins - 1):
            if occupancy[b] <= 1:
                if current_gutter_start == -1:
                    current_gutter_start = b
            else:
                if current_gutter_start != -1:
                    gutter_width = b - current_gutter_start
                    if gutter_width >= min_gutter_bins:
                        g_left = min_x + current_gutter_start * bin_size
                        g_right = min_x + b * bin_size
                        gutters.append((g_left, g_right))
                    current_gutter_start = -1
                    
        # Filter gutters that lie too close to the left/right margins (within 15%)
        margin_threshold = page_width * 0.15
        valid_gutters = []
        for g_left, g_right in gutters:
            if (g_left - min_x) > margin_threshold and (max_x - g_right) > margin_threshold:
                valid_gutters.append((g_left, g_right))
                
        # Sort gutters horizontally
        valid_gutters.sort(key=lambda g: g[0])
        
        # Partition columns
        col_bounds = []
        last_x = min_x
        for g_left, g_right in valid_gutters:
            col_bounds.append((last_x, g_left))
            last_x = g_right
        col_bounds.append((last_x, max_x))
        
        # Assign words to their matching columns
        columns_words = [[] for _ in col_bounds]
        for w in valid_words:
            w_center_x = w["bbox"][0] + w["bbox"][2] / 2.0
            
            assigned = False
            for col_idx, (c_start, c_end) in enumerate(col_bounds):
                if c_start <= w_center_x <= c_end:
                    columns_words[col_idx].append(w)
                    assigned = True
                    break
            if not assigned:
                closest_idx = 0
                min_dist = float("inf")
                for col_idx, (c_start, c_end) in enumerate(col_bounds):
                    dist = min(abs(w_center_x - c_start), abs(w_center_x - c_end))
                    if dist < min_dist:
                        min_dist = dist
                        closest_idx = col_idx
                columns_words[closest_idx].append(w)
                
        # Process each column sequentially
        reconstructed_paragraphs = []
        for col_idx, col_w in enumerate(columns_words):
            if not col_w:
                continue
            col_paras = self._reconstruct_column_layout(col_w, col_idx + 1)
            reconstructed_paragraphs.extend(col_paras)
            
        return reconstructed_paragraphs

    def _reconstruct_column_layout(self, valid_words: list, column_num: int) -> list:
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
                
                # Group words into same line if vertical variance is under 50% average word height
                if abs(w_y_center - line_avg_y_center) < (line_avg_h * 0.5):
                    line.append(word)
                    placed = True
                    break
            
            if not placed:
                lines.append([word])

        # Sort words horizontally inside each line
        for line in lines:
            line.sort(key=lambda w: w["bbox"][0])

        # Sort lines by average y coordinate
        lines.sort(key=lambda line: sum(w["bbox"][1] for w in line) / len(line))

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
                
                # Check for large paragraph break vertical gaps
                if vertical_gap > (last_line_height * 1.8):
                    is_break = True
                
                # Check for paragraph indentation
                first_word_x = line[0]["bbox"][0]
                if len(line) > 1 and first_word_x > (lines[0][0]["bbox"][0] + 40):
                    is_break = True

                if is_break:
                    paragraphs.append({
                        "lines": current_paragraph_lines,
                        "text": " ".join(current_paragraph_lines),
                        "column": column_num
                    })
                    current_paragraph_lines = [line_text]
                else:
                    current_paragraph_lines.append(line_text)

            last_line_y_bottom = line_y_bottom
            last_line_height = line_height

        if current_paragraph_lines:
            paragraphs.append({
                "lines": current_paragraph_lines,
                "text": " ".join(current_paragraph_lines),
                "column": column_num
            })

        return paragraphs

    def _reconstruct_layout_textually(self, text: str) -> list:
        paragraphs = []
        raw_paragraphs = text.split("\n\n")
        for rp in raw_paragraphs:
            rp_clean = rp.strip()
            if not rp_clean:
                continue
            lines = [line.strip() for line in rp_clean.split("\n") if line.strip()]
            paragraphs.append({
                "lines": lines,
                "text": " ".join(lines),
                "column": 1
            })
        return paragraphs

    def _segment_sentences(self, text: str) -> list:
        # Split text into sentences, preserving common title abbreviations
        sentence_end = re.compile(
            r'(?<!\bMr)(?<!\bMrs)(?<!\bMs)(?<!\bDr)(?<!\bJan)(?<!\bFeb)(?<!\bMar)(?<!\bApr)(?<!\bAug)(?<!\bSept)(?<!\bOct)(?<!\bNov)(?<!\bDec)(?<=[.?!])\s+(?=[A-Z"“])'
        )
        sentences = sentence_end.split(text)
        return [s.strip() for s in sentences if s.strip()]

    def _extract_typography(self, words: list, paragraphs: list) -> dict:
        typo = {
            "indented_paragraphs_count": 0,
            "uppercase_lines": []
        }
        
        # Check Indentations of paragraph starts
        for p in paragraphs:
            lines = p.get("lines", [])
            if lines:
                p_text_words = p["text"].split()
                if p_text_words:
                    first_w = p_text_words[0].strip(".,?!\"'();:")
                    matching_words = [w for w in words if w.get("text", "").strip(".,?!\"'();:") == first_w and w.get("bbox")]
                    if matching_words:
                        x_coord = matching_words[0]["bbox"][0]
                        if x_coord > 60:
                            typo["indented_paragraphs_count"] += 1
                            
        # Scan for Uppercase lines
        for p in paragraphs:
            for line in p.get("lines", []):
                letters = "".join(c for c in line if c.isalpha())
                if letters and letters.isupper() and len(letters) > 3:
                    typo["uppercase_lines"].append(line)
                    
        return typo

    def _extract_page_structure(self, words: list, raw_text: str) -> dict:
        structure = {
            "content_boundaries": {
                "top": None,
                "bottom": None,
                "left": None,
                "right": None
            },
            "page_continuation_hyphenated": False
        }
        
        valid_boxes = [w["bbox"] for w in words if w.get("bbox") and len(w["bbox"]) == 4]
        if valid_boxes:
            structure["content_boundaries"]["left"] = min(b[0] for b in valid_boxes)
            structure["content_boundaries"]["top"] = min(b[1] for b in valid_boxes)
            structure["content_boundaries"]["right"] = max(b[0] + b[2] for b in valid_boxes)
            structure["content_boundaries"]["bottom"] = max(b[1] + b[3] for b in valid_boxes)
            
        clean_text = raw_text.strip()
        if clean_text and clean_text[-1] in ("-", "–", "—"):
            structure["page_continuation_hyphenated"] = True
            
        return structure

    def _detect_language(self, text: str) -> tuple:
        words = [w.lower().strip(".,?!\"'();:") for w in text.split()]
        if not words:
            return "unknown", 0.0
            
        english_word_count = sum(1 for w in words if w in self.ENGLISH_STOP_WORDS)
        english_ratio = english_word_count / len(words)
        
        if english_ratio > 0.08:
            conf = min(0.5 + english_ratio * 3, 0.99)
            return "English (en)", conf
            
        return "Unknown", 0.3

    def _count_proverbs(self, text: str) -> int:
        text_lower = text.lower()
        count = 0
        for p in self.COMMON_PROVERBS:
            if p in text_lower:
                count += 1
        return count

    def _classify_document(self, text: str, line_count: int, word_count: int) -> dict:
        text_lower = text.lower()
        proverb_count = self._count_proverbs(text)
        
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
