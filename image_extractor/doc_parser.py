from image_extractor.base_analyzer import BaseAnalyzer


class DocParser(BaseAnalyzer):
    """
    Groups OCR words into lines and paragraphs using spatial bounding boxes.
    Performs language detection and document type classification.
    """
    VERSION = "1.1.0"

    # Common English stop words to assist language heuristic
    ENGLISH_STOP_WORDS = {"the", "of", "and", "to", "a", "in", "is", "it", "you", "that", 
                          "he", "was", "for", "on", "are", "as", "with", "his", "they", "i"}

    def analyze(self, file_path: str, img, context: dict) -> dict:
        results = {
            "facts": {
                "paragraphs": [],
                "statistics": {
                    "word_count": 0,
                    "line_count": 0,
                    "paragraph_count": 0
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
                "classification": "Non-Text Image (e.g., Photograph/Graphic)",
                "confidence": 0.90
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

        # Calculate counts
        word_count = len(raw_text.split())
        line_count = sum(len(p["lines"]) for p in paragraphs_reconstructed)
        paragraph_count = len(paragraphs_reconstructed)

        results["facts"]["statistics"] = {
            "word_count": word_count,
            "line_count": line_count,
            "paragraph_count": paragraph_count
        }

        # 2. Language Detection Heuristic
        lang_res, lang_conf = self._detect_language(raw_text)
        results["assessments"]["language_detection"] = {
            "language": lang_res,
            "confidence": round(lang_conf, 3)
        }

        # 3. Document Type Classification
        doc_type, doc_conf = self._classify_document(raw_text, line_count, word_count, context)
        results["assessments"]["document_classification"] = {
            "classification": doc_type,
            "confidence": round(doc_conf, 3)
        }

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
        # Two words are in the same line if their vertical overlap is high
        lines = []
        for word in valid_words:
            w_x, w_y, w_w, w_h = word["bbox"]
            w_y_center = w_y + w_h / 2.0
            
            # Try to place word in an existing line
            placed = False
            for line in lines:
                # Calculate average height and average y_center of current line
                line_y_centers = [w["bbox"][1] + w["bbox"][3]/2.0 for w in line]
                line_avg_h = sum(w["bbox"][3] for w in line) / len(line)
                line_avg_y_center = sum(line_y_centers) / len(line)
                
                # Threshold: vertical difference less than 40% of average line height
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
        # Standard spacing between consecutive lines is roughly 1.0 to 1.5 times line height.
        # A gap of >1.8x line height or an indented first word indicates a paragraph break.
        paragraphs = []
        current_paragraph_lines = []
        
        last_line_y_bottom = -1
        last_line_height = 20
        
        for line in lines:
            # Average y_top, y_bottom, and height of this line
            line_y_top = sum(w["bbox"][1] for w in line) / len(line)
            line_height = sum(w["bbox"][3] for w in line) / len(line)
            line_y_bottom = line_y_top + line_height
            line_text = " ".join(w["text"] for w in line)
            
            if not current_paragraph_lines:
                current_paragraph_lines.append(line_text)
            else:
                vertical_gap = line_y_top - last_line_y_bottom
                # Check for paragraph breaks
                is_break = False
                
                # 1. Large vertical gap
                if vertical_gap > (last_line_height * 1.8):
                    is_break = True
                
                # 2. Horizontal indent of first word in line
                first_word_x = line[0]["bbox"][0]
                # Compare to starting x of previous line
                prev_line_words = valid_words # fallback
                # Simple heuristic: if first_word_x is significantly indented compared to usual left margin
                # We can check if it's indented by > 30px
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

    def _detect_language(self, text: str) -> tuple:
        """
        Heuristic language detection based on common English words.
        """
        words = [w.lower().strip(".,?!\"'();:") for w in text.split()]
        if not words:
            return "unknown", 0.0
            
        english_word_count = sum(1 for w in words if w in self.ENGLISH_STOP_WORDS)
        english_ratio = english_word_count / len(words)
        
        # If stop word ratio is high (typically > 8% for any natural English text)
        if english_ratio > 0.08:
            # High confidence if ratio is high
            conf = min(0.5 + english_ratio * 3, 0.99)
            return "English (en)", conf
            
        return "Unknown", 0.3

    def _classify_document(self, text: str, line_count: int, word_count: int, context: dict) -> tuple:
        """
        Heuristic classification of document type.
        """
        text_lower = text.lower()
        
        # Check keywords
        invoice_keywords = {"invoice", "receipt", "total due", "billing", "amount due", "payment"}
        book_keywords = {"chapter", "said", "cried", "shook", "she", "he", "replied"}
        code_keywords = {"import ", "def ", "class ", "function", "const ", "let ", "public class"}

        has_exif = context.get("file_analyzer", {}).get("facts", {}).get("file_info", {}).get("md5_hash") is not None
        
        # Check density of keywords
        inv_matches = sum(1 for kw in invoice_keywords if kw in text_lower)
        book_matches = sum(1 for kw in book_keywords if kw in text_lower)
        code_matches = sum(1 for kw in code_keywords if kw in text_lower)

        if code_matches >= 3 or ("def " in text_lower and ":" in text_lower):
            return "Source Code File / Text Dump", 0.85
            
        if inv_matches >= 3 or ("total" in text_lower and "invoice" in text_lower):
            return "Business Invoice / Receipt Document", 0.90
            
        if book_matches >= 3 and line_count > 5:
            return "Scanned Printed Book Page", 0.85
            
        # Default fallback
        if line_count > 3:
            return "Standard Printed Document Page", 0.70
            
        return "General Graphic / Image", 0.50
