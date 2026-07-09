import os
import json
from image_extractor.base_analyzer import BaseAnalyzer

# Check for Tesseract and EasyOCR dependencies
try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False

try:
    import easyocr
    HAS_EASYOCR = True
except ImportError:
    HAS_EASYOCR = False


class OcrEngine(BaseAnalyzer):
    """
    Coordinates Optical Character Recognition. Supports EasyOCR, Tesseract,
    and falls back to loading sidecar (.ocr/.txt) files for offline/test environments.
    """
    VERSION = "1.1.0"

    def analyze(self, file_path: str, img, context: dict) -> dict:
        results = {
            "facts": {
                "raw_text": "",
                "words": []
            },
            "indicators": [],
            "assessments": {},
            "errors": []
        }

        # Check sidecar fallback first
        base_path, _ = os.path.splitext(file_path)
        sidecar_ocr = base_path + ".ocr"
        sidecar_txt = base_path + ".txt"
        sidecar_json = base_path + ".json"
        
        selected_sidecar = None
        if os.path.exists(sidecar_ocr):
            selected_sidecar = sidecar_ocr
        elif os.path.exists(sidecar_txt):
            selected_sidecar = sidecar_txt
        elif os.path.exists(sidecar_json):
            selected_sidecar = sidecar_json

        if selected_sidecar:
            try:
                with open(selected_sidecar, "r", encoding="utf-8") as sf:
                    content = sf.read().strip()
                
                # Check if sidecar is structured JSON
                try:
                    json_data = json.loads(content)
                    if isinstance(json_data, dict):
                        raw_text = ""
                        words_list = []
                        
                        # Support multiple key variations
                        if "raw_text" in json_data:
                            raw_text = json_data["raw_text"]
                        elif "ocr_data" in json_data:
                            if isinstance(json_data["ocr_data"], dict):
                                raw_text = json_data["ocr_data"].get("raw_text", "")
                                words_list = json_data["ocr_data"].get("words", [])
                            else:
                                raw_text = str(json_data["ocr_data"])
                        elif "facts" in json_data and "raw_text" in json_data["facts"]:
                            raw_text = json_data["facts"]["raw_text"]
                            words_list = json_data["facts"].get("words", [])
                            
                        if "words" in json_data:
                            words_list = json_data["words"]
                            
                        if raw_text or words_list:
                            results["facts"]["raw_text"] = raw_text
                            results["facts"]["words"] = words_list if words_list else self._mock_words_from_text(raw_text)
                        else:
                            raise ValueError()
                    else:
                        raise ValueError()
                except Exception:
                    # Plain text sidecar
                    results["facts"]["raw_text"] = content
                    results["facts"]["words"] = self._mock_words_from_text(content)

                results["indicators"].append({
                    "type": "ocr_sidecar_loaded",
                    "description": f"Loaded OCR sidecar file: {os.path.basename(selected_sidecar)}",
                    "severity": "low"
                })
                return results
            except Exception as e:
                results["errors"].append({
                    "plugin": self.get_name(),
                    "severity": "warning",
                    "message": f"Failed to load OCR sidecar: {str(e)}"
                })

        # Try EasyOCR
        if HAS_EASYOCR:
            try:
                reader = easyocr.Reader(["en"]) # default English
                # reader.readtext returns list: [([x, y], [x, y], ...), text, confidence]
                ocr_results = reader.readtext(file_path)
                
                raw_lines = []
                words_list = []
                for bbox, text, conf in ocr_results:
                    raw_lines.append(text)
                    # Convert easyocr bbox [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] to [x_min, y_min, w, h]
                    xs = [pt[0] for pt in bbox]
                    ys = [pt[1] for pt in bbox]
                    x_min, y_min = min(xs), min(ys)
                    w, h = max(xs) - x_min, max(ys) - y_min
                    
                    words_list.append({
                        "text": text,
                        "bbox": [x_min, y_min, w, h],
                        "confidence": round(float(conf), 3)
                    })
                
                results["facts"]["raw_text"] = "\n".join(raw_lines)
                results["facts"]["words"] = words_list
                return results
            except Exception as e:
                results["errors"].append({
                    "plugin": self.get_name(),
                    "severity": "warning",
                    "message": f"EasyOCR execution failed: {str(e)}"
                })

        # Try Tesseract
        if HAS_TESSERACT:
            try:
                # Configure Tesseract path if specified or check default paths on Windows
                tess_path = self.config.get("tesseract_path")
                if tess_path:
                    pytesseract.pytesseract.tesseract_cmd = tess_path
                elif os.name == "nt":
                    default_paths = [
                        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe")
                    ]
                    for path in default_paths:
                        if os.path.exists(path):
                            pytesseract.pytesseract.tesseract_cmd = path
                            break

                # Tesseract OCR
                # We can retrieve detailed word boxes using image_to_data
                data_str = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
                
                raw_lines = []
                words_list = []
                n_boxes = len(data_str["text"])
                
                current_line = []
                last_line_num = -1
                
                for i in range(n_boxes):
                    text = data_str["text"][i].strip()
                    if not text:
                        continue
                    
                    conf = float(data_str["conf"][i]) / 100.0 if "conf" in data_str else 0.8
                    if conf < 0:
                        conf = 0.8 # Tesseract -1 indicates block headings without confidence
                        
                    x = data_str["left"][i]
                    y = data_str["top"][i]
                    w = data_str["width"][i]
                    h = data_str["height"][i]
                    
                    words_list.append({
                        "text": text,
                        "bbox": [x, y, w, h],
                        "confidence": round(conf, 3)
                    })
                    
                    line_num = data_str["line_num"][i]
                    if line_num != last_line_num:
                        if current_line:
                            raw_lines.append(" ".join(current_line))
                        current_line = [text]
                        last_line_num = line_num
                    else:
                        current_line.append(text)
                        
                if current_line:
                    raw_lines.append(" ".join(current_line))
                    
                results["facts"]["raw_text"] = "\n".join(raw_lines)
                results["facts"]["words"] = words_list
                return results
            except Exception as e:
                results["errors"].append({
                    "plugin": self.get_name(),
                    "severity": "warning",
                    "message": f"Tesseract execution failed: {str(e)}"
                })

        # If no OCR is available
        results["errors"].append({
            "plugin": self.get_name(),
            "severity": "warning",
            "message": "No local OCR engine (EasyOCR or Tesseract) is installed, and no sidecar OCR file was found."
        })
        return results

    def _mock_words_from_text(self, text: str) -> list:
        """
        Generates simulated words with structured coordinates so that downstream
        layout, entities, and relationship analyzers can run on raw text inputs.
        """
        words = []
        lines = text.split("\n")
        
        y_offset = 20
        for line in lines:
            line_words = line.strip().split()
            if not line_words:
                y_offset += 30
                continue
                
            x_offset = 20
            for lw in line_words:
                w_len = len(lw) * 10
                words.append({
                    "text": lw,
                    "bbox": [x_offset, y_offset, w_len, 20],
                    "confidence": 0.95
                })
                x_offset += w_len + 15
            y_offset += 35
        return words
