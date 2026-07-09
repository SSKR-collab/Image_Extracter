import os
import json
from PIL import Image
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

try:
    import cv2
    import numpy as np
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False


class OcrEngine(BaseAnalyzer):
    """
    Coordinates Optical Character Recognition. Supports EasyOCR, Tesseract,
    and falls back to loading sidecar (.ocr/.txt/.json) files for offline/test environments.
    Implements advanced local Tesseract preprocessing (tiling, multi-scale, multi-PSM, CLAHE, sharpening)
    to maximize text extraction accuracy on complex layouts without heavy models.
    """
    VERSION = "1.3.0"

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
                # Convert PIL Image to byte buffer
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format=img.format or 'PNG')
                img_bytes = img_byte_arr.getvalue()
                
                reader = easyocr.Reader(['en'], gpu=False)
                ocr_results = reader.readtext(img_bytes)
                
                raw_lines = []
                words_list = []
                for bbox, text, conf in ocr_results:
                    raw_lines.append(text)
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

                # Tesseract OCR execution
                # Check if we can run advanced local preprocessing
                if HAS_OPENCV:
                    words_list = self._run_tesseract_with_enhancements(img, pytesseract.pytesseract.tesseract_cmd)
                else:
                    data_str = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
                    words_list = self._parse_tesseract_dict(data_str)

                # Sort words by line_num and then by x coordinate
                words_list.sort(key=lambda w: (w.get("line_num", 0), w["bbox"][0]))
                
                raw_lines = []
                current_line = []
                last_line_num = -1
                for w in words_list:
                    text = w["text"]
                    line_num = w.get("line_num", 0)
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
                
                # Clean line_num helper keys
                for w in words_list:
                    if "line_num" in w:
                        del w["line_num"]
                
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

    def _run_tesseract_with_enhancements(self, img, tess_cmd) -> list:
        """
        Runs Tesseract OCR using multiple scales, PSM config values, and
        OpenCV image preprocessing filters to maximize extraction accuracy.
        """
        # Convert Pillow image to OpenCV BGR format
        open_cv_image = np.array(img.convert("RGB"))
        gray = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2GRAY)
        
        # Pass 1: Run raw image first
        data_str = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        words = self._parse_tesseract_dict(data_str)
        
        # If text is extremely short, run advanced tile-based preprocessing passes
        if len(words) < 15:
            # Attempt grid-based overlapping tile scans (captures small, curved, and stylized text segments)
            tile_words = self._run_tile_based_ocr(gray)
            for tw in tile_words:
                if not any(self._is_overlapping(tw["bbox"], w["bbox"]) for w in words):
                    words.append(tw)
                    
            try:
                # Rotation pass: test 90-degrees clockwise rotation for vertical/skewed text banners
                h, w = gray.shape
                rotated_90 = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
                data_str_90 = pytesseract.image_to_data(Image.fromarray(rotated_90), output_type=pytesseract.Output.DICT)
                words_90 = self._parse_tesseract_dict(data_str_90)
                
                for w90 in words_90:
                    x, y, width, height = w90["bbox"]
                    # Map coordinates back: x_new = y, y_new = h - x - width
                    orig_x = y
                    orig_y = h - x - width
                    w90["bbox"] = [orig_x, orig_y, height, width]
                    
                    if not any(self._is_overlapping(w90["bbox"], w["bbox"]) for w in words):
                        words.append(w90)
            except Exception:
                pass
                
        return words

    def _run_tile_based_ocr(self, gray_img) -> list:
        """
        Divides the image into overlapping tiles and runs upscaled + enhanced
        multi-PSM Tesseract checks on each crop region.
        """
        h, w = gray_img.shape
        tile_size = 512
        step_size = 384
        
        words = []
        
        # Grid bounds calculation
        y_steps = list(range(0, max(1, h - tile_size), step_size))
        if not y_steps or y_steps[-1] + tile_size < h:
            y_steps.append(max(0, h - tile_size))
            
        x_steps = list(range(0, max(1, w - tile_size), step_size))
        if not x_steps or x_steps[-1] + tile_size < w:
            x_steps.append(max(0, w - tile_size))
            
        # Run sliding window
        for y in y_steps:
            for x in x_steps:
                tile = gray_img[y:y+tile_size, x:x+tile_size]
                th, tw = tile.shape
                if th == 0 or tw == 0:
                    continue
                    
                # Preprocess tile: Upscale 3x, CLAHE contrast stretch, filter sharpening
                resized_tile = cv2.resize(tile, (tw * 3, th * 3), interpolation=cv2.INTER_CUBIC)
                
                clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
                contrast_tile = clahe.apply(resized_tile)
                
                kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                sharpened_tile = cv2.filter2D(contrast_tile, -1, kernel)
                
                pil_tile = Image.fromarray(sharpened_tile)
                
                # Check multiple PSMs (default, sparse text, uniform block)
                for psm in ["3", "11", "6"]:
                    custom_config = f"--oem 3 --psm {psm}"
                    try:
                        data_str_tile = pytesseract.image_to_data(
                            pil_tile, config=custom_config, output_type=pytesseract.Output.DICT
                        )
                        tile_words = self._parse_tesseract_dict(data_str_tile)
                        
                        # Downscale bounding boxes back and translate to image space
                        for tw_word in tile_words:
                            tw_word["bbox"] = [
                                x + (tw_word["bbox"][0] // 3),
                                y + (tw_word["bbox"][1] // 3),
                                tw_word["bbox"][2] // 3,
                                tw_word["bbox"][3] // 3
                            ]
                            
                        # Add word if it doesn't overlap an existing detection
                        for tw_word in tile_words:
                            if not any(self._is_overlapping(tw_word["bbox"], mw["bbox"]) for mw in words):
                                words.append(tw_word)
                    except Exception:
                        pass
        return words

    def _parse_tesseract_dict(self, data_str) -> list:
        words = []
        n_boxes = len(data_str["text"])
        for i in range(n_boxes):
            text = data_str["text"][i].strip()
            if not text:
                continue
            
            conf = float(data_str["conf"][i]) / 100.0 if "conf" in data_str else 0.8
            if conf < 0:
                conf = 0.8
                
            x = data_str["left"][i]
            y = data_str["top"][i]
            w = data_str["width"][i]
            h = data_str["height"][i]
            
            words.append({
                "text": text,
                "bbox": [x, y, w, h],
                "confidence": round(conf, 3),
                "line_num": data_str.get("line_num", [0]*n_boxes)[i]
            })
        return words

    def _is_overlapping(self, boxA, boxB) -> bool:
        # boxA/boxB format: [x, y, w, h]
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
        
        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea > 0:
            areaA = boxA[2] * boxA[3]
            areaB = boxB[2] * boxB[3]
            # Overlap ratio relative to smaller box
            overlap = interArea / float(min(areaA, areaB))
            return overlap > 0.4
        return False

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
