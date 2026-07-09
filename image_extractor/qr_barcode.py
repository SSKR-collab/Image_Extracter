import os
import re
import json
from image_extractor.base_analyzer import BaseAnalyzer

try:
    import cv2
    import numpy as np
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

try:
    from pyzbar import pyzbar
    HAS_PYZBAR = True
except ImportError:
    HAS_PYZBAR = False


class QrBarcode(BaseAnalyzer):
    """
    Scans the image for QR codes and barcodes.
    Pluggably integrates OpenCV QRCodeDetector and PyZbar.
    Supports a two-stage QR scanning architecture with image enhancements,
    rotation sweeps, pyramid scaling, and semantic payload classification.
    """
    VERSION = "1.4.0"
    
    def analyze(self, file_path: str, img, context: dict) -> dict:
        results = {
            "facts": {
                "qr_codes": [],
                "barcodes": []
            },
            "indicators": [],
            "errors": []
        }

        # 1. Check sidecar JSON first (e.g. qr.json next to qr.png)
        base_path, _ = os.path.splitext(file_path)
        sidecar_paths = [
            base_path + ".json",
            base_path + ".visual.json"
        ]

        loaded_sidecar = False
        for sp in sidecar_paths:
            if os.path.exists(sp):
                try:
                    with open(sp, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    
                    found_keys = False
                    for key in ["qr_codes", "barcodes"]:
                        if key in data:
                            results["facts"][key].extend(data[key])
                            found_keys = True
                        elif "facts" in data and key in data["facts"]:
                            results["facts"][key].extend(data["facts"][key])
                            found_keys = True
                            
                    if found_keys:
                        results["indicators"].append({
                            "type": "qr_sidecar_loaded",
                            "description": f"Loaded QR/Barcode sidecar metadata file: {os.path.basename(sp)}",
                            "severity": "low"
                        })
                        loaded_sidecar = True
                        break
                except Exception as e:
                    results["errors"].append({
                        "plugin": self.get_name(),
                        "severity": "warning",
                        "message": f"Failed to parse QR sidecar {os.path.basename(sp)}: {str(e)}"
                    })

        # If sidecar loaded successfully, we return immediately
        if loaded_sidecar:
            return results

        # If image didn't load, we can't scan pixels
        if not img:
            return results

        # 2. Try scanning with pyzbar if available
        if HAS_PYZBAR:
            try:
                decoded_objects = pyzbar.decode(img)
                for obj in decoded_objects:
                    text = obj.data.decode("utf-8", errors="replace")
                    obj_type = obj.type
                    
                    if obj_type == "QRCODE":
                        classification = self._classify_qr_payload(text)
                        results["facts"]["qr_codes"].append({
                            "present": True,
                            "decoded": True,
                            "data": text,
                            "payload_info": classification,
                            "bbox": list(obj.rect) if obj.rect else None,
                            "confidence": 1.0
                        })
                        results["indicators"].append({
                            "type": "qr_code_detected",
                            "description": f"QR Code ({classification['type']}) detected: '{text[:40]}...'",
                            "severity": "low"
                        })
                    else:
                        results["facts"]["barcodes"].append({
                            "data": text,
                            "type": obj_type,
                            "bbox": list(obj.rect) if obj.rect else None,
                            "confidence": 1.0
                        })
                        results["indicators"].append({
                            "type": "barcode_detected",
                            "description": f"Barcode type {obj_type} detected: '{text[:40]}...'",
                            "severity": "low"
                        })
                
                # If we parsed successfully with pyzbar, return
                if results["facts"]["qr_codes"] or results["facts"]["barcodes"]:
                    return results
            except Exception as e:
                results["errors"].append({
                    "plugin": self.get_name(),
                    "severity": "warning",
                    "message": f"PyZbar barcode detection failed: {str(e)}"
                })

        # 3. Try scanning with OpenCV if available
        if HAS_OPENCV:
            try:
                open_cv_image = np.array(img.convert("RGB"))
                open_cv_image = open_cv_image[:, :, ::-1].copy()
                
                detector = cv2.QRCodeDetector()
                has_multi = hasattr(detector, "detectMulti") and hasattr(detector, "decodeMulti")
                
                if has_multi:
                    # Run Multi-QR sweep with pyramid and rotation fallbacks
                    matches = self._detect_and_decode_with_fallback(open_cv_image, detector)
                    
                    for match in matches:
                        pts = match["points"]
                        data = match["data"]
                        angle = match["angle"]
                        
                        bbox_coords = pts.tolist() if hasattr(pts, "tolist") else pts
                        local_quality = self._evaluate_qr_image_quality(open_cv_image, np.expand_dims(pts, axis=0))
                        
                        enhanced_attempted = False
                        if not data:
                            enhanced_attempted = True
                            data = self._enhance_and_decode(open_cv_image, np.expand_dims(pts, axis=0), detector)
                            
                        if data:
                            classification = self._classify_qr_payload(data)
                            results["facts"]["qr_codes"].append({
                                "present": True,
                                "decoded": True,
                                "data": data,
                                "payload_info": classification,
                                "bbox": bbox_coords,
                                "local_quality": local_quality,
                                "error_correction_attempted": enhanced_attempted,
                                "estimated_rotation": angle,
                                "confidence": 0.95
                            })
                            results["indicators"].append({
                                "type": "qr_code_detected",
                                "description": f"QR Code ({classification['type']}) decoded successfully: '{data[:40]}...'",
                                "severity": "low"
                            })
                        else:
                            results["facts"]["qr_codes"].append({
                                "present": True,
                                "decoded": False,
                                "bbox": bbox_coords,
                                "local_quality": local_quality,
                                "error_correction_attempted": enhanced_attempted,
                                "estimated_rotation": angle,
                                "status": "QR code detected but could not be decoded.",
                                "reason": "Stylized artwork or poor image quality (e.g. low contrast / noise) obscuring modules.",
                                "confidence": 0.80
                            })
                            results["indicators"].append({
                                "type": "qr_code_obscured",
                                "description": "QR Code detected but could not be decoded (obscured).",
                                "severity": "low"
                            })
                else:
                    # Fallback to single QR detection
                    retval, points = detector.detect(open_cv_image)
                    if retval and points is not None and len(points) > 0:
                        local_quality = self._evaluate_qr_image_quality(open_cv_image, points)
                        data, straight_qrcode = detector.decode(open_cv_image, points)
                        bbox_coords = points[0].tolist() if hasattr(points, "tolist") else points.tolist()
                        
                        enhanced_attempted = False
                        if not data:
                            enhanced_attempted = True
                            data = self._enhance_and_decode(open_cv_image, points, detector)
                            
                        if data:
                            classification = self._classify_qr_payload(data)
                            results["facts"]["qr_codes"].append({
                                "present": True,
                                "decoded": True,
                                "data": data,
                                "payload_info": classification,
                                "bbox": bbox_coords,
                                "local_quality": local_quality,
                                "error_correction_attempted": enhanced_attempted,
                                "confidence": 0.95
                            })
                            results["indicators"].append({
                                "type": "qr_code_detected",
                                "description": f"QR Code ({classification['type']}) decoded successfully: '{data[:40]}...'",
                                "severity": "low"
                            })
                        else:
                            results["facts"]["qr_codes"].append({
                                "present": True,
                                "decoded": False,
                                "bbox": bbox_coords,
                                "local_quality": local_quality,
                                "error_correction_attempted": enhanced_attempted,
                                "status": "QR code detected but could not be decoded.",
                                "reason": "Stylized artwork or poor image quality (e.g. low contrast / noise) obscuring modules.",
                                "confidence": 0.80
                            })
                            results["indicators"].append({
                                "type": "qr_code_obscured",
                                "description": "QR Code detected but could not be decoded (obscured).",
                                "severity": "low"
                            })
            except Exception as e:
                results["errors"].append({
                    "plugin": self.get_name(),
                    "severity": "warning",
                    "message": f"OpenCV QR Code detection failed: {str(e)}"
                })

        # Log dependency warning if neither is installed
        if not HAS_PYZBAR and not HAS_OPENCV:
            results["errors"].append({
                "plugin": self.get_name(),
                "severity": "warning",
                "message": "QR/Barcode scanning skipped. PyZbar or OpenCV is required but not installed."
            })

        return results

    def _detect_and_decode_with_fallback(self, open_cv_image, detector) -> list:
        """
        Runs a multi-QR detection sweep, trying scale upscaling pyramids
        and 90/180/270 degree rotation angles when standard scans fail.
        """
        detected_codes = []
        h, w = open_cv_image.shape[:2]
        
        # Helper to process raw detector results
        def process_detections(pts_array, decoded_list, success_flag):
            items = []
            for i, pts in enumerate(pts_array):
                data = decoded_list[i] if (success_flag and i < len(decoded_list)) else ""
                items.append((pts, data))
            return items
        
        # Pass 1: Standard Scan
        retval, points = detector.detectMulti(open_cv_image)
        if retval and points is not None and len(points) > 0:
            success, decoded_info, _ = detector.decodeMulti(open_cv_image, points)
            for pts, data in process_detections(points, decoded_info, success):
                detected_codes.append({
                    "points": pts,
                    "data": data,
                    "angle": 0
                })
            return detected_codes
            
        # Pass 2: Pyramid Scale-up (2x)
        try:
            scaled = cv2.resize(open_cv_image, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
            retval, points = detector.detectMulti(scaled)
            if retval and points is not None and len(points) > 0:
                success, decoded_info, _ = detector.decodeMulti(scaled, points)
                for pts, data in process_detections(points, decoded_info, success):
                    pts_orig = pts / 2.0
                    detected_codes.append({
                        "points": pts_orig,
                        "data": data,
                        "angle": 0
                    })
                return detected_codes
        except Exception:
            pass
            
        # Pass 3: Rotations (90, 180, 270)
        for angle_type, angle_deg in [(cv2.ROTATE_90_CLOCKWISE, 90), (cv2.ROTATE_180, 180), (cv2.ROTATE_90_COUNTERCLOCKWISE, 270)]:
            try:
                rotated = cv2.rotate(open_cv_image, angle_type)
                retval, points = detector.detectMulti(rotated)
                if retval and points is not None and len(points) > 0:
                    success, decoded_info, _ = detector.decodeMulti(rotated, points)
                    for pts, data in process_detections(points, decoded_info, success):
                        # Map coordinates back to original non-rotated canvas
                        pts_orig = []
                        for pt in pts:
                            rx, ry = pt
                            if angle_type == cv2.ROTATE_90_CLOCKWISE:
                                ox = ry
                                oy = h - rx
                            elif angle_type == cv2.ROTATE_180:
                                ox = w - rx
                                oy = h - ry
                            else: # ROTATE_90_COUNTERCLOCKWISE
                                ox = w - ry
                                oy = rx
                            pts_orig.append([ox, oy])
                        detected_codes.append({
                            "points": np.array(pts_orig),
                            "data": data,
                            "angle": angle_deg
                        })
                    return detected_codes
            except Exception:
                pass
                
        return detected_codes

    def _enhance_and_decode(self, open_cv_image, points, detector) -> str:
        """
        Applies local contrast stretching, denoising, sharpening, and binarization
        methods to extract data from difficult modules.
        """
        try:
            gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
            
            # Method 1: Histogram Equalization (Contrast stretch)
            equalized = cv2.equalizeHist(gray)
            data, _ = detector.decode(cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR), points)
            if data:
                return data
                
            # Method 2: CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl_img = clahe.apply(gray)
            data, _ = detector.decode(cv2.cvtColor(cl_img, cv2.COLOR_GRAY2BGR), points)
            if data:
                return data
                
            # Method 3: Gaussian Blur + Otsu Thresholding
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            data, _ = detector.decode(cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR), points)
            if data:
                return data
                
            # Method 4: Adaptive Thresholding
            adapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            data, _ = detector.decode(cv2.cvtColor(adapt, cv2.COLOR_GRAY2BGR), points)
            if data:
                return data

            # Method 5: Sharpening kernel
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(gray, -1, kernel)
            data, _ = detector.decode(cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR), points)
            if data:
                return data
        except Exception:
            pass

        return ""

    def _evaluate_qr_image_quality(self, open_cv_image, points) -> dict:
        """
        Evaluates the local image quality surrounding the QR bounding box.
        """
        try:
            pts = points[0]
            x_coords = [p[0] for p in pts]
            y_coords = [p[1] for p in pts]
            min_x, max_x = int(min(x_coords)), int(max(x_coords))
            min_y, max_y = int(min(y_coords)), int(max(y_coords))
            
            h, w = open_cv_image.shape[:2]
            min_x, max_x = max(0, min_x), min(w, max_x)
            min_y, max_y = max(0, min_y), min(h, max_y)
            
            crop = open_cv_image[min_y:max_y, min_x:max_x]
            if crop.size == 0:
                return {}
                
            gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            
            # Contrast
            std_val = np.std(gray_crop)
            contrast = "High" if std_val > 50 else ("Low contrast" if std_val < 25 else "Medium")
            
            # Brightness
            mean_val = np.mean(gray_crop)
            brightness = "Washed-out (Bright)" if mean_val > 220 else ("Poor lighting / Dark" if mean_val < 60 else "Good")
            
            # Sharpness / Blur
            laplacian_var = cv2.Laplacian(gray_crop, cv2.CV_64F).var()
            sharpness = "Sharp" if laplacian_var > 100 else "Low sharpness"
            
            return {
                "contrast": contrast,
                "brightness": brightness,
                "sharpness": sharpness,
                "raw_contrast_std": round(float(std_val), 2),
                "raw_sharpness_var": round(float(laplacian_var), 2)
            }
        except Exception:
            return {
                "contrast": "Unknown",
                "brightness": "Unknown",
                "sharpness": "Unknown"
            }

    def _classify_qr_payload(self, data: str) -> dict:
        """
        Identifies and classifies the decoded payload format (URL, WIFI, vCard, Email, SMS, Crypto).
        """
        payload_type = "Generic Text"
        details = {}
        
        data_lower = data.lower().strip()
        
        # 1. URL
        if re.match(r'^https?://[^\s/$.?#].[^\s]*$', data, re.IGNORECASE):
            payload_type = "URL / Web Link"
            details["url"] = data
            suspicious_keywords = ["signin", "login", "verify", "secure", "bank", "update", "billing"]
            is_suspicious = any(kw in data_lower for kw in suspicious_keywords)
            details["suspicious_link"] = is_suspicious
        # 2. WIFI
        elif data_lower.startswith("wifi:"):
            payload_type = "Wi-Fi Credentials"
            ssid = re.search(r'S:([^;]+)', data, re.IGNORECASE)
            security = re.search(r'T:([^;]+)', data, re.IGNORECASE)
            details["ssid"] = ssid.group(1) if ssid else "Unknown"
            details["security"] = security.group(1) if security else "None"
        # 3. vCard / Contact
        elif "begin:vcard" in data_lower:
            payload_type = "Contact Card (vCard)"
            name = re.search(r'FN:([^\n\r]+)', data, re.IGNORECASE)
            phone = re.search(r'TEL;?([^:]*):([^\n\r]+)', data, re.IGNORECASE)
            details["name"] = name.group(1).strip() if name else "Unknown"
            details["phone"] = phone.group(2).strip() if phone else "Unknown"
        # 4. Email
        elif data_lower.startswith("mailto:") or re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', data_lower):
            payload_type = "Email Link"
            details["email"] = data.replace("mailto:", "")
        # 5. SMS
        elif data_lower.startswith("sms:") or data_lower.startswith("smsto:"):
            payload_type = "SMS Link"
            parts = data.split(":")
            details["phone"] = parts[1] if len(parts) > 1 else "Unknown"
        # 6. Cryptocurrency address
        elif re.match(r'^(bitcoin:|ethereum:|litecoin:|doge:)?(bc1|[13]|[a-zA-Z0-9]{30,42})', data, re.IGNORECASE):
            payload_type = "Cryptocurrency Address"
            details["address"] = data
            
        return {
            "type": payload_type,
            "details": details
        }
