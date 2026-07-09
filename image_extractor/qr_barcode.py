import os
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
    Supports a two-stage QR process: Stage 1 (Detection) and Stage 2 (Decoding).
    """
    VERSION = "1.2.0"
    
    def analyze(self, file_path: str, img, context: dict) -> dict:
        results = {
            "facts": {
                "qr_codes": [],
                "barcodes": []
            },
            "indicators": [],
            "errors": []
        }

        # 1. Check sidecar JSON first (e.g. dino.json next to dino.png)
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
                    
                    # Merge qr_codes/barcodes if present
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

        # If sidecar loaded successfully, we can skip pixel detection or append to it
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
                        results["facts"]["qr_codes"].append({
                            "present": True,
                            "decoded": True,
                            "data": text,
                            "bbox": list(obj.rect) if obj.rect else None,
                            "confidence": 1.0
                        })
                        results["indicators"].append({
                            "type": "qr_code_detected",
                            "description": f"QR Code detected: '{text[:40]}...'",
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
                # Convert Pillow image to OpenCV BGR format
                open_cv_image = np.array(img.convert("RGB"))
                open_cv_image = open_cv_image[:, :, ::-1].copy()
                
                detector = cv2.QRCodeDetector()
                # Run Stage 1: Detection
                retval, points = detector.detect(open_cv_image)
                
                if retval and points is not None and len(points) > 0:
                    # Run Stage 2: Decoding
                    data, straight_qrcode = detector.decode(open_cv_image, points)
                    
                    # points is shape (1, 4, 2)
                    bbox_coords = points[0].tolist() if hasattr(points, "tolist") else points.tolist()
                    
                    if data:
                        results["facts"]["qr_codes"].append({
                            "present": True,
                            "decoded": True,
                            "data": data,
                            "bbox": bbox_coords,
                            "confidence": 0.95
                        })
                        results["indicators"].append({
                            "type": "qr_code_detected",
                            "description": f"QR Code detected and decoded: '{data[:40]}...'",
                            "severity": "low"
                        })
                    else:
                        results["facts"]["qr_codes"].append({
                            "present": True,
                            "decoded": False,
                            "bbox": bbox_coords,
                            "status": "QR code detected but could not be decoded.",
                            "reason": "Stylized artwork or decorative overlay obscuring finder patterns or modules.",
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
