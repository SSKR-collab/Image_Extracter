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
    """
    VERSION = "1.1.0"
    
    def analyze(self, file_path: str, img, context: dict) -> dict:
        results = {
            "facts": {
                "qr_codes": [],
                "barcodes": []
            },
            "indicators": [],
            "errors": []
        }

        # If image didn't load, we can't scan pixels
        if not img:
            return results

        # Try scanning with pyzbar if available
        if HAS_PYZBAR:
            try:
                decoded_objects = pyzbar.decode(img)
                for obj in decoded_objects:
                    text = obj.data.decode("utf-8", errors="replace")
                    obj_type = obj.type
                    
                    if obj_type == "QRCODE":
                        results["facts"]["qr_codes"].append({
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
                
                # If we parsed successfully, return immediately
                if results["facts"]["qr_codes"] or results["facts"]["barcodes"]:
                    return results
            except Exception as e:
                results["errors"].append({
                    "plugin": self.get_name(),
                    "severity": "warning",
                    "message": f"PyZbar barcode detection failed: {str(e)}"
                })

        # Try scanning with OpenCV if available
        if HAS_OPENCV:
            try:
                # Convert Pillow image to OpenCV BGR format
                open_cv_image = np.array(img.convert("RGB"))
                # Convert RGB to BGR
                open_cv_image = open_cv_image[:, :, ::-1].copy()
                
                # Scan QR Code
                detector = cv2.QRCodeDetector()
                data, bbox, _ = detector.detectAndDecode(open_cv_image)
                if data:
                    results["facts"]["qr_codes"].append({
                        "data": data,
                        "bbox": bbox.tolist() if bbox is not None else None,
                        "confidence": 0.95
                    })
                    results["indicators"].append({
                        "type": "qr_code_detected",
                        "description": f"QR Code detected via OpenCV: '{data[:40]}...'",
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
