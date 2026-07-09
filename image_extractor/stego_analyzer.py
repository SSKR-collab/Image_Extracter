import math
import os
from PIL import Image
from image_extractor.base_analyzer import BaseAnalyzer


class StegoAnalyzer(BaseAnalyzer):
    """
    Performs steganography analysis: Shannon entropy, block entropy,
    PNG chunk parsing, JPEG comment checks, and Pixel LSB analysis.
    """
    VERSION = "1.1.0"

    _EOF_MARKERS = {
        "JPEG": b"\xff\xd9",
        "PNG": b"\x49\x45\x4e\x44\xae\x42\x60\x82",
        "GIF": b"\x3b"
    }

    def analyze(self, file_path: str, img, context: dict) -> dict:
        results = {
            "facts": {},
            "indicators": [],
            "assessments": {},
            "errors": []
        }

        # Retrieve file bytes
        file_size = os.path.getsize(file_path)
        try:
            with open(file_path, "rb") as f:
                content = f.read()
        except Exception as e:
            results["errors"].append({
                "plugin": self.get_name(),
                "severity": "error",
                "message": f"Could not read file bytes: {str(e)}"
            })
            return results

        # 1. File Entropy calculation
        try:
            entropy = self._calculate_shannon_entropy(content)
            results["facts"]["file_entropy"] = round(entropy, 4)
            
            # Indicator for high overall file entropy
            if entropy > 7.9:
                results["indicators"].append({
                    "type": "high_file_entropy",
                    "description": f"Overall file entropy ({entropy:.4f}) is extremely close to 8.0, indicating encryption or high compression.",
                    "severity": "medium"
                })
        except Exception as e:
            results["errors"].append({
                "plugin": self.get_name(),
                "severity": "warning",
                "message": f"File entropy calculation failed: {str(e)}"
            })

        # 2. Check trailing bytes overlay
        img_format = context.get("file_analyzer", {}).get("facts", {}).get("image_info", {}).get("format")
        if not img_format and img:
            img_format = img.format

        if img_format in self._EOF_MARKERS:
            marker = self._EOF_MARKERS[img_format]
            idx = content.rfind(marker)
            if idx != -1:
                expected_end = idx + len(marker)
                actual_end = len(content)
                if actual_end > expected_end:
                    overlay_len = actual_end - expected_end
                    overlay_bytes = content[expected_end:]
                    overlay_entropy = self._calculate_shannon_entropy(overlay_bytes)
                    
                    results["facts"]["overlay_details"] = {
                        "overlay_size_bytes": overlay_len,
                        "overlay_entropy": round(overlay_entropy, 4),
                        "offset": expected_end
                    }
                    
                    results["indicators"].append({
                        "type": "trailing_overlay",
                        "description": f"Detected trailing overlay of {overlay_len} bytes starting at offset {expected_end}.",
                        "severity": "high"
                    })
                    
                    if overlay_entropy > 7.5:
                        results["indicators"].append({
                            "type": "high_entropy_overlay",
                            "description": f"Trailing overlay has very high entropy ({overlay_entropy:.4f}), strongly suggesting a compressed or encrypted payload.",
                            "severity": "high"
                        })
                else:
                    results["facts"]["overlay_details"] = None
            else:
                results["facts"]["overlay_details"] = None

        # 3. Custom PNG chunk parsing
        if img_format == "PNG":
            try:
                png_chunks = self._parse_png_chunks(content)
                results["facts"]["png_chunks"] = png_chunks
                
                # Check for suspicious custom chunks
                standard_chunks = {"IHDR", "PLTE", "IDAT", "IEND", "tRNS", "cHRM", "gAMA", "iCCP", 
                                   "sBIT", "sRGB", "text", "zTXt", "iTXt", "bKGD", "hIST", "pHYs", 
                                   "sPLT", "tIME", "dSIG", "eXIf"}
                
                for chunk in png_chunks:
                    c_type = chunk["type"]
                    if c_type not in standard_chunks:
                        results["indicators"].append({
                            "type": "custom_png_chunk",
                            "description": f"Non-standard PNG chunk type '{c_type}' detected (size: {chunk['size']} bytes).",
                            "severity": "medium"
                        })
            except Exception as e:
                results["errors"].append({
                    "plugin": self.get_name(),
                    "severity": "warning",
                    "message": f"PNG chunk parsing failed: {str(e)}"
                })

        # 4. JPEG comments checks
        if img_format == "JPEG":
            try:
                comments = self._parse_jpeg_comments(content)
                results["facts"]["jpeg_comments"] = comments
                for comment in comments:
                    results["indicators"].append({
                        "type": "jpeg_comment",
                        "description": f"JPEG comment segment containing: '{comment[:50]}...'",
                        "severity": "low"
                    })
            except Exception as e:
                results["errors"].append({
                    "plugin": self.get_name(),
                    "severity": "warning",
                    "message": f"JPEG comments check failed: {str(e)}"
                })

        # 5. Pixel LSB Heuristic Analysis
        if img:
            try:
                lsb_entropy = self._analyze_pixel_lsb(img)
                results["facts"]["pixel_lsb_entropy"] = round(lsb_entropy, 4)
                
                # If LSB bit distribution is extremely random (entropy close to 1.0)
                # in a normal image, it can indicate LSB steganography.
                if lsb_entropy > 0.9995:
                    # Check if the image is just a single solid color
                    # For a solid color image, LSB entropy would be 0.
                    # High LSB entropy on a rich image is a slight indicator.
                    results["indicators"].append({
                        "type": "suspicious_lsb_entropy",
                        "description": f"Pixel LSB entropy is highly random ({lsb_entropy:.4f}), which can be an indicator of LSB steganography.",
                        "severity": "low"
                    })
            except Exception as e:
                results["errors"].append({
                    "plugin": self.get_name(),
                    "severity": "warning",
                    "message": f"Pixel LSB analysis failed: {str(e)}"
                })

        # 6. Overall Steganography Assessment
        has_overlay = "overlay_details" in results["facts"] and results["facts"]["overlay_details"] is not None
        custom_chunks = any(ind["type"] == "custom_png_chunk" for ind in results["indicators"])
        high_entropy_overlay = any(ind["type"] == "high_entropy_overlay" for ind in results["indicators"])

        stego_suspected = False
        confidence = 0.0
        reasons = []

        if high_entropy_overlay:
            stego_suspected = True
            confidence = 0.90
            reasons.append("High entropy overlay bytes appended after the standard file EOF.")
        elif has_overlay:
            stego_suspected = True
            confidence = 0.70
            reasons.append("Extra trailing bytes detected after the standard file EOF.")
        
        if custom_chunks:
            stego_suspected = True
            confidence = max(confidence, 0.50)
            reasons.append("Non-standard chunk types embedded inside PNG file structure.")

        if stego_suspected:
            results["assessments"]["steganography_detected"] = {
                "result": True,
                "confidence": confidence,
                "reason": " | ".join(reasons)
            }
        else:
            results["assessments"]["steganography_detected"] = {
                "result": False,
                "confidence": 0.90,
                "reason": "No anomalies found in EOF metadata, PNG chunks, or overall byte structure."
            }

        return results

    def _calculate_shannon_entropy(self, data: bytes) -> float:
        if not data:
            return 0.0
        entropy = 0
        length = len(data)
        counts = [0] * 256
        for byte in data:
            counts[byte] += 1
        for count in counts:
            if count > 0:
                p = count / length
                entropy -= p * math.log2(p)
        return entropy

    def _parse_png_chunks(self, data: bytes) -> list:
        chunks = []
        # PNG signature is 8 bytes
        idx = 8
        limit = len(data)
        while idx + 8 < limit:
            try:
                # Read length (4 bytes)
                length = int.from_bytes(data[idx:idx+4], "big")
                # Read type (4 bytes)
                chunk_type = data[idx+4:idx+8].decode("ascii", errors="ignore")
                
                # Check for safety
                if length < 0 or idx + 8 + length > limit:
                    break
                
                chunks.append({
                    "type": chunk_type,
                    "size": length,
                    "offset": idx
                })
                
                # Advance: 4 (length) + 4 (type) + length (data) + 4 (crc)
                idx += 12 + length
                if chunk_type == "IEND":
                    break
            except Exception:
                break
        return chunks

    def _parse_jpeg_comments(self, data: bytes) -> list:
        comments = []
        idx = 0
        limit = len(data)
        # JPEG start marker is \xff\xd8
        if data[:2] != b"\xff\xd8":
            return comments
            
        idx = 2
        while idx + 4 < limit:
            try:
                # Check for marker prefix
                if data[idx] != 0xff:
                    # Search next marker
                    idx = data.find(b"\xff", idx)
                    if idx == -1:
                        break
                    continue
                
                marker = data[idx+1]
                # EOF or SOS marker
                if marker == 0xd9 or marker == 0xda:
                    break
                
                # Read length of the segment (2 bytes)
                length = int.from_bytes(data[idx+2:idx+4], "big")
                
                # Check if it is a COM marker (0xfe)
                if marker == 0xfe:
                    # Comment data is length - 2 (since length includes length field itself)
                    comment_bytes = data[idx+4 : idx+2+length]
                    comments.append(comment_bytes.decode("utf-8", errors="replace").strip())
                
                idx += 2 + length
            except Exception:
                break
        return comments

    def _analyze_pixel_lsb(self, img: Image.Image) -> float:
        # Convert image to grayscale, take first 2000 pixels
        gray = img.convert("L")
        pixels = list(gray.getdata())
        sample_size = min(len(pixels), 2000)
        
        # Collect LSB bits
        lsb_bits = [p & 1 for p in pixels[:sample_size]]
        
        # Calculate entropy of the LSB bit stream
        count_0 = lsb_bits.count(0)
        count_1 = lsb_bits.count(1)
        
        if count_0 == 0 or count_1 == 0:
            return 0.0
            
        p0 = count_0 / sample_size
        p1 = count_1 / sample_size
        return -(p0 * math.log2(p0) + p1 * math.log2(p1))
