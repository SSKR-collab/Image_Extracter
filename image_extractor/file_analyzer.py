import os
import hashlib
import datetime
import math
import zipfile
import io
from image_extractor.base_analyzer import BaseAnalyzer
from image_extractor.perceptual_hash import PerceptualHash


class FileAnalyzer(BaseAnalyzer):
    """
    Analyzes file system metadata, basic image structure, perceptual hashes,
    scans for embedded executables/archives, and inspects ZIP overlays.
    """
    VERSION = "1.1.0"
    
    # Common file signatures to detect embedded payloads
    SIGNATURES = {
        b"PK\x03\x04": "ZIP Archive / OpenXML Document (e.g., Office Doc)",
        b"MZ": "Windows PE Executable / DLL",
        b"%PDF-": "PDF Document",
        b"\x7fELF": "Linux ELF Executable",
        b"Rar!\x1a\x07\x00": "RAR Archive",
        b"7z\xbc\xaf\x27\x1c": "7z Archive"
    }

    def analyze(self, file_path: str, img, context: dict) -> dict:
        results = {
            "facts": {},
            "indicators": [],
            "assessments": {},
            "errors": []
        }

        # Retrieve configuration limits
        max_file_size = self.config.get("max_file_size_bytes", 100 * 1024 * 1024)  # Default 100MB
        scan_buf_limit = self.config.get("scan_buffer_size_bytes", 10 * 1024 * 1024)  # Default 10MB

        # 1. Check file size limits first
        file_size = os.path.getsize(file_path)
        if file_size > max_file_size:
            results["errors"].append({
                "plugin": self.get_name(),
                "severity": "error",
                "message": f"File size ({file_size} bytes) exceeds configured limit of {max_file_size} bytes. Skipping deep analysis."
            })
            return results

        # 2. File stats and hashes
        try:
            stat = os.stat(file_path)
            ctime = datetime.datetime.fromtimestamp(stat.st_ctime, datetime.timezone.utc).isoformat()
            mtime = datetime.datetime.fromtimestamp(stat.st_mtime, datetime.timezone.utc).isoformat()
            
            md5_hash = hashlib.md5()
            sha256_hash = hashlib.sha256()
            
            # Read file in chunks
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    md5_hash.update(chunk)
                    sha256_hash.update(chunk)

            results["facts"]["file_info"] = {
                "file_name": os.path.basename(file_path),
                "file_path": os.path.abspath(file_path),
                "size_bytes": file_size,
                "size_formatted": self._format_size(file_size),
                "created_time": ctime,
                "modified_time": mtime,
                "md5_hash": md5_hash.hexdigest(),
                "sha256_hash": sha256_hash.hexdigest()
            }
        except Exception as e:
            results["errors"].append({
                "plugin": self.get_name(),
                "severity": "error",
                "message": f"File stat or hash error: {str(e)}"
            })

        # 3. Image attributes and perceptual hashes
        if img:
            try:
                width, height = img.size
                gcd = math.gcd(width, height)
                aspect_ratio_str = f"{width // gcd}:{height // gcd}" if gcd > 0 else "unknown"
                
                # Check animation / multi-frame
                is_animated = getattr(img, "is_animated", False)
                n_frames = getattr(img, "n_frames", 1)

                results["facts"]["image_info"] = {
                    "format": img.format,
                    "width": width,
                    "height": height,
                    "aspect_ratio": width / height if height > 0 else 0,
                    "aspect_ratio_string": aspect_ratio_str,
                    "mode": img.mode,
                    "is_animated": is_animated,
                    "frames": n_frames,
                    "perceptual_hashes": {
                        "ahash": PerceptualHash.ahash(img),
                        "dhash": PerceptualHash.dhash(img)
                    }
                }
            except Exception as e:
                results["errors"].append({
                    "plugin": self.get_name(),
                    "severity": "warning",
                    "message": f"Failed to parse image characteristics: {str(e)}"
                })

        # 4. Embedded signature scanning & overlay check
        try:
            with open(file_path, "rb") as f:
                # If the file is very large, limit signature scan buffer to the first and last parts
                if file_size > scan_buf_limit:
                    content_start = f.read(scan_buf_limit // 2)
                    f.seek(-scan_buf_limit // 2, os.SEEK_END)
                    content_end = f.read()
                    content = content_start + content_end
                    results["indicators"].append({
                        "type": "partial_scan",
                        "description": f"File size exceeds scan buffer limit. Scanned only first/last {scan_buf_limit // 2} bytes.",
                        "severity": "low"
                    })
                else:
                    content = f.read()

            # Scan magic signatures
            embedded_detections = []
            for sig, label in self.SIGNATURES.items():
                offset = 0
                while True:
                    idx = content.find(sig, offset)
                    if idx == -1:
                        break
                    
                    # Ignore signature if it is at offset 0 (typical for zip files or pdf files if we are analyzing them, 
                    # but since this is an image extractor, any of these magic signatures occurring inside the image is suspicious).
                    # JPEG starts with \xff\xd8, PNG starts with \x89PNG, GIF starts with GIF8
                    is_image_header = False
                    if idx == 0:
                        img_format = results["facts"].get("image_info", {}).get("format")
                        if sig == b"PK\x03\x04" and img_format in ("ZIP", "DOCX", "ODT"):
                            is_image_header = True # not suspicious if file itself is zip
                    
                    if not is_image_header:
                        embedded_detections.append({
                            "signature_label": label,
                            "offset": idx,
                            "hex": sig.hex()
                        })
                        results["indicators"].append({
                            "type": "embedded_signature",
                            "description": f"Embedded {label} signature found at byte offset {idx}.",
                            "severity": "high" if sig == b"MZ" else "medium"
                        })
                    offset = idx + len(sig)

            results["facts"]["embedded_files"] = embedded_detections

            # Archive inspection (ZIP)
            results["facts"]["archive_details"] = None
            if embedded_detections:
                # Find the first embedded ZIP signature
                zip_sigs = [d for d in embedded_detections if d["signature_label"].startswith("ZIP")]
                if zip_sigs:
                    first_zip_offset = zip_sigs[0]["offset"]
                    # If this offset is near the end, try extracting the bytes as a ZIP file
                    zip_data_bytes = content[first_zip_offset:]
                    try:
                        with zipfile.ZipFile(io.BytesIO(zip_data_bytes)) as zf:
                            file_list = []
                            for info in zf.infolist():
                                file_list.append({
                                    "filename": info.filename,
                                    "file_size": info.file_size,
                                    "compress_size": info.compress_size,
                                    "is_dir": info.is_dir()
                                })
                            results["facts"]["archive_details"] = {
                                "type": "ZIP",
                                "offset": first_zip_offset,
                                "file_count": len(file_list),
                                "files": file_list
                            }
                            results["indicators"].append({
                                "type": "valid_zip_payload",
                                "description": f"Successfully parsed hidden ZIP file containing {len(file_list)} entries at offset {first_zip_offset}.",
                                "severity": "high"
                            })
                    except Exception:
                        pass # Not a valid ZIP file, signature was a false positive or encrypted

            # Formulate embedded file assessments
            has_mz = any(d["signature_label"].startswith("Windows") for d in embedded_detections)
            has_zip = results["facts"]["archive_details"] is not None

            if has_mz:
                results["assessments"]["embedded_executable"] = {
                    "risk_level": "High",
                    "confidence": 0.95,
                    "reason": "Windows Executable signature (MZ) discovered inside the image byte stream."
                }
            if has_zip:
                results["assessments"]["embedded_archive"] = {
                    "risk_level": "High",
                    "confidence": 0.99,
                    "reason": f"A fully functional ZIP archive was extracted from byte offset {zip_sigs[0]['offset']}."
                }

        except Exception as e:
            results["errors"].append({
                "plugin": self.get_name(),
                "severity": "error",
                "message": f"Signature scan execution failed: {str(e)}"
            })

        return results

    def _format_size(self, size_bytes: int) -> str:
        if size_bytes == 0:
            return "0 B"
        size_name = ("B", "KB", "MB", "GB")
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return f"{s} {size_name[i]}"
