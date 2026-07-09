import os
import json
from PIL import Image, ImageFilter, ImageStat
from image_extractor.base_analyzer import BaseAnalyzer

try:
    import cv2
    import numpy as np
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False


class VisualAnalyzer(BaseAnalyzer):
    """
    Performs visual content analysis.
    Optionally merges sidecar JSON visual annotations, and runs classical
    OpenCV contour object finders, skin-color face/hand trackers, directional blur auditors,
    and background/foreground accent color palette generators.
    """
    VERSION = "1.2.0"

    # Reference colors for mapping
    COLOR_MAP = {
        "Golden": (218, 165, 32),
        "Brown": (139, 69, 19),
        "Blue": (70, 130, 180),
        "Green": (34, 139, 34),
        "Yellow": (255, 255, 0),
        "White": (245, 245, 245),
        "Black": (20, 20, 20),
        "Gray": (128, 128, 128),
        "Red": (220, 20, 60),
        "Orange": (255, 140, 0),
        "Sky Blue": (135, 206, 235)
    }

    def analyze(self, file_path: str, img, context: dict) -> dict:
        results = {
            "facts": {
                "visual_metadata": {
                    "caption": None,
                    "objects": [],
                    "scenic_attributes": {},
                    "aesthetics": {},
                    "dominant_colors": [],
                    "color_palette": {}
                },
                "image_quality": {
                    "sharpness_score": None,
                    "exposure_assessment": None,
                    "noise_estimate": None,
                    "blur_classification": "Sharp"
                }
            },
            "indicators": [],
            "assessments": {},
            "errors": []
        }

        # 1. Check sidecar JSON first (e.g. Lion.json or qr_hard.json)
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
                    
                    v_meta = results["facts"]["visual_metadata"]
                    for key in ["caption", "objects", "scenic_attributes", "aesthetics"]:
                        if key in data:
                            v_meta[key] = data[key]
                            
                    if "assessments" in data:
                        results["assessments"].update(data["assessments"])
                        
                    results["indicators"].append({
                        "type": "visual_sidecar_loaded",
                        "description": f"Loaded computer vision sidecar metadata file: {os.path.basename(sp)}",
                        "severity": "low"
                    })
                    loaded_sidecar = True
                    break
                except Exception as e:
                    results["errors"].append({
                        "plugin": self.get_name(),
                        "severity": "warning",
                        "message": f"Failed to parse visual sidecar {os.path.basename(sp)}: {str(e)}"
                    })

        # 2. Native Quality & Color calculation
        if img:
            try:
                results["facts"]["image_quality"] = self._calculate_quality_metrics(img)
                dominant = self._calculate_dominant_colors(img)
                results["facts"]["visual_metadata"]["dominant_colors"] = dominant
                
                # Formulate a structured color palette
                if len(dominant) >= 1:
                    results["facts"]["visual_metadata"]["color_palette"] = {
                        "primary_background": dominant[0]["color"],
                        "secondary_foreground": dominant[1]["color"] if len(dominant) > 1 else "Unknown",
                        "accent_colors": [d["color"] for d in dominant[2:]] if len(dominant) > 2 else []
                    }
            except Exception as e:
                results["errors"].append({
                    "plugin": self.get_name(),
                    "severity": "warning",
                    "message": f"Failed to calculate visual quality/colors: {str(e)}"
                })

        # 3. Classical OpenCV Object & Face Detection fallback
        if not loaded_sidecar and img and HAS_OPENCV:
            try:
                open_cv_image = np.array(img.convert("RGB"))
                open_cv_image = open_cv_image[:, :, ::-1].copy() # RGB to BGR
                
                # Detect objects via OpenCV contours
                detected_objects = self._detect_classical_objects(open_cv_image)
                results["facts"]["visual_metadata"]["objects"].extend(detected_objects)
                
                # Detect skin face/hand regions
                skin_regions = self._detect_skin_regions(open_cv_image)
                results["facts"]["visual_metadata"]["objects"].extend(skin_regions)
                
                # Classify local scene type
                scene_info = self._classify_scene_class(open_cv_image, context)
                results["facts"]["visual_metadata"]["scenic_attributes"] = scene_info
                
            except Exception as e:
                results["errors"].append({
                    "plugin": self.get_name(),
                    "severity": "warning",
                    "message": f"OpenCV classical vision processing failed: {str(e)}"
                })

        # Heuristic visual classification assessments if no sidecar loaded
        if not loaded_sidecar and img:
            results["assessments"]["visual_analysis_summary"] = {
                "description": "Basic image attributes and quality stats computed. Detailed labels require a sidecar metadata file.",
                "confidence": 0.90
            }

        return results

    def _calculate_quality_metrics(self, img: Image.Image) -> dict:
        metrics = {
            "sharpness_score": 0.0,
            "exposure_assessment": "Unknown",
            "noise_estimate": 0.0,
            "blur_classification": "Sharp"
        }

        gray = img.convert("L")
        stat = ImageStat.Stat(gray)

        # Exposure Assessment
        mean_lum = stat.mean[0]
        if mean_lum < 50:
            metrics["exposure_assessment"] = "Underexposed (Dark)"
        elif mean_lum > 220:
            metrics["exposure_assessment"] = "Overexposed (Bright)"
        else:
            metrics["exposure_assessment"] = "Well-exposed (Balanced)"

        # Sharpness Score using FIND_EDGES
        try:
            edges = gray.filter(ImageFilter.FIND_EDGES)
            edge_stat = ImageStat.Stat(edges)
            metrics["sharpness_score"] = round(edge_stat.mean[0], 2)
            
            # Blur Classification
            if edge_stat.mean[0] < 5.0:
                metrics["blur_classification"] = "Defocus / Blurry"
            else:
                metrics["blur_classification"] = "Sharp"
        except Exception:
            pass

        # Noise Estimate (standard deviation of difference with blurred version)
        try:
            blurred = gray.filter(ImageFilter.BLUR)
            from PIL import ImageChops
            diff = ImageChops.difference(gray, blurred)
            diff_stat = ImageStat.Stat(diff)
            metrics["noise_estimate"] = round(diff_stat.mean[0], 2)
        except Exception:
            pass

        return metrics

    def _calculate_dominant_colors(self, img: Image.Image) -> list:
        thumb = img.resize((16, 16)).convert("RGB")
        pixels = list(thumb.getdata())

        color_counts = {}
        for r, g, b in pixels:
            closest_name = self._get_closest_color_name(r, g, b)
            color_counts[closest_name] = color_counts.get(closest_name, 0) + 1

        sorted_colors = sorted(color_counts.items(), key=lambda item: item[1], reverse=True)
        total_pixels = len(pixels)

        dominant = []
        for name, count in sorted_colors[:4]: # top 4 colors
            dominant.append({
                "color": name,
                "percentage": round((count / total_pixels) * 100.0, 1)
            })
        return dominant

    def _get_closest_color_name(self, r, g, b) -> str:
        min_dist = float("inf")
        closest_name = "Other"
        for name, rgb in self.COLOR_MAP.items():
            dist = (r - rgb[0])**2 + (g - rgb[1])**2 + (b - rgb[2])**2
            if dist < min_dist:
                min_dist = dist
                closest_name = name
        return closest_name

    def _detect_classical_objects(self, open_cv_image) -> list:
        """
        Uses OpenCV contour bounding boxes to isolate panel, screen, or package layouts.
        """
        objects = []
        try:
            gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edged = cv2.Canny(blurred, 50, 150)
            contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                # Filter out small noise boxes
                if w > 60 and h > 60:
                    aspect_ratio = w / float(h)
                    
                    label = "Generic Object"
                    if 0.8 <= aspect_ratio <= 1.2:
                        label = "Panel / Logo region"
                    elif aspect_ratio > 3.0:
                        label = "Banner / Billboard"
                    elif aspect_ratio < 0.35:
                        label = "Bottle / Column"
                    
                    objects.append({
                        "label": label,
                        "bbox": [x, y, w, h],
                        "confidence": 0.70
                    })
        except Exception:
            pass
        return objects

    def _detect_skin_regions(self, open_cv_image) -> list:
        """
        Skin color range heuristic to locate faces or hands.
        """
        regions = []
        try:
            # Skin range in YCrCb color space
            ycrcb = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2YCrCb)
            mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                if w > 40 and h > 40:
                    regions.append({
                        "label": "Face / Hand candidate",
                        "bbox": [x, y, w, h],
                        "confidence": 0.65
                    })
        except Exception:
            pass
        return regions

    def _classify_scene_class(self, open_cv_image, context) -> dict:
        """
        Analyzes edge line orientations and text yields to guess environment types.
        """
        scene_type = "General Graphic"
        environment = "Unknown"
        
        try:
            gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            
            # Look for lines to flag screenshots or billboards
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=80, maxLineGap=10)
            line_count = len(lines) if lines is not None else 0
            
            # Text count from OCR Engine
            ocr_text = context.get("ocr_engine", {}).get("facts", {}).get("raw_text", "")
            word_count = len(ocr_text.split())
            
            if word_count > 10 and line_count > 15:
                scene_type = "Advertisement / Document"
            elif line_count > 30:
                scene_type = "Screenshot / Diagram"
            elif line_count < 10:
                scene_type = "Natural Photo / Landscape"
                
            # Outdoor / Indoor heuristics based on sky colors in top half
            # Sample upper 20% of image
            top_half = open_cv_image[0:int(h*0.2), :]
            top_rgb = cv2.mean(top_half)[:3]  # BGR average
            
            # Sky Blue / Light Blue: high B and G values
            if top_rgb[0] > 180 and top_rgb[1] > 140: # High Blue and Green
                environment = "Outdoor"
            else:
                environment = "Indoor / Artificial Studio"
        except Exception:
            pass
            
        return {
            "scene": scene_type,
            "environment": environment
        }
