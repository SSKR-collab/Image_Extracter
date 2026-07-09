import os
import json
from PIL import Image, ImageFilter, ImageStat
from image_extractor.base_analyzer import BaseAnalyzer


class VisualAnalyzer(BaseAnalyzer):
    """
    Performs visual content analysis. Automatically loads sidecar JSON metadata
    for advanced computer vision data (labels, captions, bounding boxes), 
    and calculates native image quality metrics & dominant colors using Pillow.
    """
    VERSION = "1.1.0"

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
                    "dominant_colors": []
                },
                "image_quality": {
                    "sharpness_score": None,
                    "exposure_assessment": None,
                    "noise_estimate": None
                }
            },
            "indicators": [],
            "assessments": {},
            "errors": []
        }

        # Check sidecar JSON first (e.g. Lion.json next to Lion.png)
        # We can also check if the json is named Lion.visual.json or similar.
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
                    
                    # Merge visual content if it matches target structures
                    v_meta = results["facts"]["visual_metadata"]
                    for key in ["caption", "objects", "scenic_attributes", "aesthetics"]:
                        if key in data:
                            v_meta[key] = data[key]
                            
                    # Optionally copy assessments from sidecar if they exist
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

        # Calculate native visual features if image loaded
        if img:
            try:
                # 1. Native Quality metrics
                results["facts"]["image_quality"] = self._calculate_quality_metrics(img)
                
                # 2. Native Dominant Colors
                results["facts"]["visual_metadata"]["dominant_colors"] = self._calculate_dominant_colors(img)
            except Exception as e:
                results["errors"].append({
                    "plugin": self.get_name(),
                    "severity": "warning",
                    "message": f"Failed to calculate visual quality/colors: {str(e)}"
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
            "noise_estimate": 0.0
        }

        # Convert to Grayscale for quality calculations
        gray = img.convert("L")
        stat = ImageStat.Stat(gray)

        # 1. Exposure Assessment based on mean luminance
        mean_lum = stat.mean[0]
        if mean_lum < 50:
            metrics["exposure_assessment"] = "Underexposed (Dark)"
        elif mean_lum > 220:
            metrics["exposure_assessment"] = "Overexposed (Bright)"
        else:
            metrics["exposure_assessment"] = "Well-exposed (Balanced)"

        # 2. Sharpness Score using average edge gradient
        try:
            edges = gray.filter(ImageFilter.FIND_EDGES)
            edge_stat = ImageStat.Stat(edges)
            # Mean value of edge image acts as a sharpness/complexity score
            metrics["sharpness_score"] = round(edge_stat.mean[0], 2)
        except Exception:
            pass

        # 3. Noise Estimate (rough calculation based on local variance)
        # Blur the image slightly, compare differences
        try:
            blurred = gray.filter(ImageFilter.BLUR)
            diff = Image.new("L", gray.size)
            # Compute difference pixel by pixel (absolute difference)
            # Using ImageChops if available is faster, but simple Stat on diff is fine
            from PIL import ImageChops
            diff = ImageChops.difference(gray, blurred)
            diff_stat = ImageStat.Stat(diff)
            metrics["noise_estimate"] = round(diff_stat.mean[0], 2)
        except Exception:
            pass

        return metrics

    def _calculate_dominant_colors(self, img: Image.Image) -> list:
        # Resize image to 16x16 to average colors out and speed up computation
        thumb = img.resize((16, 16)).convert("RGB")
        pixels = list(thumb.getdata())

        color_counts = {}
        for r, g, b in pixels:
            # Map pixel RGB to closest named color
            closest_name = self._get_closest_color_name(r, g, b)
            color_counts[closest_name] = color_counts.get(closest_name, 0) + 1

        # Sort by frequency
        sorted_colors = sorted(color_counts.items(), key=lambda item: item[1], reverse=True)
        total_pixels = len(pixels)

        dominant = []
        for name, count in sorted_colors[:3]: # top 3 colors
            dominant.append({
                "color": name,
                "percentage": round((count / total_pixels) * 100.0, 1)
            })
        return dominant

    def _get_closest_color_name(self, r, g, b) -> str:
        min_dist = float("inf")
        closest_name = "Other"
        for name, rgb in self.COLOR_MAP.items():
            # Euclidean distance in color space
            dist = (r - rgb[0])**2 + (g - rgb[1])**2 + (b - rgb[2])**2
            if dist < min_dist:
                min_dist = dist
                closest_name = name
        return closest_name
