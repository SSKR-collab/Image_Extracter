import os
import datetime
from PIL import Image
from image_extractor.utils import Timer

# Import default plugins
from image_extractor.ocr_engine import OcrEngine
from image_extractor.doc_parser import DocParser


class ImageInfoExtractor:
    """
    Coordinator class that manages a plugin pipeline to perform
    robust OCR text extraction and visual document layout parsing.
    """
    SCHEMA_VERSION = "1.0.0"

    DEFAULT_PLUGINS = [
        OcrEngine,
        DocParser
    ]

    def __init__(self, file_path: str, config: dict = None):
        """
        Initialize with image file path and optional configuration.
        """
        self.file_path = os.path.abspath(file_path)
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")
        if not os.path.isdir(self.file_path) is False:
            raise ValueError(f"Path is a directory, not a file: {self.file_path}")
        if not os.path.isfile(self.file_path):
            raise ValueError(f"Path is not a file: {self.file_path}")

        self.config = config or {}
        self.plugins = []
        
        # Load and instantiate registered plugins
        plugin_classes = self.config.get("plugins", self.DEFAULT_PLUGINS)
        for cls in plugin_classes:
            self.plugins.append(cls(self.config.get(cls.__name__, {})))

    def extract_all(self) -> dict:
        """
        Executes active plugins in sequence, measuring resource metrics
        and aggregating findings under a versioned JSON schema.
        """
        total_timer = Timer()
        with total_timer:
            # 1. Initialize schema structure
            results = {
                "schema_version": self.SCHEMA_VERSION,
                "metadata": {
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "analyzer_capabilities": {
                        "active_plugins": [],
                        "missing_dependencies": []
                    }
                },
                "facts": {},
                "indicators": [],
                "assessments": {},
                "metrics": {
                    "total_execution_ms": 0.0,
                    "bytes_processed": os.path.getsize(self.file_path),
                    "plugin_timings_ms": {}
                },
                "errors": []
            }

            # 2. Load the image using Pillow
            img = None
            try:
                img = Image.open(self.file_path)
            except Exception as e:
                results["errors"].append({
                    "plugin": "core_loader",
                    "severity": "warning",
                    "message": f"Failed to load image via Pillow: {str(e)}."
                })

            # 3. Execute Plugins in order
            context = {}
            for plugin in self.plugins:
                p_name = plugin.get_name()
                
                # Register capabilities
                results["metadata"]["analyzer_capabilities"]["active_plugins"].append(p_name)
                missing_deps = plugin.check_dependencies()
                if missing_deps:
                    results["metadata"]["analyzer_capabilities"]["missing_dependencies"].extend(missing_deps)

                # Run analyzer and record duration
                p_timer = Timer()
                try:
                    with p_timer:
                        plugin_output = plugin.analyze(self.file_path, img, context)
                except Exception as e:
                    plugin_output = {
                        "errors": [{
                            "plugin": p_name,
                            "severity": "error",
                            "message": f"Plugin crashed with exception: {str(e)}"
                        }]
                    }

                # Record execution metrics
                results["metrics"]["plugin_timings_ms"][p_name] = round(p_timer.interval, 2)
                context[p_name] = plugin_output

                # Aggregate results into schema sections
                self._merge_outputs(results, plugin_output)

            # Close image if opened
            if img:
                try:
                    img.close()
                except Exception:
                    pass

        # Record total execution duration
        results["metrics"]["total_execution_ms"] = round(total_timer.interval, 2)
        return results

    def _merge_outputs(self, target: dict, source: dict):
        """
        Merges plugin outputs into the main unified JSON schema sections.
        """
        # 1. Merge Facts
        if "facts" in source and isinstance(source["facts"], dict):
            target["facts"].update(source["facts"])

        # 2. Merge Indicators
        if "indicators" in source and isinstance(source["indicators"], list):
            target["indicators"].extend(source["indicators"])

        # 3. Merge Assessments
        if "assessments" in source and isinstance(source["assessments"], dict):
            target["assessments"].update(source["assessments"])

        # 4. Merge Errors
        if "errors" in source and isinstance(source["errors"], list):
            target["errors"].extend(source["errors"])
