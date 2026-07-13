import os
from image_extractor.ocr_engine import OcrEngine


class ImageTextExtractor:
    """
    Performs robust offline text extraction from images, PDFs, Word documents,
    PowerPoint presentations, Excel spreadsheets, and plain text files.
    """
    def __init__(self, file_path: str, config: dict = None):
        self.file_path = os.path.abspath(file_path)
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")
        if os.path.isdir(self.file_path):
            raise ValueError(f"Path is a directory, not a file: {self.file_path}")
            
        self.config = config or {}
        self.ocr_engine = OcrEngine(self.config)

    def extract_text(self) -> str:
        """
        Extracts clean raw text from the file and returns it as a string.
        """
        results = {
            "facts": {
                "raw_text": ""
            },
            "errors": []
        }
        
        # Load image via Pillow only if it's an image file
        img = None
        ext = os.path.splitext(self.file_path)[1].lower()
        is_image = ext in (".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp", ".gif")
        if is_image:
            try:
                from PIL import Image
                img = Image.open(self.file_path)
            except Exception as e:
                results["errors"].append({
                    "severity": "warning",
                    "message": f"Failed to load image via Pillow: {str(e)}"
                })

        output = self.ocr_engine.analyze(self.file_path, img, {})
        
        if img:
            try:
                img.close()
            except Exception:
                pass
                
        return output.get("facts", {}).get("raw_text", "").strip()
