import os
import tempfile
import unittest
import shutil
from PIL import Image
from image_extractor.extractor import ImageInfoExtractor


class TestImageTextLayoutExtractor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a temporary directory for test assets
        cls.test_dir = tempfile.mkdtemp()
        
        # 1. Simple dummy image
        cls.dummy_path = os.path.join(cls.test_dir, "dummy.png")
        Image.new("RGB", (100, 100), color="white").save(cls.dummy_path)

        # 2. Document page scan simulation (PNG + Sidecar OCR Text file)
        cls.doc_path = os.path.join(cls.test_dir, "test_doc.png")
        Image.new("RGB", (120, 150), color="white").save(cls.doc_path)
        
        # Write sidecar .txt file next to it
        cls.doc_txt_path = os.path.join(cls.test_dir, "test_doc.txt")
        cls.doc_text = (
            "Mrs. Russell shook her head and cried.\n"
            "\"Never mind, darling, you shall have them one day.\"\n\n"
            "\"I know it sounds lovely,\" said Bob. \"Southend is nice, but Uncle Edward is waiting.\"\n"
            "Here is an IP address: 192.168.1.100 and a URL: https://example.com/api\n"
            "when in rome, do as the romans do\n"
            "the customer is always right\n"
            "east, west, home's best"
        )
        with open(cls.doc_txt_path, "w", encoding="utf-8") as f:
            f.write(cls.doc_text)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir)

    def test_invalid_paths(self):
        with self.assertRaises(FileNotFoundError):
            ImageInfoExtractor("missing_file.jpg")
        with self.assertRaises(ValueError):
            ImageInfoExtractor(self.test_dir)

    def test_text_and_layout_extraction(self):
        extractor = ImageInfoExtractor(self.doc_path)
        results = extractor.extract_all()
        
        # Verify sidecar loaded indicator is logged
        self.assertTrue(any(ind["type"] == "ocr_sidecar_loaded" for ind in results["indicators"]))
        
        # Verify paragraph and layout reconstruction
        facts = results["facts"]
        self.assertIn("paragraphs", facts)
        self.assertGreaterEqual(len(facts["paragraphs"]), 2)
        
        # Ensure we have columns information
        self.assertEqual(facts["paragraphs"][0]["column"], 1)
        
        stats = facts["statistics"]
        self.assertGreater(stats["word_count"], 15)
        self.assertGreater(stats["line_count"], 3)
        
        # Verify language detection and classification
        assessments = results["assessments"]
        self.assertIn("language_detection", assessments)
        self.assertIn("English", assessments["language_detection"]["language"])
        
        self.assertIn("document_classification", assessments)
        # Should detect Book Page because of proverbs present in cls.doc_text
        self.assertEqual(assessments["document_classification"]["document_type"], "Book Page")
        self.assertEqual(assessments["document_classification"]["content_type"], "Collection of English Proverbs/Idioms")


if __name__ == "__main__":
    unittest.main()
