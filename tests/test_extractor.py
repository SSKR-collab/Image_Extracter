import os
import io
import json
import zipfile
import tempfile
import unittest
import shutil
from PIL import Image
from image_extractor.extractor import ImageInfoExtractor
from image_extractor.perceptual_hash import PerceptualHash


class TestImageInfoExtractorSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a temporary directory for test assets
        cls.test_dir = tempfile.mkdtemp()
        
        # 1. Standard PNG
        cls.png_path = os.path.join(cls.test_dir, "test.png")
        img_png = Image.new("RGB", (100, 100), color="blue")
        img_png.save(cls.png_path)
        
        # Write sidecar JSON for test.png
        cls.png_json_path = os.path.join(cls.test_dir, "test.json")
        cls.png_json_content = {
            "caption": "A simple blue graphic image.",
            "scenic_attributes": {
                "habitat": "artificial studio",
                "theme": "solid color"
            },
            "objects": [
                {"label": "blue rectangle", "bbox": [0, 0, 100, 100], "confidence": 0.99}
            ]
        }
        with open(cls.png_json_path, "w", encoding="utf-8") as f:
            json.dump(cls.png_json_content, f)

        # 2. PNG with a hidden ZIP Archive Overlay
        cls.overlay_path = os.path.join(cls.test_dir, "test_overlay.png")
        img_overlay = Image.new("RGB", (50, 50), color="green")
        img_overlay.save(cls.overlay_path)
        
        # Build in-memory ZIP payload
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            zf.writestr("hidden_secret.txt", "Top secret steganography data.")
            zf.writestr("info.json", '{"id": 42}')
        cls.zip_payload = zip_buffer.getvalue()
        
        # Append ZIP to image bytes
        with open(cls.overlay_path, "ab") as f:
            f.write(cls.zip_payload)

        # 3. Document page scan simulation (PNG + Sidecar OCR Text file)
        cls.doc_path = os.path.join(cls.test_dir, "test_doc.png")
        Image.new("RGB", (120, 150), color="white").save(cls.doc_path)
        
        # Write sidecar .txt file
        cls.doc_txt_path = os.path.join(cls.test_dir, "test_doc.txt")
        cls.doc_text = (
            "Mrs. Russell shook her head and cried.\n"
            "\"Never mind, darling, you shall have them one day.\"\n\n"
            "\"I know it sounds lovely,\" said Bob. \"Southend is nice, but Uncle Edward is waiting.\"\n"
            "Here is an IP address: 192.168.1.100 and a URL: https://example.com/api\n"
            "For security testing: powershell -enc AB12CD and AWS key AKIAIOSFODNN7EXAMPLE\n"
            "This is the sixth line to satisfy the line count check."
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

    def test_perceptual_hashes(self):
        with Image.open(self.png_path) as img:
            ahash = PerceptualHash.ahash(img)
            dhash = PerceptualHash.dhash(img)
            
            self.assertIsNotNone(ahash)
            self.assertIsNotNone(dhash)
            self.assertEqual(len(ahash), 16)
            self.assertEqual(len(dhash), 16)
            # Verify they are hexadecimal strings
            int(ahash, 16)
            int(dhash, 16)

    def test_file_and_image_facts(self):
        extractor = ImageInfoExtractor(self.png_path)
        results = extractor.extract_all()
        
        self.assertEqual(results["schema_version"], "1.0.0")
        
        # Check facts
        facts = results["facts"]
        self.assertIn("file_info", facts)
        self.assertEqual(facts["file_info"]["file_name"], "test.png")
        self.assertGreater(facts["file_info"]["size_bytes"], 0)
        
        self.assertIn("image_info", facts)
        self.assertEqual(facts["image_info"]["format"], "PNG")
        self.assertEqual(facts["image_info"]["width"], 100)
        self.assertEqual(facts["image_info"]["aspect_ratio_string"], "1:1")
        self.assertIn("ahash", facts["image_info"]["perceptual_hashes"])

    def test_zip_archive_inspection(self):
        extractor = ImageInfoExtractor(self.overlay_path)
        results = extractor.extract_all()
        
        facts = results["facts"]
        # Stego overlay detected
        self.assertIn("overlay_details", facts)
        self.assertIsNotNone(facts["overlay_details"])
        self.assertEqual(facts["overlay_details"]["overlay_size_bytes"], len(self.zip_payload))
        
        # ZIP entries parsed
        self.assertIn("archive_details", facts)
        arch = facts["archive_details"]
        self.assertIsNotNone(arch)
        self.assertEqual(arch["type"], "ZIP")
        self.assertEqual(arch["file_count"], 2)
        
        filenames = [f["filename"] for f in arch["files"]]
        self.assertIn("hidden_secret.txt", filenames)
        self.assertIn("info.json", filenames)
        
        # Risk assessment for steganography and archives
        assessments = results["assessments"]
        self.assertTrue(assessments["steganography_detected"]["result"])
        self.assertGreaterEqual(assessments["steganography_detected"]["confidence"], 0.70)
        self.assertIn("embedded_archive", assessments)
        self.assertEqual(assessments["embedded_archive"]["risk_level"], "High")

    def test_document_and_nlp_sidecar(self):
        extractor = ImageInfoExtractor(self.doc_path)
        results = extractor.extract_all()
        
        # Verify sidecar loaded indicator
        self.assertTrue(any(ind["type"] == "ocr_sidecar_loaded" for ind in results["indicators"]))
        
        # Verify paragraph and layout reconstruction
        facts = results["facts"]
        self.assertIn("paragraphs", facts)
        self.assertGreaterEqual(len(facts["paragraphs"]), 2)
        
        stats = facts["statistics"]
        self.assertGreater(stats["word_count"], 20)
        self.assertGreater(stats["line_count"], 3)
        
        # Verify language detection and classification
        assessments = results["assessments"]
        self.assertIn("language_detection", assessments)
        self.assertIn("English", assessments["language_detection"]["language"])
        
        self.assertIn("document_classification", assessments)
        # Should detect printed page due to book keywords in content
        self.assertEqual(assessments["document_classification"]["document_type"], "Scanned Printed Book Page")
        self.assertEqual(assessments["document_classification"]["content_type"], "Narrative Text / Prose")

        # NLP Named Entities
        nlp = results["nlp_insights"]
        entities = [e["text"] for e in nlp["entities"]]
        self.assertIn("Mrs. Russell", entities)
        self.assertIn("Bob", entities)
        self.assertIn("Uncle Edward", entities)
        self.assertIn("Southend", entities)
        
        # Dialogue quotes
        quotes = nlp["dialogue"]
        self.assertGreaterEqual(len(quotes), 2)
        
        # Test speaker resolution
        bob_quotes = [q for q in quotes if q["speaker"] == "Bob"]
        self.assertEqual(len(bob_quotes), 1)
        self.assertIn("I know it sounds lovely", bob_quotes[0]["text"])

        # Relationships
        relations = nlp["relationships"]
        self.assertTrue(any(r["person1"] == "Bob" and r["person2"] == "Uncle Edward" for r in relations))

        # Sentiment
        self.assertEqual(nlp["sentiment"]["emotion"], "sadness") # due to "shook her head", "sad"
        self.assertGreater(nlp["sentiment"]["confidence"], 0.5)

    def test_security_scanner_rules(self):
        extractor = ImageInfoExtractor(self.doc_path)
        results = extractor.extract_all()
        
        facts = results["facts"]
        # Check IP/URL facts
        self.assertIn("192.168.1.100", facts["ips"])
        self.assertIn("https://example.com/api", facts["urls"])
        
        # Check security secret findings
        secrets_types = [s["type"] for s in facts["secrets"]]
        self.assertIn("aws_access_key", secrets_types)
        
        # Check command findings
        command_types = [c["type"] for c in facts["suspicious_commands"]]
        self.assertIn("powershell", command_types)

        # Check threat risk assessment
        assessments = results["assessments"]
        self.assertIn("security_risk", assessments)
        self.assertEqual(assessments["security_risk"]["level"], "High") # high due to aws key + powershell + IP
        self.assertGreater(assessments["security_risk"]["score"], 50)

    def test_custom_risk_weights(self):
        # Configure AWS key weight to be very low, and evaluate threat score
        config = {
            "SecurityScanner": {
                "risk_weights": {
                    "secret_aws_access_key": 2, # standard is 40
                    "suspicious_command": 5,     # standard is 35
                    "suspicious_ip": 1           # standard is 15
                }
            }
        }
        extractor = ImageInfoExtractor(self.doc_path, config=config)
        results = extractor.extract_all()
        
        score = results["assessments"]["security_risk"]["score"]
        # With custom low weights, risk score should be much lower than the standard high score
        self.assertLess(score, 50)
        self.assertEqual(results["assessments"]["security_risk"]["level"], "Medium")

    def test_resource_limits(self):
        # Set file limit very small (e.g. 5 bytes) and check for structured limits warning
        config = {
            "FileAnalyzer": {
                "max_file_size_bytes": 5
            }
        }
        extractor = ImageInfoExtractor(self.png_path, config=config)
        results = extractor.extract_all()
        
        # Should log structured error list containing warning/error
        self.assertTrue(any(e["plugin"] == "file_analyzer" and e["severity"] == "error" for e in results["errors"]))
        # Facts should be empty since file analyzer was skipped
        self.assertNotIn("file_info", results["facts"])

    def test_visual_analysis(self):
        extractor = ImageInfoExtractor(self.png_path)
        results = extractor.extract_all()
        
        # Verify visual sidecar indicator
        self.assertTrue(any(ind["type"] == "visual_sidecar_loaded" for ind in results["indicators"]))
        
        # Verify loaded details
        vm = results["facts"]["visual_metadata"]
        self.assertEqual(vm["caption"], "A simple blue graphic image.")
        self.assertEqual(vm["scenic_attributes"]["habitat"], "artificial studio")
        self.assertEqual(len(vm["objects"]), 1)
        self.assertEqual(vm["objects"][0]["label"], "blue rectangle")
        
        # Verify native color extraction (should be 100% Blue!)
        colors = vm["dominant_colors"]
        self.assertGreater(len(colors), 0)
        self.assertEqual(colors[0]["color"], "Blue")
        self.assertEqual(colors[0]["percentage"], 100.0)
        
        # Verify quality metrics
        q = results["facts"]["image_quality"]
        self.assertEqual(q["exposure_assessment"], "Underexposed (Dark)")
        self.assertLess(q["sharpness_score"], 2.0) # solid flat color has minimal edge response
        self.assertEqual(q["blur_classification"], "Defocus / Blurry")
        
        # Verify color palette mapping
        self.assertEqual(vm["color_palette"]["primary_background"], "Blue")


if __name__ == "__main__":
    unittest.main()
