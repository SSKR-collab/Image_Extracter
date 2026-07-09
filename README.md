# Enterprise Image & Document Security and Extraction Suite

A modular, dependency-optional forensic analysis, optical character recognition (OCR), named entity NLP, visual analysis, and security auditing pipeline for images and document scans.

---

## 1. Quick Start & How to Run

### Installation Requirements
The suite is designed to be lightweight, running out of the box with Python's standard library and **Pillow**. Standard visual, NLP, and security forensic scanning routines operate without heavy external dependencies.

To enable advanced OCR, QR, and barcode capabilities, install optional packages:
```bash
pip install pillow opencv-python pyzbar
```

#### Local Tesseract OCR Setup (Windows)
To use Tesseract OCR:
1. Install Tesseract OCR using the official Windows installer.
2. The suite automatically probes standard installation paths, including `C:\Program Files\Tesseract-OCR\tesseract.exe`.
3. If Tesseract is not installed or not in your PATH, the suite gracefully falls back to looking for **Sidecar files** (e.g. `image_name.txt` or `image_name.ocr`) containing pre-extracted text.

---

### Command Line Interface (CLI)
Scan any image or document file using the CLI tool:
```bash
# Basic scan (outputs human-readable summary to console)
python -m image_extractor.cli test_page.png

# Scan and export full structured JSON schema
python -m image_extractor.cli test_page.png --output results.json
```

---

### Programmatic Python Integration
Initialize the pipeline manager and extract all metadata, visual details, and security indicators:
```python
from image_extractor import ImageInfoExtractor

# Run extraction with default configurations
extractor = ImageInfoExtractor("test_page.png")
results = extractor.extract_all()

# Access data programmatically
print("Format:", results["facts"]["image_info"]["format"])
print("Risk Level:", results["assessments"]["security_risk"]["level"])
```

---

### Using Sidecars for Offline/Advanced Loading
If you don't want to install heavier local engines (like Tesseract, EasyOCR, or CLIP), place sidecar files next to the image. The tool will automatically parse and merge them:
1. **OCR Sidecar**: Create `image_name.txt` containing the document text next to `image_name.png`. The OCR engine will skip image processing and load this text directly.
2. **Visual Metadata Sidecar**: Create `image_name.json` next to `image_name.png` to load pre-calculated objects, bounding boxes, habitat descriptions, or captions.
3. **QR Sidecar**: Create `image_name.json` containing a `"qr_codes"` key to specify coordinates of stylized or obscured QR codes.

---

## 2. Component Pipeline Architecture

The suite operates as a series of pluggable analyzer modules extending `BaseAnalyzer`. Below is an explanation of what each component extracts and the underlying logic used.

```
                  [ Input Image / Document ]
                              │
  ┌───────────────────────────┼───────────────────────────┐
  ▼                           ▼                           ▼
[File Analyzer]        [Stego Analyzer]          [Visual Analyzer]
  - Hashes / Size        - Shannon Entropy         - Exposure/Sharpness
  - Magic Signatures     - LSB Pixel Check         - Dominant Colors
  - ZIP Overlays         - PNG/JPEG Comments       - Sidecar Annotations
  │                           │                           │
  └───────────────────────────┼───────────────────────────┘
                              ▼
                        [OCR Engine]
                         - Tesseract
                         - Text Sidecar Fallback
                              │
                              ▼
                     [Entity NLP Analyzer]
                      - Named Entities (NER)
                      - Proverbs / Idioms
                      - Speakers / Dialogue
                      - Family Relations / Sentiment
                              │
                              ▼
                   [Document Layout Parser]
                    - Sentence / Paragraph Blocks
                    - Indentations & Capitals
                    - Content Boundaries
                    - Classification (Invoice, Book)
                              │
  ┌───────────────────────────┴───────────────────────────┐
  ▼                                                       ▼
[QR & Barcode Decoder]                           [Security Scanner]
  - Stage 1: Bounding Box                          - Aggregates Indicators
  - Stage 2: Data Payloads                         - Secrets/Shell Commands
  - Obscurity Handling                             - Deduplicated Risk Score
                              │
                              ▼
                     [ JSON / CLI Report ]
```

---

### 1. File Analyzer (`file_analyzer.py`)
Extracts fundamental byte metadata and scans for hidden executable signatures.
* **Metadata Extraction**: Gathers file size, extensions, dimensions, color modes, frame counts, and SHA-256 / MD5 hashes.
* **Magic Signatures**: Scans the raw bytes of the file for embedded executable signatures (`MZ` for Windows Portable Executables/DLLs, `ELF` for Linux binaries, `%PDF` for documents, and `PK` for ZIP archives).
* **Overlay ZIP Parser**: If an archive signature `PK` is found appended past the image dimensions, the parser inspects the ZIP central directory structures, listing files hidden in the trapdoor overlay.
* **Resource Limits**: Enforces configurable `max_file_size_bytes` and buffer offsets to safeguard host memory against decompression bombs.

---

### 2. Stego Analyzer (`stego_analyzer.py`)
Runs statistical audits to flag potential steganographic modifications and hidden channels.
* **Shannon Entropy**: Computes overall file entropy. Values exceeding `7.8` (maximum is 8.0) indicate that raw bytes are highly compressed, encrypted, or contain high-density stego payloads.
* **LSB Pixel Randomness**: Analyzes the least significant bits (LSB) of color channels. Natural images exhibit spatial correlation in their lower bits (lower LSB entropy). If LSB entropy approaches `1.0`, it indicates high-density randomness typical of LSB steganography.
* **Chunk Payloads**: Inspects PNG chunk tables to alert on non-standard chunk headers (like custom metadata injections) and extracts JPEG comment tags (`COM`).

---

### 3. Visual Analyzer (`visual_analyzer.py`)
Extracts native quality metrics, classifies dominant colors, and merges advanced object annotations.
* **Native Sharpness & Exposure**: Applies a native Laplacian edge-detection filter to compute pixel gradient averages. Exposure is classified based on mean luminance ranges (Underexposed, Balanced, Overexposed).
* **Dominant Colors**: Resizes the image to `16x16` to calculate average color bins. Using Euclidean distance mapping, it translates RGB values to friendly color tags (e.g. *Golden, Brown, Blue*).
* **Vision Sidecar Merging**: Loads metadata from adjacent JSON configurations (e.g., `Lion.json`) containing advanced visual attributes like image captions, scenic habitats (savanna), aesthetics (wildlife photography), and object bounding boxes.

---

### 4. OCR Engine (`ocr_engine.py`)
Orchestrates text extraction layers.
* **Tesseract & EasyOCR hooks**: Directly binds to local command line engines and library bindings.
* **Sidecar Loaders**: Prompts the user or pipeline when engines are missing, falling back to read text from `.txt` or `.ocr` file structures next to the target image.

---

### 5. Document Layout Parser (`doc_parser.py`)
Converts raw word coordinates into logical text flows, reading margins, and classifying document categories.
* **Paragraph Reconstruction**: Uses vertical overlap margins of word bounding boxes to group characters into lines, and groups lines into paragraphs based on vertical spacing thresholds.
* **Sentence Segmenter**: Segments raw text blocks into lists of sentences, utilizing abbreviation filters to avoid breaking on strings like `Mr.`, `Dr.`, or month names.
* **Typography Analysis**: Detects paragraph indentations based on coordinate margins and captures uppercase header lines.
* **Marginal Boundaries**: Finds the outermost coordinates of the text content to define layout boundaries (left, top, right, bottom).
* **Page Continuation**: Checks if the trailing word in the OCR text ends with a hyphen (e.g. `Sab-`), indicating that text continues onto a subsequent page.
* **Semantic Document Classification**: Combines keyword density audits (e.g. invoice tags, prose keywords, coding markers) and proverb statistics to classify the page format (`Book Page`, `Business Document`, etc.) and content type (`Collection of English Proverbs/Idioms`, `Invoice / Receipt`).

---

### 6. Entity NLP Analyzer (`entity_nlp.py`)
Performs lightweight named entity matching, proverb detection, relationship mapping, and sentiment profiling.
* **Proverb Engine**: Houses a semantic dictionary matching 45+ classic English proverbs.
* **Rule-based NER**: Scans text for capitalized honorific pairings (e.g. `Mrs. Russell`, `Uncle Edward`) and dictionary matches to isolate names and places.
* **Dialogue Speakers**: Uses regular expressions to match spoken quotes and resolve speaking entities.
* **Social Relationships**: Traces family structures using syntax trees (e.g., `"Bob is the nephew of Uncle Edward"`).
* **Sentiment Profiling**: Maps vocabulary vectors to basic emotional categories (Joy, Sadness, Anger) and computes sentiment confidence scores.

---

### 7. QR & Barcode Decoder (`qr_barcode.py`)
Integrates a robust two-stage QR scanning architecture.
* **Stage 1 (Detection)**: Uses OpenCV's QR detector to find finder pattern vertices, rotation angles, and bounding coordinates, even if the QR code is obscured by artwork (e.g. a dragon illustration) or uses custom colors.
* **Stage 2 (Decoding)**: Tries to decode the underlying payload. If decoding fails, it reports `decoded: False` alongside the position coordinates and a failure reason.
* **Sidecar Loader**: Merges manually annotated QR grids from sidecars if the local decoder fails on highly stylized inputs.

---

### 8. Security Scanner (`security_scanner.py`)
Aggregates alerts from previous plugins and executes local threat scanner patterns.
* **Pattern Audits**: Matches regex templates for high-risk keys and scripts:
  * Credentials: AWS access keys, private keys, JWT tokens, API keys.
  * Suspicious Commands: PowerShell executions, reverse shells, curl/wget payloads.
* **Deduplicated Threat Scoring**: Maps indicators to customizable severity scores (0-100). Indicators are deduplicated by type to avoid inflating the risk score from multiple identical detections (e.g., multiple MZ headers in a single file).
