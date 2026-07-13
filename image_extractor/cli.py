import sys
import argparse
from image_extractor.extractor import ImageTextExtractor


def safe_print(*args, **kwargs):
    try:
        sys.stdout.write(" ".join(str(arg) for arg in args) + kwargs.get("end", "\n"))
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or 'ascii'
        sys.stdout.write(" ".join(str(arg).encode(encoding, errors='replace').decode(encoding) for arg in args) + kwargs.get("end", "\n"))

# Override print
print = safe_print


def main():
    parser = argparse.ArgumentParser(
        description="Robust Offline Multi-Format Text Extractor."
    )
    parser.add_argument("file_path", help="Path to the file to analyze (image, PDF, DOCX, PPTX, XLSX, TXT).")
    parser.add_argument(
        "--output", "-o",
        help="Path to save the extracted plain text to."
    )

    args = parser.parse_args()

    try:
        extractor = ImageTextExtractor(args.file_path)
        text = extractor.extract_text()
    except Exception as e:
        sys.stderr.write(f"Error: {e}\n")
        sys.exit(1)

    # Save to file if output is specified
    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(text)
            sys.exit(0)
        except Exception as e:
            sys.stderr.write(f"Error writing to output file: {e}\n")
            sys.exit(1)

    # Otherwise print to stdout
    print(text)


if __name__ == "__main__":
    main()
