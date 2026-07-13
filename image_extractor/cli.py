import sys
import json
import argparse
from image_extractor.extractor import ImageInfoExtractor


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
        description="Robust Offline Image Text & Layout Extractor."
    )
    parser.add_argument("image_path", help="Path to the image file to analyze.")
    parser.add_argument(
        "--json", 
        action="store_true", 
        help="Output results in JSON format."
    )
    parser.add_argument(
        "--pretty", 
        action="store_true", 
        help="If --json is set, output formatted JSON."
    )
    parser.add_argument(
        "--output", 
        help="Path to save the JSON output (implies --json)."
    )

    args = parser.parse_args()

    try:
        extractor = ImageInfoExtractor(args.image_path)
        data = extractor.extract_all()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Save to file if output is specified
    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            print(f"Extraction results written to {args.output}")
            sys.exit(0)
        except Exception as e:
            print(f"Error writing to output file: {e}", file=sys.stderr)
            sys.exit(1)

    # Output to stdout
    if args.json or args.pretty:
        indent = 4 if args.pretty or args.json else None
        print(json.dumps(data, indent=indent, ensure_ascii=False))
    else:
        print_human_readable(data)


def print_human_readable(data: dict):
    """
    Renders a clean terminal report highlighting page structure, column layouts,
    margins, paragraph reading order, and performance timings.
    """
    print("=" * 70)
    print(" ENTERPRISE IMAGE TEXT & LAYOUT EXTRACTOR REPORT ")
    print("=" * 70)

    meta = data.get("metadata", {})
    active = meta.get("analyzer_capabilities", {}).get("active_plugins", [])
    
    print(f"Schema Version: {data.get('schema_version')}")
    print(f"Scan Time:      {meta.get('timestamp')}")
    print(f"Active Plugins: {', '.join(active)}")
    print("-" * 70)

    # Errors & Warnings
    errors = data.get("errors", [])
    if errors:
        print("\n[!] WARNINGS & DIAGNOSTICS:")
        for err in errors:
            print(f"  [{err.get('plugin')}] {err.get('severity').upper()}: {err.get('message')}")
        print("-" * 70)

    # Text & Document Layout Facts
    facts = data.get("facts", {})
    raw_text = facts.get("raw_text", "")
    stats = facts.get("statistics", {})
    
    if raw_text:
        print("\n[+] Extraction & Layout Statistics:")
        print(f"  Words:      {stats.get('word_count')}")
        print(f"  Lines:      {stats.get('line_count')}")
        print(f"  Paragraphs: {stats.get('paragraph_count')}")
        print(f"  Sentences:  {stats.get('sentence_count', 0)}")
        
        # Margins & continuation
        ps = facts.get("page_structure", {})
        bounds = ps.get("content_boundaries", {})
        if bounds and bounds.get("left") is not None:
            print(f"  Content Box: L:{bounds['left']} T:{bounds['top']} R:{bounds['right']} B:{bounds['bottom']}")
        if ps.get("page_continuation_hyphenated"):
            print(f"  Continuation: Suspected continuation hyphenation at page end.")

        # Language & Classification
        lang = data.get("assessments", {}).get("language_detection", {})
        print(f"  Language:   {lang.get('language')} (confidence: {lang.get('confidence')})")
        
        doc_cls = data.get("assessments", {}).get("document_classification", {})
        if doc_cls:
            print(f"  Doc Type:   {doc_cls.get('document_type')} (confidence: {doc_cls.get('document_confidence')})")
            print(f"  Content:    {doc_cls.get('content_type')} (confidence: {doc_cls.get('content_confidence')})")

        # Typography
        typo = facts.get("typography", {})
        if typo:
            if typo.get("indented_paragraphs_count", 0) > 0:
                print(f"  Typography: {typo['indented_paragraphs_count']} indented paragraphs detected")
            if typo.get("uppercase_lines"):
                print(f"  Typography: Uppercase lines: {', '.join(typo['uppercase_lines'][:3])}")

        # Paragraph Layout & Reconstructed Reading Order Flow
        pages = facts.get("pages", [])
        if pages:
            for page in pages:
                page_num = page["page_number"]
                page_name = page.get("page_name")
                header_str = f"Page {page_num}"
                if page_name:
                    header_str += f" ({page_name})"
                print(f"\n==================== {header_str} ====================")
                paras = page.get("paragraphs", [])
                current_col = -1
                for idx, p in enumerate(paras):
                    col = p.get("column", 1)
                    if col != current_col:
                        print(f"\n  --- COLUMN {col} ---")
                        current_col = col
                    print(f"  [Paragraph {idx+1}]")
                    for line in p.get("lines", []):
                        print(f"    {line}")
                    print()
        else:
            paras = facts.get("paragraphs", [])
            if paras:
                print("\n[+] Reconstructed Page Layout & Reading Order Flow:")
                current_col = -1
                for idx, p in enumerate(paras):
                    col = p.get("column", 1)
                    if col != current_col:
                        print(f"\n  --- COLUMN {col} ---")
                        current_col = col
                    print(f"  [Paragraph {idx+1}]")
                    for line in p.get("lines", []):
                        print(f"    {line}")
                    print()

    # Metrics
    metrics = data.get("metrics", {})
    print("-" * 70)
    print(" PERFORMANCE TIMING METRICS")
    print("-" * 70)
    print(f"  Total Scan Time:  {metrics.get('total_execution_ms')} ms")
    print(f"  Bytes Processed:  {metrics.get('bytes_processed')} bytes")
    print("  Plugin Duration breakdown:")
    for p_name, duration in metrics.get("plugin_timings_ms", {}).items():
        print(f"    {p_name:20}: {duration:8.2f} ms")
    print("=" * 70)


if __name__ == "__main__":
    main()
