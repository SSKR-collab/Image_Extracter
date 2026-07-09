import sys
import json
import argparse
from image_extractor.extractor import ImageInfoExtractor


def main():
    parser = argparse.ArgumentParser(
        description="Enterprise-Grade Image & Document Security and Extraction Suite."
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
    Renders a detailed terminal report highlighting facts, indicators,
    nlp summaries, security risks, and plugin timing metrics.
    """
    print("=" * 70)
    print(" ENTERPRISE IMAGE & DOCUMENT ANALYSIS REPORT ")
    print("=" * 70)

    # Metadata & Capabilities
    meta = data.get("metadata", {})
    active = meta.get("analyzer_capabilities", {}).get("active_plugins", [])
    missing = meta.get("analyzer_capabilities", {}).get("missing_dependencies", [])
    
    print(f"Schema Version: {data.get('schema_version')}")
    print(f"Scan Time:      {meta.get('timestamp')}")
    print(f"Active Plugins: {', '.join(active)}")
    if missing:
        print(f"Missing (Opt):  {', '.join(missing)}")
    print("-" * 70)

    # Errors & Warnings
    errors = data.get("errors", [])
    if errors:
        print("\n[!] WARNINGS & DIAGNOSTICS:")
        for err in errors:
            print(f"  [{err.get('plugin')}] {err.get('severity').upper()}: {err.get('message')}")
        print("-" * 70)

    # File & Image Facts
    facts = data.get("facts", {})
    fi = facts.get("file_info", {})
    if fi:
        print("\n[+] File Details:")
        print(f"  Name:     {fi.get('file_name')}")
        print(f"  Size:     {fi.get('size_formatted')} ({fi.get('size_bytes')} bytes)")
        print(f"  SHA-256:  {fi.get('sha256_hash')}")
        
    ii = facts.get("image_info", {})
    if ii:
        print("\n[+] Image Properties:")
        print(f"  Format:     {ii.get('format')} ({ii.get('width')}x{ii.get('height')})")
        print(f"  Color Mode: {ii.get('mode')} (Frames: {ii.get('frames')})")
        ph = ii.get("perceptual_hashes", {})
        if ph:
            print(f"  aHash:      {ph.get('ahash')}")
            print(f"  dHash:      {ph.get('dhash')}")

    # Visual Content & Quality Analysis
    vm = facts.get("visual_metadata", {})
    iq = facts.get("image_quality", {})
    if vm or iq:
        print("\n[+] Visual Content & Quality:")
        if vm.get("caption"):
            print(f"  Description: {vm.get('caption')}")
        if vm.get("dominant_colors"):
            colors_str = ", ".join([f"{c['color']} ({c['percentage']}%)" for c in vm.get("dominant_colors")])
            print(f"  Colors:      {colors_str}")
        if iq.get("exposure_assessment"):
            print(f"  Exposure:    {iq.get('exposure_assessment')}")
            print(f"  Sharpness:   {iq.get('sharpness_score')} / 255")
            print(f"  Noise:       {iq.get('noise_estimate')}")
        if vm.get("scenic_attributes"):
            scenic = ", ".join([f"{k}: {v}" for k, v in vm["scenic_attributes"].items()])
            if scenic:
                print(f"  Scene:       {scenic}")
        if vm.get("aesthetics"):
            aesthetics = ", ".join([f"{k}: {v}" for k, v in vm["aesthetics"].items()])
            if aesthetics:
                print(f"  Aesthetic:   {aesthetics}")
        if vm.get("objects"):
            print("  Detected Objects:")
            for obj in vm["objects"][:5]:
                bbox_str = f" bbox: {obj['bbox']}" if obj.get("bbox") else ""
                conf_str = f" (conf: {obj['confidence']})" if obj.get("confidence") is not None else ""
                print(f"    - {obj['label']}{conf_str}{bbox_str}")
            if len(vm["objects"]) > 5:
                print(f"    ... and {len(vm['objects']) - 5} more objects.")

    # QR & Barcodes
    qrs = facts.get("qr_codes", [])
    barcodes = facts.get("barcodes", [])
    if qrs or barcodes:
        print("\n[+] QR & Barcode Detections:")
        if qrs:
            print(f"  QR Codes:   {len(qrs)}")
            for q in qrs:
                if q.get("decoded"):
                    data_val = q.get("data")
                    data_str = f"'{data_val[:40]}...'" if len(data_val) > 40 else f"'{data_val}'"
                    print(f"    - Decoded successfully: {data_str}")
                    pl = q.get("payload_info", {})
                    if pl:
                        print(f"      Type: {pl.get('type')}")
                        if pl.get("details", {}).get("suspicious_link"):
                            print("      [WARNING] Suspicious link detected in URL payload!")
                else:
                    print(f"    - Decoded: False (Status: {q.get('status', 'Failed')})")
                    if q.get("reason"):
                        print(f"      Reason: {q.get('reason')}")
                if q.get("local_quality"):
                    lq = q["local_quality"]
                    print(f"      Local Quality: Contrast: {lq.get('contrast')}, Brightness: {lq.get('brightness')}, Sharpness: {lq.get('sharpness')}")
                if q.get("bbox"):
                    print(f"      Location: {q.get('bbox')}")
                if q.get("error_correction_attempted"):
                    print("      Advanced pre-processing enhancement was attempted.")
        if barcodes:
            print(f"  Barcodes:   {len(barcodes)}")
            for b in barcodes:
                data_val = b.get("data", "")
                data_str = f"'{data_val[:40]}...'" if len(data_val) > 40 else f"'{data_val}'"
                print(f"    - Type: {b.get('type')}, Data: {data_str}")

    # Stego observations & archives
    stego_obs = facts.get("overlay_details")
    if stego_obs:
        print("\n[!] Forensic Overlay Detected:")
        print(f"  Offset:     {stego_obs.get('offset')}")
        print(f"  Size:       {stego_obs.get('overlay_size_bytes')} bytes")
        print(f"  Entropy:    {stego_obs.get('overlay_entropy')}")
        
    arch = facts.get("archive_details")
    if arch:
        print(f"  [+] Extracted ZIP Archive Details ({arch.get('file_count')} files):")
        for f in arch.get("files", [])[:10]: # show first 10
            is_dir = "DIR" if f.get("is_dir") else "FILE"
            print(f"    - {f.get('filename')} ({is_dir}, {f.get('file_size')} bytes)")
        if arch.get("file_count") > 10:
            print(f"    ... and {arch.get('file_count') - 10} more files.")

    # Text & Document Layout Facts
    raw_text = facts.get("raw_text", "")
    stats = facts.get("statistics", {})
    
    if raw_text:
        print("\n[+] Document Layout & OCR Stats:")
        print(f"  Words:      {stats.get('word_count')}")
        print(f"  Lines:      {stats.get('line_count')}")
        print(f"  Paragraphs: {stats.get('paragraph_count')}")
        print(f"  Sentences:  {stats.get('sentence_count', 0)}")
        
        # Margins & continuation
        ps = facts.get("page_structure", {})
        bounds = ps.get("content_boundaries", {})
        if bounds and bounds.get("left") is not None:
            print(f"  Boundaries: L:{bounds['left']} T:{bounds['top']} R:{bounds['right']} B:{bounds['bottom']}")
        if ps.get("page_continuation_hyphenated"):
            print(f"  Continuation: Page continuation hyphenation suspected at end.")

        # Typography
        typo = facts.get("typography", {})
        if typo:
            if typo.get("indented_paragraphs_count", 0) > 0:
                print(f"  Typography: {typo['indented_paragraphs_count']} indented paragraphs")
            if typo.get("uppercase_lines"):
                print(f"  Typography: Uppercase lines detected: {', '.join(typo['uppercase_lines'])}")

        lang = data.get("assessments", {}).get("language_detection", {})
        print(f"  Language:   {lang.get('language')} (confidence: {lang.get('confidence')})")
        
        doc_cls = data.get("assessments", {}).get("document_classification", {})
        if doc_cls:
            print(f"  Doc Type:   {doc_cls.get('document_type')} (confidence: {doc_cls.get('document_confidence')})")
            print(f"  Content:    {doc_cls.get('content_type')} (confidence: {doc_cls.get('content_confidence')})")

    # NLP Insights
    nlp = data.get("nlp_insights", {})
    if nlp:
        proverbs = nlp.get("proverbs", [])
        if proverbs:
            print("\n[+] Extracted Proverbs / Idioms:")
            for p in proverbs:
                print(f"  - \"{p.get('text')}\" (confidence: {p.get('confidence')})")

        entities = nlp.get("entities", [])
        if entities:
            print("\n[+] NLP Named Entities:")
            for ent in entities:
                print(f"  - {ent.get('text')} ({ent.get('type')}, conf: {ent.get('confidence')})")
                
        dialogue = nlp.get("dialogue", [])
        if dialogue:
            print("\n[+] Extracted Spoken Dialogue:")
            for d in dialogue[:5]: # show first 5
                print(f"  - \"{d.get('text')[:60]}...\" (Speaker: {d.get('speaker')})")
            if len(dialogue) > 5:
                print(f"  ... and {len(dialogue) - 5} more quotes.")

        relations = nlp.get("relationships", [])
        if relations:
            print("\n[+] Semantic Relationships:")
            for r in relations:
                note = f" ({r.get('note')})" if r.get("note") else ""
                print(f"  - {r.get('person1')} is {r.get('relation')} of {r.get('person2')}{note}")

        sentiment = nlp.get("sentiment", {})
        if sentiment and sentiment.get("emotion") != "neutral":
            print(f"\n[+] Emotional Sentiment: {sentiment.get('emotion').upper()} (confidence: {sentiment.get('confidence')})")

    # Security Audits
    assessments = data.get("assessments", {})
    sec_risk = assessments.get("security_risk", {})
    if sec_risk:
        print("\n" + "!" * 70)
        print(f" SECURITY AUDIT ASSESSMENT (Risk Level: {sec_risk.get('level').upper()})")
        print("!" * 70)
        print(f"  Threat Score:  {sec_risk.get('score')} / 100")
        print(f"  Confidence:    {sec_risk.get('confidence')}")
        print("  Detections:")
        for reason in sec_risk.get("reason_summary", []):
            print(f"    - {reason}")

    # Metrics
    metrics = data.get("metrics", {})
    print("\n" + "-" * 70)
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
