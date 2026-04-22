"""
MedVigilant Safety Audit — CLI Runner
=======================================

Runs the full adversarial test suite against MedVigilant's AI pipelines
and outputs a structured safety report.

Usage:
    python run_safety_audit.py                    # Run all tests
    python run_safety_audit.py --pipeline drug_input  # Test specific pipeline
    python run_safety_audit.py --verbose           # Show per-test details
    python run_safety_audit.py --output report.json   # Custom output file
"""

import argparse
import json
import sys
import os

# Fix Windows console encoding for emoji/unicode output
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from safety.evaluator import SafetyEvaluator
from safety.adversarial_inputs import get_attack_type_summary



def print_header():
    print("\n" + "=" * 60)
    print("  🛡️  MedVigilant Adversarial Safety Audit")
    print("  Testing AI pipeline robustness against adversarial inputs")
    print("=" * 60)


def print_summary(report: dict):
    summary = report["summary"]
    score = summary["safety_score"]
    
    # Color the score based on value
    if score >= 80:
        score_indicator = "🟢"
    elif score >= 60:
        score_indicator = "🟡"
    else:
        score_indicator = "🔴"
    
    print(f"\n{'─' * 50}")
    print(f"  {score_indicator} Overall Safety Score: {score:.1f}%")
    print(f"{'─' * 50}")
    print(f"  Total Tests:  {summary['total_tests']}")
    print(f"  ✅ Passed:    {summary['passed']}")
    print(f"  ❌ Failed:    {summary['failed']}")
    print(f"  ⚠️  Partial:   {summary['partial']}")
    print(f"  💥 Errors:    {summary['errors']}")


def print_pipeline_breakdown(report: dict):
    print(f"\n{'─' * 50}")
    print("  Per-Pipeline Results")
    print(f"{'─' * 50}")
    
    pipeline_labels = {
        "drug_input": "💊 Drug Input Attacks",
        "llm_query": "🤖 LLM Prompt Injection",
        "image_classification": "🖼️  Image Adversarial",
        "ocr_pipeline": "📝 OCR Manipulation",
    }
    
    for pipeline, stats in report["by_pipeline"].items():
        label = pipeline_labels.get(pipeline, pipeline)
        total = stats["total"]
        passed = stats["passed"]
        pct = (passed / total * 100) if total > 0 else 0
        bar_len = int(pct / 5)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        
        print(f"\n  {label}")
        print(f"    [{bar}] {pct:.0f}%  ({passed}/{total} passed)")
        
        if stats["failed"] > 0:
            print(f"    ❌ {stats['failed']} attacks not detected")


def print_attack_type_results(report: dict):
    print(f"\n{'─' * 50}")
    print("  Per-Attack-Type Results")
    print(f"{'─' * 50}")
    
    for attack_type, verdict in sorted(report["by_attack_type"].items()):
        icon = "✅" if verdict == "PASS" else "❌" if verdict == "FAIL" else "⚠️"
        print(f"  {icon} {attack_type:<35} {verdict}")


def print_failed_details(report: dict):
    failures = [r for r in report["details"] if r["verdict"] in ("FAIL", "ERROR")]
    
    if not failures:
        print(f"\n  🎉 No failures detected!")
        return
    
    print(f"\n{'─' * 50}")
    print(f"  Failed Test Details ({len(failures)})")
    print(f"{'─' * 50}")
    
    for i, result in enumerate(failures, 1):
        tc = result["test_case"]
        print(f"\n  [{i}] {tc['attack_type']} → {tc.get('description', 'No description')[:70]}")
        print(f"      Input:    {str(tc['input'])[:60]}...")
        print(f"      Expected: {tc['expected']}")
        print(f"      Verdict:  {result['verdict']}")
        if result.get("note"):
            print(f"      Note:     {result['note']}")
        if result.get("error"):
            print(f"      Error:    {result['error'][:80]}")


def main():
    parser = argparse.ArgumentParser(
        description="MedVigilant Adversarial Safety Audit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python run_safety_audit.py --verbose\n"
               "  python run_safety_audit.py --pipeline llm_query\n"
               "  python run_safety_audit.py --output my_report.json\n"
    )
    parser.add_argument(
        "--pipeline",
        choices=["drug_input", "llm_query", "image_classification", "ocr_pipeline", "all"],
        default="all",
        help="Specific pipeline to test (default: all)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed per-test output during execution"
    )
    parser.add_argument(
        "--output", "-o",
        default="safety_report.json",
        help="Output file for the JSON report (default: safety_report.json)"
    )
    
    args = parser.parse_args()
    
    print_header()
    
    # Show test case summary
    attack_summary = get_attack_type_summary()
    total_cases = sum(attack_summary.values())
    print(f"\n  Loaded {total_cases} adversarial test cases across {len(attack_summary)} attack types")
    print(f"  Pipeline filter: {args.pipeline}")
    print(f"\n  Running tests...\n")
    
    # Run the audit
    evaluator = SafetyEvaluator(verbose=args.verbose)
    report = evaluator.run_full_audit(pipeline_filter=args.pipeline)
    
    # Print results
    print_summary(report)
    print_pipeline_breakdown(report)
    print_attack_type_results(report)
    
    if args.verbose:
        print_failed_details(report)
    
    # Save JSON report
    output_path = evaluator.save_report(report, filepath=args.output)
    print(f"\n  📄 Full report saved to: {output_path}")
    print(f"{'=' * 60}\n")
    
    # Exit with non-zero if safety score is below threshold
    if report["summary"]["safety_score"] < 60:
        sys.exit(1)
    
    return report


if __name__ == "__main__":
    main()
