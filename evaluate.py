"""
15-Question Evaluation Script
-------------------------------
Runs the full RAG pipeline against 15 curated questions (including edge cases)
and produces a JSON report + a printed summary table.

Question categories:
  - HR (4 questions)
  - Technical (3 questions)
  - Compliance (3 questions)
  - Edge Cases (5 questions): cross-document, ambiguous, out-of-scope

Usage:
    python evaluate.py
Output:
    evaluation_report.json  ← detailed per-question results
    Console summary table
"""
import json
import time
from src.graph import run_query
from src.config import get_active_provider

# ─── 15 Evaluation Questions ─────────────────────────────────────────────────

EVAL_QUESTIONS = [
    # ── HR (4) ──────────────────────────────────────────────────────────────
    {
        "id": 1, "category": "HR",
        "question": "How many days of annual leave do employees get per year?",
        "expected_namespace": "hr",
        "expected_answerable": True,
        "note": "Direct factual HR question — should retrieve exact number (15 days).",
    },
    {
        "id": 2, "category": "HR",
        "question": "What is the remote work policy? How many days per week are allowed?",
        "expected_namespace": "hr",
        "expected_answerable": True,
        "note": "Multi-part factual question — tests answer completeness.",
    },
    {
        "id": 3, "category": "HR",
        "question": "How does the company's 401k matching work?",
        "expected_namespace": "hr",
        "expected_answerable": True,
        "note": "Benefits question — should retrieve 5% match detail.",
    },
    {
        "id": 4, "category": "HR",
        "question": "What happens if an employee receives a 'Below Expectations' rating twice in a row?",
        "expected_namespace": "hr",
        "expected_answerable": True,
        "note": "Tests retrieval of specific policy consequence (PIP).",
    },

    # ── Technical (3) ─────────────────────────────────────────────────────
    {
        "id": 5, "category": "Technical",
        "question": "How do I authenticate with the ACME API?",
        "expected_namespace": "technical",
        "expected_answerable": True,
        "note": "Common developer question — should return OAuth 2.0 + Bearer token steps.",
    },
    {
        "id": 6, "category": "Technical",
        "question": "What is the API rate limit for the standard tier, and what happens when I exceed it?",
        "expected_namespace": "technical",
        "expected_answerable": True,
        "note": "Multi-part technical question — tests retrieval of rate limit + 429 behaviour.",
    },
    {
        "id": 7, "category": "Technical",
        "question": "What does error code 403 mean and how is it different from 401?",
        "expected_namespace": "technical",
        "expected_answerable": True,
        "note": "Tests retrieval and distinction between two error codes.",
    },

    # ── Compliance (3) ────────────────────────────────────────────────────
    {
        "id": 8, "category": "Compliance",
        "question": "What are the GDPR data retention requirements for employee records?",
        "expected_namespace": "compliance",
        "expected_answerable": True,
        "note": "Specific compliance question — should return 3-year retention detail.",
    },
    {
        "id": 9, "category": "Compliance",
        "question": "How quickly must we notify the supervisory authority after a data breach?",
        "expected_namespace": "compliance",
        "expected_answerable": True,
        "note": "Incident response timing — should return 72-hour GDPR rule.",
    },
    {
        "id": 10, "category": "Compliance",
        "question": "What steps are involved in approving a new third-party vendor?",
        "expected_namespace": "compliance",
        "expected_answerable": True,
        "note": "Multi-step process question — tests ordered retrieval.",
    },

    # ── Edge Cases (5) ────────────────────────────────────────────────────
    {
        "id": 11, "category": "Edge Case — Cross-Document",
        "question": "How do the remote work policy and the data security policy interact? What security steps must remote employees follow?",
        "expected_namespace": "general",
        "expected_answerable": True,
        "note": "Intentionally spans HR and Compliance — tests general namespace + answer synthesis.",
    },
    {
        "id": 12, "category": "Edge Case — Out of Scope",
        "question": "What is our company's stock price today?",
        "expected_namespace": "general",
        "expected_answerable": False,
        "note": "Question the knowledge base cannot answer — should get graceful 'not found' response.",
    },
    {
        "id": 13, "category": "Edge Case — Ambiguous",
        "question": "Tell me about policies.",
        "expected_namespace": "general",
        "expected_answerable": True,
        "note": "Extremely vague question — tests whether the bot asks for clarification or gives a broad overview.",
    },
    {
        "id": 14, "category": "Edge Case — Multi-Document Span",
        "question": "What are ACME's API uptime SLA and how does that relate to our incident response obligations?",
        "expected_namespace": "general",
        "expected_answerable": True,
        "note": "Deliberately spans Technical (SLA) and Compliance (incident response) — tests multi-namespace retrieval.",
    },
    {
        "id": 15, "category": "Edge Case — Wrong Namespace Trap",
        "question": "What is the 401k contribution limit set by the IRS?",
        "expected_namespace": "hr",
        "expected_answerable": False,
        "note": "Looks like an HR question but the answer is NOT in the knowledge base. Tests hallucination resistance.",
    },
]


# ─── Evaluation Runner ────────────────────────────────────────────────────────

def score_routing(expected: str, actual: str) -> bool:
    """Routing is correct if namespaces match, or if expected is 'general'."""
    if expected == "general":
        return True   # general = we don't care which namespace is chosen
    return expected == actual


def score_answerability(expected_answerable: bool, generation: str) -> bool:
    """Rough heuristic: if expected not answerable, check the answer admits it."""
    if expected_answerable:
        return True   # Any non-empty answer is considered a pass
    not_found_phrases = [
        "couldn't find", "not covered", "not available",
        "no information", "out of scope", "knowledge base",
        "don't have", "cannot find", "unable to find",
    ]
    return any(p in generation.lower() for p in not_found_phrases)


def run_evaluation():
    print("\n" + "=" * 70)
    print("  ENTERPRISE RAG BOT — 15-QUESTION EVALUATION REPORT")
    print(f"  LLM: {get_active_provider()}")
    print("=" * 70)

    results     = []
    routing_ok  = 0
    answer_ok   = 0

    for item in EVAL_QUESTIONS:
        print(f"\n[Q{item['id']:02d}] {item['category']}")
        print(f"  Q: {item['question']}")

        start_time = time.time()
        try:
            result  = run_query(item["question"])
            elapsed = round(time.time() - start_time, 2)

            generation  = result.get("generation", "")
            doc_type    = result.get("doc_type", "unknown")
            docs        = result.get("documents", [])
            retries     = result.get("retry_count", 0)
            grading     = result.get("grading_result", "")
            hallucination = result.get("hallucination_result", "")

            routing_correct     = score_routing(item["expected_namespace"], doc_type)
            answerability_ok    = score_answerability(item["expected_answerable"], generation)

            if routing_correct:
                routing_ok += 1
            if answerability_ok:
                answer_ok += 1

            print(f"  Routed to:   {doc_type} (expected: {item['expected_namespace']}) {'✓' if routing_correct else '✗'}")
            print(f"  Docs found:  {len(docs)}")
            print(f"  Grading:     {grading}")
            print(f"  Hallucination: {hallucination}")
            print(f"  Retries:     {retries}")
            print(f"  Latency:     {elapsed}s")
            print(f"  Answer:      {generation[:120]}{'...' if len(generation) > 120 else ''}")

            results.append({
                "id":                     item["id"],
                "category":               item["category"],
                "question":               item["question"],
                "note":                   item["note"],
                "expected_namespace":     item["expected_namespace"],
                "actual_namespace":       doc_type,
                "routing_correct":        routing_correct,
                "expected_answerable":    item["expected_answerable"],
                "answerability_pass":     answerability_ok,
                "docs_retrieved":         len(docs),
                "sources":                [d.metadata.get("source", "?") for d in docs],
                "grading_result":         grading,
                "hallucination_result":   hallucination,
                "retry_count":            retries,
                "latency_seconds":        elapsed,
                "answer":                 generation,
            })

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "id": item["id"], "question": item["question"],
                "error": str(e),
            })

    # ─── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Total questions:      15")
    print(f"  Routing accuracy:     {routing_ok}/15  ({routing_ok/15*100:.0f}%)")
    print(f"  Answerability pass:   {answer_ok}/15  ({answer_ok/15*100:.0f}%)")
    print("=" * 70)

    # Save report
    report_path = "evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "llm_provider":       get_active_provider(),
                "total_questions":    15,
                "routing_accuracy":   f"{routing_ok}/15",
                "answerability_pass": f"{answer_ok}/15",
            },
            "results": results,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n✅  Detailed report saved to: {report_path}")


if __name__ == "__main__":
    run_evaluation()
