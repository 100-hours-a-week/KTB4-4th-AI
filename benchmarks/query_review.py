"""Compare generated queries with explicit persona concepts, without model calls."""

import re
import unicodedata

VERSION = "persona-query-review-v1"


def normalized(text):
    return re.sub(r"[^\w]", "", unicodedata.normalize("NFKC", text).casefold())


def contains_any(text, aliases):
    value = normalized(text)
    return any(normalized(alias) in value for alias in aliases if normalized(alias))


def review_queries(persona, queries, exclusions):
    references = (persona or {}).get("expected_queries", [])
    if not references:
        return {"version": VERSION, "score": None, "rows": [], "reason": "no_reference"}
    rows = []
    for reference in references:
        concepts = reference["concepts"]
        candidates = [
            query
            for query in queries
            if query["space"] == reference["space"]
            and reference.get("mode", "self") in query.get("modes", [])
        ]
        best = max(
            candidates,
            key=lambda query: sum(contains_any(query["text"], group) for group in concepts),
            default=None,
        )
        matches = [contains_any(best["text"], group) if best else False for group in concepts]
        rows.append(
            {
                "expected": reference["text"],
                "space": reference["space"],
                "mode": reference.get("mode", "self"),
                "actual": best["text"] if best else None,
                "query_id": best.get("id") if best else None,
                "score": sum(matches) / len(concepts),
                "matched": [
                    group[0] for group, match in zip(concepts, matches, strict=True) if match
                ],
                "missing": [
                    group[0] for group, match in zip(concepts, matches, strict=True) if not match
                ],
            }
        )
    matched = sum(len(row["matched"]) for row in rows)
    total = sum(len(reference["concepts"]) for reference in references)
    exclusion_rows = [
        {
            "expected": reference["text"],
            "actual": next(
                (text for text in exclusions if contains_any(text, reference["aliases"])), None
            ),
        }
        for reference in persona.get("expected_exclusions", [])
    ]
    forbidden = [
        {"query": query["text"], "terms": terms}
        for query in queries
        if (
            terms := [
                term
                for term in persona.get("forbidden_query_terms", [])
                if contains_any(query["text"], [term])
            ]
        )
    ]
    return {
        "version": VERSION,
        "persona_version": persona.get("version"),
        "score": matched / total,
        "matched_conditions": matched,
        "total_conditions": total,
        "rows": rows,
        "exclusions": exclusion_rows,
        "exclusion_score": sum(row["actual"] is not None for row in exclusion_rows)
        / len(exclusion_rows)
        if exclusion_rows
        else None,
        "forbidden_queries": forbidden,
    }
