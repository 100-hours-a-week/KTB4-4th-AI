from copy import deepcopy

import pytest

from benchmarks.persona_dialogue import persona_system
from benchmarks.personas import PERSONAS, VERSION, persona_snapshot
from benchmarks.query_review import review_queries


def reference():
    return {
        "version": "frozen-test",
        "expected_queries": [
            {
                "space": "content",
                "mode": "self",
                "text": "가벼운 베이지색 캠핑 제품",
                "concepts": [["가벼운", "경량"], ["베이지색", "베이지"], ["캠핑"]],
            }
        ],
        "expected_exclusions": [{"text": "강한 향", "aliases": ["강한 향", "진한 향"]}],
        "forbidden_query_terms": ["파란색"],
    }


def query(text, space="content", modes=None):
    return {"id": "q1", "text": text, "space": space, "modes": modes or ["self", "gift"]}


def test_synonyms_spacing_and_case_match_without_extra_model_requests():
    result = review_queries(reference(), [query("경량 베이지 캠핑 관련 제품")], ["진한 향"])
    assert result["score"] == 1
    assert result["rows"][0]["missing"] == []
    assert result["exclusion_score"] == 1
    assert result["persona_version"] == "frozen-test"
    assert result["forbidden_queries"] == []


def test_partial_matches_show_missing_conditions_and_use_the_best_single_query():
    result = review_queries(reference(), [query("가벼운 캠핑 제품"), query("베이지색 소품")], [])
    assert result["score"] == pytest.approx(2 / 3)
    assert result["rows"][0]["actual"] == "가벼운 캠핑 제품"
    assert result["rows"][0]["missing"] == ["베이지색"]
    assert result["matched_conditions"] == 2
    assert result["total_conditions"] == 3
    assert result["exclusion_score"] == 0


@pytest.mark.parametrize(
    "queries",
    [
        [],
        [query("가벼운 베이지색 캠핑 제품", "usage")],
        [query("가벼운 베이지색 캠핑 제품", modes=["gift"])],
    ],
)
def test_empty_or_wrong_space_or_wrong_recipient_is_not_a_match(queries):
    result = review_queries(reference(), queries, [])
    assert result["score"] == 0
    assert result["rows"][0]["actual"] is None


def test_exclusions_and_forbidden_queries_are_reported_separately_from_similarity():
    queries = [query("가벼운 베이지색 캠핑 제품"), query("파란색 캠핑 가방")]
    result = review_queries(reference(), queries, [])
    assert result["score"] == 1
    assert result["exclusions"][0]["actual"] is None
    assert result["forbidden_queries"] == [{"query": "파란색 캠핑 가방", "terms": ["파란색"]}]


def test_legacy_snapshot_has_no_invented_score_or_current_reference():
    assert review_queries({"id": "p01"}, [query("캠핑")], [])["score"] is None
    frozen = deepcopy(reference())
    current = reference()
    current["expected_queries"][0]["concepts"] = [["커피"]]
    assert review_queries(frozen, [query("가벼운 베이지색 캠핑 제품")], [])["score"] == 1


def test_original_and_additional_personas_have_valid_five_turns_and_query_references():
    assert len(PERSONAS) == 30
    assert len({p["id"] for p in PERSONAS}) == 30
    assert [p["age"] for p in PERSONAS[:10]] == [29, 32, 27, 35, 31, 26, 30, 28, 34, 38]
    assert all(p["group"] == "기존 페르소나" for p in PERSONAS[:10])
    assert all(20 <= p["age"] <= 26 and p["group"] == "20~26세 추가" for p in PERSONAS[10:20])
    assert all(27 <= p["age"] <= 34 and p["group"] == "27~34세 추가" for p in PERSONAS[20:])
    assert {p["age"] for p in PERSONAS[20:]} == set(range(27, 35))
    assert len({p["category"] for p in PERSONAS}) >= 12
    for persona in PERSONAS:
        assert len(persona["turns"]) == 5
        assert all(0 < len(turn) <= 500 for turn in persona["turns"])
        assert persona["lifestyle"] and persona["speech_style"]
        snapshot = persona_snapshot(persona["id"])
        assert snapshot["version"] == VERSION
        for expected in persona["expected_queries"]:
            assert expected["space"] in {"content", "usage", "gift"}
            assert expected["mode"] in {"self", "gift"}
            assert expected["concepts"] and all(all(group) for group in expected["concepts"])
        # A reference query must satisfy its own concepts, regardless of other references.
        for expected in persona["expected_queries"]:
            one_reference = {"expected_queries": [expected]}
            result = review_queries(
                one_reference,
                [query(expected["text"], expected["space"], [expected["mode"]])],
                [],
            )
            assert result["score"] == 1, (persona["id"], expected["text"])
        system = persona_system(snapshot)
        assert persona["speech_style"] in system
        for key in ("expected_queries", "expected_exclusions", "forbidden_query_terms"):
            assert key not in system
        snapshot["expected_queries"][0]["concepts"][0][0] = "changed"
        assert persona["expected_queries"][0]["concepts"][0][0] != "changed"
