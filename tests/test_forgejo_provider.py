"""Focused Forgejo provider API contract tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from mira.exceptions import ProviderError
from mira.models import PRInfo, ReviewComment, ReviewResult, Severity
from mira.providers.forgejo import ForgejoProvider


def _provider() -> ForgejoProvider:
    provider = ForgejoProvider.__new__(ForgejoProvider)
    provider._api = "https://forgejo.example/api/v1"
    provider._token = "token"
    provider._username = None
    return provider


def _response(data: object) -> MagicMock:
    response = MagicMock()
    response.json.return_value = data
    return response


@pytest.mark.asyncio
async def test_get_repo_topics_uses_authoritative_endpoint_and_normalizes() -> None:
    provider = _provider()
    provider._request = AsyncMock(return_value=_response({"topics": ["Mira", " SECURITY "]}))

    topics = await provider.get_repo_topics("org/name", "repo name")

    assert topics == {"mira", "security"}
    provider._request.assert_awaited_once_with(
        "GET", "https://forgejo.example/api/v1/repos/org%2Fname/repo%20name/topics"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "data",
    [None, [], {}, {"topics": None}, {"topics": "mira"}, {"topics": ["mira", 1]}],
)
async def test_get_repo_topics_rejects_malformed_responses(data: object) -> None:
    provider = _provider()
    provider._request = AsyncMock(return_value=_response(data))

    with pytest.raises(ProviderError, match="Malformed"):
        await provider.get_repo_topics("o", "r")


@pytest.mark.asyncio
async def test_get_repo_topics_wraps_api_failures() -> None:
    provider = _provider()
    provider._request = AsyncMock(side_effect=RuntimeError("offline"))

    with pytest.raises(ProviderError, match="Failed to fetch"):
        await provider.get_repo_topics("o", "r")


@pytest.mark.asyncio
async def test_approve_pr_posts_exact_formal_review_payload() -> None:
    provider = _provider()
    provider._request = AsyncMock(return_value=_response({}))
    pr_info = PRInfo(
        title="PR",
        description="",
        base_branch="main",
        head_branch="feature",
        url="https://forgejo.example/o/r/pulls/7",
        number=7,
        owner="o",
        repo="r",
        head_sha="deadbeef",
        platform="forgejo",
    )

    await provider.approve_pr(pr_info, "Mira review completed.")

    provider._request.assert_awaited_once_with(
        "POST",
        "https://forgejo.example/api/v1/repos/o/r/pulls/7/reviews",
        json={
            "event": "APPROVED",
            "body": "Mira review completed.",
            "commit_id": "deadbeef",
        },
    )


@pytest.mark.asyncio
async def test_post_review_raises_when_summary_fallback_fails() -> None:
    provider = _provider()
    inline_error = ProviderError("invalid inline review")
    inline_error.status_code = 422  # type: ignore[attr-defined]
    provider._request = AsyncMock(
        side_effect=[inline_error, ProviderError("summary failed"), _response({})]
    )
    result = ReviewResult(
        summary="Review summary",
        comments=[
            ReviewComment(
                path="app.py",
                line=3,
                end_line=None,
                severity=Severity.WARNING,
                category="correctness",
                title="Problem",
                body="Details",
                confidence=0.9,
            )
        ],
    )
    pr_info = PRInfo(
        title="PR",
        description="",
        base_branch="main",
        head_branch="feature",
        url="https://forgejo.example/o/r/pulls/7",
        number=7,
        owner="o",
        repo="r",
        head_sha="deadbeef",
        platform="forgejo",
    )

    with pytest.raises(ProviderError, match="summary"):
        await provider.post_review(pr_info, result)
