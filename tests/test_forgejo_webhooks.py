"""Tests for the Forgejo webhook route + author filter dispatch."""

from __future__ import annotations

import hashlib
import hmac
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from mira.config import FilterConfig, ForgejoConfig, MiraConfig, ReviewConfig
from mira.exceptions import ProviderError
from mira.models import PRInfo, ReviewResult
from mira.platforms.forgejo.auth import ForgejoTokenAuth
from mira.platforms.forgejo.webhook import (
    _approve_skipped_pr,
    handle_forgejo_note,
    handle_forgejo_pr,
)
from mira.platforms.server import create_app

FJ_SECRET = "fj-secret"
BOT = "mira-bot"


@pytest.fixture
def forgejo_auth():  # noqa: ANN201
    auth = ForgejoTokenAuth("tok")
    auth.get_bot_identity = AsyncMock(return_value="mira-bot")  # type: ignore[method-assign]
    return auth


@pytest.fixture
def app(forgejo_auth):  # noqa: ANN001, ANN201
    return create_app(
        app_auth=None,
        webhook_secret=None,
        bot_name=BOT,
        forgejo_auth=forgejo_auth,
        forgejo_webhook_secret=FJ_SECRET,
    )


@pytest.fixture
async def client(app) -> AsyncClient:  # noqa: ANN001
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def _sign(payload_bytes: bytes) -> str:
    return hmac.new(FJ_SECRET.encode(), payload_bytes, hashlib.sha256).hexdigest()


def _pr_payload(action: str, login: str):
    return {
        "action": action,
        "pull_request": {
            "number": 7,
            "title": "PR",
            "body": "",
            "labels": [],
            "html_url": "https://forgejo.example/o/r/pulls/7",
        },
        "repository": {
            "full_name": "o/r",
            "html_url": "https://forgejo.example/o/r",
            "default_branch": "main",
            "private": False,
        },
        "sender": {"login": login},
    }


def _comment_payload(body: str, login: str):
    return {
        "action": "created",
        "is_pull": True,
        "comment": {"id": 12, "body": body, "user": {"username": login}},
        "issue": {"number": 7},
        "repository": {
            "full_name": "o/r",
            "html_url": "https://forgejo.example/o/r",
        },
        "sender": {"login": login},
    }


def _push_payload(login: str):
    return {
        "ref": "refs/heads/main",
        "commits": [{"added": ["new.py"], "modified": [], "removed": []}],
        "repository": {"full_name": "o/r", "default_branch": "main"},
        "sender": {"login": login},
    }


def _pr_info() -> PRInfo:
    return PRInfo(
        title="PR",
        description="",
        base_branch="main",
        head_branch="feature",
        url="https://forgejo.example/o/r/pulls/7",
        number=7,
        owner="o",
        repo="r",
        head_sha="abc123",
        platform="forgejo",
    )


def _review_result(**kwargs) -> ReviewResult:
    kwargs.setdefault("reviewed_sha", "abc123")
    return ReviewResult(**kwargs)


async def _post_event(client: AsyncClient, event: str, payload: dict) -> object:
    body = json.dumps(payload).encode()
    return await client.post(
        "/forgejo/webhook",
        content=body,
        headers={
            "X-Forgejo-Event": event,
            "X-Forgejo-Signature": _sign(body),
        },
    )


@pytest.mark.asyncio
async def test_pr_opened_blocked_author_filtered(client):
    """Blocked author with [bot] suffix is filtered for PR opened."""
    with (
        patch("mira.platforms.forgejo.webhook.handle_forgejo_pr", new=AsyncMock()) as h,
        patch(
            "mira.platforms.forgejo.webhook.load_config",
            return_value=MiraConfig(filter=FilterConfig(blocked_authors=["dependabot"])),
        ),
    ):
        payload = _pr_payload("opened", "dependabot[bot]")
        body = json.dumps(payload).encode()
        resp = await client.post(
            "/forgejo/webhook",
            content=body,
            headers={
                "X-Forgejo-Event": "pull_request",
                "X-Forgejo-Signature": _sign(body),
            },
        )
    assert resp.status_code == 200
    assert resp.json()["status"] == "ignored"
    h.assert_not_called()


@pytest.mark.asyncio
async def test_pr_opened_filtered_author_gets_policy_skip_approval(client):
    config = MiraConfig(
        filter=FilterConfig(blocked_authors=["dependabot"]),
        forgejo=ForgejoConfig(approve_skipped_reviews=True),
    )
    with (
        patch("mira.platforms.forgejo.webhook.load_config", return_value=config),
        patch("mira.platforms.forgejo.webhook._approve_skipped_pr", new=AsyncMock()) as approve,
        patch("mira.platforms.forgejo.webhook.handle_forgejo_pr", new=AsyncMock()) as handler,
    ):
        resp = await _post_event(
            client,
            "pull_request",
            _pr_payload("opened", "dependabot[bot]"),
        )

    assert resp.json()["status"] == "ignored"
    handler.assert_not_awaited()
    approve.assert_awaited_once()
    assert "excluded by policy" in approve.await_args.args[2]


@pytest.mark.asyncio
async def test_pr_opened_allowed_author_not_filtered(client):
    """Non-blocked author passes through the filter."""
    with (
        patch("mira.platforms.forgejo.webhook.handle_forgejo_pr", new=AsyncMock()) as h,
        patch(
            "mira.platforms.forgejo.webhook.load_config",
            return_value=MiraConfig(filter=FilterConfig(blocked_authors=["dependabot"])),
        ),
    ):
        payload = _pr_payload("opened", "alice")
        body = json.dumps(payload).encode()
        resp = await client.post(
            "/forgejo/webhook",
            content=body,
            headers={
                "X-Forgejo-Event": "pull_request",
                "X-Forgejo-Signature": _sign(body),
            },
        )
    assert resp.status_code == 200
    assert resp.json()["status"] == "processing"
    h.assert_awaited_once()


@pytest.mark.asyncio
async def test_pr_opened_allowlist_filters_off_list(client):
    """Allowlist set but author not on it → filtered."""
    with (
        patch("mira.platforms.forgejo.webhook.handle_forgejo_pr", new=AsyncMock()) as h,
        patch(
            "mira.platforms.forgejo.webhook.load_config",
            return_value=MiraConfig(filter=FilterConfig(allowed_authors=["alice"])),
        ),
    ):
        payload = _pr_payload("opened", "bob")
        body = json.dumps(payload).encode()
        resp = await client.post(
            "/forgejo/webhook",
            content=body,
            headers={
                "X-Forgejo-Event": "pull_request",
                "X-Forgejo-Signature": _sign(body),
            },
        )
    assert resp.status_code == 200
    assert resp.json()["status"] == "ignored"
    h.assert_not_called()


@pytest.mark.asyncio
async def test_comment_review_bypass(client):
    """Manual @mira-bot review comment bypasses the author filter."""
    with (
        patch("mira.platforms.forgejo.webhook.handle_forgejo_note", new=AsyncMock()) as h,
        patch(
            "mira.platforms.forgejo.webhook.load_config",
            return_value=MiraConfig(filter=FilterConfig(blocked_authors=["dependabot"])),
        ),
    ):
        payload = _comment_payload("@mira-bot review", "dependabot[bot]")
        body = json.dumps(payload).encode()
        resp = await client.post(
            "/forgejo/webhook",
            content=body,
            headers={
                "X-Forgejo-Event": "issue_comment",
                "X-Forgejo-Signature": _sign(body),
            },
        )
    assert resp.status_code == 200
    assert resp.json()["status"] == "processing"
    h.assert_awaited_once()


@pytest.mark.asyncio
async def test_comment_non_review_no_bypass(client):
    """Non-review command (pause) by blocked author does NOT bypass filter."""
    with (
        patch("mira.platforms.forgejo.webhook.handle_forgejo_note", new=AsyncMock()) as h,
        patch(
            "mira.platforms.forgejo.webhook.load_config",
            return_value=MiraConfig(filter=FilterConfig(blocked_authors=["alice"])),
        ),
    ):
        payload = _comment_payload("@mira-bot pause", "alice")
        body = json.dumps(payload).encode()
        resp = await client.post(
            "/forgejo/webhook",
            content=body,
            headers={
                "X-Forgejo-Event": "issue_comment",
                "X-Forgejo-Signature": _sign(body),
            },
        )
    assert resp.status_code == 200
    assert resp.json()["status"] == "ignored"
    h.assert_not_called()


@pytest.mark.parametrize(
    ("event", "payload", "handler_path"),
    [
        (
            "pull_request",
            _pr_payload("opened", "alice"),
            "mira.platforms.forgejo.webhook.handle_forgejo_pr",
        ),
        (
            "push",
            _push_payload("alice"),
            "mira.platforms.forgejo.webhook.handle_forgejo_push",
        ),
        (
            "issue_comment",
            _comment_payload("@mira-bot review", "alice"),
            "mira.platforms.forgejo.webhook.handle_forgejo_note",
        ),
    ],
)
@pytest.mark.parametrize(
    ("topic_result", "expected_status"),
    [("present", "processing"), ("absent", "ignored"), ("error", "ignored")],
)
@pytest.mark.asyncio
async def test_required_topic_gates_relevant_dispatch(
    client, event, payload, handler_path, topic_result, expected_status
):
    payload["repository"]["topics"] = ["required-topic"]  # Payload topics are untrusted.
    provider = MagicMock()
    if topic_result == "present":
        provider.get_repo_topics = AsyncMock(return_value={"required-topic"})
    elif topic_result == "absent":
        provider.get_repo_topics = AsyncMock(return_value=set())
    else:
        provider.get_repo_topics = AsyncMock(side_effect=ProviderError("topics unavailable"))

    config = MiraConfig(forgejo=ForgejoConfig(required_review_topic=" Required-Topic "))
    with (
        patch("mira.platforms.forgejo.webhook.create_provider", return_value=provider),
        patch("mira.platforms.forgejo.webhook.load_config", return_value=config),
        patch(handler_path, new=AsyncMock()) as handler,
    ):
        resp = await _post_event(client, event, payload)

    assert resp.status_code == 200
    assert resp.json()["status"] == expected_status
    provider.get_repo_topics.assert_awaited_once_with("o", "r")
    if expected_status == "processing":
        handler.assert_awaited_once()
    else:
        handler.assert_not_awaited()


@pytest.mark.asyncio
async def test_absent_topic_schedules_formal_skipped_approval(client):
    provider = MagicMock()
    provider.get_repo_topics = AsyncMock(return_value=set())
    config = MiraConfig(
        forgejo=ForgejoConfig(
            required_review_topic="required-topic",
            approve_skipped_reviews=True,
        )
    )
    with (
        patch("mira.platforms.forgejo.webhook.create_provider", return_value=provider),
        patch("mira.platforms.forgejo.webhook.load_config", return_value=config),
        patch("mira.platforms.forgejo.webhook._approve_skipped_pr", new=AsyncMock()) as approve,
        patch("mira.platforms.forgejo.webhook.handle_forgejo_pr", new=AsyncMock()) as handler,
    ):
        resp = await _post_event(client, "pull_request", _pr_payload("opened", "alice"))

    assert resp.json()["status"] == "ignored"
    handler.assert_not_awaited()
    approve.assert_awaited_once()
    assert approve.await_args.args[1] == "https://forgejo.example/o/r/pulls/7"
    assert "Mira review skipped" in approve.await_args.args[2]
    assert "required topic `required-topic` is absent" in approve.await_args.args[2]


@pytest.mark.asyncio
async def test_topic_lookup_error_never_approves_filtered_author(client):
    provider = MagicMock()
    provider.get_repo_topics = AsyncMock(side_effect=ProviderError("topics unavailable"))
    config = MiraConfig(
        filter=FilterConfig(blocked_authors=["alice"]),
        forgejo=ForgejoConfig(
            required_review_topic="required-topic",
            approve_skipped_reviews=True,
        ),
    )
    with (
        patch("mira.platforms.forgejo.webhook.create_provider", return_value=provider),
        patch("mira.platforms.forgejo.webhook.load_config", return_value=config),
        patch("mira.platforms.forgejo.webhook._approve_skipped_pr", new=AsyncMock()) as approve,
    ):
        resp = await _post_event(client, "pull_request", _pr_payload("opened", "alice"))

    assert resp.json()["status"] == "ignored"
    approve.assert_not_awaited()


@pytest.mark.parametrize("policy", ["synchronize", "ignore", "paused"])
@pytest.mark.asyncio
async def test_enrolled_policy_skip_submits_approval(forgejo_auth, policy):
    payload = _pr_payload("opened", "alice")
    review = ReviewConfig()
    if policy == "synchronize":
        payload["action"] = "synchronized"
        review = ReviewConfig(review_on_synchronize=False)
    elif policy == "ignore":
        payload["pull_request"]["body"] = "@mira-bot ignore"
    else:
        payload["pull_request"]["labels"] = [{"name": "mira-paused"}]

    config = MiraConfig(
        review=review,
        forgejo=ForgejoConfig(approve_skipped_reviews=True),
    )
    with (
        patch("mira.platforms.forgejo.webhook.load_config", return_value=config),
        patch("mira.platforms.forgejo.webhook._approve_skipped_pr", new=AsyncMock()) as approve,
    ):
        await handle_forgejo_pr(payload, forgejo_auth, BOT)

    approve.assert_awaited_once()
    assert "Mira review skipped" in approve.await_args.args[2]


@pytest.mark.asyncio
async def test_skipped_approval_requires_current_sha(forgejo_auth):
    provider = MagicMock()
    provider.get_pr_info = AsyncMock(return_value=PRInfo(**{**_pr_info().__dict__, "head_sha": ""}))
    provider.approve_pr = AsyncMock()

    with patch("mira.platforms.forgejo.webhook.create_provider", return_value=provider):
        await _approve_skipped_pr(forgejo_auth, _pr_info().url, "Skipped")

    provider.approve_pr.assert_not_awaited()


@pytest.mark.parametrize(
    ("review_outcome", "should_approve"),
    [
        (_review_result(), True),
        (_review_result(reviewed_sha=""), False),
        (_review_result(skipped_reason="No changed files required review"), False),
        (None, False),
        (_review_result(delivery_failed=True), False),
        (RuntimeError("review failed"), False),
    ],
)
@pytest.mark.asyncio
async def test_automatic_review_approves_only_actual_success(
    forgejo_auth, review_outcome, should_approve
):
    provider = MagicMock()
    provider.get_pr_info = AsyncMock(return_value=_pr_info())
    provider.approve_pr = AsyncMock()
    if isinstance(review_outcome, Exception):
        review = AsyncMock(side_effect=review_outcome)
    else:
        review = AsyncMock(return_value=review_outcome)

    with (
        patch(
            "mira.platforms.forgejo.webhook.load_config",
            return_value=MiraConfig(forgejo=ForgejoConfig(approve_successful_reviews=True)),
        ),
        patch("mira.platforms.forgejo.webhook.create_provider", return_value=provider),
        patch("mira.platforms.index_handlers._get_app_db", return_value=MagicMock()),
        patch("mira.platforms.handlers.run_pr_review", new=review),
    ):
        await handle_forgejo_pr(_pr_payload("opened", "alice"), forgejo_auth, BOT)

    if should_approve:
        provider.approve_pr.assert_awaited_once_with(_pr_info(), "Mira review completed.")
    else:
        provider.approve_pr.assert_not_awaited()


@pytest.mark.asyncio
async def test_automatic_successful_approval_requires_opt_in(forgejo_auth):
    provider = MagicMock()
    provider.approve_pr = AsyncMock()

    with (
        patch("mira.platforms.forgejo.webhook.load_config", return_value=MiraConfig()),
        patch("mira.platforms.forgejo.webhook.create_provider", return_value=provider),
        patch("mira.platforms.index_handlers._get_app_db", return_value=MagicMock()),
        patch(
            "mira.platforms.handlers.run_pr_review",
            new=AsyncMock(return_value=_review_result()),
        ),
    ):
        await handle_forgejo_pr(_pr_payload("opened", "alice"), forgejo_auth, BOT)

    provider.approve_pr.assert_not_awaited()


@pytest.mark.asyncio
async def test_automatic_skipped_review_approval_requires_opt_in(forgejo_auth):
    provider = MagicMock()
    provider.get_pr_info = AsyncMock(return_value=_pr_info())
    provider.approve_pr = AsyncMock()
    config = MiraConfig(forgejo=ForgejoConfig(approve_skipped_reviews=True))
    result = _review_result(skipped_reason="No changed files required review")

    with (
        patch("mira.platforms.forgejo.webhook.load_config", return_value=config),
        patch("mira.platforms.forgejo.webhook.create_provider", return_value=provider),
        patch("mira.platforms.index_handlers._get_app_db", return_value=MagicMock()),
        patch("mira.platforms.handlers.run_pr_review", new=AsyncMock(return_value=result)),
    ):
        await handle_forgejo_pr(_pr_payload("opened", "alice"), forgejo_auth, BOT)

    provider.approve_pr.assert_awaited_once_with(
        _pr_info(), "Mira review skipped: No changed files required review."
    )


@pytest.mark.asyncio
async def test_automatic_review_never_approves_stale_sha(forgejo_auth):
    provider = MagicMock()
    provider.get_pr_info = AsyncMock(
        return_value=PRInfo(**{**_pr_info().__dict__, "head_sha": "new-sha"})
    )
    provider.approve_pr = AsyncMock()
    review = AsyncMock(return_value=_review_result())

    with (
        patch(
            "mira.platforms.forgejo.webhook.load_config",
            return_value=MiraConfig(forgejo=ForgejoConfig(approve_successful_reviews=True)),
        ),
        patch("mira.platforms.forgejo.webhook.create_provider", return_value=provider),
        patch("mira.platforms.index_handlers._get_app_db", return_value=MagicMock()),
        patch("mira.platforms.handlers.run_pr_review", new=review),
    ):
        await handle_forgejo_pr(_pr_payload("opened", "alice"), forgejo_auth, BOT)

    assert review.await_count == 2
    provider.approve_pr.assert_not_awaited()


@pytest.mark.parametrize(
    ("body", "result", "should_approve"),
    [
        ("@mira-bot review", _review_result(), True),
        ("@mira-bot review-rest", _review_result(), True),
        ("@mira-bot review", _review_result(reviewed_sha=""), False),
        ("@mira-bot review", _review_result(skipped_reason="No files"), False),
        ("@mira-bot review", None, False),
        ("@mira-bot review", _review_result(delivery_failed=True), False),
        ("@mira-bot help", _review_result(), False),
    ],
)
@pytest.mark.asyncio
async def test_manual_review_approves_only_completed_review_commands(
    forgejo_auth, body, result, should_approve
):
    provider = MagicMock()
    provider.get_pr_info = AsyncMock(return_value=_pr_info())
    provider.approve_pr = AsyncMock()
    command = AsyncMock(return_value=result)

    with (
        patch("mira.platforms.forgejo.webhook.create_provider", return_value=provider),
        patch(
            "mira.platforms.forgejo.webhook.load_config",
            return_value=MiraConfig(forgejo=ForgejoConfig(approve_successful_reviews=True)),
        ),
        patch("mira.platforms.handlers.run_pr_command", new=command),
    ):
        await handle_forgejo_note(_comment_payload(body, "alice"), forgejo_auth, BOT)

    if should_approve:
        provider.approve_pr.assert_awaited_once_with(_pr_info(), "Mira review completed.")
    else:
        provider.approve_pr.assert_not_awaited()


@pytest.mark.asyncio
async def test_manual_successful_approval_requires_opt_in(forgejo_auth):
    provider = MagicMock()
    provider.get_pr_info = AsyncMock(return_value=_pr_info())
    provider.approve_pr = AsyncMock()

    with (
        patch("mira.platforms.forgejo.webhook.create_provider", return_value=provider),
        patch("mira.platforms.forgejo.webhook.load_config", return_value=MiraConfig()),
        patch(
            "mira.platforms.handlers.run_pr_command",
            new=AsyncMock(return_value=_review_result()),
        ),
    ):
        await handle_forgejo_note(_comment_payload("@mira-bot review", "alice"), forgejo_auth, BOT)

    provider.approve_pr.assert_not_awaited()
