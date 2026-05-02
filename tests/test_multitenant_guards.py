from __future__ import annotations

from types import SimpleNamespace

import bcrypt

from src.redinha_stats.infrastructure.supabase import matches_repository


class _RecordingQuery:
    def __init__(self, calls: list[dict[str, object]]) -> None:
        self._calls = calls
        self._filters: list[tuple[str, object]] = []
        self._payload: object = None
        self._operation = "select"

    def update(self, payload):
        self._operation = "update"
        self._payload = payload
        return self

    def delete(self):
        self._operation = "delete"
        return self

    def eq(self, field, value):
        self._filters.append((field, value))
        return self

    def execute(self):
        self._calls.append(
            {
                "operation": self._operation,
                "payload": self._payload,
                "filters": list(self._filters),
            }
        )
        return SimpleNamespace(data=[{"id": "match-1"}], error=None)


class _RecordingClient:
    def __init__(self, calls: list[dict[str, object]]) -> None:
        self._calls = calls

    def table(self, _table_name):
        return _RecordingQuery(self._calls)


class _UsersQuery:
    def __init__(self, user):
        self._user = user

    def select(self, *_args, **_kwargs):
        return self

    def eq(self, *_args, **_kwargs):
        return self

    def limit(self, *_args, **_kwargs):
        return self

    def execute(self):
        data = [self._user] if self._user else []
        return SimpleNamespace(data=data, error=None)


class _UsersClient:
    def __init__(self, user):
        self._user = user

    def table(self, _table_name):
        return _UsersQuery(self._user)


def test_update_match_scopes_by_group(monkeypatch):
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(matches_repository, "_get_client", lambda: _RecordingClient(calls))

    matches_repository.update_match(
        "match-1",
        {"winner1": "Ana"},
        id_field="id",
        group_id="group-1",
    )

    assert calls == [
        {
            "operation": "update",
            "payload": {"winner1": "Ana"},
            "filters": [("id", "match-1"), ("group_id", "group-1")],
        }
    ]


def test_delete_match_scopes_by_group(monkeypatch):
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(matches_repository, "_get_client", lambda: _RecordingClient(calls))

    matches_repository.delete_match("match-1", id_field="id", group_id="group-1")

    assert calls == [
        {
            "operation": "delete",
            "payload": None,
            "filters": [("id", "match-1"), ("group_id", "group-1")],
        }
    ]


def test_group_login_requires_membership(client, monkeypatch):
    user = {"id": "user-1", "name": "Ana", "password_hash": "hashed"}

    monkeypatch.setattr(
        "src.redinha_stats.web.group_middleware.fetch_group_by_slug",
        lambda slug: {"id": "group-1", "slug": slug, "name": "Grupo 1"},
    )
    monkeypatch.setattr(
        "src.redinha_stats.web.group_blueprint.is_user_member_of_group",
        lambda user_id, group_id: False,
    )
    monkeypatch.setattr("supabase.create_client", lambda *_args, **_kwargs: _UsersClient(user))
    monkeypatch.setattr(bcrypt, "checkpw", lambda *_args, **_kwargs: True)

    response = client.post(
        "/g/grupo-1/login",
        data={"email": "ana@example.com", "password": "secret"},
    )

    assert response.status_code == 200
    assert "Voce nao faz parte deste grupo." in response.get_data(as_text=True)
    with client.session_transaction() as session:
        assert "player_user_id" not in session
        assert "player_group_id" not in session


def test_group_login_stores_group_scoped_session(client, monkeypatch):
    user = {"id": "user-1", "name": "Ana", "password_hash": "hashed"}

    monkeypatch.setattr(
        "src.redinha_stats.web.group_middleware.fetch_group_by_slug",
        lambda slug: {"id": "group-1", "slug": slug, "name": "Grupo 1"},
    )
    monkeypatch.setattr(
        "src.redinha_stats.web.group_blueprint.is_user_member_of_group",
        lambda user_id, group_id: True,
    )
    monkeypatch.setattr("supabase.create_client", lambda *_args, **_kwargs: _UsersClient(user))
    monkeypatch.setattr(bcrypt, "checkpw", lambda *_args, **_kwargs: True)

    response = client.post(
        "/g/grupo-1/login",
        data={"email": "ana@example.com", "password": "secret"},
    )

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/g/grupo-1/perfil")
    with client.session_transaction() as session:
        assert session["player_user_id"] == "user-1"
        assert session["player_name"] == "Ana"
        assert session["player_group_id"] == "group-1"
