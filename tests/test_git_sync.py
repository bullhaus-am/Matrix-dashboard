"""Sync con el remoto antes de calcular (sync_with_remote) y publicación (publish).

Cada test arma un "origin" bare y dos clones en tmp_path: `cloud` hace de
runner de GitHub Actions (commitea a diario) y `local` es el Mac. Sin red.
"""
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import generate_dashboard as gd  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


def git(repo: Path, *args) -> str:
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    assert r.returncode == 0, f"git {' '.join(args)}: {r.stderr or r.stdout}"
    return r.stdout.strip()


def _identity(repo: Path):
    git(repo, "config", "user.name", "test")
    git(repo, "config", "user.email", "test@example.com")
    git(repo, "config", "commit.gpgsign", "false")


def _commit_all(repo: Path, msg: str):
    git(repo, "add", "-A")
    git(repo, "commit", "-m", msg)


@pytest.fixture
def repos(tmp_path):
    """(cloud, local): dos clones del mismo origin, ambos en main y al día."""
    seed = tmp_path / "seed"
    seed.mkdir()
    git(seed, "init", "-b", "main")
    _identity(seed)
    (seed / "index.html").write_text("dashboard v1\n")
    (seed / "macro_etf_dashboard.html").write_text("dashboard v1\n")
    (seed / "regime_history.json").write_text('[{"date": "Oct 30, 2026", "score": 0.2}]\n')
    (seed / "generate_dashboard.py").write_text("# code v1\n")
    (seed / "snapshots").mkdir()
    (seed / "snapshots" / "2026-10-30.json").write_text("{}\n")
    (seed / "insights").mkdir()
    (seed / "insights" / "2026-10-30.md").write_text("nota\n")
    _commit_all(seed, "seed")
    origin = tmp_path / "origin.git"
    subprocess.run(["git", "clone", "--bare", str(seed), str(origin)], check=True, capture_output=True)
    clones = []
    for name in ("cloud", "local"):
        subprocess.run(["git", "clone", str(origin), str(tmp_path / name)], check=True, capture_output=True)
        _identity(tmp_path / name)
        clones.append(tmp_path / name)
    return clones


def _cloud_daily_run(cloud: Path, session: str = "2026-11-02"):
    """Lo que commitea el runner cada día."""
    (cloud / "index.html").write_text(f"dashboard {session}\n")
    (cloud / "regime_history.json").write_text(
        '[{"date": "Oct 30, 2026", "score": 0.2}, {"date": "Nov 02, 2026", "score": 0.3}]\n')
    (cloud / "snapshots" / f"{session}.json").write_text("{}\n")
    _commit_all(cloud, f"dashboard {session}")
    git(cloud, "push", "origin", "main")


def _in_rebase(repo: Path) -> bool:
    return (repo / ".git" / "rebase-merge").exists() or (repo / ".git" / "rebase-apply").exists()


# ── sync_with_remote ──────────────────────────────────────────────────────────

def test_sync_is_a_noop_when_up_to_date(repos):
    _, local = repos
    head = git(local, "rev-parse", "HEAD")
    ok, detail = gd.sync_with_remote(local)
    assert ok and "up to date" in detail
    assert git(local, "rev-parse", "HEAD") == head


def test_sync_fast_forwards_a_stale_checkout(repos):
    cloud, local = repos
    _cloud_daily_run(cloud)
    ok, detail = gd.sync_with_remote(local)
    assert ok and "fast-forwarded 1" in detail
    assert "Nov 02, 2026" in (local / "regime_history.json").read_text()     # el estado que hay que leer
    assert (local / "snapshots" / "2026-11-02.json").exists()


def test_sync_tolerates_a_dirty_file_the_remote_does_not_touch(repos):
    """macro_etf_dashboard.html queda modificado y sin commitear tras cada corrida."""
    cloud, local = repos
    (local / "macro_etf_dashboard.html").write_text("corrida local anterior\n")
    _cloud_daily_run(cloud)
    ok, _ = gd.sync_with_remote(local)
    assert ok
    assert (local / "macro_etf_dashboard.html").read_text() == "corrida local anterior\n"
    assert "Nov 02, 2026" in (local / "regime_history.json").read_text()


def test_sync_refuses_without_touching_anything_when_the_remote_changes_a_dirty_file(repos):
    cloud, local = repos
    (local / "regime_history.json").write_text("estado local sin commitear\n")
    head = git(local, "rev-parse", "HEAD")
    _cloud_daily_run(cloud)
    ok, detail = gd.sync_with_remote(local)
    assert not ok and "regime_history.json" in detail
    assert git(local, "rev-parse", "HEAD") == head
    assert (local / "regime_history.json").read_text() == "estado local sin commitear\n"


def test_sync_rebases_unpushed_local_commits(repos):
    cloud, local = repos
    (local / "generate_dashboard.py").write_text("# code v2\n")
    _commit_all(local, "fix local sin pushear")
    _cloud_daily_run(cloud)
    ok, detail = gd.sync_with_remote(local)
    assert ok and "rebased 1" in detail
    assert git(local, "log", "--format=%s", "-2").splitlines() == ["fix local sin pushear", "dashboard 2026-11-02"]
    assert "Nov 02, 2026" in (local / "regime_history.json").read_text()


def test_sync_aborts_a_conflicting_rebase_and_leaves_the_checkout_as_it_was(repos):
    cloud, local = repos
    (local / "index.html").write_text("dashboard local\n")
    _commit_all(local, "dashboard local")
    head = git(local, "rev-parse", "HEAD")
    _cloud_daily_run(cloud)
    ok, detail = gd.sync_with_remote(local)
    assert not ok and "rebase" in detail
    assert not _in_rebase(local)
    assert git(local, "rev-parse", "HEAD") == head
    assert git(local, "rev-parse", "--abbrev-ref", "HEAD") == "main"
    assert (local / "index.html").read_text() == "dashboard local\n"


def test_sync_refuses_outside_the_publish_branch(repos):
    cloud, local = repos
    git(local, "checkout", "-b", "experimento")
    _cloud_daily_run(cloud)
    ok, detail = gd.sync_with_remote(local)
    assert not ok and "experimento" in detail


def test_sync_reports_an_unreachable_remote(repos, tmp_path):
    _, local = repos
    git(local, "remote", "set-url", "origin", str(tmp_path / "no-existe.git"))
    ok, detail = gd.sync_with_remote(local)
    assert not ok and "fetch" in detail


# ── publish ───────────────────────────────────────────────────────────────────

def _local_run(local: Path, session: str = "2026-11-03"):
    """Lo que deja una corrida en el árbol antes de publicar."""
    (local / "macro_etf_dashboard.html").write_text(f"dashboard {session}\n")   # nunca se commitea
    (local / "index.html").write_text(f"dashboard {session}\n")
    (local / "regime_history.json").write_text(f'[{{"date": "{session}"}}]\n')
    (local / "snapshots" / f"{session}.json").write_text("{}\n")


def test_publish_commits_the_state_and_pushes(repos):
    cloud, local = repos
    _local_run(local)
    assert gd.publish(local, "dashboard test") is True
    git(cloud, "pull", "origin", "main")
    assert git(cloud, "log", "--format=%s", "-1") == "dashboard test"
    assert (cloud / "snapshots" / "2026-11-03.json").exists()
    assert (cloud / "macro_etf_dashboard.html").read_text() == "dashboard v1\n"   # no se publica
    assert (local / "macro_etf_dashboard.html").read_text() == "dashboard 2026-11-03\n"


def test_publish_with_nothing_new_still_succeeds(repos):
    _, local = repos
    head = git(local, "rev-parse", "HEAD")
    assert gd.publish(local, "dashboard test") is True
    assert git(local, "rev-parse", "HEAD") == head                           # sin commit vacío


def test_publish_rebases_and_retries_when_the_remote_moved_during_the_run(repos):
    cloud, local = repos
    _local_run(local)
    (cloud / "insights" / "2026-11-02.md").write_text("nota del runner\n")   # no pisa nada local
    _commit_all(cloud, "insights 2026-11-02")
    git(cloud, "push", "origin", "main")

    assert gd.publish(local, "dashboard test") is True
    assert not _in_rebase(local)
    assert (local / "macro_etf_dashboard.html").read_text() == "dashboard 2026-11-03\n"   # autostash devuelto
    git(cloud, "pull", "origin", "main")
    assert git(cloud, "log", "--format=%s", "-2").splitlines() == ["dashboard test", "insights 2026-11-02"]


def test_publish_gives_up_cleanly_when_another_run_published_the_same_files(repos):
    cloud, local = repos
    _local_run(local)
    _cloud_daily_run(cloud, "2026-11-03")                                    # misma sesión, otro publisher
    remote_head = git(cloud, "rev-parse", "HEAD")

    assert gd.publish(local, "dashboard test") is False
    assert not _in_rebase(local)
    assert git(local, "log", "--format=%s", "-1") == "dashboard test"        # el commit queda local
    assert (local / "macro_etf_dashboard.html").read_text() == "dashboard 2026-11-03\n"
    git(cloud, "fetch", "origin", "main")
    assert git(cloud, "rev-parse", "FETCH_HEAD") == remote_head              # el remoto no se tocó
