"""Tests for the sandboxed filesystem layer the agent's tools are built on."""

from pathlib import Path

import pytest

from file_organizer.workspace import Workspace, WorkspaceError


@pytest.fixture
def root(tmp_path: Path) -> Path:
    (tmp_path / "invoice.txt").write_text("Invoice #42: total due 120 EUR")
    (tmp_path / "flight.txt").write_text("Boarding pass LH123 to Lisbon")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "nested.md").write_text("nested")
    return tmp_path


@pytest.fixture
def ws(root: Path) -> Workspace:
    return Workspace(root)


class TestConstruction:
    def test_root_must_exist(self, tmp_path: Path):
        with pytest.raises(WorkspaceError, match="does not exist"):
            Workspace(tmp_path / "missing")

    def test_root_must_be_a_directory(self, tmp_path: Path):
        f = tmp_path / "file.txt"
        f.write_text("x")
        with pytest.raises(WorkspaceError, match="not a directory"):
            Workspace(f)


class TestListFiles:
    def test_lists_top_level_entries_sorted_with_dir_marker(self, ws: Workspace):
        assert ws.list_files() == ["docs/", "flight.txt", "invoice.txt"]

    def test_lists_a_subdirectory_with_paths_relative_to_root(self, ws: Workspace):
        assert ws.list_files("docs") == ["docs/nested.md"]

    def test_hides_dotfiles(self, ws: Workspace, root: Path):
        (root / ".DS_Store").write_text("")
        assert ".DS_Store" not in ws.list_files()

    def test_missing_directory_raises(self, ws: Workspace):
        with pytest.raises(WorkspaceError, match="not found"):
            ws.list_files("nope")


class TestReadFile:
    def test_reads_text(self, ws: Workspace):
        assert ws.read_file("invoice.txt").endswith("\nInvoice #42: total due 120 EUR")

    def test_truncates_long_files(self, root: Path):
        (root / "long.txt").write_text("a" * 100)
        ws = Workspace(root, max_read_chars=10)
        content = ws.read_file("long.txt")
        assert "truncated" in content
        assert "a" * 5 not in content  # 4 + 3 + 3 characters from the beginning, middle and end

    def test_binary_files_are_not_dumped(self, ws: Workspace, root: Path):
        (root / "blob.dat").write_bytes(b"\x00\x01secret\x00")
        content = ws.read_file("blob.dat")
        assert "binary" in content.lower()
        assert "blob.dat" in content and "secret" not in content

    def test_reading_a_directory_raises(self, ws: Workspace):
        with pytest.raises(WorkspaceError, match="not a file"):
            ws.read_file("docs")

    def test_missing_file_raises(self, ws: Workspace):
        with pytest.raises(WorkspaceError, match="not found"):
            ws.read_file("ghost.txt")


class TestCreateDirectory:
    def test_creates_directory(self, ws: Workspace, root: Path):
        msg = ws.create_directory_change("finance").message
        assert (root / "finance").is_dir()
        assert "finance" in msg

    def test_creates_nested_directories(self, ws: Workspace, root: Path):
        ws.create_directory_change("finance/2024")
        assert (root / "finance" / "2024").is_dir()

    def test_existing_directory_is_ok(self, ws: Workspace):
        assert "already exists" in ws.create_directory_change("docs").message

    def test_conflicting_file_raises(self, ws: Workspace):
        with pytest.raises(WorkspaceError, match="not a directory"):
            ws.create_directory_change("invoice.txt")


class TestMoveFile:
    def test_moves_file_into_existing_directory(self, ws: Workspace, root: Path):
        ws.create_directory_change("finance")
        ws.move_file_change("invoice.txt", "finance")
        assert not (root / "invoice.txt").exists()
        assert (root / "finance" / "invoice.txt").read_text().startswith("Invoice")

    def test_moves_file_into_nested_directory(self, ws: Workspace, root: Path):
        ws.create_directory_change("docs/finance")
        ws.move_file_change("invoice.txt", "docs/finance")
        assert (root / "docs" / "finance" / "invoice.txt").exists()

    @pytest.mark.parametrize("destination", ["ghost", "finance/invoice-42.txt", "docs/renamed.txt"])
    def test_never_renames_destination_must_be_an_existing_directory(
        self, ws: Workspace, root: Path, destination: str
    ):
        ws.create_directory_change("finance")
        with pytest.raises(WorkspaceError, match="does not exist"):
            ws.move_file_change("invoice.txt", destination)
        assert (root / "invoice.txt").exists()  # untouched
        assert not (root / destination).exists()

    def test_destination_cannot_be_an_existing_file(self, ws: Workspace, root: Path):
        with pytest.raises(WorkspaceError, match="not a directory"):
            ws.move_file_change("invoice.txt", "flight.txt")
        assert (root / "invoice.txt").exists()
        assert (root / "flight.txt").read_text().startswith("Boarding")

    def test_moving_into_its_current_directory_is_rejected(self, ws: Workspace, root: Path):
        with pytest.raises(WorkspaceError, match="already in"):
            ws.move_file_change("invoice.txt", ".")
        assert (root / "invoice.txt").exists()

    def test_never_overwrites(self, ws: Workspace, root: Path):
        (root / "docs" / "invoice.txt").write_text("other")
        with pytest.raises(WorkspaceError, match="already exists"):
            ws.move_file_change("invoice.txt", "docs")
        assert (root / "docs" / "invoice.txt").read_text() == "other"
        assert (root / "invoice.txt").exists()

    def test_source_must_be_a_file(self, ws: Workspace):
        with pytest.raises(WorkspaceError, match="not a file"):
            ws.move_file_change("docs", "elsewhere")

    def test_missing_source_raises(self, ws: Workspace):
        with pytest.raises(WorkspaceError, match="not found"):
            ws.move_file_change("ghost.txt", "docs")


class TestSandbox:
    @pytest.mark.parametrize("path", ["../outside.txt", "/etc/passwd", "docs/../../x"])
    def test_paths_outside_root_are_rejected(self, ws: Workspace, path: str):
        with pytest.raises(WorkspaceError, match="outside"):
            ws.read_file(path)
        with pytest.raises(WorkspaceError, match="outside"):
            ws.create_directory_change(path)
        with pytest.raises(WorkspaceError, match="outside"):
            ws.move_file_change("invoice.txt", path)

    def test_symlink_escape_is_rejected(self, ws: Workspace, root: Path, tmp_path_factory):
        outside = tmp_path_factory.mktemp("outside")
        (outside / "secret.txt").write_text("secret")
        (root / "link").symlink_to(outside)
        with pytest.raises(WorkspaceError, match="outside"):
            ws.read_file("link/secret.txt")


class TestGuardrails:
    def test_hidden_files_cannot_be_read_or_moved(self, ws: Workspace, root: Path):
        (root / ".env").write_text("SECRET=1")
        with pytest.raises(WorkspaceError, match="hidden"):
            ws.read_file(".env")
        with pytest.raises(WorkspaceError, match="hidden"):
            ws.move_file_change(".env", "docs")
        with pytest.raises(WorkspaceError, match="hidden"):
            ws.create_directory_change(".git/hooks")

    def test_symlinks_are_refused_and_not_listed(self, ws: Workspace, root: Path):
        (root / "link.txt").symlink_to(root / "invoice.txt")
        (root / "linkdir").symlink_to(root / "docs")
        assert "link.txt" not in ws.list_files()
        with pytest.raises(WorkspaceError, match="symlink"):
            ws.move_file_change("link.txt", "docs")
        with pytest.raises(WorkspaceError, match="symlink"):
            ws.read_file("linkdir/nested.md")
        assert (root / "invoice.txt").exists()

    @pytest.mark.parametrize("bad", ["a\x00b", "x" * 5000])
    def test_invalid_paths_raise_workspace_error(self, ws: Workspace, bad: str):
        with pytest.raises(WorkspaceError):
            ws.read_file(bad)

    def test_read_is_bounded(self, root: Path):
        (root / "big.txt").write_text("a" * 100_000)
        text = Workspace(root, max_read_chars=10).read_file("big.txt")
        content = text.split("[content:", 1)[1]
        assert content.count("truncated") == 2 and "a" * 5 not in content and len(content) < 500

    def test_listing_is_capped(self, root: Path):
        listing = Workspace(root, max_list_entries=1).list_files()
        assert listing == ["docs/", "[... 2 more entries not shown]"]

    def test_directory_depth_is_limited(self, ws: Workspace):
        with pytest.raises(WorkspaceError, match="levels deep"):
            ws.create_directory_change("a/b/c/d")
        with pytest.raises(WorkspaceError, match="root"):
            ws.create_directory_change(".")


class TestRenameOnClash:
    @pytest.fixture
    def ws(self, root: Path) -> Workspace:
        (root / "docs" / "invoice.txt").write_text("other")
        return Workspace(root, rename_on_clash=True)

    def test_off_by_default(self, root: Path):
        assert Workspace(root).rename_on_clash is False

    def test_takes_the_first_free_name_and_overwrites_nothing(self, ws: Workspace, root: Path):
        (root / "docs" / "invoice (2).txt").write_text("other 2")
        change = ws.move_file_change("invoice.txt", "docs")
        assert (root / "docs" / "invoice (3).txt").read_text().startswith("Invoice")
        assert (root / "docs" / "invoice.txt").read_text() == "other"
        assert (root / "docs" / "invoice (2).txt").read_text() == "other 2"
        assert "renamed to 'invoice (3).txt'" in change.message
        assert change.undo == {"op": "move", "from": "invoice.txt", "to": "docs/invoice (3).txt"}

    def test_planned_destination_changes_nothing(self, ws: Workspace, root: Path):
        assert ws.planned_destination("invoice.txt", "docs") == "docs/invoice (2).txt"
        assert ws.planned_destination("ghost.txt", "docs") is None
        assert (root / "invoice.txt").exists()

    def test_planned_destination_counts_claimed_destinations(self, ws: Workspace, root: Path):
        (root / "docs" / "invoice.txt").unlink()
        assert ws.planned_destination("invoice.txt", "docs") == "docs/invoice.txt"
        claimed = {"docs/invoice.txt"}
        assert ws.planned_destination("invoice.txt", "docs", claimed) == "docs/invoice (2).txt"

    def test_moving_into_its_own_directory_is_still_rejected(self, ws: Workspace, root: Path):
        with pytest.raises(WorkspaceError, match="already in"):
            ws.move_file_change("invoice.txt", ".")
        assert sorted(p.name for p in root.iterdir()) == ["docs", "flight.txt", "invoice.txt"]


class TestUndoSupport:
    def test_create_directory_reports_every_directory_it_created(self, ws: Workspace):
        assert ws.create_directory_change("a/b/c").undo == {"op": "mkdir", "paths": ["a", "a/b", "a/b/c"]}
        assert ws.create_directory_change("docs/x").undo == {"op": "mkdir", "paths": ["docs/x"]}
        assert ws.create_directory_change("docs").undo is None  # already existed

    def test_undo_move_puts_the_file_back(self, ws: Workspace, root: Path):
        ws.move_file_change("invoice.txt", "docs")
        ws.undo_move("docs/invoice.txt", "invoice.txt")
        assert (root / "invoice.txt").exists() and not (root / "docs" / "invoice.txt").exists()

    def test_undo_move_never_overwrites(self, ws: Workspace, root: Path):
        ws.move_file_change("invoice.txt", "docs")
        (root / "invoice.txt").write_text("new file")
        with pytest.raises(WorkspaceError, match="refusing to overwrite"):
            ws.undo_move("docs/invoice.txt", "invoice.txt")
        assert (root / "invoice.txt").read_text() == "new file"

    def test_undo_move_is_sandboxed(self, ws: Workspace):
        with pytest.raises(WorkspaceError, match="outside"):
            ws.undo_move("invoice.txt", "../stolen.txt")

    def test_remove_empty_directory_only_when_empty(self, ws: Workspace, root: Path):
        (root / "empty").mkdir()
        (root / "hidden").mkdir()
        (root / "hidden" / ".DS_Store").write_text("")
        ws.remove_empty_directory("empty")
        assert not (root / "empty").exists()
        assert "already gone" in ws.remove_empty_directory("empty")
        for path in ("docs", "hidden"):
            with pytest.raises(WorkspaceError, match="not empty"):
                ws.remove_empty_directory(path)
        with pytest.raises(WorkspaceError, match="root"):
            ws.remove_empty_directory(".")
