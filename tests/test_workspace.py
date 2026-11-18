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
        assert ws.read_file("invoice.txt") == "Invoice #42: total due 120 EUR"

    def test_truncates_long_files(self, root: Path):
        (root / "long.txt").write_text("a" * 100)
        ws = Workspace(root, max_read_chars=10)
        content = ws.read_file("long.txt")
        assert content.startswith("a" * 10)
        assert "truncated" in content
        assert "a" * 11 not in content

    def test_binary_files_are_not_dumped(self, ws: Workspace, root: Path):
        (root / "photo.jpg").write_bytes(b"\xff\xd8\xff\x00\x10JFIF\x00")
        content = ws.read_file("photo.jpg")
        assert "binary" in content.lower()
        assert "photo.jpg" in content

    def test_reading_a_directory_raises(self, ws: Workspace):
        with pytest.raises(WorkspaceError, match="not a file"):
            ws.read_file("docs")

    def test_missing_file_raises(self, ws: Workspace):
        with pytest.raises(WorkspaceError, match="not found"):
            ws.read_file("ghost.txt")


class TestCreateDirectory:
    def test_creates_directory(self, ws: Workspace, root: Path):
        msg = ws.create_directory("finance")
        assert (root / "finance").is_dir()
        assert "finance" in msg

    def test_creates_nested_directories(self, ws: Workspace, root: Path):
        ws.create_directory("finance/2024")
        assert (root / "finance" / "2024").is_dir()

    def test_existing_directory_is_ok(self, ws: Workspace):
        assert "already exists" in ws.create_directory("docs")

    def test_conflicting_file_raises(self, ws: Workspace):
        with pytest.raises(WorkspaceError, match="not a directory"):
            ws.create_directory("invoice.txt")


class TestMoveFile:
    def test_moves_file_into_existing_directory(self, ws: Workspace, root: Path):
        ws.create_directory("finance")
        ws.move_file("invoice.txt", "finance")
        assert not (root / "invoice.txt").exists()
        assert (root / "finance" / "invoice.txt").read_text().startswith("Invoice")

    def test_moves_file_into_nested_directory(self, ws: Workspace, root: Path):
        ws.create_directory("docs/finance")
        ws.move_file("invoice.txt", "docs/finance")
        assert (root / "docs" / "finance" / "invoice.txt").exists()

    @pytest.mark.parametrize("destination", ["ghost", "finance/invoice-42.txt", "docs/renamed.txt"])
    def test_never_renames_destination_must_be_an_existing_directory(
        self, ws: Workspace, root: Path, destination: str
    ):
        ws.create_directory("finance")
        with pytest.raises(WorkspaceError, match="does not exist"):
            ws.move_file("invoice.txt", destination)
        assert (root / "invoice.txt").exists()  # untouched
        assert not (root / destination).exists()

    def test_destination_cannot_be_an_existing_file(self, ws: Workspace, root: Path):
        with pytest.raises(WorkspaceError, match="not a directory"):
            ws.move_file("invoice.txt", "flight.txt")
        assert (root / "invoice.txt").exists()
        assert (root / "flight.txt").read_text().startswith("Boarding")

    def test_moving_into_its_current_directory_is_rejected(self, ws: Workspace, root: Path):
        with pytest.raises(WorkspaceError, match="already exists"):
            ws.move_file("invoice.txt", ".")
        assert (root / "invoice.txt").exists()

    def test_never_overwrites(self, ws: Workspace, root: Path):
        (root / "docs" / "invoice.txt").write_text("other")
        with pytest.raises(WorkspaceError, match="already exists"):
            ws.move_file("invoice.txt", "docs")
        assert (root / "docs" / "invoice.txt").read_text() == "other"
        assert (root / "invoice.txt").exists()

    def test_source_must_be_a_file(self, ws: Workspace):
        with pytest.raises(WorkspaceError, match="not a file"):
            ws.move_file("docs", "elsewhere")

    def test_missing_source_raises(self, ws: Workspace):
        with pytest.raises(WorkspaceError, match="not found"):
            ws.move_file("ghost.txt", "docs")


class TestSandbox:
    @pytest.mark.parametrize("path", ["../outside.txt", "/etc/passwd", "docs/../../x"])
    def test_paths_outside_root_are_rejected(self, ws: Workspace, path: str):
        with pytest.raises(WorkspaceError, match="outside"):
            ws.read_file(path)
        with pytest.raises(WorkspaceError, match="outside"):
            ws.create_directory(path)
        with pytest.raises(WorkspaceError, match="outside"):
            ws.move_file("invoice.txt", path)

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
            ws.move_file(".env", "docs")
        with pytest.raises(WorkspaceError, match="hidden"):
            ws.create_directory(".git/hooks")

    def test_symlinks_are_refused_and_not_listed(self, ws: Workspace, root: Path):
        (root / "link.txt").symlink_to(root / "invoice.txt")
        (root / "linkdir").symlink_to(root / "docs")
        assert "link.txt" not in ws.list_files()
        with pytest.raises(WorkspaceError, match="symlink"):
            ws.move_file("link.txt", "docs")
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
        assert text.startswith("a" * 10) and "truncated" in text and len(text) < 100

    def test_listing_is_capped(self, root: Path):
        listing = Workspace(root, max_list_entries=1).list_files()
        assert listing == ["docs/", "[... 2 more entries not shown]"]

    def test_directory_depth_is_limited(self, ws: Workspace):
        with pytest.raises(WorkspaceError, match="levels deep"):
            ws.create_directory("a/b/c/d")
        with pytest.raises(WorkspaceError, match="root"):
            ws.create_directory(".")
