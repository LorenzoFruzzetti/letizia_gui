"""roi_editor_spec.md 7.3 (pure parts): pages from a manifest / cohort, the image cache."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import tifffile  # noqa: E402

import channel_rectangle as cr  # noqa: E402
from wfci_gui.roi_editor import logic, sources  # noqa: E402

HEADER = ("day,animal,group,n_recordings,recordings,total_frames,bregma_row,bregma_col,profile,"
          "channel_order,animal_path,recording_paths\n")


def make_recording(folder: Path, n_frames: int = 6, size: int = 32) -> None:
    """Interleaved frames: odd ones dim (GCaMP-like), even ones bright."""
    folder.mkdir(parents=True)
    rng = np.random.default_rng(len(str(folder)))
    for i in range(n_frames):
        level = 1000.0 if i % 2 == 0 else 3000.0          # file 1, 3, 5 = odd frames
        image = (level + rng.random((size, size)) * 10).astype(np.uint16)
        tifffile.imwrite(folder / f"frame_{i + 1:05d}.tif", image)


def settings(tmp_path, **changes):
    values = dict(logic.default_values(), folder=[], roi_set_dir=str(tmp_path / "reviewed"),
                  roi_seed_dir=None, frames_per_parity=2)
    values.update(changes)
    return logic.make_settings(values, need_folder=False)


def write_manifest(path: Path, rows: list[str]) -> Path:
    path.write_text(HEADER + "".join(row + "\n" for row in rows), encoding="utf-8")
    return path


def test_manifest_pages_keys_order_group_and_order_per_row(tmp_path):
    raw = tmp_path / "raw"
    for day, animal, t in [("230711", "B", "t0"), ("230711", "B", "t2"), ("230714", "A", "t1"),
                           ("230711", "A", "t1"), ("230711", "A", "t0")]:
        make_recording(raw / day / animal / t)
    paths = lambda day, animal, ts: ";".join(str(raw / day / animal / t) for t in ts)  # noqa: E731
    # Grouped style (one row per unit) and explicit style (one row per recording) together.
    manifest = write_manifest(tmp_path / "m.csv", [
        f"230711,B,R,2,t0;t2,0,120,130,cortical_gsr,auto,x,{paths('230711', 'B', ['t0', 't2'])}",
        f"230714,A,P,1,t1,0,,,cortical_gsr,emo_first,x,{paths('230714', 'A', ['t1'])}",
        f"230711,A,P,1,t1,0,,,cortical_gsr,gcamp_first,x,{paths('230711', 'A', ['t1'])}",
        f"230711,A,P,1,t0,0,,,cortical_gsr,emo_first,x,{paths('230711', 'A', ['t0'])}",
    ])
    pages, args, profile, preview_only, notes = sources.pages_from_manifest(manifest, settings(tmp_path))
    assert [p.key for p in pages] == ["230711_A_t0", "230711_A_t1", "230714_A_t1", "230711_B_t0", "230711_B_t2"]
    assert [p.channel_order for p in pages] == ["emo_first", "gcamp_first", "emo_first", "auto", "auto"]
    assert [p.group for p in pages] == ["P", "P", "P", "R", "R"]
    # The row's own Bregma seeds a page without a set; empty -> the editor default.
    assert (pages[3].y_1, pages[3].x_2) == (60, 65) and pages[0].bregma_row == 120
    assert not preview_only and notes == [] and profile == "cortical_gsr"
    assert all(p.image is None for p in pages)            # images are read lazily


def test_manifest_without_raw_frames_uses_the_cache_and_lists_the_rest(tmp_path):
    manifest = write_manifest(tmp_path / "m.csv", [
        f"230711,A,P,2,t0;t1,0,,,cortical_gsr,auto,x,{tmp_path / 'gone' / 't0'};{tmp_path / 'gone' / 't1'}"])
    images = tmp_path / "reviewed" / "alignment_images"
    images.mkdir(parents=True)
    tifffile.imwrite(images / "230711_A_t1.tif", np.ones((128, 128), np.float32))
    pages, *_rest, preview_only, notes = sources.pages_from_manifest(manifest, settings(tmp_path))
    assert [p.key for p in pages] == ["230711_A_t1"] and preview_only
    assert len(notes) == 1 and notes[0].startswith("230711_A_t0")
    sources.load_images(pages[0], settings(tmp_path))
    assert pages[0].image_source == "legacy" and pages[0].odd_even_128[1] is None
    assert pages[0].odd_even is None                       # no decision without raw frames


def test_manifest_with_mismatched_lists_is_refused(tmp_path):
    manifest = write_manifest(tmp_path / "m.csv", ["230711,A,P,2,t0;t1,0,,,cortical_gsr,auto,x,only_one"])
    with pytest.raises(ValueError, match="2 recordings but 1"):
        sources.manifest_recordings(manifest)


def test_cohort_lookup_match_none_and_ambiguous(tmp_path):
    cohorts = tmp_path / "cohorts"
    cohorts.mkdir()
    manifest = tmp_path / "m.csv"
    manifest.write_text(HEADER, encoding="utf-8")
    (cohorts / "a.yaml").write_text(f"name: a\nmanifest: {manifest}\n", encoding="utf-8")
    (cohorts / "b.yaml").write_text("name: b\nmanifest: somewhere_else.csv\n", encoding="utf-8")
    assert sources.cohort_for_manifest(manifest, cohorts) == cohorts / "a.yaml"
    assert sources.cohort_for_manifest(tmp_path / "other.csv", cohorts) is None
    (cohorts / "c.yaml").write_text(f"name: c\nmanifest: {manifest}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="ambiguous"):
        sources.cohort_for_manifest(manifest, cohorts)


def test_the_real_cohorts_name_their_own_manifests():
    """No two committed cohorts share a manifest; the rectangle manifest has none (D10)."""
    for cohort in sorted((REPO_ROOT / "cohorts").glob("*.yaml")):
        import yaml
        manifest = yaml.safe_load(cohort.read_text(encoding="utf-8")).get("manifest")
        if manifest:
            assert sources.cohort_for_manifest(REPO_ROOT / manifest) == cohort
    assert sources.cohort_for_manifest(REPO_ROOT / "manifests" / "botox_2023_manifest_rectangle.csv") is None


def test_raw_read_writes_the_pair_and_the_sidecar_decides_validity(tmp_path):
    folder = tmp_path / "raw" / "230711" / "A" / "t0"
    make_recording(folder, n_frames=8, size=32)
    s = settings(tmp_path)
    page = sources.make_page("230711_A_t0", str(folder), sources.editor_args(s),
                             dict(sources._profile(sources.editor_args(s)).atlas), 121, 134)
    sources.load_images(page, s)
    assert page.image_source == "raw"
    odd_256, even_256 = page.odd_even
    assert odd_256.shape == (16, 16) and page.image.shape == (8, 8)
    assert odd_256.mean() < even_256.mean()                 # odd files were the dim ones
    assert np.array_equal(page.odd_even_128[0], sources.to_128(odd_256))
    paths = sources.cache_paths(s.roi_set_dir, page.key)
    assert not paths["legacy"].exists()                       # the old preview is never written
    cached = sources.read_cache(s, page.key)
    assert np.allclose(cached[0], page.odd_even_128[0]) and np.allclose(cached[1], page.odd_even_128[1])
    assert not list(paths["odd"].parent.glob("*.partial"))
    # Other frames-per-parity: the cache no longer counts.
    assert sources.read_cache(settings(tmp_path, frames_per_parity=3), page.key) is None
    assert sources.cache_page(page.key, str(folder), s) == "already cached"
    assert sources.cache_page(page.key, str(folder), settings(tmp_path, frames_per_parity=3)) == "cached"


def test_a_page_from_its_own_set_loads_its_rectangle(tmp_path):
    from roi_editor import save_roi_set
    from wfci.atlases import CORTEX_22
    s = settings(tmp_path)
    own = Path(s.roi_set_dir) / "230711_A_t0.yaml"
    save_roi_set(own, dict(CORTEX_22.boxes), 120, 134, grid=(128, 128), name="x", source="x", lambda_offset=40)
    factors = cr.RectangleFactors(0.5, 0.2, 1.0)
    cr.append_roi_set_rectangle(own, factors)
    args = sources.editor_args(s)
    page = sources.make_page("230711_A_t0", "nowhere", args, dict(CORTEX_22.boxes), 121, 134)
    assert page.has_own_set and page.rectangle == factors == page.seed_rectangle
    assert page.seed_source == "own set"
    other = sources.make_page("230711_A_t1", "nowhere", args, dict(CORTEX_22.boxes), 121, 134)
    assert other.rectangle is None and other.seed_source == "seeded from the cortical_gsr atlas"


def test_recent_list_is_most_recent_first_without_duplicates():
    entries = []
    for i in range(12):
        entries = sources.push_recent(entries, "manifest", f"m{i}.csv")
    entries = sources.push_recent(entries, "cohort", "m5.csv")
    entries = sources.push_recent(entries, "manifest", "m11.csv")
    assert len(entries) == 10 and entries[0][1].endswith("m11.csv") and entries[1][0] == "cohort"
    assert len(set(entries)) == len(entries)


def test_bake_alignment_images_writes_the_pair_the_editor_reads(tmp_path):
    import argparse
    import bake_alignment_images as bake
    folder = tmp_path / "raw" / "230711" / "A" / "t0"
    make_recording(folder, n_frames=8, size=32)
    args = argparse.Namespace(pattern="*.tif", frame_order="name", rot90_k=0, roi_set_dir=str(tmp_path / "reviewed"),
                              frames_per_parity=2, legacy_preview=False)
    image_dir = tmp_path / "reviewed" / "alignment_images"
    assert not bake.is_baked("230711_A_t0", image_dir, args)
    key, _seconds, skipped = bake.bake_pair(("230711_A_t0", str(folder), args))
    assert key == "230711_A_t0" and not skipped and bake.is_baked(key, image_dir, args)
    assert bake.bake_pair((key, str(folder), args))[2]               # second run: skipped
    odd, even = sources.read_cache(settings(tmp_path), key)
    odd_256, even_256 = sources.read_raw(folder, settings(tmp_path))
    assert np.allclose(odd, sources.to_128(odd_256)) and np.allclose(even, sources.to_128(even_256))
    assert not (image_dir / f"{key}.tif").exists()
