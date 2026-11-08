import pytest

from app.imaging.sizing import SizeError, compute_target, plan_geometry, subject_bbox


@pytest.mark.parametrize("width,rw,rh,expected_h,exact", [
    (1200, 10, 10, 1200, True),
    (1000, 10, 3, 300, True),
    (1001, 10, 3, 300, False),  # 300.3 -> 300
    (1005, 10, 3, 302, False),  # 301.5 -> 302 (round half up)
    (750, 3, 4, 1000, True),
    (1920, 16, 9, 1080, True),
    (800, 10, 13.5, 1080, True),
    (801, 10, 13.5, 1081, False),  # 1081.35
])
def test_compute_target(width, rw, rh, expected_h, exact):
    t = compute_target(width, rw, rh)
    assert (t.width, t.height, t.exact_ratio) == (width, expected_h, exact)


def test_ratio_10_10_equals_1_1():
    assert compute_target(900, 10, 10) == compute_target(900, 1, 1)


def test_suggested_widths_make_ratio_exact():
    t = compute_target(1001, 10, 3)
    assert t.suggested_widths == (1000, 1010)
    for w in t.suggested_widths:
        assert compute_target(w, 10, 3).exact_ratio
    t = compute_target(801, 10, 13.5)
    for w in t.suggested_widths:
        assert compute_target(w, 10, 13.5).exact_ratio


def test_explicit_height():
    t = compute_target(640, height=480)
    assert (t.width, t.height) == (640, 480)


@pytest.mark.parametrize("args", [(8, 1, 1), (12000, 1, 1), (1000, 10, 200), (1000, 0, 1), (1000, 10, -3), (9000, 1, 1)])
def test_invalid_sizes(args):
    with pytest.raises(SizeError):
        compute_target(*args)


def test_plan_pad_keeps_aspect_and_centres():
    t = compute_target(1200, 1, 1)
    p = plan_geometry(400, 200, t, "pad")
    assert p.content_size == (1200, 600)
    assert p.offset == (0, 300)
    assert p.canvas_size == (1200, 1200)


def test_plan_crop_covers_canvas():
    t = compute_target(1000, 10, 5)
    p = plan_geometry(300, 300, t, "crop")
    x0, y0, x1, y1 = p.crop_box
    assert (x1 - x0, y1 - y0) == (300, 150)
    assert p.content_size == (1000, 500)


def test_plan_crop_anchor_follows_subject():
    import numpy as np
    img = np.full((300, 600, 3), 255, np.uint8)
    img[100:200, 20:120] = (10, 10, 10)  # product near the left edge
    box = subject_bbox(img)
    assert box == (20, 100, 120, 200)
    t = compute_target(300, 1, 1)
    p = plan_geometry(600, 300, t, "crop", box)
    assert p.crop_box[0] == 0  # window shifted left instead of centred (150)


def test_subject_bbox_none_for_busy_photo():
    import numpy as np
    rng = np.random.default_rng(0)
    assert subject_bbox(rng.integers(0, 255, (200, 200, 3), dtype=np.uint8)) is None
