"""Pure checks for precision dense-progress stages. No Isaac."""

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from env.reward_manager.dense_progress import (  # noqa: E402
    _blend,
    build_tower_groups,
    exp_progress,
    fasten_screws_groups,
    insert_key_groups,
    insert_tubes_groups,
    opposite_x,
    play_xylophone_groups,
    split_weight,
    stage_phi,
)


def _weight(groups):
    return sum(stage["weight"] for group in groups for stage in group["stages"])


def test_exp_progress_is_one_at_zero_and_decays():
    assert exp_progress(0.0, 0.01) == 1.0
    assert math.isclose(exp_progress(0.01, 0.01), math.exp(-1.0))
    assert exp_progress(0.02, 0.01) < exp_progress(0.01, 0.01)


def test_blend_keeps_a_weak_factor_in_the_same_order():
    factors = [0.4, 0.5, 0.6, 0.4]
    product = math.prod(factors)
    blended = _blend(factors)
    assert product < 0.05
    assert blended > 0.4


def test_stage_phi_fills_ninety_five_percent_of_the_open_stage():
    assert math.isclose(stage_phi(0.0, 1.0, 15.0), 0.95 * 0.15)
    assert math.isclose(stage_phi(15.0, 0.0, 15.0), 0.15)


def test_same_side_weight_is_split_onto_later_stages():
    stages = [{"weight": 20}, {"weight": 20}, {"weight": 15}]
    split_weight(20, stages)
    assert math.isclose(sum(stage["weight"] for stage in stages), 75.0)


def test_opposite_sides_use_x_sign():
    assert opposite_x(-0.2, 0.3)
    assert not opposite_x(0.2, 0.3)


def test_static_stage_weights():
    assert _weight(insert_key_groups()) == 90
    assert _weight(fasten_screws_groups()) == 90
    assert _weight(insert_tubes_groups()) == 90
    assert _weight(build_tower_groups()) == 90
    assert _weight(play_xylophone_groups()) == 90
    keys = play_xylophone_groups()[0]["stages"]
    assert len(keys) == 15
    assert keys[-1]["terms"][0]["kind"] == "hit"
