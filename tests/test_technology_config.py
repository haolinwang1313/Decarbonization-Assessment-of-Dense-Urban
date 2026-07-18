from __future__ import annotations

from paper05.model.technology import technology_config


def test_technology_config_infers_ev_penetration_levels_from_technology_name() -> None:
    low = technology_config({"technology_set": "base_with_EV_low_smart"})
    medium = technology_config({"technology_set": "base_with_EV_medium_smart"})
    high = technology_config({"technology_set": "base_with_EV_high_smart"})
    combo = technology_config({"technology_set": "base_with_DR10_EV_medium_thermal10"})

    assert low["ev_penetration"] == "low"
    assert medium["ev_penetration"] == "medium"
    assert high["ev_penetration"] == "high"
    assert combo["ev_penetration"] == "medium"
    assert combo["dr_fraction"] == 0.1
    assert combo["thermal_fraction"] == 0.1
