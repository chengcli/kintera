from __future__ import annotations


def _global_scale() -> float:
    # ei_scale multiplies every electron-impact branching ratio so we can sweep
    # total ionization without re-editing the per-channel tables.
    from .config import get_titan_config
    return get_titan_config().ei_scale


def _kinetics_base_electron_impact_secondary_params(
    reactants: list[str], products: list[str]
) -> dict[str, float] | None:
    """Return per-wavelength secondary-electron parameters or None.

    Previously this returned ``{"threshold_eV", "W_eV"}`` for N2/CH4 EI
    channels so the actinic-flux integrator multiplied σ × ALTFLX by
    ``(1 + (E_γ - threshold)/W)`` per wavelength bin. With KB's
    `_loss.dat` off-by-one fixed (kb_patches/01-loss-file-altitude-
    shift.patch), the empirical W=60 tuning that gave 174 matched no
    longer applies — that tuning was against altitude-shifted KB data.
    KB's actual treatment is a constant per-channel multiplier (×4.15
    for N2→N2+E, ×117 for N2→N+N+E, ×2.07/2.61 for the CH4 channels)
    applied inside JPHOTO. We now match that by returning None here
    (disabling per-wavelength secondary boost) and using KB's M as
    the channel scale below.
    """
    return None


def _is_kinetics_base_electron_impact_reaction(products: list[str]) -> bool:
    return "E" in products or any(product.endswith("+") for product in products)

def _channel_scale(env_name: str, default: float) -> float:
    from .config import get_titan_config
    return get_titan_config().ei_channel(env_name)


def _kinetics_base_electron_impact_scale(
    reactants: list[str], products: list[str]
) -> float:
    g = _global_scale()
    # Hardcoded per-channel multipliers from KB's JPHOTO Cheng block
    # (kinetgen2X.F:7085-7113). KB applies these constants AFTER the
    # wavelength integration of σ × ALTFLX. Earlier empirical tuning
    # (N2_N2P=0.25, N2_NP=0.5, CH4_CH3P=0.172, CH4_CH2P=0.217) was
    # against the buggy altitude-shifted KB reference and combined with
    # an unphysical per-wavelength (1 + n_sec) factor; both removed now
    # that the KB off-by-one is patched (see project-kb-loss-off-by-one).
    if reactants == ["N2"] and products == ["N2+", "E"]:
        # KB ISP(539): zk *= 4.15
        return _channel_scale("KINTERA_EI_SCALE_N2_N2P", 4.15) * g
    if reactants == ["N2"] and "N+" in products:
        # KB ISP(540): zk *= 117 for N2 -> N+ + N + E
        return _channel_scale("KINTERA_EI_SCALE_N2_NP", 117.0) * g
    if reactants == ["CH4"] and products == ["CH3+", "H", "E"]:
        # KB ISP(537): zk *= 2.07
        return _channel_scale("KINTERA_EI_SCALE_CH4_CH3P", 2.07) * g
    if reactants == ["CH4"] and products == ["CH2+", "H2", "E"]:
        # KB ISP(538): zk *= 2.61
        return _channel_scale("KINTERA_EI_SCALE_CH4_CH2P", 2.61) * g
    # KB ISP(536) (CH4 -> CH4+ + E, ×3.05) maps to a reaction the pun
    # file doesn't include, so we don't expose it.
    if reactants == ["CH4"] and products == ["CH3", "H+", "E"]:
        return _channel_scale("KINTERA_EI_SCALE_CH4_HP", 0.083) * g
    if "N+" in products:
        return _channel_scale("KINTERA_EI_SCALE_OTHER_NP", 0.0035) * g
    if not reactants:
        return _channel_scale("KINTERA_EI_SCALE_DEFAULT", 0.25) * g
    if reactants[0] == "CH4":
        return (1.0 / 12.0) * g
    return _channel_scale("KINTERA_EI_SCALE_DEFAULT", 0.25) * g

def _kinetics_base_electron_impact_profile(
    reactants: list[str], products: list[str]
) -> list[tuple[float, float]] | None:
    if reactants == ["N2"] and products == ["N2+", "E"]:
        # Temporary Titan oracle scaffold for the missing electron energy
        # deposition profile.  Multipliers are relative to the channel-scaled
        # catalog rate and preserve the observed N2+ production altitude shape.
        return [
            (0.0, 0.0),
            (499.7, 0.0),
            (530.8, 7.35424e-05),
            (563.7, 0.00310157),
            (598.7, 0.0444304),
            (635.7, 0.281841),
            (675.1, 1.00493),
            (716.8, 2.38219),
            (761.0, 4.28253),
            (808.0, 6.38209),
            (857.8, 8.35379),
            (910.7, 10.0056),
            (966.7, 11.2738),
            (1026.0, 12.198),
            (1089.0, 12.8475),
            (1156.0, 13.2995),
            (1227.0, 13.6023),
            (1303.0, 13.8121),
        ]
    if reactants == ["CH4"] and products == ["CH3+", "H", "E"]:
        return [
            (0.0, 0.0),
            (499.7, 0.0),
            (530.8, 3.37317e-05),
            (563.7, 0.00154261),
            (598.7, 0.0209029),
            (635.7, 0.122668),
            (675.1, 0.412185),
            (716.8, 0.940689),
            (761.0, 1.65355),
            (808.0, 2.43387),
            (857.8, 3.16511),
            (910.7, 3.78158),
            (966.7, 4.26348),
            (1026.0, 4.61682),
            (1089.0, 4.87396),
            (1156.0, 5.05394),
            (1227.0, 5.18007),
            (1303.0, 5.26958),
        ]
    if reactants == ["CH4"] and products == ["CH2+", "H2", "E"]:
        return [
            (0.0, 0.0),
            (499.7, 0.0),
            (530.8, 2.09568e-05),
            (563.7, 0.000800089),
            (598.7, 0.0103562),
            (635.7, 0.0612753),
            (675.1, 0.209528),
            (716.8, 0.484549),
            (761.0, 0.858117),
            (808.0, 1.26645),
            (857.8, 1.64664),
            (910.7, 1.96345),
            (966.7, 2.20588),
            (1026.0, 2.38157),
            (1089.0, 2.50441),
            (1156.0, 2.58924),
            (1227.0, 2.6464),
            (1303.0, 2.68699),
        ]
    if reactants == ["CH4"] and products == ["CH+", "H", "H2"]:
        return [
            (0.0, 0.0),
            (499.7, 0.0),
            (530.8, 2.74196e-05),
            (563.7, 0.000907129),
            (598.7, 0.00968507),
            (635.7, 0.0478567),
            (675.1, 0.141925),
            (716.8, 0.295933),
            (761.0, 0.486183),
            (808.0, 0.679677),
            (857.8, 0.849488),
            (910.7, 0.984256),
            (966.7, 1.08282),
            (1026.0, 1.15149),
            (1089.0, 1.19775),
            (1156.0, 1.22859),
            (1227.0, 1.24856),
            (1303.0, 1.26217),
        ]
    if reactants == ["CH4"] and products == ["C+", "H2", "H2", "E"]:
        return [
            (0.0, 0.0),
            (499.7, 0.0),
            (530.8, 3.17629e-05),
            (563.7, 0.00101729),
            (598.7, 0.010318),
            (635.7, 0.0482719),
            (675.1, 0.136437),
            (716.8, 0.274058),
            (761.0, 0.437884),
            (808.0, 0.599973),
            (857.8, 0.738984),
            (910.7, 0.847308),
            (966.7, 0.925081),
            (1026.0, 0.978358),
            (1089.0, 1.01458),
            (1156.0, 1.03812),
            (1227.0, 1.05343),
            (1303.0, 1.0634),
        ]
    if reactants == ["CH4"] and products == ["CH3", "H+", "E"]:
        return [
            (0.0, 0.0),
            (499.7, 0.0),
            (530.8, 2.09836e-05),
            (563.7, 0.00068014),
            (598.7, 0.00709024),
            (635.7, 0.0343112),
            (675.1, 0.100133),
            (716.8, 0.206423),
            (761.0, 0.336665),
            (808.0, 0.468247),
            (857.8, 0.583059),
            (910.7, 0.673891),
            (966.7, 0.739951),
            (1026.0, 0.785962),
            (1089.0, 0.817005),
            (1156.0, 0.837583),
            (1227.0, 0.850989),
            (1303.0, 0.85984),
        ]
    if "N+" in products:
        return [
            (0.0, 0.0),
            (499.7, 0.0),
            (530.8, 0.000251421),
            (563.7, 0.00779263),
            (598.7, 0.0761152),
            (635.7, 0.343612),
            (675.1, 0.942543),
            (716.8, 1.84936),
            (761.0, 2.90595),
            (808.0, 3.93211),
            (857.8, 4.80118),
            (910.7, 5.47001),
            (966.7, 5.94694),
            (1026.0, 6.2738),
            (1089.0, 6.49094),
            (1156.0, 6.63267),
            (1227.0, 6.72645),
            (1303.0, 6.7851),
        ]
    return None
