from __future__ import annotations

import math
from pathlib import Path

import torch

from .models import (
    KBTitanActiveNetwork,
    KBTitanBoundaryEntry,
    KBTitanSpecialEntry,
    KBTitanSpecialIndex,
)


def parse_kinetics_base_special(path: str | Path) -> list[KBTitanSpecialEntry]:
    entries: list[KBTitanSpecialEntry] = []
    for line in Path(path).read_text().splitlines():
        data, _, comment = line.partition("!")
        parts = data.split()
        if len(parts) < 3:
            continue
        try:
            entries.append(
                KBTitanSpecialEntry(
                    index=int(parts[0]),
                    kind=int(parts[1]),
                    target_id=int(parts[2]),
                    comment=comment.strip(),
                )
            )
        except ValueError:
            continue
    return entries

def parse_kinetics_base_special_index(path: str | Path) -> KBTitanSpecialIndex:
    """Parse KINETICS-base's special file as an ISP lookup table."""

    return KBTitanSpecialIndex(parse_kinetics_base_special(path))

def parse_kinetics_base_truncate(path: str | Path) -> KBTitanActiveNetwork:
    """Parse KINETICS-base truncation mappings.

    The reaction mapping is Fortran's IRX array: keys are original .pun reaction
    ids and nonzero values are operational reaction ids in the truncated model.
    """
    lines = Path(path).read_text().splitlines()
    species_values = _parse_kinetics_base_mapping_array(
        lines, "MAPPING ARRAY FOR SPECIES"
    )
    reaction_values = _parse_kinetics_base_mapping_array(
        lines, "MAPPING ARRAY FOR REACTIONS"
    )
    return KBTitanActiveNetwork(
        species_mapping={
            original_id: operational_id
            for original_id, operational_id in enumerate(species_values, start=1)
        },
        reaction_mapping={
            original_id: operational_id
            for original_id, operational_id in enumerate(reaction_values, start=1)
        },
    )

def _parse_kinetics_base_mapping_array(lines: list[str], label: str) -> list[int]:
    values: list[int] = []
    for index, line in enumerate(lines):
        if not line.strip().upper().startswith(label):
            continue
        for current in lines[index + 1 :]:
            parsed = _parse_float_tokens(current)
            if values and not parsed:
                return values
            if not parsed:
                continue
            values.extend(int(value) for value in parsed)
        return values
    raise ValueError(f"missing KINETICS-base truncate section: {label}")

def _parse_kinetics_base_run_radiation_inputs(
    path: str | Path | None,
) -> dict[str, object]:
    if path is None:
        return {}
    file_path = Path(path)
    if not file_path.exists():
        return {}
    lines = file_path.read_text().splitlines()
    values: dict[str, object] = {}

    def numeric_line_after(label: str, skip: int = 1) -> list[float]:
        for index, line in enumerate(lines):
            if line.strip().upper().startswith(label):
                target = index + skip + 1
                if target < len(lines):
                    return _parse_float_tokens(lines[target])
        return []

    planet = numeric_line_after("PLANET PARAMETERS")
    if len(planet) >= 7:
        solar_semimajor_axis_au = planet[0]
        eccentricity = planet[1]
        obliquity = planet[2]
        year_days = planet[3]
        values["planet_day_hours"] = planet[4]
        perihelion_day = planet[5]
        spring_day = planet[6]
    else:
        obliquity = 0.0
        year_days = 1.0
        spring_day = 0.0

    latitude = numeric_line_after("DLAT", skip=0)
    if latitude:
        values["solar_latitude_deg"] = latitude[0]

    grid = numeric_line_after("NALT1", skip=0)
    if len(grid) >= 2:
        # KINETICS-base radiation arrays are dimensioned by NALT2; NALTTP can
        # carry extra storage rows that should not participate in attenuation.
        values["radiation_active_nlyr"] = int(grid[1])

    update = numeric_line_after("UPDATE PARAMETERS")
    if len(update) >= 7:
        values["freeze_actinic_flux"] = int(update[6]) == 0

    radiation = numeric_line_after("BASIC RADIATION PARAMETERS")
    if radiation:
        values["diurnal_average"] = int(radiation[0]) < 2

    photolysis = numeric_line_after("NPHOTO", skip=0)
    if len(photolysis) >= 8:
        nphoto = int(photolysis[0])
        nphots = int(photolysis[1])
        nphotr = int(photolysis[2])
        nphotd = int(photolysis[3])
        nzen = int(photolysis[7])
        ndisort = int(photolysis[12]) if len(photolysis) > 12 else 0
        values["aerosol_extinction_enabled"] = nphotd != 0
        values["aerosol_scattering_enabled"] = nzen != 0
        values["kinetics_direct_radiation"] = nzen == 0 and ndisort == 0
        active_photo_ids = _collect_numeric_block_after_label(
            lines, "IPHOTO", nphoto + nphots + nphotr + nphotd
        )
        values["active_photo_reaction_ids"] = active_photo_ids
        values["active_opacity_reaction_ids"] = active_photo_ids[:nphoto]

    timing = numeric_line_after("ICYEAR", skip=0)
    if len(timing) >= 3:
        day = timing[1]
        values["solar_hour"] = timing[2]
        if year_days != 0.0:
            distance, declination = _kinetics_base_orbit_distance_declination(
                solar_semimajor_axis_au,
                eccentricity,
                obliquity,
                year_days,
                values.get("planet_day_hours", 1.0),
                perihelion_day,
                spring_day,
                day,
            )
            values["solar_distance_au"] = distance
            values["solar_declination_deg"] = declination

    latitude_deg = values.get("solar_latitude_deg")
    declination_deg = values.get("solar_declination_deg")
    hour = values.get("solar_hour")
    day_hours = values.get("planet_day_hours")
    if (
        isinstance(latitude_deg, float)
        and isinstance(declination_deg, float)
        and isinstance(hour, float)
        and isinstance(day_hours, float)
        and day_hours > 0.0
    ):
        hour_angle = 2.0 * math.pi * (hour / day_hours - 0.5)
        lat = math.radians(latitude_deg)
        dec = math.radians(declination_deg)
        mu0 = math.sin(dec) * math.sin(lat) + math.cos(dec) * math.cos(lat) * math.cos(
            hour_angle
        )
        values["solar_mu0"] = max(min(mu0, 1.0), 1.0e-6)

    return values

def _parse_float_tokens(line: str) -> list[float]:
    values: list[float] = []
    for token in line.replace(",", " ").split():
        try:
            values.append(float(token))
        except ValueError:
            continue
    return values

def _collect_numeric_block_after_label(
    lines: list[str],
    label: str,
    count: int,
) -> list[int]:
    values: list[int] = []
    for index, line in enumerate(lines):
        if not line.strip().upper().startswith(label):
            continue
        for current in lines[index + 1 :]:
            for value in _parse_float_tokens(current):
                values.append(int(value))
                if len(values) >= count:
                    return values
            if values and not _parse_float_tokens(current):
                return values
        return values
    return values

def _kinetics_base_orbit_distance_declination(
    semimajor_axis_au: float,
    eccentricity: float,
    obliquity_deg: float,
    year_days: float,
    day_hours: float,
    perihelion_day: float,
    spring_day: float,
    day: float,
) -> tuple[float, float]:
    period = year_days * day_hours * 3600.0
    perihelion_offset = perihelion_day * day_hours * 3600.0
    season_day = (day - spring_day) % year_days
    time_since_equinox = season_day * day_hours * 3600.0

    def eccentric_anomaly(mean_anomaly: float) -> float:
        value = eccentricity * math.sin(mean_anomaly) + mean_anomaly
        for _ in range(20):
            next_value = value + (
                mean_anomaly - value + eccentricity * math.sin(value)
            ) / (1.0 - eccentricity * math.cos(value))
            if abs(next_value - value) < 1.0e-7:
                return next_value
            value = next_value
        return value

    mean_anomaly = 2.0 * math.pi * (time_since_equinox + perihelion_offset) / period
    anomaly = eccentric_anomaly(mean_anomaly)
    distance = semimajor_axis_au * (1.0 - eccentricity * math.cos(anomaly))
    true_anomaly = 2.0 * math.atan(
        math.sqrt((1.0 + eccentricity) / (1.0 - eccentricity))
        * math.tan(0.5 * anomaly)
    )

    perihelion_mean = 2.0 * math.pi * perihelion_offset / period
    perihelion_anomaly = eccentric_anomaly(perihelion_mean)
    perihelion_angle = 2.0 * math.atan(
        math.sqrt((1.0 + eccentricity) / (1.0 - eccentricity))
        * math.tan(0.5 * perihelion_anomaly)
    )
    declination = obliquity_deg * math.sin(true_anomaly - perihelion_angle)
    return distance, declination

def _parse_kinetics_base_catalog(path: str | Path) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        if len(line) > 60:
            equation = line[:60].strip()
            filename = line[60:].strip()
        else:
            parts = line.rsplit(None, 1)
            if len(parts) != 2:
                continue
            equation, filename = parts[0].strip(), parts[1].strip()
        if equation and filename:
            entries.append((equation, filename))
    return entries

def _parse_kinetics_base_flux(path: str | Path) -> list[tuple[float, float, float]]:
    rows: list[tuple[float, float, float]] = []
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            wavelength = float(parts[0])
            width = float(parts[1])
            value = float(parts[2])
        except ValueError:
            continue
        rows.append((wavelength, width, value))
    return rows

def _parse_kinetics_base_aerosol_extinction(
    path: str | Path | None,
    flux: list[tuple[float, float, float]],
) -> dict[str, Any] | None:
    return _parse_kinetics_base_aerosol_property(path, flux, outside_value=0.0)

def _infer_kinetics_base_aerosol_path(
    source_path: str | Path | None,
    source_token: str,
    target_token: str,
) -> Path | None:
    if source_path is None:
        return None
    path = Path(source_path)
    inferred = path.with_name(path.name.replace(source_token, target_token))
    return inferred if inferred.exists() else None

def _parse_kinetics_base_aerosol_property(
    path: str | Path | None,
    flux: list[tuple[float, float, float]],
    *,
    outside_value: float = 0.0,
) -> dict[str, Any] | None:
    if path is None:
        return None
    file_path = Path(path)
    if not file_path.exists():
        return None
    tokens = file_path.read_text().split()
    try:
        nlat_pos = tokens.index("NLAT")
        nalt = int(tokens[nlat_pos + 6])
        nwave = int(tokens[nlat_pos + 7])
        wave_pos = tokens.index("WAVELENGTH") + 1
    except (ValueError, IndexError):
        return None

    wavelengths = [float(value) for value in tokens[wave_pos : wave_pos + nwave]]
    cursor = wave_pos + nwave
    values_by_wave: list[list[float]] = []
    while cursor < len(tokens) and len(values_by_wave) < nwave:
        if tokens[cursor] != "NWAVE":
            cursor += 1
            continue
        cursor += 2
        block: list[float] = []
        while cursor < len(tokens) and len(block) < nalt:
            try:
                block.append(float(tokens[cursor]))
            except ValueError:
                break
            cursor += 1
        if len(block) == nalt:
            values_by_wave.append(block)
    if len(values_by_wave) != nwave:
        return None

    flux_wavelengths = [row[0] for row in flux]
    aligned_by_alt: list[list[float]] = []
    for layer in range(nalt):
        layer_values = [values_by_wave[wave][layer] for wave in range(nwave)]
        aligned_by_alt.append(
            [
                _linear_interpolate_with_outside(
                    wavelengths, layer_values, wavelength, outside_value
                )
                for wavelength in flux_wavelengths
            ]
        )
    return {
        # KINETICS-base aerosol files do not repeat the altitude grid; Titan's
        # interpolation inputs use 91 levels spanning 0-900 km.
        "altitude_km": [float(i) * 900.0 / float(nalt - 1) for i in range(nalt)],
        "values": aligned_by_alt,
    }

def _parse_kinetics_base_cross_section(
    path: Path,
) -> dict[int, list[tuple[float, float]]]:
    if not path.exists():
        return {}
    lines = path.read_text().splitlines()
    if len(lines) < 5:
        return {}
    try:
        n_datasets = int(lines[1].strip())
    except ValueError:
        return {}

    line_idx = 2
    datasets: dict[int, list[tuple[float, float]]] = {}
    for _ in range(n_datasets):
        if line_idx + 1 >= len(lines):
            break
        try:
            dataset_type = int(float(lines[line_idx].split()[0]))
        except (ValueError, IndexError):
            break
        line_idx += 1

        try:
            meta = [int(float(value)) for value in lines[line_idx].split()]
        except ValueError:
            break
        if len(meta) >= 2 and meta[0] > 0 and meta[1] > meta[0]:
            n_points = meta[1] - meta[0] + 1
        elif meta:
            n_points = meta[-1]
        else:
            break
        line_idx += 1

        rows: list[tuple[float, float]] = []
        for _ in range(n_points):
            if line_idx >= len(lines):
                break
            parts = lines[line_idx].split()
            line_idx += 1
            if len(parts) < 2:
                continue
            try:
                rows.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
        if rows and dataset_type not in datasets:
            datasets[dataset_type] = rows

    return datasets

def _parse_kinetics_base_cross_section_on_flux(
    path: Path,
    flux: list[tuple[float, float, float]],
) -> dict[int, list[float]]:
    if not path.exists():
        return {}
    lines = path.read_text().splitlines()
    if len(lines) < 5:
        return {}
    try:
        n_datasets = int(lines[1].strip())
    except ValueError:
        return {}

    line_idx = 2
    datasets: dict[int, list[float]] = {}
    for _ in range(n_datasets):
        if line_idx + 1 >= len(lines):
            break
        try:
            dataset_type = int(float(lines[line_idx].split()[0]))
        except (ValueError, IndexError):
            break
        line_idx += 1

        try:
            meta = [int(float(value)) for value in lines[line_idx].split()]
        except ValueError:
            break
        if len(meta) < 2:
            break
        first_bin, last_bin = meta[0], meta[1]
        n_points = last_bin - first_bin + 1
        if n_points <= 0:
            break
        line_idx += 1

        values = [0.0 for _ in flux]
        for offset in range(n_points):
            if line_idx >= len(lines):
                break
            parts = lines[line_idx].split()
            line_idx += 1
            if len(parts) < 2:
                continue
            original_bin = first_bin + offset
            # KINETICS-base stores cross sections by wavelength-bin number and
            # ignores the text wavelength column during the Fortran read.  The
            # first flux row is bin 1; bin 0, when present, has no flux row.
            flux_index = original_bin - 1
            if flux_index < 0 or flux_index >= len(values):
                continue
            try:
                values[flux_index] = float(parts[1])
            except ValueError:
                continue
        if dataset_type not in datasets:
            datasets[dataset_type] = values

    return datasets

def _multiply_kinetics_base_cross_sections(
    absorption: list[tuple[float, float]],
    branch: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    wavelengths = [row[0] for row in absorption]
    values = [row[1] for row in absorption]
    return [
        (wavelength, value * _linear_interpolate(wavelengths, values, wavelength))
        for wavelength, value in branch
    ]

def _integrate_kinetics_base_photo_rate(
    cross_section: list[tuple[float, float]],
    flux: list[tuple[float, float, float]],
) -> float:
    return sum(
        cross_value * flux_value
        for _, _, flux_value, cross_value in _align_kinetics_base_flux_cross_section(
            cross_section, flux
        )
    )

def _align_kinetics_base_flux_cross_section(
    cross_section: list[tuple[float, float]],
    flux: list[tuple[float, float, float]],
) -> list[tuple[float, float, float, float]]:
    wavelengths = [row[0] for row in cross_section]
    values = [row[1] for row in cross_section]
    rows: list[tuple[float, float, float, float]] = []
    for wavelength, width, flux_value in flux:
        sigma = _linear_interpolate(wavelengths, values, wavelength)
        rows.append((wavelength, width, flux_value, sigma))
    return rows

def _align_kinetics_base_absorption_cross_section(
    absorption: list[tuple[float, float]],
    flux: list[tuple[float, float, float]],
) -> list[float]:
    wavelengths = [row[0] for row in absorption]
    values = [row[1] for row in absorption]
    return [
        _linear_interpolate(wavelengths, values, wavelength)
        for wavelength, _, _ in flux
    ]

def _linear_interpolate(xs: list[float], ys: list[float], x: float) -> float:
    return _linear_interpolate_with_outside(xs, ys, x, 0.0)

def _linear_interpolate_with_outside(
    xs: list[float],
    ys: list[float],
    x: float,
    outside_value: float,
) -> float:
    if not xs or not ys or x < xs[0] or x > xs[-1]:
        return outside_value
    if x == xs[0]:
        return ys[0]
    for i in range(1, len(xs)):
        if x <= xs[i]:
            if x == xs[i]:
                return ys[i]
            frac = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
            return ys[i - 1] + frac * (ys[i] - ys[i - 1])
    return ys[-1]

def _parse_kinetics_base_equation(equation: str) -> tuple[list[str], list[str]]:
    left, sep, right = equation.partition("=")
    if not sep:
        return [], []
    return _parse_kinetics_base_side(left), _parse_kinetics_base_side(right)

def _parse_kinetics_base_side(side: str) -> list[str]:
    # Catalog entries sometimes concatenate a species and a following
    # coefficient without whitespace, e.g. "1,3-C4H6+2H" (from
    # CROSS_C4H8=1,3-C4H6+2H_LORES1.DAT) or "CL+2F". Insert a separator
    # before "+digit" so the whitespace split picks up both tokens. Guard
    # against breaking doubly-charged cations like "C++" by requiring the
    # preceding character to be alphanumeric (not "+").
    import re
    side = re.sub(r"(?<=[A-Za-z0-9])\+(?=\d)", " +", side)
    species: list[str] = []
    for raw in side.split():
        raw = raw.lstrip("+")
        if raw == "+":
            continue
        # Skip bath gas / electron symbols. The previous ``"-" in raw`` check
        # was too aggressive: it also dropped legitimate hyphenated species
        # names like ``l-C3H``, ``c-C3H2``, ``1,2-C4H6`` and their cations.
        # Result: KB's photo reactions on these species (rxn 22 and similar)
        # were never built in kintera, leaving a permanent KB-only gap.
        if raw in {"M", "E", "-"}:
            continue
        if raw.endswith("-") and raw != "-":
            # Anion notation like "E-" (we don't have any in this network,
            # but keep the guard so future networks parse correctly).
            continue
        count = 1
        name = raw
        if raw and raw[0].isdigit():
            # Only treat leading digits as a stoichiometric coefficient when
            # they are immediately followed by a letter (e.g. "2H", "2C2H2").
            # Tokens like "1-C4H6", "1,2-C4H6", "1,3-C4H6" begin with a digit
            # that is part of the species name; previously the digits were
            # silently stripped, turning "1-C4H6" into "-C4H6" and breaking
            # the catalog→pun key match for every 1-C4H6 / 1,2-C4H6 / 1,3-C4H6
            # photolysis branch (KB-only rxns 65, 66, 72, 73 etc).
            digit_end = 0
            while digit_end < len(raw) and raw[digit_end].isdigit():
                digit_end += 1
            if digit_end < len(raw) and raw[digit_end].isalpha():
                count = int(raw[:digit_end])
                name = raw[digit_end:]
        if name:
            species.extend([_normalize_kinetics_base_photo_species(name)] * count)
    return species

def _normalize_kinetics_base_photo_species(name: str) -> str:
    normalized = name.strip()
    if normalized.startswith("aN"):
        normalized = "N" + normalized[2:]
    if normalized.startswith("a"):
        normalized = normalized[1:]
    return normalized

def _kinetics_base_reaction_key(reactants: list[str], products: list[str]) -> str:
    return (
        _kinetics_base_side_key(reactants)
        + "\x1e"
        + _kinetics_base_side_key(products)
    )

def _kinetics_base_side_key(species: list[str]) -> str:
    counts: dict[str, int] = {}
    order: list[str] = []
    for name in species:
        normalized = _normalize_kinetics_base_photo_species(name).upper()
        if normalized in {"M", "E"}:
            continue
        if normalized not in counts:
            order.append(normalized)
            counts[normalized] = 0
        counts[normalized] += 1
    return "\x1f".join(
        (str(counts[name]) if counts[name] > 1 else "") + name for name in order
    )

def _kinetics_base_vapor_coefficients_from_pun(
    pun: str | Path,
) -> dict[str, tuple[float, float, float, float, float]]:
    coefficients: dict[str, tuple[float, float, float, float, float]] = {}
    lines = Path(pun).read_text().splitlines()
    for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) < 2 or not parts[0].endswith("."):
            continue
        name = parts[1]
        if i + 4 >= len(lines):
            continue
        values = lines[i + 4].split()
        if len(values) < 5:
            continue
        try:
            coeffs = tuple(float(value) for value in values[:5])
        except ValueError:
            continue
        if coeffs[0] != 0.0:
            coefficients[name] = coeffs  # type: ignore[assignment]
    return coefficients

def _read_kinetics_base_boundary_file(
    boundary_path: str | Path,
) -> list[tuple[int, float, str]]:
    return [
        (entry.lower_kind, entry.lower_value, entry.species)
        for entry in parse_kinetics_base_boundary(boundary_path)
    ]

def parse_kinetics_base_boundary(
    boundary_path: str | Path,
) -> list[KBTitanBoundaryEntry]:
    entries: list[KBTitanBoundaryEntry] = []
    for line in Path(boundary_path).read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            entries.append(
                KBTitanBoundaryEntry(
                    lower_kind=int(parts[0]),
                    lower_value=float(parts[1]),
                    upper_kind=int(parts[2]),
                    upper_value=float(parts[3]),
                    species=parts[4],
                )
            )
        except ValueError:
            continue
    return entries

def _normalize_kinetics_base_boundary_species(name: str) -> str:
    if len(name) > 1 and name[0].isdigit():
        return f"({name[0]}){name[1:]}"
    return name

def _canonical_kinetics_base_species_name(
    name: str,
    canonical_species: dict[str, str],
) -> str:
    return canonical_species.get(name.upper(), name)

def _apply_cheng_cold_trap_boundaries(
    concentration: torch.Tensor,
    conversion: dict[str, str],
    density: torch.Tensor,
    species: list[str],
    profile_values: dict[str, torch.Tensor] | None = None,
) -> None:
    """Apply the KINETICS-base ``__CHENG`` cold trap boundary condition for CH4.

    The Cheng Titan branch sets the CH4 cold-trap boundary at level 24
    (1-indexed, 0-indexed 23, ~500 km altitude). KB's bc_save lists
    ``CH4: lower_kind=5 lower_value=4e-4`` as a generic lower-mixing-ratio
    BC, but that value is the historic surface flux rather than the
    atmospheric mixing ratio (which is 4e-3 in the atm file). KB ignores
    the bc_save row for CH4 because the Cheng cold-trap mechanism
    overrides it.  Match that behaviour by restoring CH4 below the cold
    trap (lev 0–23) to its atm-file profile value, undoing the rounded
    4e-4 the lower-BC pass installed.
    """
    _COLD_TRAP_LEVEL = 23  # 0-indexed; Fortran level 24, ~500 km on Titan

    if _COLD_TRAP_LEVEL >= density.numel() or density[_COLD_TRAP_LEVEL] == 0:
        return
    species_lower = {name.lower(): i for i, name in enumerate(species)}
    ch4_idx = species_lower.get("ch4")
    if ch4_idx is None:
        return
    canonical = species[ch4_idx]
    conversion[canonical] = "kinetics_base_cheng_cold_trap_mixing_ratio"

    # Restore CH4 concentration in the lower atmosphere (lev 0 through the
    # cold trap) from the atm-file profile, overriding any prior lower-BC
    # mixing-ratio overwrite. This matches KB which uses the atm-file
    # mixing ratio (4e-3) instead of the bc_save value (4e-4).
    if profile_values is not None and canonical in profile_values:
        ch4_profile = profile_values[canonical]
        # Profile is in mixing ratio (matches the .pun_zero_conc atm file);
        # convert to number density at the levels we're restoring.
        nlevs = min(_COLD_TRAP_LEVEL + 1, ch4_profile.shape[0])
        for L in range(nlevs):
            concentration[L, ch4_idx] = ch4_profile[L] * density[L]


def _apply_lower_mixing_ratio_boundaries(
    concentration: torch.Tensor,
    conversion: dict[str, str],
    density: torch.Tensor,
    species: list[str],
    boundary_path: str | Path,
) -> None:
    # Build case-insensitive lookup so that bc entries like "gch4" match species "GCH4".
    species_index_lower = {name.lower(): i for i, name in enumerate(species)}
    # Last level with non-zero density; escape velocities pin concentration there to 0.
    nonzero_levels = (density > 0).nonzero(as_tuple=True)[0]
    last_real = int(nonzero_levels[-1]) if nonzero_levels.numel() > 0 else len(density) - 1
    for entry in parse_kinetics_base_boundary(boundary_path):
        normalized = _normalize_kinetics_base_boundary_species(entry.species)
        if normalized.lower() not in species_index_lower:
            continue
        j = species_index_lower[normalized.lower()]
        canonical = species[j]
        # Lower boundary conditions.
        if entry.lower_kind == 5:
            concentration[0, j] = entry.lower_value * density[0]
            conversion[canonical] = "lower_boundary_mixing_ratio_times_density"
        elif entry.lower_kind == 2 and entry.lower_value < 0:
            # Type-2 velocity BC with a negative (downward) deposition velocity.
            # In KINETICS-base's implicit diffusion-chemistry solver the deposition
            # term VSTAR1 = GAMA1 * RAMD1 * XLOWER is added to the diagonal and the
            # RHS of the lowest-level equation, effectively acting as a very strong
            # loss that drives the surface concentration to zero.  Pin the initial
            # concentration and the held boundary to 0 to match this behaviour.
            concentration[0, j] = 0.0
            conversion[canonical] = "lower_boundary_deposition_velocity_zero"
        # Upper boundary conditions.
        if entry.upper_kind == 2 and entry.upper_value > 0:
            # Type-2 velocity BC with a positive (upward) escape velocity at the top.
            # In KINETICS-base's BNDRY1, the escape term VSTARN = GAMAN * RAMDN * XUPPER
            # adds a strong loss on the diagonal of the topmost active level, driving
            # its concentration to near zero.  Pin the last real level to 0 to reproduce
            # this behaviour and prevent accumulation that triggers numerical instability.
            concentration[last_real, j] = 0.0
            conversion[canonical] = "upper_boundary_escape_velocity_zero"

