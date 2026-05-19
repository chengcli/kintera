from __future__ import annotations


def kinetics_base_species_charge(name: str) -> int:
    """Return the integer charge encoded in a KINETICS-base species name."""

    if name == "E":
        return -1
    if name.endswith("+"):
        return 1
    if name.endswith("-"):
        return -1
    return 0


def kinetics_base_reaction_charge_summary(
    reactants: list[str],
    products: list[str],
) -> dict[str, object]:
    reactant_charge = sum(kinetics_base_species_charge(name) for name in reactants)
    product_charge = sum(kinetics_base_species_charge(name) for name in products)
    charged_reactants = [
        name for name in reactants if kinetics_base_species_charge(name) != 0
    ]
    charged_products = [
        name for name in products if kinetics_base_species_charge(name) != 0
    ]
    return {
        "reactant_charge": reactant_charge,
        "product_charge": product_charge,
        "net_charge_delta": product_charge - reactant_charge,
        "charge_balanced": reactant_charge == product_charge,
        "charged_reactants": charged_reactants,
        "charged_products": charged_products,
    }


def is_kinetics_base_dissociative_recombination(
    reactants: list[str],
    products: list[str],
) -> bool:
    del products
    return "E" in reactants and any(
        kinetics_base_species_charge(name) > 0 for name in reactants
    )


def is_kinetics_base_charged_reaction(
    reactants: list[str],
    products: list[str],
) -> bool:
    return any(
        kinetics_base_species_charge(name) != 0 for name in [*reactants, *products]
    )


def kinetics_base_thermal_ion_kind(
    reactants: list[str],
    products: list[str],
) -> str:
    if is_kinetics_base_dissociative_recombination(reactants, products):
        return "pun_dissociative_recombination"
    if is_kinetics_base_charged_reaction(reactants, products):
        return "pun_ion_mass_action_reaction"
    return "pun_thermal_reaction"
