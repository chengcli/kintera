from kintera import SpeciesThermo, ThermoOptions, ThermoX

if __name__ == "__main__":
    # Create an instance of ThermoOptions
    op = ThermoOptions.from_yaml("jupiter.yaml")
    thermo = ThermoX(op)

    print(thermo)
