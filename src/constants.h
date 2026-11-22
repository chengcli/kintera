#pragma once

namespace kintera {
namespace constants {
double constexpr Avogadro = 6.02214076e23;  // mol^-1, Avogadro's number
double constexpr Rgas = 8.31446261815324;  // J/(mol*K), universal gas constant,
                                           // exact SI (Avogadro * KBoltz)
double constexpr ElectronCharge = 1.602176634e-19;  // C, elementary charge
double constexpr OneAtm = 1.01325e5;     // Pa, standard atmospheric pressure
double constexpr KBoltz = 1.380649E-23;  // J/K, Boltzmann constant
double constexpr JoulePerCal = 4.184;    // J/cal, thermochemical calorie
}  // namespace constants
}  // namespace kintera
