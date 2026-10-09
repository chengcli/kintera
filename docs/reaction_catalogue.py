"""Offline, reviewed coefficient records. No network access or native imports.

All curves return ln Q with numerical partial pressures in Pa. The records
explicitly distinguish implementation, primary data and conditional derivations.
"""
import numpy as np

LN10 = np.log(10.)
NIST = 'https://webbook.nist.gov/cgi/cbook.cgi?ID={}&Mask=4&Type=ANTOINE&Plot=on'
MORLEY = 'https://arxiv.org/html/1206.4313'
VISSCHER = 'https://arxiv.org/abs/astro-ph/0511136'
VISSCHER10 = 'https://arxiv.org/html/1001.3639'


def curve(name, kind, params, interval, status):
    return dict(name=name, kind=kind, params=params, interval=interval, status=status)


def ideal(name, params, interval):
    return curve(name, 'ideal', params, interval, 'legacy; coefficient provenance unresolved')


def antoine(name, params, interval):
    return curve(name, 'antoine', params, interval, 'NIST WebBook Antoine table')


def linear(name, a, b, interval, status, base=10):
    return curve(name, 'linear', [a, b, base], interval, status)


# Species composition and stoichiometry also drive isolated native tests.
COMPOSITION = {
    'He': {'He': 1}, 'H2': {'H': 2}, 'H2O': {'H': 2, 'O': 1},
    'NH3': {'N': 1, 'H': 3}, 'H2S': {'H': 2, 'S': 1},
    'CH4': {'C': 1, 'H': 4}, 'SO2': {'S': 1, 'O': 2}, 'CO2': {'C': 1, 'O': 2},
    'KCl': {'K': 1, 'Cl': 1}, 'K': {'K': 1}, 'HCl': {'H': 1, 'Cl': 1},
    'NH4SH': {'N': 1, 'H': 5, 'S': 1}, 'Mn': {'Mn': 1}, 'MnS': {'Mn': 1, 'S': 1},
    'Zn': {'Zn': 1}, 'ZnS': {'Zn': 1, 'S': 1}, 'Na': {'Na': 1}, 'Na2S': {'Na': 2, 'S': 1},
    'Mg': {'Mg': 1}, 'SiH4': {'Si': 1, 'H': 4}, 'MgSiO3': {'Mg': 1, 'Si': 1, 'O': 3},
}
REACTIONS = {}


def add(key, title, reactants, cloud, curves, sources, note='', h2=0, cloud_nu=1):
    gas = dict(reactants)
    if h2:
        gas['H2'] = -h2
    REACTIONS[key] = dict(title=title, gas=gas, cloud=cloud, cloud_nu=cloud_nu,
                          curves=curves, sources=sources, note=note)


add('h2o', 'Water', {'H2O': 1}, 'H2O', [
    ideal('h2o_ideal', [273.16, 611.7, 24.845, 4.986009, 22.98, .52], [230, 303]),
    ideal('h2o_bryan', [273.16, 611.7, 24.815845, 4.986009, 24.815845, 4.986009], [230, 303]),
    antoine('NIST liquid', [5.40221, 1838.675, -31.737], [273, 303])],
    [('Bridgeman and Aldrich (1964), NIST Antoine table, 273–303 K', NIST.format('C7732185'))],
    'The ideal branches switch at 273.16 K. h2o_bryan continues the liquid branch below the triple point; it is not an ice fit. The displayed legacy interval is a plotting interval, not a verified validity range.')
add('nh3', 'Ammonia', {'NH3': 1}, 'NH3', [
    ideal('nh3_ideal', [195.4, 6060, 20.08, 5.62, 20.64, 1.43], [164, 300]),
    antoine('NIST low T', [3.18757, 506.713, -80.78], [164, 239.6]),
    antoine('NIST high T', [4.86886, 1113.928, -10.409], [239.6, 300])],
    [('Stull (1947), NIST Antoine table (high branch valid to 371.5 K)', NIST.format('C7664417'))],
    'The ideal solid/liquid switch is 195.4 K. Its fitted beta/gamma provenance is unresolved; the comparison does not certify either branch.')
add('h2s', 'Hydrogen sulfide', {'H2S': 1}, 'H2S', [
    ideal('h2s_ideal', [187.63, 23300, 11.89, 5.04, 11.89, 5.04], [138.8, 349.5]),
    antoine('h2s_antoine low T', [4.43681, 829.439, -25.412], [138.8, 212.8]),
    antoine('h2s_antoine high T', [4.52887, 958.587, -.539], [212.8, 349.5])],
    [('Stull (1947), NIST Antoine table', NIST.format('C7783064'))],
    'Both Antoine C signs were corrected to match NIST. The legacy ideal solid branch repeats the liquid coefficients and remains unverified.')
add('ch4', 'Methane', {'CH4': 1}, 'CH4', [
    ideal('ch4_ideal', [90.67, 11690, 10.15, 2.1, 10.41, .9], [70, 189.99]),
    antoine('NIST liquid', [3.9895, 443.028, -.49], [90.99, 189.99])],
    [('Prydz and Goodwin (1972), NIST Antoine table', NIST.format('C74828'))],
    'The ideal branches switch at 90.67 K. No uncertainty or validity range for the legacy beta/gamma coefficients has been verified.')
add('so2', 'Sulfur dioxide', {'SO2': 1}, 'SO2', [
    antoine('so2_antoine', [3.48586, 668.225, -72.252], [177.7, 263]),
    antoine('NIST high T (comparison only)', [4.37798, 966.575, -42.071], [263, 414.9])],
    [('Stull (1947), NIST Antoine table', NIST.format('C7446095'))],
    'The registered function is only the low-temperature fit. It does not select the high-temperature branch automatically.')
add('co2', 'Carbon dioxide', {'CO2': 1}, 'CO2', [
    antoine('co2_antoine', [6.81228, 1301.679, -3.494], [154.26, 195.89]),
    curve('old transcription (rejected)', 'antoine', [6.81228, 1301.679, -34.94], [154.26, 195.89], 'incorrect C; comparison only')],
    [('Giauque and Egan (1937), NIST Antoine table', NIST.format('C124389'))],
    'C = -3.494 K replaces the erroneous -34.94 K in both value and derivative. This is the solid sublimation temperature range.')
add('kcl', 'Potassium chloride', {'KCl': 1}, 'KCl', [
    linear('kcl_lodders', 12.611, 11382, [500, 1040], 'verified conversion of Morley Eq. 18'),
    linear('PR108 alternative (unverified)', 30.39, 27077, [500, 1040], 'comparison only; not registered', base=np.e)],
    [('Morley et al. (2012), Eq. 18: log10 p_KCl(bar) = 7.611 - 11382/T', MORLEY),
     ('NIST-JANAF KCl(cr), standard-state thermochemistry', 'https://janaf.nist.gov/tables/Cl-036.html')],
    'Convert bar to Pa by adding 5 before multiplying by ln(10). The old implementation multiplied by log10(e), and its derivative omitted ln(10). No primary source was located for the alternative coefficients. The plot stays below the 1044 K solid melting point; this is not a published fit interval.')
add('nh4sh', 'Ammonium hydrosulfide', {'NH3': 1, 'H2S': 1}, 'NH4SH', [
    linear('nh3_h2s_lewis (legacy atm)', 14.82 + 2*np.log10(101325), 4705, [180, 300], 'legacy atm-squared convention'),
    linear('NASA TM bar convention', 24.82, 4705, [180, 300], 'comparison only; unit ambiguity unresolved')],
    [('Larson et al. (1984), NASA TM 86661, Eqs. 5a–5b, citing Lewis (1969)', 'https://ntrs.nasa.gov/api/citations/19850004528/downloads/19850004528.pdf')],
    'Q = p_NH3 p_H2S. The implementation uses log10 and atm squared. NASA TM gives the same numbers with bar pressures. The original Lewis coefficient table was not verified: preserve the public implementation and expose the 2.67 percent pressure-product difference; do not silently change conventions. The plotted 180–300 K interval is illustrative.')
add('k_hcl', 'Potassium and hydrogen chloride', {'K': 2, 'HCl': 2}, 'KCl', [
    linear('PR108 (unverified)', 65.06, 81230, [500, 1000], 'comparison only; not registered', base=np.e),
    curve('JANAF reconstruction', 'table', [[500,600,700,800,900,1000],
        list((15-2*(np.array([40.632,33.008,27.573,23.505,20.350,17.834])-np.array([-4.481,-3.017,-1.983,-1.215,-.625,-.158])-np.array([10.151,8.530,7.369,6.494,5.812,5.265])))*LN10)], [500,1000], 'tabulated standard-state reconstruction; linear interpolation for display')],
    [('NIST-JANAF KCl(cr), log Kf column', 'https://janaf.nist.gov/tables/Cl-036.html'),
     ('NIST-JANAF K(g), log Kf column', 'https://janaf.nist.gov/tables/K-005.html'),
     ('NIST-JANAF HCl(g), log Kf column', 'https://janaf.nist.gov/tables/Cl-026.html')],
    'For 2K + 2HCl -> 2KCl(s) + H2, log10 Q(Pa) = 15 - 2(logKf_KCl - logKf_K - logKf_HCl), using a 1 bar standard state. H2 has zero formation Gibbs energy. The PR fit differs by orders of magnitude and is not enabled. No replacement regression is installed.', h2=1, cloud_nu=2)
r = -4.56 - np.log10(.84)
for key, metal, cloud, a, b, pr_a, pr_b, base in [
    ('mns','Mn','MnS',11.532,23810,27.58,54823,np.e),
    ('zns','Zn','ZnS',12.812,15873,13.24,15873,10),
    ('na2s','Na','Na2S',8.550,13889,22.48,27778,10)]:
    n = 2 if key == 'na2s' else 1
    curves = [linear('PR108 (unverified)',pr_a,pr_b,[800,1400],'comparison only; not registered',base=base),
              linear('conditional solar reconstruction',n*(a+5)+r,n*b,[800,1400],'derived under fixed solar H2S/H2; not a published Q fit')]
    if key == 'na2s':
        curves[0] = linear('na_h2s_visscher',22.48,27778,[800,1400],'adopted PR108 approximation; exact coefficient provenance unresolved')
        curves.append(linear('superseded legacy fit',32.1,27778,[800,1400],'historical comparison only; not registered'))
    add(key, cloud+' formation', {metal:n,'H2S':1}, cloud, curves,
        [('Morley et al. (2012), Eqs. 9, 12, 15 (metal partial pressures)',MORLEY),
         ('Visscher et al. (2006), Eqs. 16, 25–30 (solar sulfur chemistry)',VISSCHER)],
        'Metal saturation partial pressure is not the full reaction quotient. The comparison reconstructs Q using log10 X_H2S = -4.56 and X_H2 = 0.84 at solar metallicity, giving log10(X_H2S/X_H2) = '+f'{r:.9f}'+'. Its 800–1400 K plotting interval is not a verified validity range. No unconditional coefficient correction follows from this assumption. '+
        ('For Na2S, production now adopts the PR108 intercept 22.48 instead of the legacy 32.1. The conditional reconstruction gives 22.61572: a factor of 1.367 in Q, or 1.169 in sodium saturation pressure at fixed H2S/H2. This supports the approximate correction but does not establish exact coefficient provenance or a validity interval. The Pa quotient has net pressure exponent two: log10 Q(Pa) = log10 Q(bar) + 10; no additional conversion is applied to 22.48. ' if key=='na2s' else '')+
        ('For the PR natural-log Mn fit, the derivative is 54823/T²; multiplying by ln(10) again was an error. This derivative correction does not verify the fitted coefficients.' if key=='mns' else ''),h2=1)
add('mgsio3', 'Magnesium silicate', {'Mg':1,'SiH4':1,'H2O':3}, 'MgSiO3', [
    linear('PR108 (unverified)',9.63,50971,[800,2500],'comparison only; not registered'),
    linear('conditional solar reconstruction',5.86-5*np.log10(.84),49895,[800,2500],'derived above-cloud approximation')],
    [('Visscher et al. (2010), Tables 2 and 3, above-cloud Mg and SiH4',VISSCHER10)],
    'Combining the Mg and SiH4 expressions with three H2O factors cancels the solar water abundance and pressure factors. With X_H2 = 0.84, log10 Q = 5.86 - 5 log10(0.84) - 49895/T. This is a conditional reconstruction, not a directly quoted reaction fit; the tables cover 800–2500 K and metallicities up to [Fe/H] = 0.5. Here the pressure exponent sum is zero, so the bar-to-Pa shift vanishes.',h2=5)


def evaluate(c, t):
    t = np.asarray(t, dtype=float)
    p = c['params']
    if c['kind'] == 'ideal':
        tr, pr, bl, gl, bs, gs = p
        b, g = np.where(t > tr, bl, bs), np.where(t > tr, gl, gs)
        return np.log(pr) + b*(1-tr/t) - g*np.log(t/tr)
    if c['kind'] == 'antoine':
        a,b,d = p
        return np.log(1e5) + LN10*(a-b/(t+d))
    if c['kind'] == 'linear':
        a,b,base = p
        return (a-b/t)*np.log(base)
    return np.interp(t, *p)


def source_fingerprint():
    """Fingerprint native coefficient/solver sources and the validation model."""
    import hashlib
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    paths = sorted([*root.glob('src/thermo/*'), root/'src/species.cpp',
                    root/'src/vapors/vapor_functions.h', root/'src/func_table.cpp',
                    root/'src/func_table.cu', Path(__file__),
                    Path(__file__).with_name('reaction_validation.py')])
    return hashlib.sha256(b''.join(p.read_bytes() for p in paths if p.is_file())).hexdigest()
