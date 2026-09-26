"""effective_cp on CPU must give one answer whatever the heap held before the call.

On CPU, effective_cp solves its (reaction x reaction) gain system with torch.linalg.lstsq.
torch's default CPU driver, gelsy, hands LAPACK a pivot array (JPVT) that it never
initialises (pytorch#187411), and ?gelsy reads JPVT on entry: a nonzero entry pins that
column to the front of the pivoted QR. The column order, and so the answer, then follows
leftover heap bytes. In the state below reaction 0 cannot condense, so its gain row and
column are zero; when that zero column is pinned first, gelsy reports rank 0, returns a
zero solution, and effective_cp silently drops the latent heat of reaction 1.

Each sample runs in a FRESH interpreter (kintera's species registry is process-global)
under a different glibc MALLOC_PERTURB_, which sets the byte every malloc'd block starts
with (255 gives zeros, i.e. free pivoting; any other value pins every column). Every
sample must return bitwise the same effective_cp, and it must carry the latent term.
MALLOC_PERTURB_ is ignored off glibc; the samples then see whatever the heap holds.
CPU only: the CUDA branch of effective_cp uses pinv, not lstsq.
"""

import os
import subprocess
import sys
import textwrap

# Antoine form: logsvp = log(1e5) + (A - B/(T+C)) * log(10). At 300 K, B(l) has svp = 100 Pa
# (supersaturated at x = 0.02, 1 bar) and A(l) has svp = 1e12 Pa (never condenses).
YAML = """
reference-state: {Tref: 0., Pref: 1.e5}
species:
- {name: dry, composition: {N: 1.56, O: 0.42}, cv_R: 2.5}
- {name: A, composition: {H: 2, O: 1}, cv_R: 1.5, u0_R: 0.}
- {name: B, composition: {H: 3, N: 1}, cv_R: 1.5, u0_R: 0.}
- {name: A(l), composition: {H: 2, O: 1}, cv_R: 7.5, u0_R: -6786.66}
- {name: B(l), composition: {H: 3, N: 1}, cv_R: 7.5, u0_R: -5000.00}
reactions:
- {equation: 'A <=> A(l)', type: nucleation, rate-constant: {formula: antoine, A: 10.0, B: 900., C: 0.}}
- {equation: 'B <=> B(l)', type: nucleation, rate-constant: {formula: antoine, A: 0.0, B: 900., C: 0.}}
"""

WORKER = textwrap.dedent("""
    import sys, torch
    from kintera import ThermoOptions, ThermoX
    torch.set_default_dtype(torch.float64)
    th = ThermoX(ThermoOptions.from_yaml(sys.argv[1]))
    temp = torch.tensor([[300.0]])
    pres = torch.tensor([[1.0e5]])
    xfrac = torch.tensor([[[0.96, 0.02, 0.02, 0.0, 0.0]]])
    gain = th.forward(temp, pres, xfrac)
    assert float(gain[..., 0, :].abs().max()) == 0.0, gain
    assert float(gain[..., :, 0].abs().max()) == 0.0, gain
    assert float(gain[..., 1, 1].abs()) > 0.0, gain
    cp = float(th.effective_cp(temp, pres, xfrac, gain))
    frozen = float(th.effective_cp(temp, pres, xfrac, torch.zeros_like(gain)))
    print(cp.hex(), frozen.hex())
""")

# None = the inherited environment; 255 = zeroed blocks; the rest = nonzero fill bytes.
PERTURB = [None, None, 255, 165, 1, 2, 16, 64, 85, 128, 170, 254]


def _sample(card, perturb):
    env = dict(os.environ)
    env.pop("MALLOC_PERTURB_", None)
    if perturb is not None:
        env["MALLOC_PERTURB_"] = str(perturb)
    out = subprocess.run([sys.executable, "-c", WORKER, str(card)], env=env,
                         capture_output=True, text=True, check=True).stdout
    cp, frozen = out.split()[-2:]
    return float.fromhex(cp), float.fromhex(frozen)


def test_effective_cp_does_not_depend_on_heap_contents(tmp_path):
    card = tmp_path / "two_condensates.yaml"
    card.write_text(YAML)
    got = [(p,) + _sample(card, p) for p in PERTURB]
    cps = sorted({cp for _, cp, _ in got})
    table = "\n".join("MALLOC_PERTURB_=%s: cp %r frozen %r" % g for g in got)
    print(table)
    assert len(cps) == 1, "%d distinct effective_cp values:\n%s" % (len(cps), table)
    for p, cp, frozen in got:
        assert cp > frozen, "latent term dropped under MALLOC_PERTURB_=%s:\n%s" % (p, table)
