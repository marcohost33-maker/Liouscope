from pathlib import Path
import re


def sub_once(path, pattern, repl, *, flags=0, label=None):
    p = Path(path)
    text = p.read_text(encoding='utf-8')
    new, n = re.subn(pattern, repl, text, count=1, flags=flags)
    if n != 1:
        raise SystemExit(f'{path}: expected one replacement for {label or pattern[:60]!r}, got {n}')
    p.write_text(new, encoding='utf-8')


def replace_once(path, old, new, label):
    p = Path(path)
    text = p.read_text(encoding='utf-8')
    n = text.count(old)
    if n != 1:
        raise SystemExit(f'{path}: expected one exact block for {label}, got {n}')
    p.write_text(text.replace(old, new, 1), encoding='utf-8')

# 1) CAR(1): optimise the same stationary exact likelihood that AICc consumes,
# and make theta estimation invariant to residual amplitude.
sub_once(
    'src/liouscope/fitting/car1.py',
    r'def _profile_nll\(.*?\n\n\ndef estimate_car1_theta',
    '''def _profile_nll(theta: float, t: np.ndarray, residuals: np.ndarray) -> float:\n    """Profiled stationary CAR(1) ``-2 log L`` up to a constant.\n\n    This is the SAME likelihood family reported by ``fit_gls_ar1``: the\n    stationary first observation is included, the per-step innovations are\n    scaled by ``sqrt(1-a_k^2)``, and the stationary variance is profiled over\n    all ``n`` whitened residuals.  AIC/AICc comparisons require a maximised\n    likelihood; using a conditional theta estimate and then reporting a\n    stationary likelihood violates that contract.\n\n    Residuals are divided by their max magnitude before squaring.  Profiling\n    the variance makes a global positive scale factor theta-independent, so\n    this changes only floating-point conditioning, not the MLE.\n    """\n    t = np.asarray(t, dtype=float)\n    r = np.asarray(residuals, dtype=float)\n    if r.size != t.size or r.size < 3:\n        return float("inf")\n    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(r))):\n        return float("inf")\n    dt = np.diff(t)\n    if np.any(dt <= 0.0) or not np.isfinite(theta) or theta <= 0.0:\n        return float("inf")\n    scale = float(np.max(np.abs(r)))\n    if not np.isfinite(scale) or scale <= 0.0:\n        return float("inf")\n    z = r / scale\n    a = np.exp(-float(theta) * dt)\n    var = np.maximum(1.0 - a * a, _VAR_FLOOR)\n    w = np.empty_like(z)\n    w[0] = z[0]\n    w[1:] = (z[1:] - a * z[:-1]) / np.sqrt(var)\n    s2 = float(np.mean(w * w))\n    if not np.isfinite(s2) or s2 <= 0.0:\n        return float("inf")\n    return float(r.size * np.log(s2) + np.sum(np.log(var)))\n\n\ndef estimate_car1_theta''',
    flags=re.S,
    label='CAR1 stationary profile likelihood',
)

sub_once(
    'src/liouscope/fitting/car1.py',
    r'def estimate_car1_theta\(t: np\.ndarray, residuals: np\.ndarray\) -> float:\n.*?\n    return theta\n',
    '''def estimate_car1_theta(t: np.ndarray, residuals: np.ndarray) -> float:\n    """Stationary maximum-likelihood CAR(1) relaxation rate ``theta``.\n\n    The objective is :func:`_profile_nll`, the same stationary likelihood\n    family that ``fit_gls_ar1`` reports to AICc.  The first residual therefore\n    participates in both estimation and reporting; no conditional/exact hybrid\n    remains.  The search interval is expressed in the observation grid's own\n    identifiable scales and a coarse log sweep locates the basin before bounded\n    refinement.\n\n    Returns NaN when the grid/residuals cannot identify a rate.\n    """\n    t = np.asarray(t, dtype=float)\n    r = np.asarray(residuals, dtype=float)\n    if t.size < 3 or r.size != t.size:\n        return float("nan")\n    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(r))):\n        return float("nan")\n    dt = np.diff(t)\n    if np.any(dt <= 0.0):\n        return float("nan")\n    scale = float(np.max(np.abs(r)))\n    if not np.isfinite(scale) or scale <= 0.0:\n        return float("nan")\n    z = r / scale\n    centred = z - float(np.mean(z))\n    if float(np.dot(centred, centred)) <= 0.0:\n        return float("nan")\n\n    span = float(t[-1] - t[0])\n    dt_min = float(np.min(dt))\n    if span <= 0.0 or dt_min <= 0.0:\n        return float("nan")\n    lo = float(np.log(_THETA_LO_FRAC / span))\n    hi = float(np.log(_THETA_HI_FRAC / dt_min))\n    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:\n        return float("nan")\n\n    sweep = np.linspace(lo, hi, 241)\n    vals = np.array(\n        [_profile_nll(float(np.exp(u)), t, z) for u in sweep], dtype=float\n    )\n    if not np.any(np.isfinite(vals)):\n        return float("nan")\n    j = int(np.argmin(np.where(np.isfinite(vals), vals, np.inf)))\n    left = float(sweep[max(j - 1, 0)])\n    right = float(sweep[min(j + 1, sweep.size - 1)])\n    if right <= left:\n        theta = float(np.exp(sweep[j]))\n    else:\n        res = minimize_scalar(\n            lambda u: _profile_nll(float(np.exp(u)), t, z),\n            bounds=(left, right),\n            method="bounded",\n            options={"xatol": 1.0e-10},\n        )\n        theta = (\n            float(np.exp(float(res.x))) if res.success else float(np.exp(sweep[j]))\n        )\n\n    if not np.isfinite(theta) or theta <= 0.0:\n        return float("nan")\n    return theta\n''',
    flags=re.S,
    label='CAR1 theta estimator',
)

# 2) GLS residual-family override for estimator-consistent jackknife.
p='src/liouscope/fitting/gls.py'
text=Path(p).read_text(encoding='utf-8')
text=text.replace('from collections.abc import Callable\n', 'from collections.abc import Callable\nfrom typing import Literal\n', 1)
old='''    bounds: tuple[np.ndarray, np.ndarray] | None = None,\n    n_iters: int = 3,\n    max_nfev: int = 2000,\n) -> GLSFitOutput:\n'''
new='''    bounds: tuple[np.ndarray, np.ndarray] | None = None,\n    n_iters: int = 3,\n    max_nfev: int = 2000,\n    residual_family: Literal["auto", "ar1", "car1"] = "auto",\n) -> GLSFitOutput:\n'''
if text.count(old)!=1: raise SystemExit('gls signature guard')
text=text.replace(old,new,1)
old='''    # Which residual model applies is a property of the GRID, decided once.\n    # A uniform grid keeps the historical discrete AR(1) path bit-for-bit; a\n    # grid whose step varies gets the continuous-time CAR(1) model, whose\n    # per-step correlation exp(-theta dt_k) is the only whitening that is\n    # actually valid there (see :mod:`liouscope.fitting.car1`).\n    uniform = is_uniform_grid(t)\n'''
new='''    # Residual-family selection is normally a property of the observation grid.\n    # BCa jackknife is the exception: deleting an interior point from a uniform\n    # grid makes the punctured grid geometrically non-uniform, but it must remain\n    # the SAME estimator as the base fit.  ``residual_family`` therefore allows\n    # that resampling path to pin the base family while ordinary callers retain\n    # the historical auto behaviour.\n    if residual_family not in {"auto", "ar1", "car1"}:\n        raise ValueError(\n            "residual_family must be one of: auto, ar1, car1; "\n            f"got {residual_family!r}"\n        )\n    uniform = is_uniform_grid(t) if residual_family == "auto" else residual_family == "ar1"\n'''
if text.count(old)!=1: raise SystemExit('gls uniform block guard')
text=text.replace(old,new,1)
text,n=re.subn(r'    # OPEN FINDING \(external review, PR #127\), still open on purpose, with the\n.*?    n = whitened\.size\n', '    # CAR(1) theta is fitted against the same stationary likelihood reported\n    # below (PR #127 hardening, 2026-09-12), so AICc consumes a maximised\n    # likelihood rather than a conditional/exact hybrid.\n    n = whitened.size\n', text, count=1, flags=re.S)
if n!=1: raise SystemExit('gls stale likelihood comment guard')
Path(p).write_text(text,encoding='utf-8')

# 3) Jackknife pins the base residual family.
p='src/liouscope/fitting/bootstrap.py'
text=Path(p).read_text(encoding='utf-8')
old='''    bounds: tuple[np.ndarray, np.ndarray] | None,\n) -> np.ndarray:\n'''
new='''    bounds: tuple[np.ndarray, np.ndarray] | None,\n    *,\n    residual_family: str = "auto",\n) -> np.ndarray:\n'''
if text.count(old)!=1: raise SystemExit('jackknife signature guard')
text=text.replace(old,new,1)
old='''        fit_i = fit_gls_ar1(model, t_i, y_i, theta_hat, bounds=bounds)\n'''
new='''        fit_i = fit_gls_ar1(\n            model,\n            t_i,\n            y_i,\n            theta_hat,\n            bounds=bounds,\n            residual_family=residual_family,\n        )\n'''
if text.count(old)!=1: raise SystemExit('jackknife fit guard')
text=text.replace(old,new,1)
Path(p).write_text(text,encoding='utf-8')

# 4) Relaxation: count both CAR1 nuisance parameters; reuse certified spectrum;
# successful-fit metadata only; pin jackknife family; preserve worst-mode identity.
p='src/liouscope/diagnostics/relaxation.py'
text=Path(p).read_text(encoding='utf-8')
old='''    seed: int = 42,\n) -> RelaxationResult:\n'''
new='''    seed: int = 42,\n    spectral_eigenvalues: np.ndarray | None = None,\n) -> RelaxationResult:\n'''
if text.count(old)!=1: raise SystemExit('relax signature guard')
text=text.replace(old,new,1)
old='''    k = int(p0.size) + (1 if np.isfinite(fit.theta_car1) else 0)\n'''
new='''    # On the CAR(1) path both theta AND the stationary Gaussian variance are\n    # estimated from this candidate's residuals and enter its maximised\n    # likelihood.  AIC/AICc parameter counting therefore includes both nuisance\n    # parameters.  The historical uniform-AR(1) convention remains unchanged.\n    k = int(p0.size) + (2 if np.isfinite(fit.theta_car1) else 0)\n'''
if text.count(old)!=1: raise SystemExit('AIC k guard')
text=text.replace(old,new,1)
old='''        # The FAST scale comes from the spectrum, never from a guess: it is\n        # ``max(-Re lambda)``, the same eigenvalues the resolution guard reads.\n        # ``fastest_decay_rate`` returns NaN when no finite positive rate\n        # exists, and ``default_relaxation_grid`` then keeps the uniform window\n        # -- fail-closed, so an unusable spectrum degrades to the historical\n        # behaviour plus its warning rather than to an invented timescale.\n        fast_rate = fastest_decay_rate(L_super)\n'''
new='''        # Reuse the spectrum already certified by the caller when available.\n        # ``diagnose`` passes ``SpectralResult.eigenvalues`` so a repaired Schur\n        # or real-driver solve is not immediately discarded by launching a new\n        # raw ``np.linalg.eigvals`` solve for the fast scale. Direct callers that\n        # do not have a certified spectrum retain the historical fallback.\n        if spectral_eigenvalues is None:\n            fast_rate = fastest_decay_rate(L_super)\n        else:\n            ev = np.asarray(spectral_eigenvalues, dtype=complex)\n            rates = -np.real(ev)\n            positive = rates[np.isfinite(rates) & (rates > 0.0)]\n            fast_rate = float(np.max(positive)) if positive.size else float("nan")\n'''
if text.count(old)!=1: raise SystemExit('fast rate guard')
text=text.replace(old,new,1)
old='''    if grid_residual_model == "ar1":\n        residual_model = "ar1"\n    else:\n        whitened_car1 = [np.isfinite(fr.residual_theta_car1) for fr in fits.values()]\n        if not whitened_car1:\n            residual_model = "car1_unavailable"\n        elif all(whitened_car1):\n            residual_model = "car1"\n        elif any(whitened_car1):\n            residual_model = "car1_mixed"\n        else:\n            residual_model = "car1_fallback_ar1"\n'''
new='''    successful_fits = [fr for fr in fits.values() if fr.success]\n    if not successful_fits:\n        residual_model = "unavailable"\n    elif grid_residual_model == "ar1":\n        residual_model = "ar1"\n    else:\n        whitened_car1 = [\n            np.isfinite(fr.residual_theta_car1) for fr in successful_fits\n        ]\n        if all(whitened_car1):\n            residual_model = "car1"\n        elif any(whitened_car1):\n            residual_model = "car1_mixed"\n        else:\n            residual_model = "car1_fallback_ar1"\n'''
if text.count(old)!=1: raise SystemExit('residual metadata guard')
text=text.replace(old,new,1)
old='''            if t_grid.size <= 60:\n                jk = _jackknife(winner_fn, t_grid, rel_entropy, theta_hat, None)\n'''
new='''            if t_grid.size <= 60:\n                base_family = (\n                    "car1"\n                    if np.isfinite(fits[winner].residual_theta_car1)\n                    else "ar1"\n                )\n                jk = _jackknife(\n                    winner_fn,\n                    t_grid,\n                    rel_entropy,\n                    theta_hat,\n                    None,\n                    residual_family=base_family,\n                )\n'''
if text.count(old)!=1: raise SystemExit('jackknife caller guard')
text=text.replace(old,new,1)
old='''        samples_per_fast_efolding=fast_resolution,\n        residual_model=residual_model,\n'''
new='''        samples_per_fast_efolding=fast_resolution,\n        samples_per_worst_resolved_efolding=fast_resolution,\n        worst_resolved_rate=worst_rate,\n        worst_blind_interval=blind,\n        worst_blind_start=blind_start,\n        residual_model=residual_model,\n'''
if text.count(old)!=1: raise SystemExit('relax return metadata guard')
text=text.replace(old,new,1)
Path(p).write_text(text,encoding='utf-8')

# 5) Top-level diagnose forwards the already certified spectrum.
p='src/liouscope/_diagnostics.py'
text=Path(p).read_text(encoding='utf-8')
old='''        gap=spectral.gap,\n        bootstrap_B=bootstrap_B,\n'''
new='''        gap=spectral.gap,\n        spectral_eigenvalues=spectral.eigenvalues,\n        bootstrap_B=bootstrap_B,\n'''
if text.count(old)!=1: raise SystemExit('diagnose spectral forwarding guard')
text=text.replace(old,new,1)
Path(p).write_text(text,encoding='utf-8')

# 6) Additive relaxation audit fields + correct metric semantics.
p='src/liouscope/_types.py'
text=Path(p).read_text(encoding='utf-8')
old='''    # How finely this grid samples the FASTEST decaying mode, as samples per\n    # e-folding. Below ``relaxation.MIN_SAMPLES_PER_FAST_EFOLD`` that mode was\n    # stepped over rather than measured, so the reported rates describe only\n    # the slow dynamics the window resolves and an\n    # ``UnderResolvedTransientWarning`` is emitted. ``inf`` when nothing decays.\n    samples_per_fast_efolding: float = float("nan")\n'''
new='''    # Historical field name retained for schema compatibility. Since the\n    # multiscale hardening, the value is the MINIMUM sampling resolution over\n    # all decaying modes, not necessarily the fastest mode. New consumers should\n    # read ``samples_per_worst_resolved_efolding`` plus the identity fields below.\n    samples_per_fast_efolding: float = float("nan")\n    samples_per_worst_resolved_efolding: float = float("nan")\n    worst_resolved_rate: float = float("nan")\n    worst_blind_interval: float = float("nan")\n    worst_blind_start: float = float("nan")\n'''
if text.count(old)!=1: raise SystemExit('types resolution fields guard')
text=text.replace(old,new,1)
text=text.replace('''    #   "car1_unavailable"   non-uniform grid and no fit succeeded at all.\n''','''    #   "unavailable"        no fit succeeded at all, regardless of grid.\n''',1)
Path(p).write_text(text,encoding='utf-8')

# 7) E3 stale comment drift in dense + sparse. Production logic already uses coherent scale.
for p in ['src/liouscope/core/lindblad.py','src/liouscope/sparse/build.py']:
    text=Path(p).read_text(encoding='utf-8')
    text,n=re.subn(
        r'        # OPEN QUESTION \(E3, cross-family review requested\): whether a large\n        # PHYSICAL dissipation may excuse a Hermiticity defect of the coherent\n        # part at all\. The answer changes exactly this one expression -- e\.g\.\n        # to ``coherent_scale`` alone -- and nothing else in the gate\.\n        reference = coherent_scale\n',
        '        # E3 resolved 2026-09-12: dissipation is diagnostic only and does not\n        # relax the structural Hermiticity contract of the Hamiltonian input.\n        reference = coherent_scale\n',
        text,
        count=1,
    )
    if n!=1: raise SystemExit(f'{p}: E3 stale comment guard {n}')
    Path(p).write_text(text,encoding='utf-8')

# 8) Zero-mode theorem: approximate trace preservation cannot prove an exact zero
# eigenvalue for arbitrary nonnormal imported operators.
p='src/liouscope/numerics/linalg.py'
text=Path(p).read_text(encoding='utf-8')
pattern=r'def _zero_mode_applicable\(\n.*?\n\n\ndef certified_eigvals'
repl='''def _zero_mode_applicable(\n    tp_defect: float,\n    fro: float,\n    bound: float,\n    dim: int,\n    tp_rtol: float,\n) -> bool:\n    """Whether the exact-zero-mode theorem is applicable to this operator.\n\n    ``vec(I)^H L = 0`` is an exact algebraic premise.  A merely small residual\n    does *not* imply an eigenvalue lies within the same small band for a\n    nonnormal matrix: pseudospectral displacement can be much larger than the\n    perturbation norm.  Therefore an imported approximately trace-preserving\n    operator is not labelled a failed eigensolve when its spectrum correctly\n    lacks an exact zero.\n\n    The robust trace accumulator already distinguishes exact cancellation from\n    floating residual.  Only an exact zero defect activates the theorem.\n    Approximate operators simply proceed without this structural certificate;\n    this can reduce coverage but cannot create a false spectral claim.\n\n    The remaining parameters are retained for API/call-site stability and for\n    the certificate's separate tolerance logic; they do not relax the theorem's\n    premise.\n    """\n    del fro, bound, dim, tp_rtol\n    return bool(np.isfinite(tp_defect) and tp_defect == 0.0)\n\n\ndef certified_eigvals'''
text,n=re.subn(pattern,repl,text,count=1,flags=re.S)
if n!=1: raise SystemExit(f'zero mode applicable guard {n}')
Path(p).write_text(text,encoding='utf-8')

# 9) Focused regression tests.
Path('tests/test_pr127_review_hardening_20260912.py').write_text('''import numpy as np\nimport pytest\n\nfrom liouscope.diagnostics.relaxation import compute_relaxation_layer\nfrom liouscope.fitting.bootstrap import _jackknife\nfrom liouscope.fitting.car1 import estimate_car1_theta\nfrom liouscope.fitting.gls import fit_gls_ar1\nfrom liouscope.fitting.models import M0\nfrom liouscope.numerics.linalg import _zero_mode_applicable\n\n\ndef test_car1_theta_is_residual_scale_invariant_on_irregular_grid():\n    rng = np.random.default_rng(7)\n    t = np.cumsum(np.r_[0.0, np.geomspace(1e-3, 0.5, 39)])\n    theta_true = 0.37\n    a = np.exp(-theta_true * np.diff(t))\n    r = np.empty(t.size)\n    r[0] = rng.normal()\n    for k, ak in enumerate(a):\n        r[k + 1] = ak * r[k] + np.sqrt(1.0 - ak * ak) * rng.normal()\n    ref = estimate_car1_theta(t, r)\n    tiny = estimate_car1_theta(t, r * 1e-170)\n    huge = estimate_car1_theta(t, r * 1e170)\n    assert np.isfinite(ref)\n    assert tiny == pytest.approx(ref, rel=2e-10)\n    assert huge == pytest.approx(ref, rel=2e-10)\n\n\ndef test_jackknife_can_pin_ar1_family_after_interior_deletion(monkeypatch):\n    seen = []\n    import liouscope.fitting.bootstrap as bs\n    real = bs.fit_gls_ar1\n\n    def wrapped(*args, **kwargs):\n        seen.append(kwargs.get("residual_family"))\n        return real(*args, **kwargs)\n\n    monkeypatch.setattr(bs, "fit_gls_ar1", wrapped)\n    t = np.linspace(0.0, 2.0, 12)\n    y = 2.0 * np.exp(-0.7 * t) + 0.02\n    base = fit_gls_ar1(M0, t, y, np.array([2.0, 0.7, 0.02]), residual_family="ar1")\n    assert base.success\n    _jackknife(M0, t, y, base.params, None, residual_family="ar1")\n    assert seen and set(seen) == {"ar1"}\n\n\ndef test_failed_flat_hierarchy_reports_residual_model_unavailable():\n    L = np.zeros((4, 4), dtype=complex)\n    rho = np.eye(2, dtype=complex) / 2\n    with pytest.warns(RuntimeWarning):\n        out = compute_relaxation_layer(\n            L, rho_initial=rho, rho_steady_state=rho,\n            t_grid=np.linspace(0.0, 1.0, 12), bootstrap_B=5\n        )\n    assert out.residual_model == "unavailable"\n\n\ndef test_worst_resolved_identity_is_persisted():\n    L = np.diag([0.0, -1e-6, -1e-3, -1.0]).astype(complex)\n    rho = np.eye(2, dtype=complex) / 2\n    with pytest.warns((RuntimeWarning, UserWarning)):\n        out = compute_relaxation_layer(\n            L, rho_initial=rho, rho_steady_state=rho,\n            t_grid=np.r_[np.linspace(0.0, 10.0, 6), np.linspace(1e4, 1e7, 8)],\n            bootstrap_B=5,\n        )\n    assert out.samples_per_worst_resolved_efolding == out.samples_per_fast_efolding\n    assert np.isfinite(out.worst_resolved_rate)\n    assert np.isfinite(out.worst_blind_interval)\n    assert np.isfinite(out.worst_blind_start)\n\n\ndef test_approximate_tp_does_not_activate_exact_zero_theorem():\n    assert _zero_mode_applicable(0.0, 1.0, 1e-12, 4, 1e-10)\n    assert not _zero_mode_applicable(1e-300, 1.0, 1e-12, 4, 1e-10)\n''', encoding='utf-8')

print('PR127 review hardening patch applied with all guards satisfied')
