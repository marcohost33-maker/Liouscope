"""Block C: die 16 Befunde aus Issue #118 empirisch am Stand von PR #107 pruefen.

Lauf:  .venv/Scripts/python.exe .verify/2026-08-25/pruefskript_issue118_16_befunde.py
Voraussetzung: ausgecheckter Stand origin/claude/liouscope-repo-analysis-xgztfd (4a8ae9c).

Nur die Befunde mit einer im Review mitgelieferten Reproduktion sind hier
ausfuehrbar; die uebrigen sind per Quelltext-Stelle belegt (siehe Bericht).
"""
import dataclasses
import warnings

import numpy as np


def _rel():
    from liouscope._types import RelaxationResult
    kw = {f.name: 0.0 for f in dataclasses.fields(RelaxationResult)
          if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING}
    kw.update(fits={}, aicc_model="M0", beta_D=float("nan"),
              bca_ci_beta=(float("nan"), float("nan")))
    for k in list(kw):
        if "curve" in k:
            kw[k] = np.zeros(3)
        if k == "entanglement_asymmetry":
            kw[k] = None
    return RelaxationResult(**kw)


def b2_f5_ausrichtung():
    import liouscope.diagnostics.classification as C
    rel = _rel()
    ev = {"pseudospectral_radius": 10.0, "henrici_eta": 2.0}   # kein "gap"
    leiter = {r[0]: r[3] for r in C._hypothesis_ladder(ev, relaxation=rel)}
    matrix = {e["rule_id"]: e["status"] for e in C.hypothesis_evidence_matrix(ev, relaxation=rel)}
    k = "F5_PSEUDOSPECTRAL"
    ok = (matrix[k] == "SUPPORTED") == bool(leiter.get(k))
    return ok, f"Matrix={matrix[k]} Leiter={leiter.get(k)}"


def b3_nan_pflichtevidenz():
    import liouscope.diagnostics.classification as C
    rel = _rel()
    st = []
    for ev in ({"kreiss": 1e6, "petermann_max": float("nan"), "henrici_eta": 0.1, "gap": 1.0},
               {"kreiss": 1e6, "henrici_eta": 0.1, "gap": 1.0}):
        ent = {e["rule_id"]: e["status"] for e in C.hypothesis_evidence_matrix(ev, relaxation=rel)}
        st.append((ent["F1_OVERLAP_AMPLIFICATION"], ent["A12_FALLBACK"]))
    ok = st[0] == st[1] == ("UNEVALUABLE", "UNEVALUABLE")
    return ok, f"NaN={st[0]} fehlend={st[1]}"


def b4_unveraenderliche_abbildung():
    import types
    import liouscope
    o = liouscope.RESERVED_A_CLASSES
    if not isinstance(o, types.MappingProxyType):
        return False, f"typ={type(o).__name__}"
    try:
        o.pop("A6")
    except AttributeError:
        return True, "mappingproxy, pop -> AttributeError"
    return False, "mutierbar"


def b5_alias_synchron():
    from liouscope._types import ClassificationResult
    kw = {f.name: {"dominant_class": "A1", "tier": "EXPLORATION"}.get(f.name, ())
          for f in dataclasses.fields(ClassificationResult)
          if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING}
    try:
        r = ClassificationResult(**{**kw, "confidence": 0.7, "support_score": 0.2})
    except ValueError as e:
        return True, f"abgelehnt: {str(e)[:60]}"
    return r.confidence == r.support_score, f"c={r.confidence} s={r.support_score}"


def b6_m3a_saat():
    from liouscope.fitting.gls import fit_gls_ar1
    from liouscope.fitting.models import M3a, initial_guess_m3a
    verh = []
    for c in (1.0, 1e6):
        t = np.linspace(0, 10 / c, 80)
        y = (1 + 0.2 * c * t) * np.exp(-0.5 * c * t)
        r = fit_gls_ar1(M3a, t, y, initial_guess_m3a(t, y))
        verh.append(np.asarray(r.params) / np.array([1, 0.2 * c, 0.5 * c]))
    ok = all(np.allclose(v, 1.0, rtol=1e-3) for v in verh)
    return ok, f"c=1 -> {verh[0]}, c=1e6 -> {verh[1]}"


def b8_nan_eingabe():
    from liouscope.fitting.gls import fit_gls_ar1
    from liouscope.fitting.models import M0
    t = np.array([0.0, 1.0, np.nan, 3.0])
    y = np.exp(-np.array([0.0, 1.0, 2.0, 3.0]))
    try:
        r = fit_gls_ar1(M0, t, y, np.array([1.0, 1.0]))
    except ValueError as e:
        return True, f"abgelehnt: {str(e)[:60]}"
    return False, f"angenommen, success={r.success}"


def b9_negative_zerfallsrate():
    from liouscope.fitting.gls import fit_gls_ar1
    from liouscope.fitting.models import M0
    t = np.linspace(0, 1e10, 64)
    y = np.exp(-5 * t / 1e10)
    r = fit_gls_ar1(M0, t, y, np.array([1.0, -1.0]))
    p = np.asarray(r.params)
    res = float(np.linalg.norm(M0(t, p) - y))
    ok = not (r.success and p[1] < 0)
    return ok, f"success={r.success} params={p} residuum={res:.3e}"


def b10_konstruktor_reihenfolge():
    from liouscope._types import ClassificationResult
    kw_only = ClassificationResult.__dataclass_params__.kw_only
    return kw_only, f"kw_only={kw_only} (positional gar nicht mehr moeglich)"


def b15_subsplit_zerfallsmode():
    from liouscope import build_liouvillian
    from liouscope.core.lindblad import steady_state
    from liouscope.diagnostics.spectral import compute_spectral_layer
    H = np.diag([0.0, 1.0]).astype(complex)
    lower = np.array([[0, 1], [0, 0]], dtype=complex)
    sz = np.diag([1.0, -1.0]).astype(complex)
    L = build_liouvillian(H, [lower, sz], [1e-15, 1e-14])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = compute_spectral_layer(L, steady_state(L, allow_degenerate=True))
    ok = abs(s.gap - 1e-15) < 5e-16
    return ok, f"D1={s.gap:.3e} (wahr 1.0e-15), Zertifikat={s.zero_mode_certificate}"


def b16_zertifikat_aendert_verdikt():
    import pathlib
    import re
    txt = pathlib.Path("CHANGELOG.md").read_text(encoding="utf-8")
    behauptet_report_only = "report-only, additive field" in txt
    src = pathlib.Path("src/liouscope/diagnostics/classification.py").read_text(encoding="utf-8")
    liest_zertifikat = bool(re.search(r"certificate = getattr\(spectral", src))
    ok = not (behauptet_report_only and liest_zertifikat)
    return ok, f"CHANGELOG sagt report-only={behauptet_report_only}, Klassifikator liest es={liest_zertifikat}"


PRUEFUNGEN = [
    ("[2]  F5-Pflichtschluessel ausgerichtet", b2_f5_ausrichtung),
    ("[3]  NaN-Pflichtevidenz = unevaluierbar", b3_nan_pflichtevidenz),
    ("[4]  unveraenderliche Reserved-Abbildung", b4_unveraenderliche_abbildung),
    ("[5]  Alias-Synchronitaet", b5_alias_synchron),
    ("[6]  M3a-Saat gitterrelativ", b6_m3a_saat),
    ("[8]  NaN-Eingabe abgelehnt", b8_nan_eingabe),
    ("[9]  negative Zerfallsrate beschraenkt", b9_negative_zerfallsrate),
    ("[10] Konstruktor-Reihenfolge", b10_konstruktor_reihenfolge),
    ("[15] Sub-Split-Zerfallsmode nicht als Null", b15_subsplit_zerfallsmode),
    ("[16] Zertifikat-Wirkung offengelegt", b16_zertifikat_aendert_verdikt),
]

if __name__ == "__main__":
    offen = 0
    for name, fn in PRUEFUNGEN:
        try:
            ok, detail = fn()
        except Exception as exc:  # noqa: BLE001 - Diagnose-Skript
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        offen += 0 if ok else 1
        status = "BEHOBEN" if ok else "OFFEN  "
        print(f"{status}  {name:<44} {detail}")
    print(f"\n{offen} von {len(PRUEFUNGEN)} ausfuehrbaren Befunden weiterhin offen.")
