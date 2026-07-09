# LiouScope documentation

**LiouScope** is an open-source diagnostic framework for time-homogeneous
Markovian open quantum systems described by Gorini–Kossakowski–Sudarshan–Lindblad
(GKSL) generators. It quantifies *when and why* the Liouvillian gap fails as a
relaxation-time predictor, replacing a single "decay rate" number with a layered,
auditable `DiagnosticReport`.

:::{note}
This site is organised along the [Diátaxis](https://diataxis.fr/) framework
(tutorials / how-to / reference / explanation). The Reference section is
auto-generated from the package docstrings; the Tutorials, How-to and
Explanation sections are hand-written (issue #72, slices 1–2).
:::

## Where to start

- **New to LiouScope?** Read the {doc}`tutorials/index`.
- **Have a specific task?** See the {doc}`how-to/index`.
- **Need the API?** Jump to the {doc}`reference/index`.
- **Want the "why"?** Read the {doc}`explanation/index`.

For installation, the design philosophy ("no single number", explicit
uncertainty, auditable manifests) and the current citable release gates, see the
project `README.md` and `docs/RELEASE_AUDIT_v0.5.0.md` in the repository.

```{toctree}
:hidden:
:maxdepth: 2

tutorials/index
how-to/index
reference/index
explanation/index
```
