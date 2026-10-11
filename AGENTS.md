# AUVWorkPlace engineering rules

Work in this repository and preserve existing experiments. Do not reset user
changes, force push, overwrite saved result directories, or create a separate
replacement project.

Keep the original `PythonVehicleSimulator/`, REMUS hydrodynamic coefficients,
actuators and controller parameters intact. Add external adapters or explicit
experimental configurations. Separate P25 kinematic execution from original
REMUS six-DOF execution in code, figures and claims.

Do not calculate SHA256, MD5, file/model/configuration digests, Git object
checksums, or add checksum-based identity/binding checks. Ordinary Git version
management is permitted. Validate actual parameters, shapes, times, finite
values and behavior instead.

Keep all existing useful tests. Do not weaken collision geometry or safety
criteria to improve research results. Distinguish test pass, arrival, actual
collision/clearance, unsafe emergency forecasts and finite-horizon acceptance.
Record failures honestly. CA remains the stage2 REMUS obstacle predictor until
an independently tested integration changes it. Do not claim ECP, PPLNM6DOF,
electrical energy savings or real-time execution when those are absent.

The owner has requested that quality-checked development stages be committed
and pushed to `Whsjbwy/AUVWorkPlace` using research/feature branches, with a PR
to `main` when available. Use `research/stage2-baseline` for the current stage.
Confirm the remote, inspect all changes and explicitly stage reviewed files.
Run original PVS, stage1 and stage2 tests plus relevant actual simulations and
dependency checks first. Fix blocking program/test errors before publishing;
documented research performance failures may remain on a research branch and
must be disclosed in the PR. Never automatically merge, reset or overwrite
`main`. Later direct user instructions override this workflow.

Exclude environments, credentials, large archives, training states and batch
raw results from Git. Preserve them locally in separate `results_stage*`
directories and independent review ZIPs. Keep necessary small inference
fixtures, source, configs, tests, protocols and concise evidence. Never print
authentication credentials. See `docs_stage2/VERSION_CONTROL.md` for the full
workflow and `.github/workflows/tests.yml` for Windows/Linux CPU CI.
