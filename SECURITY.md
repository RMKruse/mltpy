# Security Policy

## Supported Versions

mltpy is currently pre-1.0 and follows a rolling release model: only the
latest published version on PyPI receives security fixes. Patches are not
back-ported to earlier releases.

| Version | Supported          |
| ------- | ------------------ |
| 0.4.x   | :white_check_mark: |
| < 0.4   | :x:                |

We recommend always running the most recent release. The installed version
is available via:

```python
import mltpy
print(mltpy.__version__)
```

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub
issues, pull requests, or discussions.**

Report suspected vulnerabilities privately through either of the following
channels:

1. **GitHub Security Advisories** (preferred) — open a private report at
   <https://github.com/RMKruse/mltpy/security/advisories/new>.
2. **Email** — contact the maintainer at **kontakt.ddl@proton.me**
   with the subject line `[mltpy security]`.

To help us triage quickly, please include where you can:

- A description of the issue and the potential impact.
- The mltpy version, Python version, and OS where it was observed.
- Steps to reproduce, ideally a minimal code sample or input that triggers
  the behaviour.
- Any relevant stack traces or logs.

### What to expect

mltpy is maintained on a best-effort basis. We will:

- Acknowledge your report as soon as we reasonably can.
- Provide an initial assessment and severity classification once the issue
  has been triaged.
- Keep you updated on remediation progress for confirmed issues.
- Credit you in the release notes and advisory once a fix is published,
  unless you prefer to remain anonymous.

We ask that you give us a reasonable opportunity to address the issue before
any public disclosure (coordinated disclosure).

## Scope and Threat Model

mltpy is a numerical/statistical library for fitting Conditional
Transformation Models. It performs in-process numerical computation on
NumPy arrays and has no network, filesystem-write, or subprocess
functionality at runtime. Its only hard dependencies are `numpy` and
`scipy`.

Reports that fall within scope include, for example:

- Memory-safety or code-execution issues reachable through mltpy's public
  API with well-formed inputs.
- Vulnerabilities introduced by mltpy itself (not its dependencies) that
  could compromise the host process.

The following are generally **out of scope**:

- Crashes, exceptions, `RuntimeError` convergence failures, or non-finite
  results caused by malformed, adversarial, or numerically degenerate input
  arrays. mltpy validates shapes and mathematical preconditions but does not
  treat its input arrays as a security boundary — do not feed untrusted data
  to a fitting routine and rely on it as a sandbox.
- Loading model objects from untrusted sources. mltpy model objects may be
  serialised with `pickle`; **never unpickle data you do not trust**, as
  `pickle` can execute arbitrary code by design. This is a property of
  `pickle`, not a mltpy vulnerability.
- Vulnerabilities in `numpy`, `scipy`, `matplotlib`, `pandas`, or other
  third-party packages — please report those to the respective projects.
  We will, however, update our version constraints in response to upstream
  advisories.
- Denial of service from intentionally oversized problems (e.g. extremely
  high Bernstein order or very large designs) that exhaust memory or CPU.

If you are unsure whether an issue is in scope, report it privately and we
will help determine its classification.
