# Security Policy

`chronohorn` is alpha-stage research software. There is no production support
contract, no embargo policy, and no guarantee of timely fixes — but real
security issues will be taken seriously.

## Supported Versions

Only the latest release on PyPI is supported. Older versions will not receive
patches.

| Version | Supported |
|---|---|
| latest on PyPI | ✅ |
| any older | ❌ |

## Reporting a Vulnerability

If you find a security issue:

1. **Do not** open a public issue describing the vulnerability or how to
   exploit it.
2. Use GitHub's [private vulnerability reporting](https://github.com/asuramaya/chronohorn/security/advisories/new)
   form, or email the maintainer address listed on the GitHub profile.
3. Include enough detail to reproduce — versions, environment, and a minimal
   PoC if possible.

You should expect an acknowledgement within a few days. Beyond that, fix
timing depends on severity and on whether you (the reporter) need an embargo
window before the fix lands.

## Scope

In scope:

- the published `chronohorn` Python package
- the runtime daemon (`chronohorn runtime`, dashboard, MCP server)
- the rust crates published from this workspace

Out of scope:

- third-party dependencies (report upstream)
- experiment results, manifests, or user-supplied configuration files
- GPU driver or operating-system issues exposed by the runtime
