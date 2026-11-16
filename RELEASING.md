# Release policy

`scadactl` uses Semantic Versioning. CLI output intended for machine
consumption, configuration schemas, persisted state, and documented environment
variables are public compatibility surfaces.

## Release requirements

A release commit must have:

- an updated version in `pyproject.toml`, `scadactl/__init__.py`, and
  `debian/changelog`;
- an updated `CHANGELOG.md` with the release date;
- passing Python, shell, package-build, and installation smoke tests;
- a source archive, Python wheel, and Debian binary package built from the same
  clean commit;
- verified Apache-2.0 and third-party notice content for every distributed file;
- no credentials, internal-only references, private infrastructure details, or
  non-public Git history; and
- maintainer approval for security, compatibility, dependency, and packaging
  impact.

## Release procedure

1. Start from a clean checkout of the release commit.
2. Run:

   ```sh
   python3 -m pytest -q
   bash -n scripts/manage-scada-domains
   shellcheck scripts/manage-scada-domains
   python3 -m build
   dpkg-buildpackage -us -uc -b
   ```

3. Install the wheel in an isolated environment and run `scadactl --version`
   and `scadactl --help`.
4. Install the Debian package on a supported Linux system and verify the paths
   in `packaging/README.md` plus `manage-scada-domains --help`.
5. Review the source archive and Git history for the release gate above.
6. Create an annotated `vMAJOR.MINOR.PATCH` tag on the approved commit.
7. Publish the source archive, wheel, Debian package, checksums, changelog entry,
   and any required third-party notices together.

Release artifacts are immutable. A correction is published as a new version.
