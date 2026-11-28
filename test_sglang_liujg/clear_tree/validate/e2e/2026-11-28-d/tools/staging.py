"""Stage the fixed ClearTree source and inputs without replacing existing fits."""
import hashlib
import importlib.util
import io
import json
import re
import stat
import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath

RUN = Path(__file__).resolve().parents[1]
RAW = RUN / "raw"
SYNC = RAW / "sync"
META = json.loads((RUN / "meta.json").read_text())
REFERENCE = RUN.parent / META["reference"]
WATERCAPTION = RUN.parent / META.get("watercaption_reference", "2026-11-27-b")
PRIOR_TOOLS = RUN.parent / "2026-11-27-b" / "tools"
SUPERVISOR = RUN.parent / "2026-11-28-b" / "tools/supervisor.py"

spec = importlib.util.spec_from_file_location("ct_reuse_supervisor", SUPERVISOR)
prior = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = prior
spec.loader.exec_module(prior)
base, REPO, SKILL, PKG = prior.base, prior.REPO, prior.SKILL, prior.PKG
require, fit_coverage = base.require, prior.fit_coverage
base.ROOT, base.RAW, base.SYNC = RUN, RAW, SYNC
prior.RUN, prior.RAW, prior.SYNC = RUN, RAW, SYNC

__all__ = ["StagingDriver", "RUN", "RAW", "SYNC", "META", "REPO", "SKILL", "PKG",
           "base", "require", "fit_coverage"]

FIT_PATTERN = r"clear_tree_fit.*\.json(?:\.tmp)?\Z"
SOURCE_EXCLUDES = ("**/clear_tree_fit*.json", "**/clear_tree_fit*.json.tmp",
                   "**/__pycache__/**", "**/*.pyc", "**/*.tmp")
MANIFESTS = ("head", "draft", "staged", "npu")

# A stdlib-only helper runs identically in both containers. Source and fit
# inventories are disjoint; only source paths from the verified stage are written.
PACKAGE_TOOL = r'''
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import tempfile
from pathlib import Path

operation, root, fit_pattern, *args = sys.argv[1:]
root = Path(root)
fit_name = re.compile(fit_pattern).fullmatch


def fail(error):
    raise error


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def inventory(directory, fits):
    if not directory.is_dir() or directory.is_symlink():
        raise RuntimeError(f'not a real package directory: {directory}')
    entries = {}
    for parent, dirs, files in os.walk(directory, onerror=fail):
        # Do not recurse into caches, symlinks, or anything named like a fit.
        names = dirs + files
        dirs[:] = [name for name in dirs if name != '__pycache__' and not fit_name(name)
                   and not name.endswith(('.pyc', '.tmp')) and not (Path(parent) / name).is_symlink()]
        for name in names:
            path = Path(parent) / name
            if fits != bool(fit_name(name)):
                continue
            if not fits and (name == '__pycache__' or name.endswith(('.pyc', '.tmp'))):
                continue
            info = path.lstat()
            if not fits and stat.S_ISDIR(info.st_mode):
                continue
            rel = path.relative_to(directory).as_posix()
            if fits:
                record = dict(inode=info.st_ino, device=info.st_dev, mode=info.st_mode,
                              uid=info.st_uid, gid=info.st_gid, size=info.st_size,
                              mtime_ns=info.st_mtime_ns)
                if stat.S_ISREG(info.st_mode):
                    record['sha256'] = digest(path)
                elif stat.S_ISLNK(info.st_mode):
                    record['link'] = os.readlink(path)
                else:
                    raise RuntimeError(f'unsupported fit entry: {path}')
                entries[rel] = record
            elif stat.S_ISREG(info.st_mode):
                entries[rel] = digest(path)
            else:
                raise RuntimeError(f'unsupported source entry: {path}')
    return entries


def manifest(entries):
    return ''.join(f'{entries[name]}  ./{name}\n' for name in sorted(entries))


if operation == 'fits':
    print(json.dumps(inventory(root, True), sort_keys=True))
elif operation == 'manifest':
    sys.stdout.write(manifest(inventory(root, False)))
elif operation == 'install':
    stage, expected, previous, fits = map(Path, args)
    source = inventory(stage, False)
    live = inventory(root, False)
    if manifest(source) != expected.read_text() or inventory(stage, True):
        raise RuntimeError('staged source does not match the fixed manifest or contains fits')
    if manifest(live) != previous.read_text() or inventory(root, True) != json.loads(fits.read_text()):
        raise RuntimeError('Target package changed since backup')
    # Do not silently delete unknown local files. The approved revision is an
    # in-place source update, not a cleanup of unexpected Target files.
    if extra := live.keys() - source.keys():
        raise RuntimeError(f'unexpected Target source files: {sorted(extra)}')
    for rel in source:
        dest = root / rel
        if dest.is_dir():
            raise RuntimeError(f'source file would overwrite a directory: {dest}')
        for parent in dest.parents:
            if parent == root:
                break
            if parent.is_symlink() or (parent.exists() and not parent.is_dir()):
                raise RuntimeError(f'invalid source parent: {parent}')
    for rel in sorted(source):
        src, dest = stage / rel, root / rel
        mode = stat.S_IMODE(src.stat().st_mode)
        if live.get(rel) == source[rel]:
            existing = dest.stat()
            if (existing.st_uid, existing.st_gid, stat.S_IMODE(existing.st_mode)) == (1008, 1008, mode):
                continue
        for parent in reversed(dest.parents):
            if parent != root and root in parent.parents and not parent.exists():
                parent.mkdir(mode=0o755)
                os.chown(parent, 1008, 1008)
        fd, temp = tempfile.mkstemp(prefix='.ct-source-', suffix='.tmp', dir=dest.parent)
        os.close(fd)
        try:
            shutil.copyfile(src, temp)
            os.chmod(temp, mode)
            os.chown(temp, 1008, 1008)
            os.replace(temp, dest)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)
    if inventory(root, False) != source or inventory(root, True) != json.loads(fits.read_text()):
        raise RuntimeError('installed source or preserved fit verification failed')
else:
    raise ValueError(operation)
'''


def checked_head():
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    require(head == META["commit"], "local HEAD differs from the approved commit")


def source_archive():
    archive = subprocess.check_output(
        ["git", "archive", "--format=tar.gz", f"{META['commit']}:{PKG}", "--", ".",
         *(f":(exclude,glob){pattern}" for pattern in SOURCE_EXCLUDES)], cwd=REPO,
    )
    entries = {}
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as handle:
        for member in handle:
            rel = PurePosixPath(member.name)
            require(rel.parts and not rel.is_absolute() and ".." not in rel.parts
                    and not any(re.fullmatch(FIT_PATTERN, part) for part in rel.parts),
                    f"unexpected source archive member: {member.name}")
            if member.isdir():
                continue
            require(member.isfile() and rel not in entries, f"invalid source archive member: {member.name}")
            with handle.extractfile(member) as stream:
                entries[rel] = hashlib.sha256(stream.read()).hexdigest()
    require(entries, "approved source archive is empty")
    lines = "".join(f"{entries[path]}  ./{path}\n" for path in sorted(entries, key=str))
    return archive, lines


class StagingDriver(prior.StagingDriver):
    def __init__(self):
        # The base initializer requires a jump host; this run must never open
        # a new connection. ProxyCommand=false fails closed if mux is lost.
        self.work = base.env("CT_WORK")
        require(re.fullmatch(r"/[A-Za-z0-9_./-]+", self.work)
                and str(PurePosixPath(self.work)) == self.work
                and ".." not in PurePosixPath(self.work).parts,
                "CT_WORK must be a normalized absolute path without spaces")
        self.remote = base.env("CT_REMOTE")
        socket = Path(base.env("CT_SSH_SOCKET"))
        require(stat.S_ISSOCK(socket.stat().st_mode), "authenticated SSH control socket unavailable")
        self.ssh = ["-F", "/dev/null", "-o", f"ControlPath={socket}", "-o", "ControlMaster=no",
                    "-o", "BatchMode=yes", "-o", "ProxyCommand=false", "-o", "IdentityFile=none",
                    "-o", "IdentityAgent=none"]
        self.shared = {key: base.env(f"CT_{key}") for key in ("RPC_PORT", "RPC_ADDR", "HTTP_ADDR")}
        self.cfg = {role: {key: base.env(f"CT_{role.upper()}_{key}") for key in
                           ("CONTAINER", "ROOT", "MODEL", "DEVICE", "HTTP_PORT", "TUNNEL")}
                    for role in base.ROLES}
        self.identity_path = {"draft": str(SKILL / "process_identity.py"),
                              "target": f"{self.work}/process_identity.py"}
        self.precheck_path = {"draft": str(PRIOR_TOOLS / "precheck.sh"),
                              "target": f"{self.work}/precheck.sh"}
        self.datasets = {name: tuple(base.env(f"CT_{name.upper()}_{key}") for key in ("DATA", "IMG"))
                         for name in ("rsteller", "watercaption")}
        self.data, self.images = self.datasets["rsteller"]

    def data_check(self, suffix=""):
        labels = {"rsteller": "RSTeller", "watercaption": "WaterCaption"}
        args = [value for name, label in labels.items() for value in (label, *self.datasets[name])]
        result = self.container("target", "python3", "-B", f"{self.work}/data_identity.py", *args)
        (SYNC / f"data-identity{suffix}.json").write_text(result.stdout)
        observed = json.loads(result.stdout)
        expected = {label: json.loads((reference / "raw/sync/data-identity.json").read_text())[label]
                    for label, reference in (("RSTeller", REFERENCE), ("WaterCaption", WATERCAPTION))}
        for label, identity in expected.items():
            require(observed.get(label) == identity, f"{label} dataset or ordered images differ from reference")
        require(observed.keys() == expected.keys(), "unexpected dataset in identity report")
        return observed

    def setup(self):
        checked_head()
        SYNC.mkdir(parents=True, exist_ok=True)
        files = [RUN / "tools" / name for name in ("server.sh", "run-client.sh")]
        files += [PRIOR_TOOLS / name for name in ("data_identity.py", "precheck.sh")]
        files += [SKILL / name for name in ("process_identity.py", "parse_argv.py", "package_manifest.py")]
        clients = [REPO / "test_sglang_liujg/evaluation" / name for name in (
            "eval_spectre_rsteller.py", "eval_rsteller_metrics.py",
            "eval_spectre_rsteller_serving.py", "eval_spectre_watercaption_serving.py",
        )]
        dependencies = {*files, *clients, *(RUN / "tools").glob("*.py"), SUPERVISOR,
                        Path(base.__file__), REPO / PKG / "clear_tree_prefit.py",
                        REPO / PKG / "clear_tree_width.py", SKILL / "accept_logs.py"}
        hashes = {path: base.sha_bytes(path) for path in sorted(dependencies)}
        (SYNC / "execution-script-hashes.txt").write_text("".join(
            f"{digest}  {path.relative_to(REPO)}\n" for path, digest in hashes.items()))

        self.initial_check("draft")
        self.host("target", ["mkdir", "-m", "700", self.work], label="Target staging directory")
        self.host("target", ["mkdir", f"{self.work}/client"], label="Target client staging directory")
        self.container("target", "mkdir", self.work, f"{self.work}/client")
        self.upload(*files)
        self.upload(*clients, subdir="client")
        for group, dest in ((files, self.work), (clients, f"{self.work}/client")):
            for path in group:
                require(self._digest("target", f"{dest}/{path.name}") == hashes[path],
                        f"Target container staging hash differs: {path.name}")
        for path in files:
            if path.name in {"precheck.sh", "process_identity.py"}:
                fields = self.host("target", ["sha256sum", f"{self.work}/{path.name}"],
                                   label="Target host precheck script hash").stdout.split()
                require(fields and fields[0] == hashes[path], f"Target host staging hash differs: {path.name}")
        self.initial_check("target")

        self.container("draft", "mkdir", self.work)
        for path in files:
            if path.name in {"server.sh", "parse_argv.py"}:
                self.docker("draft", "cp", path, f"{self.cfg['draft']['CONTAINER']}:{self.work}/")
                require(self._digest("draft", f"{self.work}/{path.name}") == hashes[path],
                        f"Draft container staging hash differs: {path.name}")
        self.data_check()
        (SYNC / "setup-complete.txt").write_text("Host prechecks, staged scripts, and both dataset identities verified.\n")

    def _digest(self, role, path):
        fields = self.container(role, "sha256sum", path).stdout.split()
        require(fields and re.fullmatch(r"[0-9a-f]{64}", fields[0]), f"invalid {role} digest: {path}")
        return fields[0]

    def _root(self, role):
        return f"{self.cfg[role]['ROOT']}/{PKG}"

    def _package(self, role, operation, root, *args):
        return self.container(role, "python3", "-B", "-c", PACKAGE_TOOL, operation, root, FIT_PATTERN, *args).stdout

    def _fits(self, role):
        return json.loads(self._package(role, "fits", self._root(role)))

    def _approved_fits(self, fits):
        require({mode: fits.get(name, {}).get("sha256") for mode, name in base.NAMES.items()}
                == META["fit_sha256"], "Draft fits do not match the approved hashes")

    def sync(self):
        require((SYNC / "setup-complete.txt").is_file(), "setup has not completed")
        require(not (SYNC / "sync-complete.txt").exists(), "sync has already completed")
        checked_head()
        for role in base.ROLES:
            self.precheck(role, SYNC / f"pre-sync-{role}.txt")
        archive, head = source_archive()
        manifest_path = SYNC / "head.manifest"
        manifest_path.write_text(head)
        package = SYNC / "clear_tree_remote.tar.gz"
        package.write_bytes(archive)
        draft = self._package("draft", "manifest", self._root("draft"))
        (SYNC / "draft.manifest").write_text(draft)
        require(draft == head, "shared Draft source differs from the approved git archive")
        fits = {role: self._fits(role) for role in base.ROLES}
        self._approved_fits(fits["draft"])
        for role in base.ROLES:
            base.write_json(SYNC / f"{role}-fits-before.json", fits[role])

        previous = SYNC / "npu-before.manifest"
        previous.write_text(self._package("target", "manifest", self._root("target")))
        backup = SYNC / "npu-package-before.tar.gz"
        remote_backup = f"{self.work}/{backup.name}"
        self.container("target", "tar", "-czf", remote_backup, "-C", self._root("target"), ".")
        digest = self._digest("target", remote_backup)
        self.copyout("target", remote_backup, SYNC)
        require(base.sha_bytes(backup) == digest, "full Target backup hash differs after copy")
        (SYNC / f"{backup.name}.sha256").write_text(f"{digest}  {backup.name}\n")

        staged_inputs = (package, manifest_path, previous, SYNC / "target-fits-before.json")
        self.upload(*staged_inputs)
        for path in staged_inputs:
            require(self._digest("target", f"{self.work}/{path.name}") == base.sha_bytes(path),
                    f"Target staged input hash differs: {path.name}")
        self.precheck("target", SYNC / "pre-install-target.txt")
        remote_package = f"{self.work}/{package.name}"
        stage = f"{self.work}/source-stage"
        self.container("target", "mkdir", stage)
        self.container("target", "tar", "-xzf", remote_package, "-C", stage)
        staged = self._package("target", "manifest", stage)
        (SYNC / "staged.manifest").write_text(staged)
        require(staged == head, "unpacked Target stage differs from approved source")
        self._package("target", "install", self._root("target"), stage,
                      f"{self.work}/{manifest_path.name}", f"{self.work}/{previous.name}",
                      f"{self.work}/target-fits-before.json")
        npu = self._package("target", "manifest", self._root("target"))
        (SYNC / "npu.manifest").write_text(npu)
        require(npu == head, "installed Target source differs from approved source")
        for role in base.ROLES:
            after = self._fits(role)
            base.write_json(SYNC / f"{role}-fits-after.json", after)
            require(after == fits[role], f"{role} fit data or metadata changed during source sync")
        (SYNC / "sync-complete.txt").write_text("Approved source manifests match; all existing fits preserved in place.\n")

    def verify_synced(self):
        require((SYNC / "sync-complete.txt").is_file(), "missing successful sync evidence")
        checked_head()
        _, head = source_archive()
        require(all((SYNC / f"{name}.manifest").read_text() == head for name in MANIFESTS),
                "stored source manifests differ from the approved commit")
        for role in base.ROLES:
            require(self._package(role, "manifest", self._root(role)) == head,
                    f"{role} live source differs from the approved commit")
            before = json.loads((SYNC / f"{role}-fits-before.json").read_text())
            current = self._fits(role)
            require(current == before, f"{role} fit data or metadata changed since sync")
            if role == "draft":
                self._approved_fits(current)
