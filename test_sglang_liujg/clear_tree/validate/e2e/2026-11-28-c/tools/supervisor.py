"""Adapt the reviewed dual-mode driver without changing the reference record."""
import importlib.util
import json
import re
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

RUN = Path(__file__).resolve().parents[1]
RAW = RUN / "raw"
SYNC = RAW / "sync"
META = json.loads((RUN / "meta.json").read_text())
REFERENCE = RUN.parent / META["reference"]
spec = importlib.util.spec_from_file_location("ct_reference_supervisor", REFERENCE / "tools/supervisor.py")
reference = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = reference
spec.loader.exec_module(reference)

base, REPO, SKILL, PKG = reference.base, reference.REPO, reference.SKILL, reference.PKG
reference.RUN = base.ROOT = RUN
reference.RAW = base.RAW = RAW
reference.SYNC = base.SYNC = SYNC
reference.COMMIT = META["commit"]
require, fit_coverage, load = base.require, reference.fit_coverage, reference.load
PREVIOUS = reference.PREVIOUS  # Historical parsing helpers, not the data baseline.
MANIFESTS = ("head", "draft", "npu")
FIT_NAME = re.compile(r"clear_tree_fit.*\.json(?:\.tmp)?\Z")
NO_FITS = 'shopt -s nullglob; cd "$1" && files=(clear_tree_fit*.json clear_tree_fit*.json.tmp) && (( ${#files[@]} == 0 ))'


class StagingDriver(reference.StagingDriver):
    def data_check(self, suffix=""):
        result = self.container("target", "python3", "-B", f"{self.work}/data_identity.py", self.data, self.images)
        (SYNC / f"data-identity{suffix}.json").write_text(result.stdout)
        expected = REFERENCE / "raw/sync/data-identity.json"
        require(json.loads(result.stdout) == json.loads(expected.read_text()), "RSTeller identity differs from reference")

    def setup(self):
        super().setup()
        # The inherited list covers the staged files; include imported reference code too.
        extra = [REFERENCE / "tools/supervisor.py", REFERENCE / "tools/review.py",
                 reference.PREVIOUS / "tools/review.py"]
        with (SYNC / "execution-script-hashes.txt").open("a") as stream:
            for path in extra:
                stream.write(f"{base.sha_bytes(path)}  {path.relative_to(REPO)}\n")

    def package_manifest(self, role, root, *args):
        path = str(SKILL / "package_manifest.py") if role == "draft" else f"{self.work}/package_manifest.py"
        return self.container(role, "python3", "-B", path, "--root", root, *args).stdout

    def sync(self):
        roots = {role: f"{self.cfg[role]['ROOT']}/{PKG}" for role in base.ROLES}
        for role in base.ROLES:
            self.precheck(role, SYNC / f"pre-sync-{role}.txt")
        before = "npu-package-before.tar.gz"
        before_src = f"{self.work}/{before}"
        self.container("target", "tar", "-czf", before_src, "-C", roots["target"], ".")
        digest = self.container("target", "sha256sum", before_src).stdout.split()[0]
        self.copyout("target", before_src, SYNC)
        require(base.sha_bytes(SYNC / before) == digest, "NPU full-package backup hash differs")
        (SYNC / f"{before}.sha256").write_text(f"{digest}  {before}\n")

        for role in ("target", "draft"):
            self.precheck(role, SYNC / f"pre-archive-{role}.txt")
            name = f"{role}-old-fit"
            dest = f"{self.work}/{name}" if role == "target" else str(SYNC / name)
            script = f"{self.work}/archive-fits.sh" if role == "target" else str(RUN / "tools/archive-fits.sh")
            result = self.container(role, "bash", script, roots[role], dest, check=False)
            (SYNC / f"{name}.sha256").write_text(result.stdout)
            if role == "target":
                self.copyout(role, dest, SYNC)
            require(result.returncode == 0, f"{role} fit archive failed")
            verified = set()
            for entry in result.stdout.splitlines():
                expected, filename = entry.split(maxsplit=1)
                require(FIT_NAME.fullmatch(filename), "fit archive produced an unexpected name")
                if expected != "ABSENT":
                    require(base.sha_bytes(SYNC / name / filename) == expected, f"{role} fit archive hash differs: {filename}")
                    verified.add(filename)
            require(verified == {p.name for p in (SYNC / name).iterdir()}, f"{role} fit archive manifest incomplete")
            self.container(role, "bash", "-c", NO_FITS, "fit-absence", roots[role])

        with tempfile.TemporaryDirectory(prefix="ct-fixed-head-") as temp:
            archive = subprocess.check_output(["git", "archive", META["commit"], PKG], cwd=REPO)
            subprocess.run(["tar", "-xf", "-", "-C", temp], input=archive, check=True)
            head = subprocess.check_output([sys.executable, "-B", str(SKILL / "package_manifest.py"),
                                            "--root", f"{temp}/{PKG}"], text=True)
        (SYNC / "head.manifest").write_text(head)
        package = SYNC / "clear_tree_remote.tar.gz"
        draft = self.package_manifest("draft", roots["draft"], "--tar", str(package))
        (SYNC / "draft.manifest").write_text(draft)
        require(bool(head) and draft == head, "Draft package differs from fixed HEAD")
        with tarfile.open(package) as handle:
            require(not any(FIT_NAME.fullmatch(Path(p).name) for p in handle.getnames()), "source tar contains fit data")
        self.upload(package)
        self.precheck("target", SYNC / "pre-install-target.txt")
        code = '''set -euo pipefail
pkg=$1; work=$2; digest=$3; run_id=$4
printf '%s  %s\\n' "$digest" "$work/clear_tree_remote.tar.gz" | sha256sum -c --status
stage="$pkg.$run_id-new"; old="$pkg.$run_id-old"
test ! -e "$old" && test ! -L "$old"
mkdir "$stage"
tar -xzf "$work/clear_tree_remote.tar.gz" -C "$stage"
chown -R 1008:1008 "$stage"
python3 -B "$work/package_manifest.py" --root "$stage" > "$work/staged.manifest"
diff -q "$work/head.manifest" "$work/staged.manifest"
mv -T "$pkg" "$old"
if ! mv -T "$stage" "$pkg"; then mv -T "$old" "$pkg"; exit 1; fi
if ! python3 -B "$work/package_manifest.py" --root "$pkg" | diff -q "$work/head.manifest" -; then
  mv -T "$pkg" "$stage"; mv -T "$old" "$pkg"; exit 1
fi
# The original complete package is already verified in raw/sync.
mv -T "$old" "$work/npu-before-unpacked"
test "$(stat -c %u:%g "$pkg")" = 1008:1008
[[ -z $(find "$pkg" \( ! -uid 1008 -o ! -gid 1008 \) -print -quit) ]]
'''
        self.upload(SYNC / "head.manifest")
        self.container("target", "bash", "-c", code, "sync-package", roots["target"], self.work,
                       base.sha_bytes(package), META["run_id"])
        remote = self.package_manifest("target", roots["target"])
        (SYNC / "npu.manifest").write_text(remote)
        require(remote == head, "installed Target package differs from fixed HEAD")
        self.container("target", "bash", "-c", NO_FITS, "fit-absence", roots["target"])
        (SYNC / "sync-complete.txt").write_text("Old fits archived; all three package manifests match fixed HEAD.\n")
        print("Old fits archived; Draft and Target source match the fixed commit", flush=True)

    def verify_synced(self):
        require((SYNC / "sync-complete.txt").is_file(), "missing successful sync evidence")
        head = (SYNC / "head.manifest").read_text()
        require(bool(head) and all((SYNC / f"{name}.manifest").read_text() == head for name in MANIFESTS),
                "stored source manifests differ")
        require(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip() == META["commit"],
                "local HEAD changed")
        for role in base.ROLES:
            root = f"{self.cfg[role]['ROOT']}/{PKG}"
            live = self.package_manifest(role, root)
            # The manifest helper excludes Greedy, but includes the RPD JSON.
            # Before either block that file must still be absent, so no entries are filtered here.
            require(live == head, f"{role} live package changed since sync")
