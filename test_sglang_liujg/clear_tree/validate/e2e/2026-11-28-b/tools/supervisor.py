"""Staging and schema-5 fit checks for the dual-mode RSTeller run."""
import importlib.util
import json
import re
import shlex
import stat
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

RUN = Path(__file__).resolve().parents[1]
RAW = RUN / "raw"
SYNC = RAW / "sync"
PREVIOUS = RUN.parent / "2026-11-27-c"
REPO = RUN.parents[4]
SKILL = REPO / ".cursor/skills/clear-tree-e2e-validation/scripts"
PKG = "python/sglang/srt/speculative/clear_tree_remote"
COMMIT = json.loads((RUN / "meta.json").read_text())["commit"]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Keep the reference run's process identity, wait, and cleanup logic.
base = load("ct_supervisor", PREVIOUS / "tools/drive.py")
base.ROOT, base.RAW, base.SYNC = RUN, RAW, SYNC
require = base.require
prefit = load("ct_prefit", REPO / PKG / "clear_tree_prefit.py")


def fit_coverage(fit, mode, captures):
    header, saved = fit["header"], fit["plan"]
    require(fit["schema"] == header["schema"] == 5, "not a schema 5 fit")
    require(header["draft"].get("lookahead") is True, "Draft lookahead not installed")
    target = header["target"]
    require(target.get("verify_mode") == "rpd" if mode == "rpd" else "verify_mode" not in target,
            "wrong mode fit identity")
    expected = dict(batches=list(base.BATCHES), ks=list(base.TIERS), widths=list(base.TIERS),
                    buckets=list(base.BUCKETS), max_depth=8)
    require(saved["inputs"] == expected and target["kv_buckets"] == expected["buckets"],
            "wrong fit plan inputs")
    plan = prefit.build_plan(prefit.PlanInputs(**expected))
    require(saved["fingerprint"] == plan.fingerprint, "wrong fit plan fingerprint")
    h_required = {tuple(key) for key in plan.required_h}
    layer_required = {(w, "graph", b) for b, w in captures if w in base.TIERS}
    coverage = fit["coverage"]
    require({tuple(k) for k in coverage["required_h"]} == h_required and
            {tuple(k) for k in coverage["required_layers"]} == layer_required,
            "persisted fit coverage differs from plan or runtime captures")
    hs, layers = {}, {}
    for row in fit["h"]:
        key = (row["b"], row["k"], row["bucket"])
        require(key not in hs and len(row["samples_ms"]) == len(row["ells"]), "invalid H row")
        hs[key] = len(row["ells"])
    for row in fit["layers"]:
        key = (row["w"], row["path"], row["g"])
        require(key not in layers and all(type(row[k]) is int and row[k] >= 0 for k in ("n", "dropped")),
                "invalid layer sample or spike count")
        layers[key] = row
    require(all(hs.get(k, 0) >= prefit.PREFIT_CELL_SAMPLES for k in h_required), "H coverage incomplete")
    require(all(layers.get(k, {}).get("n", 0) >= prefit.PREFIT_LAYER_SAMPLES for k in layer_required),
            "post-filter layer coverage incomplete")
    budgets = {str(b): sorted({(g["rounds"], g["max_new"]) for g in plan.groups
                              if g["kind"] == "w" and g["b"] == b}) for b in base.BATCHES}
    return dict(schema=5, fingerprint=plan.fingerprint, run=fit["run"],
                required_h=len(h_required), min_h_samples=min(hs[k] for k in h_required),
                required_layers=len(layer_required), min_layer_samples=min(layers[k]["n"] for k in layer_required),
                layers=[dict(cell=k, n=v["n"], dropped=v["dropped"]) for k, v in sorted(layers.items())],
                skipped=coverage["skipped"], width_plan_rounds_and_max_new_by_batch=budgets)


class StagingDriver(base.Driver):
    def __init__(self):
        super().__init__()
        socket = Path(base.env("CT_SSH_SOCKET"))
        require(stat.S_ISSOCK(socket.stat().st_mode), "authenticated SSH control socket unavailable")
        self.ssh = ["-o", f"ControlPath={socket}", "-o", "BatchMode=yes", *self.ssh]
        self.precheck_path["draft"] = str(RUN / "tools/precheck.sh")

    def upload(self, *sources, subdir=""):
        dest = f"{self.work}/{subdir}"
        self.run(["scp", *self.ssh, *map(str, sources), f"{self.remote}:{dest}"],
                 label="Target staging copy", capture=False)
        for source in sources:
            self.docker("target", "cp", f"{dest}/{Path(source).name}",
                        f"{self.cfg['target']['CONTAINER']}:{dest}")

    def initial_check(self, role):
        result = self.host(role, ["bash", self.precheck_path[role], role, self.cfg[role]["TUNNEL"],
                                 self.identity_path[role]], label=f"{role} precheck", check=False)
        path = SYNC / f"precheck-{role}.txt"
        path.write_text(result.stdout)
        require(result.returncode == 0 and f"{role}_ports_and_service_precheck OK" in result.stdout,
                f"{role} idle/tunnel precheck did not pass")

    def data_check(self, suffix=""):
        result = self.container("target", "python3", "-B", f"{self.work}/data_identity.py", self.data, self.images)
        (SYNC / f"data-identity{suffix}.json").write_text(result.stdout)
        expected = PREVIOUS / "raw/sync/data-identity.json"
        require(json.loads(result.stdout) == json.loads(expected.read_text()), "RSTeller identity differs from reference")

    def setup(self):
        observed = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        require(observed == COMMIT, "local HEAD changed")
        require(not subprocess.check_output(["git", "diff", "HEAD", "--", PKG], cwd=REPO),
                "tracked ClearTree package differs from fixed HEAD")
        SYNC.mkdir(parents=True, exist_ok=True)
        files = [RUN / "tools" / name for name in ("precheck.sh", "archive-fits.sh", "server.sh", "run-client.sh", "data_identity.py")]
        files += [SKILL / name for name in ("process_identity.py", "package_manifest.py", "parse_argv.py")]
        self.host("target", ["mkdir", "-m", "700", self.work], label="Target staging directory")
        for role in base.ROLES:
            self.container(role, "mkdir", self.work, f"{self.work}/client")
        self.host("target", ["mkdir", f"{self.work}/client"], label="Target client staging directory")
        self.upload(*files)
        for name in ("server.sh", "parse_argv.py"):
            source = next(path for path in files if path.name == name)
            self.docker("draft", "cp", source, f"{self.cfg['draft']['CONTAINER']}:{self.work}/")
        clients = [REPO / "test_sglang_liujg/evaluation" / f"eval_spectre_rsteller{x}.py" for x in ("", "_serving")]
        self.upload(*clients, subdir="client")
        (SYNC / "execution-script-hashes.txt").write_text("".join(
            f"{base.sha_bytes(path)}  {path.relative_to(REPO)}\n" for path in sorted(set([
                *files, *clients, *sorted((RUN / 'tools').glob('*.py')),
                PREVIOUS / 'tools/drive.py', SKILL / 'accept_logs.py',
            ]))))
        for role in base.ROLES:
            self.initial_check(role)
        self.data_check()
        print("Both hosts idle; tunnel baselines and RSTeller identity match", flush=True)

    def sync(self):
        draft = self.cfg["draft"]
        target = self.cfg["target"]
        roots = {role: f"{self.cfg[role]['ROOT']}/{PKG}" for role in base.ROLES}
        for role in base.ROLES:
            self.precheck(role, SYNC / f"pre-sync-{role}.txt")
        before = "npu-package-before.tar.gz"
        before_src = f"{self.work}/{before}"
        self.container("target", "tar", "-czf", before_src, "-C", roots["target"], ".")
        digest = self.container("target", "sha256sum", before_src).stdout.split()[0]
        self.copyout("target", before_src, SYNC)
        require(base.sha_bytes(SYNC / before) == digest, "NPU backup bytes do not match")
        (SYNC / f"{before}.sha256").write_text(f"{digest}  {before}\n")

        for role in ("target", "draft"):
            self.precheck(role, SYNC / f"pre-archive-{role}.txt")
            archived = f"{role}-old-fit"
            dest = f"{self.work}/{archived}" if role == "target" else str(SYNC / archived)
            source = f"{self.work}/archive-fits.sh" if role == "target" else str(RUN / "tools/archive-fits.sh")
            result = self.container(role, "bash", source, roots[role], dest, check=False)
            (SYNC / f"{archived}.sha256").write_text(result.stdout)
            if role == "target":
                self.copyout(role, dest, SYNC)
            require(result.returncode == 0, f"{role} archive command failed")
            for entry in result.stdout.splitlines():
                check, name = entry.split(maxsplit=1)
                if check != "ABSENT":
                    require(base.sha_bytes(SYNC / archived / name) == check, f"{role} archive hash mismatch: {name}")

        manifest = SKILL / "package_manifest.py"
        with tempfile.TemporaryDirectory(prefix="ct-fixed-head-") as path:
            archive = subprocess.check_output(["git", "archive", COMMIT, PKG], cwd=REPO)
            subprocess.run(["tar", "-xf", "-", "-C", path], input=archive, check=True)
            head = subprocess.check_output([sys.executable, "-B", str(manifest), "--root", f"{path}/{PKG}"], text=True)
        (SYNC / "head.manifest").write_text(head)
        package = SYNC / "clear_tree_remote.tar.gz"
        live = self.container("draft", "python3", "-B", str(manifest), "--root", roots["draft"], "--tar", str(package)).stdout
        (SYNC / "draft.manifest").write_text(live)
        require(live == head, "Draft package differs from fixed HEAD")
        with tarfile.open(package) as handle:
            require(not any(re.fullmatch(r"clear_tree_fit.*\.json", Path(name).name) for name in handle.getnames()),
                    "package tar contains fit JSON")
        self.upload(package)
        self.precheck("target", SYNC / "pre-install-target.txt")
        code = '''set -euo pipefail
pkg=$1; work=$2; digest=$3
printf '%s  %s\\n' "$digest" "$work/clear_tree_remote.tar.gz" | sha256sum -c --status
stage="$pkg.${work##*/}-new"; old="$work/npu-before-unpacked"
test ! -e "$old"
mkdir "$stage"
tar -xzf "$work/clear_tree_remote.tar.gz" -C "$stage"
chown -R 1008:1008 "$stage"
mv -T "$pkg" "$old"
if ! mv -T "$stage" "$pkg"; then mv -T "$old" "$pkg"; exit 1; fi
'''
        self.container("target", "bash", "-c", code, "sync-package", roots["target"], self.work,
                       base.sha_bytes(package))
        remote = self.container("target", "python3", "-B", f"{self.work}/package_manifest.py", "--root", roots["target"]).stdout
        (SYNC / "npu.manifest").write_text(remote)
        require(remote == head, "NPU package differs from fixed HEAD")
        print("Old fits archived; Draft and NPU package manifests match fixed HEAD", flush=True)

    def client(self, mode, trial, concurrency, ordinal):
        out = RAW / trial
        out.mkdir()
        dest = f"{self.work}/{trial}"
        self.container("target", "mkdir", dest)
        args = ["env", f"CT_HTTP_ADDR={self.shared['HTTP_ADDR']}",
                f"CT_HTTP_PORT={self.cfg['target']['HTTP_PORT']}", f"CT_CLIENT_DIR={self.work}/client",
                "bash", f"{self.work}/run-client.sh", trial, "rsteller", str(concurrency), self.data, self.images, dest]
        print(f"formal {trial} {mode} concurrency={concurrency} started", flush=True)
        result = self.container("target", "bash", "-c", f"{shlex.join(args)} > {shlex.quote(dest + '/client.stdout')} 2>&1",
                                capture=False, check=False)
        (out / "exit-code.txt").write_text(f"{result.returncode}\n")
        self.copyout("target", dest, RAW)
        paths = list(out.glob("client.*.jsonl"))
        require(result.returncode == 0 and len(paths) == 1, f"formal {trial} failed or JSONL missing")
        rows = [json.loads(line) for line in paths[0].read_text().splitlines() if line.strip()]
        require(len(rows) == 100 and all(not row.get("error") for row in rows), f"formal {trial}: not 100/100")
        require(all(row.get("spec_fallback_ct") == 0 for row in rows), f"formal {trial}: fallback or missing counter")
        require(paths[0].with_suffix(".csv").stat().st_size and paths[0].with_suffix(".run.json").stat().st_size,
                f"formal {trial}: missing CSV or manifest")
        flush = self.logpart("target", mode, "Cache flushed successfully|POST /flush_cache")
        (out / "target-flush.txt").write_text(flush)
        require(re.findall(r'"POST /flush_cache HTTP/\S+" (\d+)', flush) == ["200"] * ordinal and
                all(len(re.findall(rf"\bTP{rank}\].*Cache flushed successfully", flush)) == ordinal for rank in (0, 1)),
                f"formal {trial}: flush not confirmed by HTTP 200 and both ranks")
        print(f"formal {trial} complete (100/100, no fallback, flush confirmed)", flush=True)
