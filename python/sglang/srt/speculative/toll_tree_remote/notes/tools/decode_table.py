#!/usr/bin/env python3
"""Target-only step-time table from scheduler 'Decode batch' log lines.

Each cell's log is the slice of the Target log written while that cell ran.

  python3 decode_table.py --cell b=4,l=4096,rep=1:cells/b4_l4096_r1.log [...]

Only lines whose #running-req equals the cell's b are used, so warm-up,
ramp-down and prefill dips are excluded. The graph flag is reported so eager
steps can be spotted. Step time is b / gen_throughput.
"""
import argparse
import re
import statistics

LINE = re.compile(
    r"Decode batch.*?#running-req:\s*(\d+).*?#token:\s*(\d+)"
    r"(?:.*?graph:\s*(True|False))?.*?gen throughput \(token/s\):\s*([\d.]+)"
)


def parse(path, b):
    thr, tokens, graph = [], [], []
    with open(path, errors="replace") as f:
        for line in f:
            m = LINE.search(line)
            if not m or int(m.group(1)) != b:
                continue
            thr.append(float(m.group(4)))
            tokens.append(int(m.group(2)))
            if m.group(3) is not None:
                graph.append(m.group(3) == "True")
    return thr, tokens, graph


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cell", action="append", required=True,
                   help="b=<int>,l=<label>,rep=<int>:<log slice path>")
    args = p.parse_args()
    print(f"{'b':>3} {'prefix':>7} {'rep':>3} {'lines':>5} {'tok/s med':>10} {'CV':>7}"
          f" {'step ms':>8} {'mean #token':>11} {'graph':>6}")
    for spec in args.cell:
        head, path = spec.split(":", 1)
        kv = dict(x.split("=") for x in head.split(","))
        b = int(kv["b"])
        thr, tokens, graph = parse(path, b)
        if len(thr) < 3:
            print(f"{b:>3} {kv['l']:>7} {kv.get('rep', '-'):>3} {len(thr):>5}  too few lines")
            continue
        med = statistics.median(thr)
        cv = statistics.stdev(thr) / statistics.fmean(thr)
        g = f"{sum(graph)}/{len(graph)}" if graph else "n/a"
        print(f"{b:>3} {kv['l']:>7} {kv.get('rep', '-'):>3} {len(thr):>5} {med:>10.2f} {cv:>7.2%}"
              f" {1000 * b / med:>8.2f} {statistics.fmean(tokens):>11.0f} {g:>6}")


if __name__ == "__main__":
    main()
