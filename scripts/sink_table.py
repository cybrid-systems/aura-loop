"""Markdown table of a `python3 -m aura_loop --sink --out report.json` report."""
import json
import sys

for path in sys.argv[1:]:
    r = json.load(open(path))
    print(f"\n{path}: rounds={r['rounds']}, baseline threshold {r['baseline_threshold']:g}, learner {r['learner']}, "
          f"sink {r['runner'].get('sink_head', '')}\n")
    print("| arm | round | KEEP | ROLLBACK | SKIP | ABORT | kept speedup sum | kept geomean | reward |")
    print("|---|---|---|---|---|---|---|---|---|")
    for arm, ms in r["per_round"].items():
        for m in ms:
            print(f"| {arm} | {m['round']} | {m['keep']} | {m['rollback']} | {m['skip']} | {m['abort']} | "
                  f"{m['kept_speedup_sum']} | {m['kept_speedup_geomean']} | {m['reward']} |")
    print()
    for arm, t in r["totals"].items():
        a = r["totals_after_round1"][arm]
        print(f"- {arm}: total KEEP {t['keep']}, ROLLBACK {t['rollback']}, kept speedup sum {t['kept_speedup_sum']}, "
              f"reward {t['reward']}; rounds 2..{r['rounds']}: KEEP {a['keep']}, ROLLBACK {a['rollback']}, "
              f"kept speedup sum {a['kept_speedup_sum']}")
