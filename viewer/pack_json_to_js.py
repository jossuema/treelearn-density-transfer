"""Turns the exported JSON into .js files that the viewer can load over file://

Browsers block fetch over file://, but they do allow <script src>. So we wrap each
JSON in an assignment to window and save it as .js.

Use: python3 pack_json_to_js.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

PIECES = [("geo.json", "geo.js", "GEO"),
          ("trees.json", "trees.js", "TREES"),
          ("clouds.json", "clouds.js", "CLOUDS"),
          ("crowns.json", "crowns.js", "CROWNS"),
          ("instances.json", "instances.js", "INSTANCES")]


def main():
    total = 0
    for in_name, output, var in PIECES:
        pe = os.path.join(DATA, in_name)
        ps = os.path.join(DATA, output)
        if not os.path.exists(pe):
            # Without the JSON we leave an empty object so the viewer does not blow up.
            with open(ps, "w") as f:
                f.write(f"window.{var} = {{}};\n")
            print(f"  {in_name:14s} does not exist, {output} is left empty")
            continue
        with open(pe) as f:
            d = json.load(f)
        with open(ps, "w") as f:
            f.write(f"window.{var} = ")
            json.dump(d, f, separators=(",", ":"), ensure_ascii=False)
            f.write(";\n")
        n = os.path.getsize(ps)
        total += n
        print(f"  {in_name:14s} -> {output:12s} {n/1e6:7.2f} MB  {len(d)} plots")
    print(f"total {total/1e6:.2f} MB")
    print(f"open {os.path.join(HERE, 'index.html')} in the browser")


if __name__ == "__main__":
    main()
