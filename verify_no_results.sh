#!/usr/bin/env bash
# Evidence that no DFT results existed at registration.
# Every output template inside the archived tarballs must be all zeros.
set -e
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
fail=0
for t in dft_inputs/*.tar.gz; do
  tar -xzf "$t" -C "$tmp"
done
for f in $(find "$tmp" -name 'stresses.json' -o -name 'energies.json' \
                       -o -name 'barriers.json'); do
  if python3 -c "
import json,sys
d=json.load(open('$f'))
def flat(x):
    if isinstance(x,list):
        for y in x: yield from flat(y)
    else: yield x
vals=[v for x in d.values() for v in flat(x)]
sys.exit(0 if all(v==0 for v in vals) else 1)"; then
    echo "  ZERO  ${f#$tmp/}"
  else
    echo "  NONZERO (!)  ${f#$tmp/}"; fail=1
  fi
done
[ $fail -eq 0 ] && echo "PASS: all output templates are empty." \
               || { echo "FAIL: an output template contains data."; exit 1; }
