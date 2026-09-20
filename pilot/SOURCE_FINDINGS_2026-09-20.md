# Source investigation: multi-packet candidates found in Geodo

This is a local follow-up to the completed P1 Drive run, not a new Colab run.
No need to rerun notebooks 01/02. No encoder training was started.

## New positive evidence

The first 10,000 physical packets of `Malware/Geodo/Geodo.pcap` produced 2,404
candidate sessions under the documented P1 policy. **100 candidates matched USTC
image rows uniquely within this probe, had multiple packets, and had positive
IAT.** Expected class label and matching image labels agreed. This establishes
a feasible source to investigate, not globally verified sample identities.

The same bounded probe of Gmail yielded 3,585 candidates, two unique image matches,
and zero matched multi-packet candidates. Both scans are partial. The four small
captures from the original P1 were not representative of all USTC classes.

Geodo source SHA-256:
`448fdf2fe393dce570a694bc70e0147c9c3a8877e9abd28ab306c401d6f200a4`.
Gmail source SHA-256:
`6b10fa77c165de6d621842f9becb158d9630b0c913ddf6df2f943a839b142d17`.

## Preprocessing trace

The upstream OpenDetect converter uses Scapy `Raw` payload and zero-pads failures;
the current converter reads transport payload structurally and excludes invalid
packets. A separate legacy serializer diagnostic was run on the SAME packet
indices from P1 (no altered grouping or new label assignment):

| Capture | Legacy matching candidates | Legacy matching multi-packet |
|---|---:|---:|
| Facetime | 2,000 | 0 |
| Tinba (partial) | 92 | 0 |
| Skype | 107 | 0 |
| BitTorrent | 6 | 0 |

Tinba additionally had one candidate with a conflicting label. Do not accept it.
Serializer differences explain some mismatches, but do not by themselves recover
multi-packet matches in those four pilot captures. Upstream `pcap2png.py` assumes
already split flow-PCAPs; it does not establish how those original sessions were made.
The USTC-TK2016 source uses SplitCap default session mode; that is useful provenance
to investigate, not proof OpenDetect used exactly that version/settings.

## Source authenticity and missing material

Git blob hashes of all four local P1 PCAPs match the corresponding files in the
current davidyslu/USTC-TFC2016 GitHub tree. The Malicious_TLS RAR likewise matches
gcx-Yuan/Malicious_TLS and contains only `malicious_TLS.csv`. Redownloading those
same five artifacts does not supply additional source material.

The dataset author's DeepTraffic README points to a ScienceDB record. The link
was identified but this environment could not fetch its page; its actual downloadable
contents/access requirements have NOT been verified. No bulk download was attempted.
No public per-flow Malicious_TLS PCAP bundle or NPZ-row mapping was established in
the checked sources. This is a search result, not proof that none exists elsewhere.

## Next action for temporal fusion

1. Expand USTC pairing from Geodo to additional classes with genuine multi-packet
   source sessions. Complete scans before calling identities globally unique.
2. Validate session/window rules, per-class temporal coverage, label provenance,
   duplicates and split grouping. A single malware class is not an open-set pilot.
3. Prefer a manifest of actual byte-image matches. If images must be regenerated,
   state the paired cohort explicitly and compare image-only and fusion on that
   same cohort. Do not attach unrelated IAT to old image rows.
4. Keep B/C source acquisition separate: request the original timestamped
   `malicious_TLS_4_paper/<class>/<flow>.pcap`, sessionization settings, label map,
   and NPZ row-to-flow/window mapping from the source authors if not found publicly.
   No message/issue has been sent on the user's behalf.

## Reproducible tools

- `python -m pilot.probe_sources --dataset-dir <NPZ_DIR> --capture <GEODO_PCAP> Geodo --packet-limit 10000 --output <NEW_REPORT.json>`
- `python -m pilot.trace_preprocessing --manifest <P1_LOCAL_MANIFEST> --dataset-dir <NPZ_DIR> --raw-root <USTC_ROOT> --output <TRACE.json>`
  The trace resolves exact capture filenames in the supplied manifest; a Drive-ID
  prefix requires matching local filenames, not silently guessed provenance.

## Primary sources checked

- https://github.com/niebikong/Open-Detect/blob/master/data/Preprocessing/utils.py
- https://github.com/niebikong/Open-Detect/blob/master/data/Preprocessing/pcap2png.py
- https://github.com/davidyslu/USTC-TFC2016
- https://github.com/davidyslu/USTC-TK2016/blob/master/1_Pcap2Session.ps1
- https://github.com/gcx-Yuan/Malicious_TLS
- https://github.com/echowei/DeepTraffic
- https://www.scidb.cn/detail?dataSetId=53fb61fb99754abfa71b32c0c88b577a&version=V1 (unverified contents)
