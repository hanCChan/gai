# External Source Check

Evidence cutoff and observation date: **2026-10-24**. Metadata and public source
code only. No external signals, models, private credentials, paid access,
author contact, or third-dataset scoring.

## Decision

Third independent multi-RX validation is still **not satisfied**. WiFi_RFFI
cannot be structurally qualified through the currently reachable entry.
MobRFFI has stronger multi-RX/round/date documentation, but its published S3
entry is unavailable and licensing/free processed-data access is unresolved.
Neither candidate is selected or downloaded.

The IoT-J full text was **not obtained**. A repository self-identifying as the
source of the matching paper title was inspected; this is code-level evidence,
not independently confirmed publication provenance or a complete reproduction.

## Evidence Levels

“Implemented” means readable source exists; “documented” means README or code
comments claim the property. Neither means this audit executed the method or
verified signal coverage. HTTP errors are observations about the inspected
entry and this environment, not proof that all legal access routes are absent.

| Candidate | Fixed state | Evidence | RX / dates | Access / license | Decision |
|---|---|---|---|---|---|
| WiFi_RFFI | `fee6f2ea34d60c657fcab30d90c9a91c865bd1b5` | Author README points to IEEE DataPort and a March 2026 TVT paper | Counts/session coverage not verified | Code GPL-3.0; data license not inferred from code license. DataPort proxy tunnel 403 | Pending structure and legal access |
| MobRFFI / own ORBIT datasets | `4057ef3f165d7baaca31089bf3d914579405c2a5` | DatasetAPI comments and capture config | V2/V3: 4 RX; V4: 3 RX. Separate rounds and July/August 2024 filenames | Repository license absent; processed subset/free access unconfirmed. Published bucket returns NoSuchBucket | Promising structure, unavailable entry |
| MobRFFI WiSig adapters | Same commit | Explicit `wisig_old` / `wisig_new` paths | WiSig-derived | Not independent of our WiSig source | Excluded as third independent source |

## WiFi_RFFI

Primary source:
[repository README](https://github.com/eexuanyang/WiFi_RFFI/blob/fee6f2ea34d60c657fcab30d90c9a91c865bd1b5/README.md).

README associates the repository with “Division-based Receiver-agnostic RFF
Identification in WiFi Systems”, TVT early access, March 2026, and points to
the [data entry](https://ieee-dataport.org/documents/wifirffi).
GitHub API/README/tree were reachable (HTTP 200). DataPort was not: the proxy
CONNECT tunnel returned **403 Forbidden**, before the dataset page was read.
Do not describe this as a verified DataPort paywall or permission denial.

[Experiment README](https://github.com/eexuanyang/WiFi_RFFI/blob/fee6f2ea34d60c657fcab30d90c9a91c865bd1b5/Experiments/README.md)
references MATLAB WLAN Toolbox preprocessing and public InceptionTime /
receiver-agnostic model implementations. The repository tree has experiment
and simulation scripts, but no accessible raw-signal inventory establishing
TX/RX/capture coverage, bytes, or dataset license.

## MobRFFI

Primary source:
[DatasetAPI](https://github.com/I-SENSE/mobintel-rffi-paper/blob/4057ef3f165d7baaca31089bf3d914579405c2a5/fingerprinting/dataset_api.py#L15).

The code comments distinguish V1 (one RX), V2/V3 (four RX), and V4 (three RX).
The V2/V4 loader names files from 2024-07-20 and 2024-08-08; separate training
and epoch files are explicit. TX lists vary by coverage, so do not treat any
one list as the verified complete eligible label set.

[Capture README](https://github.com/I-SENSE/mobintel-rffi-paper/blob/4057ef3f165d7baaca31089bf3d914579405c2a5/orbit-capture/README.md)
documents N210r4 receivers, repeated rounds, and simultaneous capture.
[Master config](https://github.com/I-SENSE/mobintel-rffi-paper/blob/4057ef3f165d7baaca31089bf3d914579405c2a5/orbit-capture/master.py#L19)
lists four earlier RX and three active V4 RX.

[Preprocessing README](https://github.com/I-SENSE/mobintel-rffi-paper/blob/4057ef3f165d7baaca31089bf3d914579405c2a5/preprocessor/README.md)
requires licensed MATLAB R2024a OFDM decoding and warns about S3 transfer
costs. This warning does not establish that an anonymous reader would be
charged; free access remains unverified.

[Published S3 config](https://github.com/I-SENSE/mobintel-rffi-paper/blob/4057ef3f165d7baaca31089bf3d914579405c2a5/preprocessor/prepare_orbit.py#L24)
names `mobintel-orbit-dataset`. One anonymous metadata-only list request
(`max-keys=1`, July-19 prefix) returned **HTTP 404, XML NoSuchBucket** from the
published global S3 entry. No signal object was retrieved; no credentials
were supplied. This entry is currently unusable, not proof that the dataset
has been permanently withdrawn or has no alternative location.

Analysis: four RX could support a limited K-estimation/frozen-transfer design,
but not a clean fitting pool of three plus separate selection plus target
within one four-RX collection. Overlapping hardware across dates must not be
counted as extra physical receivers.

## IoT-J Matching-Title Code

Primary source:
[few_shot_open_set.py](https://github.com/iiiwan/A-Novel-Radio-Frequency-Fingerprint-Identification-Scheme-for-Few-Shot-Open-Set-Recognition/blob/cdc303c90cdeaa7fa29619b842ab631308681d68/few_shot_open_set.py).

Repository API describes itself as the source for “A Novel Radio Frequency
Fingerprint Identification Scheme for Few-Shot Open-Set Recognition”.
Inspected commit: `cdc303c90cdeaa7fa29619b842ab631308681d68`.

Readable implementation contains:

- Elementwise support-feature weighting to form the class center.
- A per-dimension sigma-weighted squared-distance sum followed by square root.
- Known-class query CE plus an open-set entropy term.
- Episodic class sampling; example paths refer to LoRa training and moving-NLOS data.

This differs from our source-RX K/Sigma/U plug-in posterior and does not justify
calling H3 a reproduction. The script imports `models.*` files not present in
the inspected repository tree. No top-level paper PDF or complete model tree
was found. No repository install, notebook, or training code was executed.

Legal full-text checks of IEEE document **10960433** and its stamp endpoint
both failed at proxy CONNECT with 403. Crossref title/DOI and arXiv requests
also failed at the proxy. The link between this repository and the IEEE record
has not been independently established from the inaccessible record.

The UAV Gaussian-prototypical DOI remains unverified:
`10.1109/LCOMM.2025.3591973`. Crossref lookup failed at proxy CONNECT; no full
text obtained, no implementation added to the frozen Stage A matrix.

## Method And Limits

GitHub API repository metadata, current branch commit, recursive tree, and
fixed-commit public source were inspected. No GitHub credentials were needed.
All shell public-network calls used the configured Venus proxy. Web search
returned no usable results in this run; claims above rest on the primary
responses identified here. Versions/commits were fixed on 2026-10-24.

Next prerequisite for an independent third-set run is an accessible author
inventory, dataset license, physical-RX identity map, and free minimum subset.
Neither data acquisition nor a new RF baseline should be described as complete.
