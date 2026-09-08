# Owner-A relay — activation required

This is not activated by updating the Colab files. Owner A must deploy/authorize
the web app. No OAuth tokens are sent to workers. Worker secrets are narrowly
scoped bearer capabilities, not verified Google email identities. Only assign
trusted collaborators; never share the Apps Script project itself with workers.

## One-time setup as A

1. Stop ALL existing A/B writers to the eight scenario outputs. Existing
   mount-based notebooks bypass relay leases. Confirm checkpoint cloud durability.
2. Open https://script.google.com as the owner of THESIS IMPLEMENTASI. New project
   named OpenDetect Owner Relay. Paste `Code.gs` from this directory.
3. Project settings → show appsscript.json; replace with the supplied manifest.
4. Script properties: `ACK_PRODUCERS_STOPPED` = `yes`.
5. Run `initializeOwner` in the editor. Approve Drive and external requests as A.
   It creates `_relay` subfolders, imports a manifest of existing files without
   altering them, and registers the six original NPZ files. Duplicate names,
   missing hashes and B-owned bootstrap files cause a stop.
6. Deploy → New deployment → Web app → **Execute as: Me (A)**.
   Access must allow the worker HTTP client (Anyone); application-level random
   secrets protect every data/control request. The public GET reveals readiness
   only. If organization policy prohibits this, STOP: do not relax policy or use
   another account; arrange approved hosting. Copy the `/exec` URL.
7. Set script properties `PENDING_WORKER=B-mn`, `PENDING_SCENARIOS=C-1`.
   Run `issueWorker`. Copy the one-time key from the PRIVATE owner execution log
   to that worker's Colab Secret `OPENDETECT_WORKER_KEY`. Set another Colab Secret
   `OPENDETECT_RELAY_URL` to the deployment URL. Enable notebook access.
8. Repeat step 7 for B-mc/B-2 etc. Give A its own key with required scenarios
   (comma-separated). This supports multiple workers, NOT unlimited platform GPU
   sessions. Don't reuse a key across people; keep keys out of source/notebook cells.

## Required acceptance test before real training

On CPU, from each worker notebook run only setup/hello first. Confirm its assigned
scenario. An unassigned scenario and a second session claim on the same scenario
must be rejected. Then test one real epoch, confirm `CLOUD COMMITTED TO A`, stop
cleanly, and start the SAME scenario as A. Confirm restore + actual model/RNG resume
before processing the remaining epochs. GPU availability is still required for
this final test; a successful API hello is not a resume test.

For an abandoned lease: STOP the old producer, set PENDING_SCENARIOS, run
`releaseLostWorker` as A. The old lease is fenced: it cannot publish after release.
Never release a live producer merely to force a second same-scenario worker.
To revoke a compromised key, set PENDING_WORKER and run revokeWorker as A.
The worker label is not a verified email; possession of the private key grants
its scenario capabilities. Keep existing Drive collaborators trusted as well:
immutable means the relay never overwrites objects, not that a Drive Editor
cannot modify them outside the relay. Restore verifies the recorded hashes.

## Storage and costs/limits

All new file objects and manifests are A-owned under
`outputs/<scenario>/_relay/{objects,snapshots}`. A single verified manifest pointer
selects the generation; a partial upload does not replace it. Source-code hashes,
config and data remain unchanged. Latest notebooks restore this layout; old mount
notebooks/recovery-v4 see only the preserved pre-relay state and must not be used
to resume new relay work. Do not delete pre-relay checkpoints during rollout.

No automatic deletion is implemented. Immutable versions consume A storage
(roughly checkpoint size × uploaded epochs, plus other changed files). Stop if A
quota is low. Snapshot history retention must be reviewed before long runs; this
implementation favors recoverability over space efficiency. One snapshot is
limited to 300 files/30 GiB and an individual upload to 2 GiB.

Payload upload uses a scoped resumable upload URI initiated by A; owner OAuth token
never leaves Apps Script. Downloads use bounded 8 MiB authenticated relay ranges.
Apps Script/Drive quotas and network costs still apply. Current Google docs list
20,000 daily URL Fetch calls for consumer accounts and 6 minutes per execution;
many parallel workers can exhaust quotas. Errors stop the trainer, not fall back
to B's Drive. Control endpoints use a global lock briefly; GPU training is parallel.

Sources: https://developers.google.com/apps-script/guides/web
https://developers.google.com/apps-script/guides/services/quotas
https://developers.google.com/workspace/drive/api/guides/manage-uploads

## Deployment status

Code generation/local tests are not evidence of live A authorization, deployment,
large-file round trips or GPU/RNG restoration. Keep activation fail-closed until
these steps are completed. Do not print credentials or upload session URLs.
