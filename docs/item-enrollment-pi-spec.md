# Personal item enrollment and Raspberry Pi inference

Status: proposed implementation specification, 2026-10-17. No enrollment,
quantized runtime, or personal item recognition is implemented by this document.
**Deferred** until after the SwampHacks demo (keys only, laptop inference). The Pi model is unconfirmed,
despite "confirmed" below.
Companion: [bridge integration](bridge-integration.md).

## Goal and confirmed constraints

Let anyone in a household add everyday belongings, teach the system what each looks
like, and retrieve its most recent supported sighting. Support multiple categories
without hardcoding keys. Target Raspberry Pi Zero 2 W; inference must execute on
the Pi for an on-device claim. Build/export work can execute on a development
machine. Network access remains necessary for existing cloud voice services.

The user has tested keys only. Other categories, personal identity matching,
quantization accuracy, and Pi performance are unverified. Example categories in
this spec are proposed evaluation targets, not supported-product claims.

## Current implementation and corrections to earlier assumptions

- The parent `lastseen` project contains YOLO-World small, medium, and XL weights,
  `memory_engine`, `lastscene_bridge`, clips, and evaluation manifests.
- `YoloWorldDetector` calls `set_classes` with prompts and optional distractors.
  `LastSeen` accepts prompts; the bridge CLI currently constructs it with the
  default keys prompt. The CLI needs an enrollment/configuration input.
- `tool/idea-forge-main` calls hosted language models and supplies no local vision
  weights for this conversion.
- ORB scene matching compares surroundings, explicitly masking the object. It
  cannot establish that a newly detected wallet is the user's enrolled wallet.
- Memories and backend last-seen currently use category labels. A track ID or
  memory ID does not establish persistent physical item identity.
- Saving examples currently does not train or personalize the detector.

Open vocabulary describes the source model's capabilities. A conventional export
with frozen text embeddings can have a fixed vocabulary. Arbitrary prompt changes
after export are not guaranteed. Export inspection and runtime measurements must
settle that question before shipping an enrollment promise.

## User experience

1. Choose **Add item**, enter a display name such as "Adam's blue wallet", and
   select or describe its category. Keep the display name separate from the
   detector prompt, such as "wallet".
2. Capture a short guided sequence or several photos. Ask for multiple sides,
   backgrounds, and lighting conditions. The user confirms the object crop;
   allow a manual crop if automatic detection fails.
3. Reject unusable examples with a concrete explanation: blurred image, object
   cut off, or more than one possible target. Collect distinct views rather than
   many adjacent frames. Start with 8–12 accepted views as a UX hypothesis, not
   an accuracy guarantee.
4. Save the item. Show **Preparing**, **Ready to try**, or **Needs more examples**.
   "Ready to try" means the package is installed, not that accuracy is proven.
5. Run a guided check with fresh views and, where available, a similar other item.
   Show detections for confirmation. Report category recognition separately from
   personal item matching.
6. During use, allow **That's mine**, **Wrong item**, and **Not sure**. Only explicit
   confirmations become positive examples. Rejections attach to the candidate
   item and do not automatically label every other enrolled item as absent.
7. Allow renaming, adding examples, pausing recognition, and deleting enrollment.
   Renaming preserves identity and history. "Found it" resolves a search; it does
   not delete the enrollment.

If recognition is ambiguous, show "Possible wallet sighting" and the evidence.
Do not attach it to a named person's wallet without sufficient identity evidence.
Nearly identical items may remain indistinguishable; offer user confirmation or a
distinctive visual tag as a separate optional approach.

## Recognition architecture

Use two separately evaluated stages:

1. **Category detection:** the quantized detector proposes boxes and categories.
2. **Personal item matching:** a small image encoder compares each candidate crop
   against confirmed reference crops for enrolled items in that category.

Store several reference vectors per item to preserve different views. Compare
similarity with a calibrated acceptance threshold and require separation from
the next-best enrolled item. An explicit unknown outcome is mandatory. Similarity
is not a probability and must not be substituted for detector confidence.

The encoder is a candidate component, not an existing capability. Select it by
measuring instance discrimination, memory, and latency on the Pi. Do not assume
the existing CLIP ViT-B/32 checkpoint is suitable for the Zero 2 W. If no encoder
meets the device budget, release category-only sightings with explicit user
confirmation; personal identity remains unavailable.

If the category detector misses an object, the matching stage cannot recover it.
Manual enrollment crops do not prove automatic detection will work. Unusual
categories may require collecting labeled images and fine-tuning off-device.

## Enrollment versus model training

Initial enrollment adds reference examples and metadata without changing neural
network weights. Compute reference vectors with the same encoder version and
preprocessing used for live candidates. Model updates invalidate old vectors and
must rebuild them from retained approved crops before activation.

Actual fine-tuning is a later, explicit workflow: gather labeled positive and
negative data, split by capture session, train off-device, evaluate, quantize,
and publish a new version. Do not silently train on unconfirmed sightings or
other households' images. User corrections enter a reviewable dataset; they do
not immediately rewrite the running model.

## Vocabulary and quantization workflow

Default proposal: build a versioned vocabulary package off-device. Each package
contains the active category prompts and distractors, frozen text embeddings
where required, and a matching label map. A new personal item in an existing
category updates references only. A new category can require a new detector
export and calibration pass; show that preparation state to the user.

A runtime with externally supplied text embeddings is an alternative only after
its exported inputs, supported operators, and device cost are demonstrated.
Do not place an unmeasured text encoder on the Pi to preserve arbitrary prompts.

Conversion steps:

1. Preserve the source checkpoints and record hashes, library versions, prompts,
   class order, input shape, preprocessing, and postprocessing.
2. Export a floating-point baseline. Preserve distractor suppression and box
   transforms. A class-map mismatch must prevent loading.
3. Calibrate static INT8 using representative frames across active categories,
   backgrounds, lighting, small objects, and empty scenes. Keys-only clips can
   support an exploratory artifact, not multi-category release validation.
4. Compare baseline and INT8 on held-out capture sessions. Keep calibration and
   evaluation frames separate; adjacent frames from the same recording must not
   be split between them. Include unknown items and similar-looking distractors.
5. Compare the existing medium model and the available small model at measured
   input sizes. Smaller images may lose small objects. Do not assume medium at
   960 pixels meets the Pi budget simply because weights fit in storage.
6. Package the successful candidate with metadata and deployment measurements.
   Keep experimental exports clearly marked until the device checks pass.

Quantization shrinks some tensors; activation memory, runtime allocation, camera
buffers, and the rest of the device application also count toward memory use.

## Proposed data and API contracts

These are additions to design and implement, not existing endpoints.

| Record | Required content |
|---|---|
| Enrolled item | UUID `item_id`, household, display name, canonical category, detector prompt, aliases, enrollment version, state, timestamps |
| Reference example | UUID, item ID, image/crop reference, crop box, capture session, explicit confirmation, quality outcome, timestamp |
| Reference vector | example ID, encoder hash/version, preprocessing version, vector |
| Model package | package ID/version, hashes, runtime requirements, vocabulary and class order, input/output specs, thresholds, evaluation status |
| Device state | paired device ID, installed and active package versions, sync state, last error |
| Feedback | sighting ID, proposed item ID, confirmed/rejected/unsure, actor, timestamp |

Proposed authenticated routes:

- `POST /api/v1/enrolled-items`: create a household-scoped draft.
- `POST /api/v1/enrolled-items/{item_id}/examples`: upload a confirmed crop and
  source-image metadata; validate bounds and image limits.
- `POST /api/v1/enrolled-items/{item_id}/prepare`: queue preparation idempotently
  for an enrollment version; return job status, not immediate readiness.
- `GET /api/v1/enrolled-items/{item_id}`: status and examples for authorized users.
- `PATCH /api/v1/enrolled-items/{item_id}`: rename, edit aliases, or pause.
- `DELETE /api/v1/enrolled-items/{item_id}`: delete references and publish a
  deletion tombstone for paired devices.
- `POST /api/v1/sighting-feedback`: record explicit feedback.
- Device package manifest/download/ack routes: restricted to the paired household.

Extend observations with optional `item_id`, `identity_status`,
`identity_score`, `enrollment_version`, and `model_package_version`.
Keep `label` and detector `confidence` compatible with existing consumers.
Validate item ownership against device scope. Unknown or ambiguous matches have
no item ID and must not advance a named item's last-seen record.

Add item-specific lookup and found routes under `/api/v1/enrolled-items/{item_id}`.
Preserve category routes for older clients; do not merge two wallets into one
personal history. Do not backfill historical category sightings with guessed IDs.
Retrieval text must distinguish category sightings from confirmed item matches.

### States, versions, and delayed events

Enrollment states are `draft`, `preparing`, `ready_to_try`,
`needs_more_examples`, `active`, `paused`, and `deleted`. Preparation failure
returns to `needs_more_examples` with an actionable reason, or remains retryable
with a recorded build error. `active` requires the guided validation outcome and
the device's acknowledgment of the current references; it never implies a
population-level accuracy guarantee. Deletion is terminal for that item ID.

Identity outcomes are `category_only`, `unknown`, `ambiguous`, and `matched`.
Only `matched` may carry an item ID into automatic personal last-seen updates.
User-confirmed associations record their separate confirmation provenance.
Pausing prevents new personal memories; resuming requires device acknowledgment.
Rejecting a sighting removes its personal association and rebuilds that item's
last-seen from remaining valid evidence without deleting the category evidence.

Version the reference set independently as immutable `reference_set_version`.
Observations identify detector package, encoder, preprocessing, reference set,
and enrollment versions. Reference-only updates must be atomic and require no
detector rebuild. The device manifest declares all compatible versions together.

The server enforces item tombstones even when a device is offline. Delayed events
for deleted items cannot recreate the item or personal history; return a terminal
item-deleted result so the queue can discard the personal association. Reject
personal associations captured during a pause interval. Earlier valid events may
remain historical but never clear a later found state. Stale reference versions
are accepted only if retained as compatible and not revoked; otherwise retain
category evidence with a rejected-personal-association reason. Upload acknowledgments
must report these per-object decisions to avoid retrying them indefinitely.

Feedback is idempotent by feedback UUID and uses an expected sighting revision.
Conflicting edits return a conflict for review, rather than silently choosing the
last upload. Deletion overrides feedback. Keep an audit of confirmations and
retractions, and exclude retracted examples from subsequent reference sets.

## Device operation and synchronization

- Inference, crop matching, and event selection run locally after package sync.
  Existing upload authentication, stable event IDs, retries, original capture
  times, and two-second upload spacing remain required.
- Stage downloads, verify hashes and compatibility, then activate the detector,
  class map, references, and thresholds as one version. Keep the previous working
  package for rollback. Partial downloads never replace the active package.
- Allow draft enrollment offline; show pending preparation when an off-device
  vocabulary build is needed. Do not claim an unsynced item is recognized.
- Use bounded queues and explicit storage limits. Benchmark with camera, audio,
  matching, and upload work active together. Avoid loading the XL search fallback
  on the Pi automatically.
- Measure actual frame cadence. Revisit temporal memory confirmation thresholds
  when inference runs slower than the current two-inferences-per-second preset.
- Deletion stops future recognition after device acknowledgment. Offline devices
  display pending deletion until they reconnect. Purge deleted references from
  staged and rollback packages as well as the active package. Historical evidence
  follows an explicit user-visible deletion/retention policy.
- Found actions use capture-time ordering so replayed pre-found events cannot
  reopen the item. A genuinely later sighting may reopen it under defined rules.

## Validation and release gates

Report each category and each enrolled item separately. Do not extend the existing
keys results to wallets, glasses, or other categories.

| Area | Required evidence |
|---|---|
| Enrollment | Accepted/rejected crops, rename, correction, pause, deletion, preparation failure and retry |
| Detection | Held-out box precision/recall, missed objects, false sightings on empty scenes, baseline versus INT8 |
| Identity | Correct item matches, wrong-person/item matches, unknown rejection, similar-item confusion |
| Memory | Correct item ID and evidence coordinates, capture time, duplicate handling, found/offline ordering |
| Device | Model size, peak resident memory, cold start, median/p95 inference and end-to-end latency, sustained thermal behavior |
| Integration | Real Pi capture through authenticated upload to household dashboard and item-specific retrieval |
| Isolation | Cross-household item IDs, examples, packages, and feedback cannot be read or written |

Provisional engineering gates: no out-of-memory failure or swap dependence during
a 30-minute combined workload; p95 processing time fits the selected capture
interval; no named-item assignments for the curated unknown/confusable negative
set. These are proposed gates, not measured results or population guarantees.
Record sample counts and failures. Choose precision/recall and quantization-loss
targets before evaluating a held-out release set, based on the baseline and the
cost of wrong memories. Do not lower thresholds after inspecting that release set.

## Implementation sequence

1. Establish Pi baseline and a reproducible multi-category export/calibration
   pipeline. Deliver measured artifacts and a report; identity is still unknown.
2. Add enrollment storage, crop confirmation, household authorization, and item
   IDs throughout observations and retrieval.
3. Evaluate and implement reference matching with unknown rejection. Deliver
   per-item results before enabling named-item automatic memories.
4. Add atomic package sync, offline state, rollback, deletion propagation, and
   configurable bridge prompts. Integrate the Pi camera and existing uploader.
5. Run the held-out recognition and full device acceptance checks. Publish the
   supported categories and limitations in the UI and handoff documentation.

Open decisions to resolve through measurements: runtime/export format, encoder,
input resolution, frame cadence, maximum active categories and enrolled items,
reference count/storage budget, and supported recognition thresholds. The Pi
target is confirmed; these performance choices are not yet established.

## Technical references

- [Ultralytics YOLO-World](https://docs.ultralytics.com/models/yolo-world/): custom
  vocabulary and export behavior.
- [ONNX Runtime quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html):
  static calibration and quantization accuracy considerations.
