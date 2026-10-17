# Training on Vultr (hackathon credit), without ever being charged

`python -m lastscene_pipeline.vultr run` rents a server for one training run, publishes the model to the
LastScene API, and destroys the server. Wearables pick the model up with `python -m lastscene_pipeline.fetch`.

## One-time setup (you, in the Vultr console)

1. **Account → API → Enable API**, then **Access Control**: allow only your own IP address (or your
   phone hotspot's). The key controls the whole account, so a leaked key is useless from anywhere else.
2. Copy the key into a file outside the repo, readable only by you:
   ```bash
   mkdir -p ~/.lastscene && printf 'VULTR_API_KEY=%s\n' '<paste key>' > ~/.lastscene/vultr.env && chmod 600 ~/.lastscene/vultr.env
   ```
   The key never leaves your machine: the training server never sees it.
3. Have the pipeline's device credentials in another env file (as for `sync`): `API_URL`, `AUTH0_DOMAIN`,
   `AUTH0_AUDIENCE`, `DEVICE_CLIENT_ID`, `DEVICE_CLIENT_SECRET`, `PAIRING_CODE`.

## Use

```bash
cd vision
python -m lastscene_pipeline.vultr status                      # credit left, expiry, any trainer servers
python -m lastscene_pipeline.vultr run --env-file ~/.lastscene/pipeline.env
python -m lastscene_pipeline.vultr cleanup                     # destroy anything labelled lastscene-trainer
```

`run` does nothing (and rents nothing) when there are no Teach photos or the published model already
matches them. The server trains from the `master` branch on GitHub, so push before running.

## Guardrails (checked before any server exists; one failure means nothing is created)

| Guardrail | Default | Env var |
|---|---|---|
| Stop the day before the credit expires (then the card would be charged) | 2026-11-17 | `VULTR_CREDIT_EXPIRES` |
| Worst case per run (hourly price × time limit, whole hours) | $2.00 | `MAX_RUN_COST` |
| Credit that must remain after that worst case | $20.00 | `MIN_CREDIT_LEFT` |
| Time limit, after which the server is destroyed | 60 min (still one billed hour) | `--max-minutes` |
| One trainer at a time | – | – |

The server is created with backups, IPv6, DDoS protection and activation emails off, and no snapshots,
block storage or reserved IPs. It powers itself off when training ends or fails (and after the time limit
in any case), and the runner **destroys** it as soon as it stops, on the time limit, on Ctrl-C or on any
error. Stopped Vultr servers still bill; only destroying ends billing. If the runner itself is killed,
run `cleanup`.

The device secret reaches the server inside its startup script (Vultr keeps that with the server until
it's destroyed) and is shredded from the disk when training ends.

## How long a run takes

Training defaults to 100 epochs with 40 COCO128 photos mixed in as "no item" backgrounds (see
`lastscene_pipeline/train.py`). On Oct 17, 2026, with 10 Pin photos at 320 px, that took 8.5 min on a
MacBook (M2, 4 CPU threads), including the INT8 export. Vultr's shared vCPUs are slower, so expect roughly
15 to 35 min of training plus 5 to 10 min to boot and install, inside the 60 min limit. More photos take
longer: if a run hits the limit, pass `--max-minutes 120` (two billed hours, still under `MAX_RUN_COST`).

## Cost

The default is the cheapest plan with 4 vCPUs and 8 GB in Atlanta (CPU training at 320 px is enough for
a few hundred photos). `--gpu` uses the cheapest GPU plan instead; the cost guardrails apply either way.
Check `status` and the Vultr console's Billing page after the first run.
