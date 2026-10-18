# Training on Vultr (hackathon credit), without ever being charged

`python -m lastscene_pipeline.vultr run` rents a server for one training run, publishes the model to the
LastScene API, and destroys the server. Wearables pick the model up with `python -m lastscene_pipeline.fetch`.

## One-time setup (you, in the Vultr console)

1. **Account → API → Enable API**, then **Access Control**: allow only your own IPv4 address (or your
   phone hotspot's; `curl -4 https://ifconfig.me` shows it). The key controls the whole account, so a
   leaked key is useless from anywhere else.
   Vultr checks the address each request actually comes from, and on a network with IPv6 that can be an
   IPv6 address even though your IPv4 one is allowed (seen Oct 17 on a Verizon hotspot: `401 Unauthorized IP
   address: 2600:1006:...`). So the runner talks to Vultr over IPv4 only. On a network without IPv4, set
   `VULTR_IPV4_ONLY=0` in the env file and allow the IPv6 address instead. If Vultr still refuses, the error
   names the address it saw: add that one. A hotspot's address can change when it reconnects.
2. Put the key in your env file (`vision/.env`, from `.env.example`; gitignored) as `VULTR_API_KEY=...`, next
   to the pipeline's device credentials (as for `sync`): `API_URL`, `AUTH0_DOMAIN`, `AUTH0_AUDIENCE`,
   `DEVICE_CLIENT_ID`, `DEVICE_CLIENT_SECRET`, `PAIRING_CODE`. Keep it readable only by you (`chmod 600 .env`).
   The key never leaves your machine: the training server only gets the device credentials.
   (An older `~/.lastscene/vultr.env` holding only the key still works.)

## Use

```bash
cd vision
python -m lastscene_pipeline.vultr status --env-file .env       # credit left, expiry, any trainer servers
python -m lastscene_pipeline.vultr run --env-file .env
python -m lastscene_pipeline.vultr cleanup --env-file .env      # destroy anything labelled lastscene-trainer
```

`--skip NAME` (repeatable, any case) leaves that Teach item's photos out of training, e.g.
`run --env-file .env --skip pin`. The item keeps its class id; the model just doesn't learn it. The
dashboard still reports the model as up to date, because the photos on the server didn't change.

`run` does nothing (and rents nothing) when there are no Teach photos or the published model already
matches them. The server trains from the `master` branch on GitHub, so push before running.

## Guardrails (checked before any server exists; one failure means nothing is created)

| Guardrail | Default | Env var |
|---|---|---|
| Stop the day before the credit expires (then the card would be charged) | 2026-11-17 | `VULTR_CREDIT_EXPIRES` |
| Worst case per run (hourly price × time limit, whole hours) | $2.00 | `MAX_RUN_COST` |
| Credit that must remain after that worst case | $20.00 | `MIN_CREDIT_LEFT` |
| Time limit, after which the server is destroyed | sized from the photo count, whole billed hours (1 h for 10 photos, about 20 h for 50 items x 30 photos) | `--max-minutes` |
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
15 to 35 min of training plus 5 to 10 min to boot and install. More photos take
longer, so the time limit is estimated from the photo count (`vultr.time_limit`: 15 min setup + 0.3 s per
photo per epoch, plus half again, rounded up to whole billed hours). At $0.055/h the largest Teach set
(50 items x 30 photos, about 20 h) still costs about $1.10, under `MAX_RUN_COST`. `--max-minutes` overrides it.

## Cost

The default is the cheapest plan with 4 vCPUs and 8 GB in Atlanta (CPU training at 320 px is enough for
a few hundred photos). `--gpu` uses the cheapest GPU plan instead; the cost guardrails apply either way.
Check `status` and the Vultr console's Billing page after the first run.
