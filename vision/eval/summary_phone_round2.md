# Summary: phone_round2

Memory columns: useful = useful memory before walk-away; false mem = phases (of 5) where a no-keys clip produced a memory; formed = phases with any memory. All at 2 inferences/s, scene matching on.

| preset | ms/inf | 01_placement @0.05 (@0.03) | 02_clutter @0.05 (@0.03) | 02b_clutter @0.05 (@0.03) | 03_no_keys @0.05 (@0.03) | 04_return @0.05 (@0.03) | recall @0.05 by category | retrieval | nav: scene recognized / keys detected (scored frames) | false 'visible' claims | deletion |
|---|---|---|---|---|---|---|---|---|---|---|---|
| efficient | 20 | useful 0/5 (1/5) | useful 0/5 (0/5) | useful 1/5 (1/5) | false mem 5/5 (5/5) | formed 0/5 (2/5) | clutter 3%, hand 18%, near 2%, placed 0%, return 0% | WRONG (hand, th=0.05) | 3 / 1 of 38 | 0 | OK (1 removed) |
| m480d | 15 | useful 2/5 (2/5) | useful 3/5 (5/5) | useful 0/5 (5/5) | false mem 0/5 (0/5) | formed 2/5 (5/5) | clutter 20%, hand 8%, near 15%, placed 9%, return 0% | OK (placed, th=0.05) | 4 / 6 of 38 | 0 | OK (1 removed) |
| m640d | 15 | useful 3/5 (5/5) | useful 2/5 (5/5) | useful 1/5 (5/5) | false mem 0/5 (0/5) | formed 5/5 (5/5) | clutter 20%, hand 6%, near 42%, placed 45%, return 0% | OK (placed, th=0.05) | 0 / 16 of 38 | 0 | OK (1 removed) |
| m960d | 16 | useful 4/5 (4/5) | useful 0/5 (3/5) | useful 0/5 (5/5) | false mem 0/5 (0/5) | formed 4/5 (5/5) | clutter 6%, hand 8%, near 39%, placed 81%, return 69% | OK (placed, th=0.05) | 0 / 15 of 38 | 0 | OK (1 removed) |
| x960d | 29 | useful 5/5 (5/5) | useful 0/5 (2/5) | useful 0/5 (3/5) | false mem 0/5 (1/5) | formed 5/5 (5/5) | clutter 3%, hand 12%, near 89%, placed 100%, return 23% | OK (placed, th=0.05) | 0 / 34 of 38 | 0 | OK (2 removed) |
