Change in march-pass GPU time against stock, summed over every (demo, view) of the scene and tier; one row per run. In-run spread: the worst (slowest - fastest) / fastest of any arm and view in that run.

| scene | tier | run | demos | stock ms | stock_again | uplight_const | in-run spread |
|---|---|---|---|---|---|---|---|
| city | minimal | run1 | congestus,marine-congestus-coarse | 12.40 | +0.0% | -27.4% | 1340% |
| city | low | run1 | congestus,marine-congestus-coarse | 32.92 | -0.4% | -21.4% | 5% |
| city | medium | run1 | congestus,marine-congestus-coarse | 70.42 | -0.1% | -19.7% | 9% |
| city | high | run1 | congestus,marine-congestus-coarse | 128.73 | +0.3% | -17.2% | 3% |

| scene | tier | run | demos | stock ms | stock_again | no_premarch | no_gradient | ocean_const | ice_const | clearance_ctl | in-run spread |
|---|---|---|---|---|---|---|---|---|---|---|---|
| day | minimal | run1 | congestus,dycoms,rce,stratified,twpice | 12.88 | -0.0% | -25.7% | -27.4% | -11.2% | -0.6% | -0.2% | 15% |
| day | low | run1 | congestus,dycoms,rce,stratified,twpice | 45.70 | +0.1% | -27.9% | -24.1% | -13.0% | -1.0% | -0.1% | 12% |
| day | medium | run1 | congestus,dycoms,rce,stratified,twpice | 197.11 | -0.0% | -19.6% | -23.9% | -17.3% | -1.8% | +0.3% | 15% |
| day | high | run1 | congestus,dycoms,rce,stratified,twpice | 497.99 | +0.1% | -11.5% | -22.7% | -21.3% | -2.0% | +0.6% | 13% |
