import os
import time
import subprocess

model_times = {0:  "2024-04-26_233604", 
               1:  "2024-04-27_000623", 
               2:  "2024-04-27_003641", 
               3:  "2024-04-27_010659", 
               4:  "2024-04-27_013718", 
               5:  "2024-04-27_020736", 
               6:  "2024-04-27_023756", 
               7:  "2024-04-27_030816", 
               8:  "2024-04-27_033836", 
               9:  "2024-04-27_040857", 
               10: "2024-04-27_122555", 
               11: "2024-04-27_125614",
               12: "2024-04-27_132632",
               13: "2024-04-27_135652",
               14: "2024-04-27_142711",
               15: "2024-04-27_145730",
               16: "2024-04-27_152750",
               17: "2024-04-27_155810",
               18: "2024-04-27_162829",
               19: "2024-04-27_165849",
              }

for i in range(6, 20):
    process = subprocess.Popen(["ns-train", "nerfrenemy", "--data", f"../datasets/PROPS-NeRF-ablate-{i}", "--pipeline.model.nerf-path", f"outputs/PROPS-NeRF-ablate-{i}/nerfacto/{model_times[i]}/nerfstudio_models/", 
                                "--steps-per-save", "250", "--max-num-iterations", "1000", "--save-only-latest-checkpoint", "False", "--vis", "wandb"])
    try:
        print('Running in process', process.pid)
        process.wait(timeout=60 * 45)
    except subprocess.TimeoutExpired:
        print('Timed out - killing', process.pid)
        process.kill()

    print("Starting Next Set")