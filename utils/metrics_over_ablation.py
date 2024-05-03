

# Read the "outputs.json" file for each ablation experiment and average the results of each metric over all experiments.
import json
import os
from collections import defaultdict
from typing import Dict, List

import numpy as np

relative_path = '../nerfstudio/ablation'

output_data: Dict[str, List[float]] = defaultdict(list)

for i in range(20):
    with open(f'{relative_path}/PROPS-NeRF-ablate-{i}/output.json', 'r') as f:
        json_data = json.load(f)
        for key, value in json_data["results"].items():
            output_data[key].append(value)

val_map = lambda k,v: f"{np.mean(v*2):.5f}" if "std" in k else f"{np.mean(v):.5f}"
output_data = {key: val_map(key, value) for key, value in output_data.items()}



loss_types = ["loss_segmentation", "loss_centermap", "loss_R"]
# iterate over the sets of keys and the values, and find the difference between the predicted and original
for type in loss_types:
    for i in range(10):
        diff = float(output_data[f"predicted_{type}_{i}"]) - float(output_data[f"original_{type}_{i}"])
        output_data[f"diff_{type}_{i}"] = f"{diff:.6f}"
    diff = float(output_data[f"predicted_{type}"]) - float(output_data[f"original_{type}"])
    output_data[f"diff_{type}"] = f"{diff:.6f}"


# # Save the averaged results to a file
with open(f'output.json', 'w') as f:
    json.dump(output_data, f)


# field_name = lambda idx, field: f'PROPS-NeRF-ablate-{idx} - Train Loss Dict/{field}'
