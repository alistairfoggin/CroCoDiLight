import os

import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluations import whdr
import json

root_dir = "./data/iiw/data/"  # replace with your IIW directory path
filenames = [file for file in os.listdir(root_dir) if file.endswith(".png") and not "_ref" in file]

# split according to the paper https://ieeexplore.ieee.org/document/7298915
filenames = sorted(filenames, key=lambda x: int(x.split(".")[0]))
filenames = filenames[::5] # take every 5th image starting from the first

whdr_measurements = {}
for filename in filenames:
    path = os.path.join(root_dir, filename)
    reflectance = whdr.load_image(filename=path.replace(".png", "_ref.png"), is_srgb=True)
    judgements = json.load(open(path.replace(".png", ".json")))
    img_whdr = whdr.compute_whdr(reflectance + 0.5, judgements, 0.1)
    print(f"WHDR: {img_whdr:.6f}, Filename: {filename}")
    whdr_measurements[filename] = img_whdr

with open("whdr_results_05.json", "w") as f:
    json.dump(whdr_measurements, f, indent=4)
average_whdr = np.mean(np.array(list(whdr_measurements.values())))
print(f"Average WHDR: {average_whdr:.6f}")
