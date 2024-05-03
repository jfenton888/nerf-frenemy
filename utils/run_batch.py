import os
import time
import subprocess


for i in range(10, 20):
    subprocess.run(["cp", "-r", "../datasets/PROPS-NeRF", f"../datasets/PROPS-NeRF-ablate-{i}"], capture_output=True)
    # os.system(f"cp -r ../datasets/PROPS-NeRF ../datasets/PROPS-NeRF-ablate-{i}")
    # time.sleep(30)
    subprocess.run(["python", "../nerfrenemy/utils/dataset_modifier.py", "--all-mask", "--all", "--source", "../datasets/PROPS-NeRF", "--dest", f"../datasets/PROPS-NeRF-ablate-{i}"], capture_output=True)
    # os.system(f"python ../nerfrenemy/utils/dataset_modifier.py --all-mask --all --source ../datasets/PROPS-NeRF --dest ../datasets/PROPS-NeRF-ablate-{i}")
    # time.sleep(30)
    print("Starting training")
    process = subprocess.Popen(["ns-train", "nerfacto", "--data", f"../datasets/PROPS-NeRF-ablate-{i}"])
    # os.system(f"ns-train nerfacto --data ../datasets/PROPS-NeRF-ablate-{i}")
    # time.sleep(60*30)
    print("Entering try")
    try:
        print('Running in process', process.pid)
        process.wait(timeout=60*30)
    except subprocess.TimeoutExpired:
        print('Timed out - killing', process.pid)
        process.kill()

    print("Starting Next Set")