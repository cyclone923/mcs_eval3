#!/usr/bin/env python3

import os
import subprocess

dir_path = os.path.dirname(os.path.realpath(__file__))
cwd = os.getcwd()
assert dir_path == cwd, "must be launched from PWD"

grav_dir = "gravity_scenes"

# create gravity dataset dir if it doesn't already exist
if not os.path.isdir("gravity_scenes"):
    subprocess.run(["mkdir", grav_dir])

os.chdir(grav_dir)

for i in range(1, 13):
    url = "https://raw.githubusercontent.com/NextCenturyCorporation/MCS/master/machine_common_sense/scenes/gravity_support_ex_"
    url += f"{i:02d}" + ".json"
    subprocess.run(["wget", "-nc", url])

os.chdir(dir_path)
