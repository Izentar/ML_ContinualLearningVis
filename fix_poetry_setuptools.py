import os

os.system("poetry run python3.9 -m pip install setuptools=='59.5.0'")
print("Changed setuptools version to v59.5.0. Fix is avaliable in pytorch nightly (at least this is what internet says).")