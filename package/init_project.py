from __future__ import annotations

from pathlib import Path
import sys


THIS_FILE = Path(__file__).resolve()
PACKAGE_DIR = THIS_FILE.parent
PROJECT_DIR = PACKAGE_DIR.parent
LAB_DIR = PROJECT_DIR.parents[1]

for path in [str(LAB_DIR), str(PROJECT_DIR), str(PACKAGE_DIR)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from constants import *
from controller import *
from data import *
from language import *
from notebook_units import *
from paths import *
from real_hri import *
from real_sh5_zmq import *
from real_stt import *
from real_zed import *
from sh5_right_arm import *
from training import *

from messages import *
from nodes import *
from ros_graph import *
from runtime import *

print("\n[LILAC_ROS] initialized.")
