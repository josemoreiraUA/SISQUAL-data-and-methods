import subprocess
import sys
from alembic.config import Config
from alembic import command
from core.config import ROOT

alembic_cfg = Config(ROOT.parent / "alembic.ini")

subprocess.run([sys.executable, "./backend_pre_start.py"])
command.upgrade(alembic_cfg, "head")
#subprocess.run([sys.executable, "./app/initial_data.py"])
