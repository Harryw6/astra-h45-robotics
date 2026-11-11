"""Workspace launcher; dependencies are explicit, never installed on execution."""
from pathlib import Path
import sys

project=Path(__file__).resolve().parent
vendor=project.parents[1]/'work/wam_demo_library/vendor'
if vendor.is_dir():sys.path.insert(0,str(vendor))
sys.path.insert(0,str(project/'src'))
from wam_astra.cli import main

if __name__=='__main__':main()
