"""
config_utils.py - Configuration helper for AIML CropMapper Cloud toolbox.
Loads settings from config.json, config_cdse.json, config_s1.json, config_s2.json,
environment variables, and defaults.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent


def load_json_file(file_path: Path) -> Dict[str, Any]:
    if file_path.exists():
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def get_project_config() -> Dict[str, Any]:
    """
    Loads unified configuration combining root config.json, config_cdse.json,
    and environment variables.
    """
    cfg = {}

    # 1. Root config.json
    root_cfg_file = PROJECT_ROOT / "config.json"
    cfg = load_json_file(root_cfg_file)

    # 2. CDSE specific config
    cdse_cfg_file = PROJECT_ROOT / "config_cdse.json"
    cdse_data = load_json_file(cdse_cfg_file)
    if cdse_data:
        cfg.setdefault("cdse", {})
        if "username" in cdse_data:
            cfg["cdse"]["username"] = cdse_data["username"]
        if "password" in cdse_data:
            cfg["cdse"]["password"] = cdse_data["password"]

    # 3. Environment variable overrides
    paths = cfg.setdefault("paths", {})
    if "AIML_WORKING_DIR" in os.environ:
        paths["working_dir"] = os.environ["AIML_WORKING_DIR"]
    if "AIML_AUX_DIR" in os.environ:
        paths["aux_dir"] = os.environ["AIML_AUX_DIR"]
    if "SNAP_GPT_EXE" in os.environ:
        paths["snap_gpt_exe"] = os.environ["SNAP_GPT_EXE"]
    if "SNAP_AUXDATA_PATH" in os.environ:
        paths["snap_auxdata_path"] = os.environ["SNAP_AUXDATA_PATH"]
    if "S1_REPO_PATH" in os.environ:
        paths["s1_repo_path"] = os.environ["S1_REPO_PATH"]
    if "S2_REPO_PATH" in os.environ:
        paths["s2_repo_path"] = os.environ["S2_REPO_PATH"]

    cdse = cfg.setdefault("cdse", {})
    if "CDSE_USERNAME" in os.environ:
        cdse["username"] = os.environ["CDSE_USERNAME"]
    if "CDSE_PASSWORD" in os.environ:
        cdse["password"] = os.environ["CDSE_PASSWORD"]

    return cfg


def get_cdse_credentials() -> tuple:
    """Returns (username, password) from config files or env."""
    cfg = get_project_config()
    cdse = cfg.get("cdse", {})
    username = cdse.get("username", "")
    password = cdse.get("password", "")
    return username, password
