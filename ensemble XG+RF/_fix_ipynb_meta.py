"""
Fix ensemble_voting_heart.ipynb so GitHub renders it correctly.

Error:  "the 'state' key is missing from 'metadata.widgets'.
         Add 'state' to each, or remove 'metadata.widgets'."

Root cause: the widgets block was exported in the old Colab format where
individual model UUIDs sit at the top level of
  metadata.widgets["application/vnd.jupyter.widget-state+json"]
instead of being nested under a "state" key as the spec requires:

  REQUIRED structure:
  {
    "application/vnd.jupyter.widget-state+json": {
      "state": { <uuid>: {...}, ... },   <- must exist
      "version_major": 2,
      "version_minor": 0
    }
  }

Fix applied:
  - Move all UUID entries into the "state" sub-dict
  - Add "version_major" / "version_minor"
  - Also patch kernelspec + language_info for JupyterLab compatibility
  - Add cell "id" fields required by nbformat >= 4.5
"""

import json, uuid, pathlib

NB = pathlib.Path(__file__).with_name("ensemble_voting_heart.ipynb")

nb = json.loads(NB.read_text(encoding="utf-8"))

# ── 1. Top-level format ────────────────────────────────────────────────────
nb["nbformat_minor"] = 5

# ── 2. kernelspec ──────────────────────────────────────────────────────────
nb["metadata"]["kernelspec"] = {
    "name"        : "python3",
    "display_name": "Python 3 (ipykernel)",
    "language"    : "python",
}

# ── 3. language_info ───────────────────────────────────────────────────────
nb["metadata"]["language_info"] = {
    "name"              : "python",
    "version"           : "3.10.0",
    "mimetype"          : "text/x-python",
    "file_extension"    : ".py",
    "pygments_lexer"    : "ipython3",
    "codemirror_mode"   : {"name": "ipython", "version": 3},
    "nbconvert_exporter": "python",
}

# ── 4. Colab toc flag ─────────────────────────────────────────────────────
nb["metadata"].setdefault("colab", {})["toc_visible"] = True

# ── 5. Fix widgets metadata ────────────────────────────────────────────────
#
# GitHub (nbconvert v7) requires:
#   metadata.widgets["application/vnd.jupyter.widget-state+json"]["state"] = {...}
#
# Old Colab notebooks dump UUID keys directly at the root level — that is
# what triggers "the 'state' key is missing" error.
#
WIDGET_KEY = "application/vnd.jupyter.widget-state+json"
widgets_meta = nb["metadata"].get("widgets", {})
raw_widget_block = widgets_meta.get(WIDGET_KEY, {})

if "state" not in raw_widget_block:
    # Collect all UUID-keyed entries into a proper "state" dict
    state_dict = {k: v for k, v in raw_widget_block.items()
                  if k not in ("version_major", "version_minor")}
    nb["metadata"]["widgets"] = {
        WIDGET_KEY: {
            "state"        : state_dict,
            "version_major": raw_widget_block.get("version_major", 2),
            "version_minor": raw_widget_block.get("version_minor", 0),
        }
    }
    print(f"[OK] Restructured widgets: {len(state_dict)} model(s) moved under 'state'")
else:
    print("[OK] widgets.state already present — no change needed")

# ── 6. Cell IDs (nbformat 4.5+ requirement) ────────────────────────────────
patched = 0
for cell in nb.get("cells", []):
    if "id" not in cell:
        cell["id"] = uuid.uuid4().hex[:8]
        patched += 1

# ── 7. Write back ──────────────────────────────────────────────────────────
NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")

print(f"[OK] Patched: {NB.name}")
print(f"     nbformat_minor -> {nb['nbformat_minor']}")
print(f"     kernelspec     -> {nb['metadata']['kernelspec']['display_name']}")
print(f"     language       -> {nb['metadata']['language_info']['name']} "
      f"v{nb['metadata']['language_info']['version']}")
print(f"     cell ids added -> {patched} cells")
print("[DONE] Re-run: git add . && git commit && git push")
