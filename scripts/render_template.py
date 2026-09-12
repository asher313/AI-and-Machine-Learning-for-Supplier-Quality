"""Substitute an operator's configuration into parsed JSON; never interpolate raw JSON."""

import argparse
import json
from pathlib import Path
from string import Template


def render(value, variables):
    if isinstance(value, str):
        return Template(value).substitute(variables)
    if isinstance(value, list):
        return [render(v, variables) for v in value]
    if isinstance(value, dict):
        return {k: render(v, variables) for k, v in value.items()}
    return value


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--template", type=Path, required=True)
    p.add_argument("--values", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    output = render(
        json.loads(a.template.read_text()),
        json.loads(a.values.read_text()),
    )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(output, indent=2) + "\n")
