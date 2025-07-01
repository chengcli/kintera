#! /bin/bash

pip uninstall kintera
rm -f dist/*whl
python -m build --wheel .
pip install dist/*.whl
