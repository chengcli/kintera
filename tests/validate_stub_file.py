#!/usr/bin/env python3
"""
Validation script for kintera.pyi stub file

This script validates that the stub file correctly represents
the API by checking that all documented classes and methods exist.
"""

import sys
import ast
from pathlib import Path

def parse_stub_file(stub_path):
    """Parse the stub file and extract all public classes and methods."""
    with open(stub_path, 'r') as f:
        tree = ast.parse(f.read(), filename=str(stub_path))

    classes = {}
    functions = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            # Get class name and methods
            methods = []
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    if not item.name.startswith('_') or item.name in ['__init__', '__repr__']:
                        methods.append(item.name)
            classes[node.name] = methods
        elif isinstance(node, ast.FunctionDef):
            # Top-level functions
            if not node.name.startswith('_') and not any(node in c.body for c in ast.walk(tree) if isinstance(c, ast.ClassDef)):
                functions.append(node.name)

    return classes, functions

def validate_stub_completeness(stub_path):
    """Validate that the stub file has comprehensive coverage."""
    print(f"Validating stub file: {stub_path}")
    print("=" * 60)

    classes, functions = parse_stub_file(stub_path)

    print(f"\n✅ Found {len(classes)} classes:")
    for class_name, methods in sorted(classes.items()):
        print(f"   - {class_name} ({len(methods)} methods)")

    print(f"\n✅ Found {len(functions)} module-level functions:")
    for func in sorted(functions):
        print(f"   - {func}")

    # Expected classes from C++ bindings
    expected_classes = [
        'SpeciesThermo',
        'Reaction',
        'NucleationOptions',
        'ThermoOptions',
        'ThermoX',
        'ThermoY',
        'ArrheniusOptions',
        'Arrhenius',
        'CoagulationOptions',
        'EvaporationOptions',
        'Evaporation',
        'KineticsOptions',
        'Kinetics',
        'constants'
    ]

    print(f"\n{'='*60}")
    print("Checking for required classes...")
    missing = []
    for cls in expected_classes:
        if cls in classes:
            print(f"   ✅ {cls}")
        else:
            print(f"   ❌ {cls} MISSING")
            missing.append(cls)

    if missing:
        print(f"\n❌ Missing classes: {', '.join(missing)}")
        return False

    # Expected functions
    expected_functions = [
        'species_names',
        'set_species_names',
        'species_weights',
        'set_species_weights',
        'species_cref_R',
        'set_species_cref_R',
        'species_uref_R',
        'set_species_uref_R',
        'species_sref_R',
        'set_species_sref_R',
        'set_search_paths',
        'get_search_paths',
        'add_resource_directory',
        'find_resource',
        'evolve_implicit',
        'relative_humidity'
    ]

    print(f"\n{'='*60}")
    print("Checking for required functions...")
    missing_funcs = []
    for func in expected_functions:
        if func in functions:
            print(f"   ✅ {func}")
        else:
            print(f"   ❌ {func} MISSING")
            missing_funcs.append(func)

    if missing_funcs:
        print(f"\n❌ Missing functions: {', '.join(missing_funcs)}")
        return False

    print(f"\n{'='*60}")
    print("✅ All required classes and functions are present!")
    print(f"\nSummary:")
    print(f"  - Classes: {len(classes)}")
    print(f"  - Module functions: {len(functions)}")
    print(f"  - Total methods: {sum(len(m) for m in classes.values())}")

    return True

def check_stub_syntax(stub_path):
    """Check if the stub file has valid Python syntax."""
    print("\nChecking stub file syntax...")
    try:
        with open(stub_path, 'r') as f:
            ast.parse(f.read(), filename=str(stub_path))
        print("✅ Stub file has valid Python syntax")
        return True
    except SyntaxError as e:
        print(f"❌ Syntax error in stub file: {e}")
        return False

def main():
    """Main validation function."""
    repo_root = Path(__file__).parent.parent
    stub_file = repo_root / "python" / "kintera.pyi"

    if not stub_file.exists():
        print(f"❌ Stub file not found: {stub_file}")
        sys.exit(1)

    # Check syntax
    if not check_stub_syntax(stub_file):
        sys.exit(1)

    # Validate completeness
    if not validate_stub_completeness(stub_file):
        sys.exit(1)

    print(f"\n{'='*60}")
    print("✅ Stub file validation PASSED!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
