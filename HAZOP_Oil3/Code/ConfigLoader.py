import json

def load_config(path="risk_config.json"):
    """
    Load configuration file from project root
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_active_unit(config):
    """
    Return the first enabled refinery unit
    """
    for name, unit in config["refinery_units"].items():
        if unit.get("enabled", True):
            return name, unit
    return None, None

def get_features_for_unit(config, unit_name):
    """
    Extract all active features for a given refinery unit
    """
    unit = config["refinery_units"][unit_name]
    features = []
    for cat in unit["risk_categories"].values():
        features += cat.get("features", [])
    return list(set(features))

def get_hazop_rules(config, unit_name=None):
    """
    Extract HAZOP rules for operational risk
    """
    if unit_name is None:
        unit_name, _ = get_active_unit(config)
    unit = config["refinery_units"][unit_name]
    operational = unit["risk_categories"].get("operational", {})
    return operational.get("hazop_rules", {})

# Quick test (optional)
if __name__ == "__main__":
    config = load_config()
    name, unit = get_active_unit(config)
    print(" Active unit:", name)
    print(" Features:", get_features_for_unit(config, name))
    print(" Risk categories:", list(unit["risk_categories"].keys()))

if __name__ == "__main__":
    config = load_config()
    name, unit = get_active_unit(config)
    print("Active unit:", name)
    print("Features:", get_features_for_unit(config, name))