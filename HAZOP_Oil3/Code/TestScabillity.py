from ConfigLoader import load_config, get_active_unit, get_features_for_unit

config = load_config()
unit_name, unit = get_active_unit(config)
print(" Active Unit:", unit["name"])
print(" Features:", get_features_for_unit(config, unit_name))
print(" Risk categories:", list(unit["risk_categories"].keys()))