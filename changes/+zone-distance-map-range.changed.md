`calculate_zone_features` raises a `ValueError` when its distance map has a value outside [0, 65535] or a NaN. Before, a negative value or a NaN removed the zone from the GLDZM silently.
