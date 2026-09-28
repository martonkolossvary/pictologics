`format_results(fmt="long")` now raises `ValueError` for an unknown `output_type`, as the wide format does. Before, it returned a DataFrame.
