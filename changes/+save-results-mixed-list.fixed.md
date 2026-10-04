`save_results()` raises ValueError for a list that mixes dicts or DataFrames with other items, as its documentation says. Before, it raised TypeError or AttributeError.
