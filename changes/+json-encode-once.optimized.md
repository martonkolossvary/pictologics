`save_results()` writes a JSON table with a NaN or infinite value in one encode: 80 ms instead of 132 ms for 100 rows of 1,020 features with a NaN in the last row, with the same bytes.
