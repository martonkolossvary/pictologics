`format_results` gets the feature names and values of a Series in two calls, not one by one. A wide row of the 6 standard configurations takes 65 µs instead of 99 µs. The values do not change.
