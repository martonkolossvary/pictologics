Slice overlays color all labels in one step, not in one pass per label. A 512 x 512 slice with 117 labels takes 8 ms instead of 122 ms, and with 1 label 5 ms instead of 7 ms. The pixels are the same.
