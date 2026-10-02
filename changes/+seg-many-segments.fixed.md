A DICOM SEG with more than 255 segments now loads. The combined label image is uint16 when a segment number is above 255. Before, the load failed, or the labels wrapped (300 became 44).
