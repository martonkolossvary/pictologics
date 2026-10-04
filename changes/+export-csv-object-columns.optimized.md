`DicomDatabase.export_csv()` looks for lists only in the columns of Python objects: 83 ms instead of 89 ms for a database of 6,000 files, with the same CSV bytes.
