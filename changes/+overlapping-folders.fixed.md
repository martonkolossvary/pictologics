`DicomDatabase.from_folders()` reads each file once when the folders overlap (a folder and its subfolder). Before, it read the files of the subfolder twice.
