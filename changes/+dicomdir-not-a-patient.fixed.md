`DicomDatabase.from_folders` no longer lists a DICOMDIR file as an image of patient "UNKNOWN". A DICOMDIR only indexes the other files of a DICOM medium, so the scan now skips it.
