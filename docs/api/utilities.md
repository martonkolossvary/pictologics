# Utilities API

::: pictologics.utilities.dicom_database
    options:
      members:
        - DicomInstance
        - DicomSeries
        - DicomStudy
        - DicomPatient
        - DicomDatabase

::: pictologics.utilities.dicom_utils
    options:
      members:
        - DicomPhaseInfo
        - get_dicom_phases
        - split_dicom_phases

::: pictologics.utilities.sr_parser
    options:
      members:
        - SRMeasurement
        - SRMeasurementGroup
        - SRDocument
        - SRBatch
        - is_dicom_sr


::: pictologics.utilities.visualization
    options:
      members:
        - visualize_slices
        - save_slices
