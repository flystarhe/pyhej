# pip install SimpleITK
# https://github.com/InsightSoftwareConsortium/SimpleITK-Notebooks
# SimpleITK可以读取和写入图像从单个文件或一组文件(例如DICOM系列)
#  - SimpleITK: image[x,y,z]
#  - numpy: image_numpy_array[z,y,x]
import os
import SimpleITK as sitk


def get_series_ids(data_directory):
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(data_directory)
    return series_ids


def read_dicom_slice(dicom_path):
    image = sitk.ReadImage(dicom_path)
    # GetMetaDataKeys, HasMetaDataKey, and GetMetaData
    return image


def read_dicom_series(data_directory, series_id):
    reader = sitk.ImageSeriesReader()
    image = sitk.ReadImage(reader.GetGDCMSeriesFileNames(data_directory, series_id))
    # GetMetaDataKeys, HasMetaDataKey, and GetMetaData
    return image


def to_numpy(image):
    image_array = sitk.GetArrayFromImage(image)
    # image.GetSize() vs image_array.shape
    return image_array


def meta_data_from_image(image, tag):
    """tag:
    "0010|0010": "Patient name: "
    "0008|0060": "Modality: "
    "0008|0021": "Series date: "
    "0008|0080": "Institution name: "
    """
    if image.HasMetaDataKey(tag):
        return True, image.GetMetaData(tag)
    return False, image.GetMetaDataKeys()


def meta_data_from_file(dicom_path, tag):
    """tag:
    "0010|0010": "Patient name: "
    "0008|0060": "Modality: "
    "0008|0021": "Series date: "
    "0008|0080": "Institution name: "
    """
    file_reader = sitk.ImageFileReader()
    file_reader.SetFileName(dicom_path)
    file_reader.ReadImageInformation()
    if file_reader.HasMetaDataKey(tag):
        return True, file_reader.GetMetaData(tag)
    return False, file_reader.GetMetaDataKeys()


# Read DICOM series and write it as a single mha file
def dicom_series_to_mha(data_directory, series_id, output_dir=".", output_name="3Dimage.mha"):
    original_image = read_dicom_series(data_directory, series_id)
    # Write the image.
    output_file_name_3D = os.path.join(output_dir, output_name)
    sitk.WriteImage(original_image, output_file_name_3D)
    # Read it back again.
    written_image = sitk.ReadImage(output_file_name_3D)
    # Check that the original and written image are the same.
    statistics_image_filter = sitk.StatisticsImageFilter()
    statistics_image_filter.Execute(original_image - written_image)
    # Check that the original and written files are the same
    print("Max, Min differences are : {0}, {1}".format(statistics_image_filter.GetMaximum(), statistics_image_filter.GetMinimum()))


# Write an image series as JPEG
def dicom_series_to_jpeg(data_directory, series_id, output_dir="."):
    original_image = read_dicom_series(data_directory, series_id)
    # rescale the image (default is [0,255]) since the JPEG format uint8 pixel type.
    output_file_names = [os.path.join(output_dir, "slice{:03d}.jpg".format(i)) for i in range(original_image.GetSize()[2])]
    sitk.WriteImage(sitk.Cast(sitk.RescaleIntensity(original_image), sitk.sitkUInt8), output_file_names)
