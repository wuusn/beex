import os
import SimpleITK as sitk
import numpy as np


def read_dcm(file_path):
    series_reader = sitk.ImageSeriesReader()
    series_files_path = series_reader.GetGDCMSeriesFileNames(file_path)

    series_reader.SetFileNames(series_files_path)

    img = series_reader.Execute()
    print("img:", img.GetDirection())
    file_reader = sitk.ImageFileReader()

    file_reader.SetFileName(series_files_path[6])
    file_reader.ReadImageInformation()
    manufacturer = file_reader.GetMetaData("0008|0070") if file_reader.HasMetaDataKey("0008|0070") else "unknown"
    print(f"**{manufacturer}**")
    return img, manufacturer


def dcm2nii(input_dir, output_dir, name):
    data,_ = read_dcm(input_dir)
    #output_nifti_filename = os.path.join(output_dir, f"img_{name[:7]}_{name.split('_')[1]}.nii.gz")
    output_nifti_filename = os.path.join(output_dir, f"{name}.nii.gz")
    output = os.path.join(output_dir, output_nifti_filename)
    sitk.WriteImage(data,output)


if __name__ == '__main__':
    
    # Example usage
    # please replace the input_dir and output_dir with your own directory
    input_dir = '/mnt/hd0/project_large_files/bee/caseDCM/a/bgy01598482/bgy01598482MRI/DWI'
    output_dir = '/mnt/hd0/project_large_files/bee/caseDCM/test_cvt'
    os.makedirs(output_dir, exist_ok=True)
    dcm2nii(input_dir, output_dir, 'bgy01598482MRI')
    print("over!")   

