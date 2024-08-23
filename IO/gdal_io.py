# Copyright (c) 2022, Jin Liu and GPCV.
# All rights reserved.
# Author: Jin Liu

import numpy as np
import gdal


def Read_Img_Tone(path, x_lu, y_lu, x_size, y_size):
    dataset = gdal.Open(path)
    if dataset == None:
        print("GDAL RasterIO Error: Opening" + path + " failed!")
        return

    data = dataset.ReadAsArray(x_lu, y_lu, x_size, y_size)

    im = np.power(data, 1.0 / 2.2)  # gamma correction

    # cut off the small values
    below_thres = np.percentile(im.reshape((-1, 1)), 0.5)
    im[im < below_thres] = below_thres
    # cut off the big values
    above_thres = np.percentile(im.reshape((-1, 1)), 99.5)
    im[im > above_thres] = above_thres
    img = 255 * (im - below_thres) / (above_thres - below_thres)

    del dataset

    return img


def gdal_read_img(path, x_lu, y_lu, xsize, ysize):
    dataset = gdal.Open(path)
    if dataset is None:
        raise Exception("GDAL RasterIO Error: Opening" + path + " failed!")

    if x_lu is None:
        x_lu = 0
    if y_lu is None:
        y_lu = 0
    if xsize is None:
        x_size = dataset.RasterXSize - x_lu
    if ysize is None:
        y_size = dataset.RasterYSize - y_lu

    data = dataset.ReadAsArray(x_lu, y_lu, xsize, ysize)

    del dataset

    return data


def gdal_get_size(path):
    dataset = gdal.Open(path)
    if dataset is None:
        raise Exception("GDAL RasterIO Error: Opening" + path + " failed!")

    width = dataset.RasterXSize
    height = dataset.RasterYSize

    del dataset
    return width, height


def gdal_write_img(filename, im_data):
    # The data types of gdal include:
    # gdal.GDT_Byte,
    # gdal.GDT_UInt16, gdal.GDT_Int16, gdal.GDT_UInt32, gdal.GDT_Int32,
    # gdal.GDT_Float32, gdal.GDT_Float64

    # Determining the data type of raster data:
    if 'uint8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'uint16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_Int16
    elif 'int32' in im_data.dtype.name:
        datatype = gdal.GDT_Int32
    elif 'uint32' in im_data.dtype.name:
        datatype = gdal.GDT_uInt32
    elif 'float32' in im_data.dtype.name:
        datatype = gdal.GDT_Float32
    else:
        datatype = gdal.GDT_Float64

    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape

    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)

    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(im_data)
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(im_data[i])

    del dataset


def gdal_write_to_tif(out_path, xlu, ylu, data):
    dataset = gdal.Open(out_path, gdal.GF_Write)
    if dataset is None:
        raise Exception("GDAL RasterIO Error: Opening" + out_path + " failed!")

    if len(data.shape) == 3:
        im_bands = data.shape[0]
    else:
        im_bands = 1

    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(data, xlu, ylu)
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(data[i], xlu, ylu)
    del dataset


def gdal_create_dsm_file(out_path, border, xuint, yuint):
    width = int((border[1] - border[0] + 0.00000001) / xuint)
    height = int((border[3] - border[2] + 0.00000001) / yuint)

    driver = gdal.GetDriverByName("GTiff")
    dst_ds = driver.Create(out_path, width, height, 1, gdal.GDT_Float32)
    del dst_ds

    text_ = ""
    text_ += str(xuint) + "\n0\n0\n" + str(-yuint) + "\n" + str(border[0]) + "\n" + str(border[3])
    tfw_path = out_path.replace(".tif", ".tfw")
    with open(tfw_path, "w") as f:
        f.write(text_)