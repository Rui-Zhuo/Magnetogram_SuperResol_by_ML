# Image size
h_HMI = 4096
w_HMI = 4096
h_HMI_edge = 48
w_HMI_edge = 48

# Upscaling factors
h_ups = 1.56
w_ups = 1.68

# Image shapes
h_shape_lr = 200
w_shape_lr = 200
h_shape_hr = int(h_shape_lr * h_ups)
w_shape_hr = int(w_shape_lr * w_ups)

# Pixel size
pixel_LR_arcsec = 0.5
arcsec2m = 7e8 / 960
pixel_LR_m = pixel_LR_arcsec * arcsec2m
pixel_HR_h_m = pixel_LR_m / h_ups
pixel_HR_w_m = pixel_LR_m / w_ups

# Pixel areas
pixel_area_LR = pixel_LR_m ** 2
pixel_area_HR = pixel_HR_h_m * pixel_HR_w_m

# Normalization factors
inp_sub = 0
inp_norm = 200
gt_sub = 0
gt_norm = 200