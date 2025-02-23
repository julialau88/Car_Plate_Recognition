# A model trained to recognise car plates
In this project, we propose an algorithm to detect car plates, along with training a neural network model to recognise the detect car plate values. The steps are detailed below: 

## Locating the vehicle in the image 
### 1. Preprocess car plate images 
The first step is to preprocess captured car plate images. Preprocessing steps include converting the image into grayscale and resizing the image. 

### 2. Canny Edge Detection 
The next step is to perform canny edge detection to get the edge image. 

### 3. Masking 
Masking is then performed to make the car in the image more protuding as compared to the background. The mask is created by using Floyd-Steinberg dithering to approximate the original image luminosity levels. The background has lower luminosity levels compared to the vehicle. After applying Floyd-Steinberg, the background has more white pixels compared to the vehicle in the image. The vehicle has higher luminosity levels compared to the background. With applying Floyd-Steinberg onto the image, the vehicle has more black pixels compared to the background. The mask is applied onto the car image. 

### 4. Locating the car by choosing quadrant with most black pixels. 
The image is split into 9 quadrants and we will locate which quadrant has the most black pixels. After obtaining the chosen quadrant with the most black pixels, the quadrant will be set as the centre of the cropping area and we will crop the car out of the image hence removing unnecessary background information. 

## Locating the car plate 
After locating the vehicle, it is time to locate the car plate. 
### 1. Preprocessing 
Preprocessing steps are unsharp, histogram equalisation, and Gaussian filtering. 

### 2. Vertical Edge Detection
Vertical Sobel is applied to detect vertical edges

### 3. Car plate location
We estimate 3 window sizes that approximate possible car plate sizes to slide across the car image (with background removed using above step) to locate the car plate. We leverage the 
