from PIL import Image, ImageEnhance, ImageOps
import numpy as np
from scipy.ndimage import filters
from skimage import draw
import random


# Need to add parser arguments

image_path = 'test.jpg'

# Scales contrast
scale_value = 1
# Scales size of mask (both main mask and outer gradient skirt)
mask_scale_inner = 0.8
mask_scale_outer = 0.9
# Standard deviation for Gaussian filter (blur)
sigma = 6
mask_sigma = 20
# Scales mean value for black/white dropoff
alpha = 0.2


def gen_mask(arr, mask_scale):
	"""
	Generates an elliptical mask proportionate to the size of the input image
	"""
	mask = np.zeros_like(arr)
	rr, cc = draw.ellipse(int(arr.shape[0]/2), int(arr.shape[1]/2), int((arr.shape[0]/2)*mask_scale), int((arr.shape[1]/2)*mask_scale))
	mask[rr, cc] = 255
	mask = filters.gaussian_filter(mask, mask_sigma)
	return mask/255


def blob(arr):
	"""
	Generates an ink splatter mask at the edges of the main shape
	"""
	blob_mask = np.zeros_like(arr)
	max_value = np.amax(arr)
	# trying to make blob size proportionate to the size of the input image
	blob_size = (arr.shape[0]+arr.shape[1])//60 # 60 seems to be a reasonable size but should be a param
	for i in range(arr.shape[0]):
		for j in range(int(arr.shape[1]/2)):
			# align the probability of the pixel receiving a blob with the pixel value
			prob = arr[i][j]/max_value
			prob_reduce = prob * 0.01 # scaled the probability a bit to make it rarer for blobs to appear, this should be a param
			if random.random() < prob_reduce:
				print("blobbin'")
				r = blob_size*prob
				rr, cc = draw.circle(i, j, r)
				blob_mask[rr, cc] = 255
	return blob_mask

# Open image
image = Image.open(image_path)
# image.show()

# Convert to grayscale
image = image.convert('L')

# Invert the image
image = ImageOps.invert(image)

# Create array layers
arr = np.array(image)
arr_layer = np.array(image)

# Generate masks
mask = gen_mask(arr, mask_scale_inner)
mask_layer = gen_mask(arr_layer, mask_scale_outer)

# Apply masks
arr = np.multiply(arr, mask, out=arr, casting='unsafe')
arr_layer = np.multiply(arr_layer, mask, out=arr_layer, casting='unsafe')

# Invert image back
image = Image.fromarray(arr)
image = ImageOps.invert(image)
image_layer = Image.fromarray(arr_layer)
image_layer = ImageOps.invert(image_layer)

# image.show()
# image_layer.show()

# Enhance contrast
image = ImageEnhance.Contrast(image).enhance(scale_value)

# image_layer.show()

# Convert back to array
arr = np.array(image)
arr_layer = np.array(image_layer)

# Blurs the image
arr = filters.gaussian_filter(arr, sigma)
arr_layer = filters.gaussian_filter(arr_layer, sigma)


# Push all values under mean*alpha to black and everything else to white
mean_value = np.mean(arr)
arr[arr < mean_value*alpha] = 0
arr[arr >= mean_value*alpha] = 255

# Create a gradient skirt around the main shape
arr_layer[arr_layer == mean_value*alpha] = 0
arr_layer[arr_layer < mean_value*alpha] = 255
image_layer = Image.fromarray(arr_layer)
image_layer = ImageOps.invert(image_layer)
arr_layer = np.array(image_layer)
# image_layer.show()

# Convert the gradient skirt layer to ink blobs
arr -= blob(arr_layer)

# Mirror the image
arr = arr[:,:-int(arr.shape[1]/2)]
mirror = np.fliplr(arr)
arr = np.hstack([arr, mirror])

# Add slight blur for polish
arr = filters.gaussian_filter(arr, 1.5)

# Convert from numpy array to image
image = Image.fromarray(arr)

# Show image
image.show()