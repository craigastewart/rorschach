from PIL import Image, ImageEnhance, ImageOps
import numpy as np
from scipy.ndimage import filters
from skimage import draw
import random, argparse, sys, os


def gen_mask(arr, mask_scale):
	"""
	Generates an elliptical mask proportionate to the size of the input image
	"""
	mask = np.zeros_like(arr)
	rr, cc = draw.ellipse(int(arr.shape[0]/2), int(arr.shape[1]/2), int((arr.shape[0]/2)*mask_scale), int((arr.shape[1]/2)*mask_scale))
	mask[rr, cc] = 255
	mask = filters.gaussian_filter(mask, mask_scale)
	return mask/255


def blob(arr):
	"""
	Generates an ink splatter mask at the edges of the main shape
	"""
	blob_mask = np.zeros_like(arr)
	max_value = np.amax(arr)
	# trying to make blob size proportionate to the size of the input image
	blob_size = (arr.shape[0]+arr.shape[1])//60 # 60 seems to be a reasonable size but should be a param
	blob_count = 0
	for i in range(arr.shape[0]):
		for j in range(int(arr.shape[1]/2)):
			# align the probability of the pixel receiving a blob with the pixel value
			prob = arr[i][j]/max_value
			prob_reduce = prob * 0.01 # scaled the probability a bit to make it rarer for blobs to appear, this should be a param
			if random.random() < prob_reduce:
				blob_count += 1
				r = blob_size*prob
				rr, cc = draw.circle(i, j, r)
				blob_mask[rr, cc] = 255
	print("Blobbed {} times!".format(blob_count))
	return blob_mask


def texture(arr, texture_dir):
	"""
	Applies a randomly chosen predefined ink blot texture
	"""
	tx_files = os.listdir(texture_dir)
	t = random.randint(1,len(tx_files))
	texture = Image.open('{}texture_{}.jpg'.format(texture_dir, t))
	texture = texture.convert('L')
	texture = texture.resize((arr.shape[1], arr.shape[0]))
	texture = ImageOps.invert(texture)
	tx_arr = np.array(texture)
	max_value = np.amax(tx_arr)
	center_shade = np.zeros_like(tx_arr)
	# center_shade[]
	tx_arr = tx_arr/max_value
	return np.multiply(arr, tx_arr, out=arr, casting='unsafe')


def parse_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument('--image_path',dest='image_path', help='Input image path', type=str, default='test.jpg')
	parser.add_argument('--texture_dir',dest='texture_dir', help='Texture directory path', type=str, default='textures/')
	parser.add_argument('--contrast',dest='contrast', help='Scales contrast applied to input image', type=float, default='1.3')
	parser.add_argument('--inner_mask',dest='inner_mask', help='Size of the inner mask (0.0, 1.0]', type=float, default='0.8')
	parser.add_argument('--outer_mask',dest='outer_mask', help='Size of the outer mask (0.0, 1.0]', type=float, default='0.9')
	parser.add_argument('--blur',dest='blur', help='Amount of blur applied to main image', type=float, default='6.0')
	parser.add_argument('--mask_blur',dest='mask_blur', help='Amount of blur applied to masks', type=float, default='20.0')
	parser.add_argument('--bw_dropoff',dest='bw_dropoff', help='Point of dropoff for black and white', type=float, default='0.2')
	parser.add_argument('--white_threshold',dest='white_threshold', help='Color value at which image is cropped [0, 255]', type=float, default='100')
	return parser.parse_args()


def main(args):
	args = parse_arguments()

	# Open image
	image = Image.open(args.image_path)
	# image.show()
	
	# Convert to grayscale
	image = image.convert('L')
	
	# Invert the image
	image = ImageOps.invert(image)
	
	# Create array layers
	arr = np.array(image)
	arr_layer = np.array(image)
	
	# Generate masks
	mask = gen_mask(arr, args.inner_mask)
	mask_layer = gen_mask(arr_layer, args.outer_mask)
	
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
	image = ImageEnhance.Contrast(image).enhance(args.contrast)
	
	# image_layer.show()
	
	# Convert back to array
	arr = np.array(image)
	arr_layer = np.array(image_layer)
	
	# Blurs the image
	arr = filters.gaussian_filter(arr, args.blur)
	arr_layer = filters.gaussian_filter(arr_layer, args.blur)
	
	
	# Push all values under mean*args.bw_dropoff to black and everything else to white
	mean_value = np.mean(arr)
	arr[arr < mean_value*args.bw_dropoff] = 0
	arr[arr >= mean_value*args.bw_dropoff] = 255
	
	# Create a gradient skirt around the main shape
	arr_layer[arr_layer == mean_value*args.bw_dropoff] = 0
	arr_layer[arr_layer < mean_value*args.bw_dropoff] = 255
	image_layer = Image.fromarray(arr_layer)
	image_layer = ImageOps.invert(image_layer)
	arr_layer = np.array(image_layer)
	# image_layer.show()
	
	# Convert the gradient skirt layer to ink blobs
	arr -= blob(arr_layer)
	
	image = Image.fromarray(arr)
	image = ImageOps.invert(image)
	arr = np.array(image)
	
	# Crop image in half
	arr = arr[:,:-arr.shape[1]//2]
	
	# Texture
	arr = texture(arr, args.texture_dir)
	
	# Crop lightest grey
	arr[arr < args.white_threshold] = 0
	
	# Mirror the image
	mirror = np.fliplr(arr)
	arr = np.hstack([arr, mirror])
	
	# Add slight blur for polish
	arr = filters.gaussian_filter(arr, 1.1)
	
	# Convert from numpy array to image
	image = Image.fromarray(arr)
	image = ImageOps.invert(image)
	
	# Show image
	image.save('output.png')


if __name__ == '__main__':
	main(sys.argv)
