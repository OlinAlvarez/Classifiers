import glob
import cv2
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
	rotation_range = 40,
	width_shift_range = 0.2,
	height_shift_range = 0.2,
	rescale = 1. / 255,
	shear_range = 0.2,
	zoom_range = 0.2,
	horizontal_flip = True,
	fill_mode = 'nearest')

dice = []
for img in glob.glob("pos_dice/*.jpg"):
	dice.append(cv2.imread(img))
for die in dice:
   
    die = die.reshape((1,) + die.shape) 
    i = 0
    for batch in datagen.flow(die, batch_size=1,save_to_dir='dice_training_images', save_prefix='die', save_format='jpeg'):
	    i += 1
	    if i > 20:
		break 




