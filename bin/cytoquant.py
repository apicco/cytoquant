# All the software here is distributed under the terms of the GNU General Public License Version 3, June 2007.  
# CytoQuant is a free software and comes with ABSOLUTELY NO WARRANTY. 
# 

import numpy as np
from os import listdir as ls
from os import mkdir
import copy as cp

from skimage import filters #import filters
from skimage import morphology
from skimage import img_as_float, img_as_uint
from skimage.external import tifffile as tiff
from skimage.measure import label
from skimage import io

from matplotlib import pyplot as plt 

def MAD( x , axis = None , k = 1.4826):

	MAD = np.median( np.absolute( x - np.median( x , axis ) ) , axis )

	return( k * MAD )

def load_images( path , reference_frame = 0 , target_frame = 1 , fv_shape = ( 1024 , 1344 ) ) :
	
	"""
	load all the images in path. Images are stacks containing a reference image at position reference_frame
	and a target image at position target_frame. load_images return two stacks containing the red
	and green images respectively, that is to make sure that the threshold is computed taking
	into account the fluorescence measurements throughout the length of the acquisition.

	- fv_shape: is a tuple with the size of the field of view, in pixels. Default is ( 1024 , 1344 )
	"""


	# List the images in path
	images = [ f for f in [ f for f in ls( path ) if 'tif' in f ] if 'BF' not in f ]

	# Define the empty matrices where the images of the two respective
	# channels will be saved
	reference_channel = np.zeros( ( len( images ) , fv_shape[ 0 ] , fv_shape[ 1 ] ) , dtype = 'uint16' )
	target_channel = np.zeros( ( len( images ) , fv_shape[ 0 ] , fv_shape[ 1 ] ) , dtype = 'uint16' )

	for i in range( len( images ) ):

		im = tiff.imread( path + '/' + images[ i ] )

		if ( im.shape[ 1 ] < fv_shape[ 0 ] ) | ( im.shape[ 2 ] < fv_shape[ 1 ] ) : 

			# images can be a crop of the filed of view. If so, they need to be 
			# sized to be added to the stack, which has the size of the field of 
			# view. The extra pixels that are added to fill the empty space have
			# the value of the lowest pixel intensity recorded in the image, so 
			# to interfere the least with the tresholding of the cells, which are
			# much brighter.
		
			reference_channel[ i , : , : ] = np.zeros( ( fv_shape[ 0 ] , fv_shape[ 1 ] ) ) \
					+ np.min( im[ reference_frame , : , : ] )
			reference_channel[ i , 0 : im.shape[ 1 ] , 0 : im.shape[ 2 ] ] = im[ reference_frame , : , : ]
			
			target_channel[ i , : , : ] = np.zeros( ( fv_shape[ 0 ] , fv_shape[ 1 ] ) ) \
					+ np.min( im[ target_frame , : , : ] )
			target_channel[ i , 0 : im.shape[ 1 ] , 0 : im.shape[ 2 ] ] = im[ target_frame , : , : ]

		else :
			
			reference_channel[ i , : , : ] = im[ reference_frame , : , : ] 
			target_channel[ i , : , : ] = im[ target_frame , : , : ] 

	return ( reference_channel , target_channel )

def select_cytoplasm( im , median_radius , exclude_spots ,  ref_threshold = [] , threshold_image_name = 'threshold.tif') : 

	# Compute the median filter of the image, which will be used to select the spots
	im_median = cp.deepcopy( im )
	for i in range( im.shape[ 0 ] ) :
		im_median[ i , : , : ] = filters.median( im[ i , : , : ] , morphology.disk( median_radius ) )

	# Make a threshold image, that will be used to output the pixel values.
	# The threshold image is the combination of the thresholds on the raw 
	# and median filtered image.
	threshold_raw_cells = filters.threshold_otsu( im )
	threshold_median_cells = filters.threshold_otsu( im_median )

	threshold_raw_image = cp.deepcopy( im )
	threshold_raw_image[ im <= threshold_raw_cells ] = 0
	threshold_raw_image[ im > threshold_raw_cells ] = 1

	threshold_median_image = cp.deepcopy( im )
	threshold_median_image[ im <= threshold_median_cells ] = 0
	threshold_median_image[ im > threshold_median_cells ] = 1

	threshold_image = cp.deepcopy( im )
	threshold_image[ : ] = 0
	threshold_image[ ( threshold_median_image == 1 ) & ( threshold_raw_image == 1 ) ] = 1

	# Exclude the spots that have been thresholded
	if exclude_spots :

		# Isolate pixels brighter than the median. These will be spots
		im_spots = cp.deepcopy( im )
		im_spots[ im <= im_median ] = 0
	
		# Remove the median value from the isolated spots as a 
		# measure of the local cytoplasmatic background. This 
		# image of the spots will be used to theshorld the spots.
		im_spots[ im_spots == im ] = im_spots[ im_spots == im ] - im_median[ im_spots == im ]

		threshold_spots = filters.threshold_yen( im_spots )
		
		im_spots[ im_spots < threshold_spots ] = 0
		im_spots[ im_spots >= threshold_spots ] = 1
		
		#spolt dilation to limit the influence of the pixel intensity in the spots
		for i in range( im_spots.shape[ 0 ] ) :
			im_spots[ i , : , : ] = morphology.dilation( im_spots[ i , : , : ] , selem = morphology.disk( 3 ) )

		threshold_image[ im_spots == 1 ] = 0


	# erode the threshold_image. The threshold image selects the pixels 
	# for the quantification. The erosion is a conservative measure to 
	# reduce the likelihood of quantifying pixels that are outside the 
	# cell.
	for i in range( threshold_image.shape[ 0 ] ) :
		threshold_image[ i , : , : ] = morphology.erosion( threshold_image[ i , : , : ] , selem = morphology.disk( 3 ) )

	ID = 1 #ID of thresholded cells cannot start at 0, as that' the background.
	
	# Compute the red (reference) channel thresholding
	if ref_threshold == [] :

		# Remove the thresholded areas that are smaller than E^8. Those are the vacuols 
		# that are autofluorescent in the red channel

		for i in range( threshold_image.shape[ 0 ] ) :
			
			lb = label( threshold_image[ i , : , : ] )
		
			for j in range( np.max( lb ) ) :

				threshold_area = len( lb[ lb == j + 1 ] )
				
				# set the area threshold limit to be exp( 8 ). If cells
				# are smaller they are likely errors in the thresholding
				# and they are removed. If they are bigger, they are kept
				# and they are assigned an id.
				if np.log( threshold_area ) < 8 :

					threshold_image[ i , : , : ][ lb == ( j + 1 ) ] = 0

				else :

					threshold_image[ i ,  : , : ][ lb == ( j + 1 ) ] = ID
					ID = ID + 1

	# Compute the green (target) channel thresholding
	else :

		# Remove the thresholded cells that belong to the reference	
		for i in range( threshold_image.shape[ 0 ] ) :
			lb = label( threshold_image[ i , : , : ] )
			for j in range( np.max( lb ) ) :
				
				# If cells are not falling into the reference threshold 
				# areas they are likely errors in the thresholding
				# and they are removed. If they are bigger, they are kept
				# and they are assigned an id.
				if np.max( ref_threshold[ i , : , : ][ lb == ( j + 1 ) ] ) > 0 :

					threshold_image[ i , : , : ][ lb == ( j + 1 ) ] = 0 

				else :

					threshold_image[ i ,  : , : ][ lb == ( j + 1 ) ] = ID
					ID = ID + 1

	tiff.imsave( threshold_image_name , threshold_image )
	tiff.imsave( 'raw_channel.tif' , im )
	
	return  threshold_image

def cytoquant( path , median_radius = 6 , exclude_spots = True , golog = True , plot_name = 'hist' , reference_threshold_mask = 'reference_threshold_mask.tif' , target_threshold_mask = 'target_threshold_mask.tif' ):

	"""
	cytoquant( path , median_radius = 17 , exclude_spots = True , golog = True ) : extract the pixel values
	measuring the cytoplasmatic intensity of the protein of interest and of the cells carrying an RFP tag, 
	which are used to measure the autofluorescence intensity of the cell.

	cytoquant outputs two cdata objects: one contains the pixel values of the cytoplasm of the cells
	expressing the protein of interest; the other contains the pixel values measuring the autofluorescence.

	- path: is a string with the path to the folder containing the images to be analysed
	- median_radius: is used to remove brigth spots that might alter cell tresholding. These spots can 
	also be excluded from the quantification is excude_spots = True. The radius of the median filter 
	should be bigger than the size of the spots
	- exclude_spots: remove bright patches (for example, endocytic events) from the quantification of the
	cytoplasmatic concentration. Default is True
	- golog: work in the log space of the fluorescence intensities. Default is True
	"""

	# list images
	channels = load_images( path )

	reference_threshold = select_cytoplasm( channels[ 0 ] , median_radius , exclude_spots , threshold_image_name = refrence_threshold_mask )
	target_threshold = select_cytoplasm( channels[ 1 ] , median_radius , exclude_spots , ref_threshold = reference_threshold , threshold_image_name = target_threshold_mask )

	print( 'thresholds done' )
	if golog : 

		print( "golog = True; I'm working in the log space of the flurescence intensities" )

		reference_values = []
		for i in range( 1 , np.max( reference_threshold ) + 1 ) :

			# cell masks can have small pixel regions, usually at the edge of the cell (hence with low intensity)
			# that are left alone and are labelled as one thresholded object. These are not cells and need to be
			# discarded. I use the same criteria as to identify vacuels. Any object larger thant exp( 8 ) pixels
			# is considered a cell and it is eligible of analysis
			if np.log( len( reference_threshold[ reference_threshold == i ] ) ) > 8 :
				reference_values.append( np.median( np.log( channels[ 1 ][ reference_threshold == i ] ) / np.log( 2 ) ) )
		reference_all_values = np.log( channels[ 1 ][ reference_threshold > 0 ] ) / np.log( 2 ) 

		target_values = []
		for i in range( 1 , np.max( target_threshold ) + 1 ) :

			# cell masks can have small pixel regions, usually at the edge of the cell (hence with low intensity)
			# that are left alone and are labelled as one thresholded object. These are not cells and need to be
			# discarded. I use the same criteria as to identify vacuels. Any object larger thant exp( 8 ) pixels
			# is considered a cell and it is eligible of analysis
			if np.log( len( target_threshold[ target_threshold == i ] ) ) > 8 :
				target_values.append( np.median( np.log( channels[ 1 ][ target_threshold == i ] ) / np.log( 2 ) ) )
		target_all_values = np.log( channels[ 1 ][ target_threshold > 0 ] ) / np.log( 2 ) 

		r = np.median( reference_values )
		s_r = MAD( reference_values ) / np.sqrt( len( reference_values ) )
		t = np.median( target_values ) 
		s_t = MAD( target_values ) / np.sqrt( len( target_values ) )

		output = [
				2 ** t - 2 ** r ,
				np.sqrt( ( s_t * np.log( 2 ) *  2 ** t ) ** 2 + ( s_r * np.log( 2 ) * 2 ** r ) ** 2 ) 
				]
	else :

		reference_values = []
		
		for i in range( 1 , np.max( reference_threshold ) + 1 ) :
			
			# cell masks can have small pixel regions, usually at the edge of the cell (hence with low intensity)
			# that are left alone and are labelled as one thresholded object. These are not cells and need to be
			# discarded. I use the same criteria as to identify vacuels. Any object larger thant exp( 8 ) pixels
			# is considered a cell and it is eligible of analysis
			if np.log( len( reference_threshold[ reference_threshold == i ] ) ) > 8 :
				reference_values.append( np.median( channels[ 1 ][ reference_threshold == i ] ) )
		reference_all_values = channels[ 1 ][ refrence_threshold > 0 ]
		
		target_values = []
		
		for i in range( 1 , np.max( target_threshold ) + 1 ) :
			
			# cell masks can have small pixel regions, usually at the edge of the cell (hence with low intensity)
			# that are left alone and are labelled as one thresholded object. These are not cells and need to be
			# discarded. I use the same criteria as to identify vacuels. Any object larger thant exp( 8 ) pixels
			# is considered a cell and it is eligible of analysis
			if np.log( len( target_threshold[ target_threshold == i ] ) ) > 8 :
				target_values.append( np.median( channels[ 1 ][ target_threshold == i ] ) )
		target_all_values = channels[ 1 ][ target_threshold > 0 ]
	

		t = np.median( target_values ) 
		s_t = MAD( target_values ) / np.sqrt( len( target_values ) )
		r = np.median( reference_values )
		s_r = MAD( reference_values ) / np.sqrt( len( reference_values ) )
		
		output = [
				t - r  ,
				np.sqrt( s_t ** 2 + s_r ** 2 ) 
				]

	plt.figure()
	plt.hist( reference_all_values , normed = True , facecolor = 'g', alpha = 0.75 , label = 'cell autoFI' )
	#plt.hist( reference_values , normed = True , facecolor = 'b', alpha = 0.75 , label = 'cell autoFI' )
	plt.hist( target_all_values , normed = True , facecolor = 'r' , alpha = 0.75 , label = 'target prot. FI' )
	#plt.hist( target_values , normed = True , facecolor = 'k' , alpha = 0.75 , label = 'target prot. FI' )
	plt.xlabel( 'log( FI )' )
	plt.title( 'ratio = ' + str( output[ 0 ] ) )
	plt.legend( loc = 'best' )
	plt.savefig( plot_name + '.png' )

	return reference_values , target_values , output


