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
from skimage.exposure import rescale_intensity
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
		#im = rescale_intensity( im , in_range = ( 0 , 2**12 - 1 ) , out_range = 'uint16' ) #DEBUG

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

def threshold_cytoplasm( im , median_radius , exclude_spots ) : 

	im_spots = cp.deepcopy( im )
	threshold_image = cp.deepcopy( im )
	threshold_image[ : ] = 0

	for i in range( im.shape[ 0 ] ) : #filter images individually

		im_frame = im[ i , : , : ]
		threshold_image_frame = threshold_image[ i , : , : ]
		# Compute the median filter of the image, which will be used to select the spots
		im_median = filters.median( im_frame , morphology.disk( median_radius ) )

		# Make a threshold image, that will be used to output the pixel values.
		# The threshold image is the combination of the thresholds on the raw 
		# and median filtered image.
		threshold_raw = filters.threshold_otsu( im_frame )
		threshold_median = filters.threshold_otsu( im_median )
	
		threshold_image_frame[ ( im_frame > threshold_median ) & ( im_frame > threshold_raw ) ] = 1
	
		# Exclude the spots that have been thresholded
		if exclude_spots :

			im_spots_tmp = im_spots[ i , : , : ]

			# Isolate pixels brighter than the median. These will be spots
			im_spots_tmp[ im_frame <= im_median ] = 0
		
			# Remove the median value from the isolated spots as a 
			# measure of the local cytoplasmatic background. This 
			# image of the spots will be used to theshorld the spots.
			im_spots_tmp[ im_spots_tmp == im_frame ] = im_spots_tmp[ im_spots_tmp == im_frame ] - im_median[ im_spots_tmp == im_frame ]
	
			threshold_spots = filters.threshold_yen( im_spots_tmp )
			
			im_spots_tmp[ im_spots_tmp < threshold_spots ] = 0
			im_spots_tmp[ im_spots_tmp >= threshold_spots ] = 1
			
			#spot dilation to limit the influence of the pixel intensity in the spots
			im_spots_tmp = morphology.dilation( im_spots_tmp , selem = morphology.disk( 3 ) )

#DEBUG		if ( ref_threshold == [] ) & ( i == 2 ) :
#DEBUG			tiff.imsave( 'tmp-mask.tif' , threshold_image_frame )
#DEBUG			tiff.imsave( 'tmp-raw.tif' , im_frame )
#DEBUG			tiff.imsave( 'tmp.tif' , im_median )
	
	#remove all the spots from the thresholded image
	threshold_image[ im_spots == 1 ] = 0

	return threshold_image

def select_cytoplasm( target , reference , threshold_image_name = 'threshold.tif') : 

	t = cp.deepcopy( target )
	r = cp.deepcopy( reference ) 

	ID = 1 #ID of thresholded cells cannot start at 0, as that' the background.
	
	for i in range( t.shape[ 0 ] ) :

		t_frame = t[ i , : , : ] 
		r_frame = r[ i , : , : ] 

		lb = label( t_frame )

		for j in range( np.max( lb ) ) :
			
			t_area = len( lb[ lb == j + 1 ] )

			# set the area threshold limit to be exp( 7 ). If cells
			# are smaller they are likely errors in the thresholding
			# and they are removed. If they are bigger, they are kept
			# and they are assigned an id, but only if the reference (r)
			# and target (t) images do not overlap.
			if np.log( t_area ) < 7 :

				t_frame[ lb == ( j + 1 ) ] = 0

			elif np.max( r_frame[ lb == ( j + 1 ) ] ) > 0 :
				
				t_frame[ lb == ( j + 1 ) ] = 0 

			else : 
				
				t_frame[ lb == ( j + 1 ) ] = ID
				ID = ID + 1

	tiff.imsave( threshold_image_name , t )
	
	return  t

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

	reference_threshold_tmp = threshold_cytoplasm( channels[ 0 ] , median_radius , exclude_spots )
	target_threshold_tmp = threshold_cytoplasm( channels[ 1 ] , median_radius , exclude_spots )

	for i in range( reference_threshold_tmp.shape[ 0 ] ) :	
	
		try :
		
			# erode the threshold_image. The threshold image selects the pixels 
			# for the quantification. The erosion is a conservative measure to 
			# reduce the likelihood of quantifying pixels that are outside the 
			# cell.
			reference_threshold_tmp[ i , : , : ] = morphology.erosion( reference_threshold_tmp[ i , : , : ] , selem = morphology.disk( 3 ) )
			target_threshold_tmp[ i , : , : ] = morphology.erosion( target_threshold_tmp[ i , : , : ] , selem = morphology.disk( 3 ) )
		
		except :
			
			print( 'reference and threshold masks might differ in size' )
			raise

	#TO DO add one round of erosion over a copy of t_frame, to remove dust and sspeed up the thresholding
	tiff.imsave( 'reference_threshold_tmp.tif' , reference_threshold_tmp )
	tiff.imsave( 'target_threshold_tmp.tif' , target_threshold_tmp )
	
	reference_threshold = select_cytoplasm( reference_threshold_tmp , target_threshold_tmp , threshold_image_name = reference_threshold_mask )
	target_threshold = select_cytoplasm( target_threshold_tmp , reference_threshold_tmp , threshold_image_name = target_threshold_mask )
	
	print( '--------thresholds done--------' )
	if golog : 

		print( "golog = True; I'm working in the log space of the flurescence intensities" )

		reference_values = []
		for i in range( 1 , np.max( reference_threshold ) + 1 ) :

			# cell masks can have small pixel regions, usually at the edge of the cell (hence with low intensity)
			# that are left alone and are labelled as one thresholded object. These are not cells and need to be
			# discarded. I use the same criteria as to identify vacuels. Any object larger thant exp( 8 ) pixels
			# is considered a cell and it is eligible of analysis
			if np.log( len( reference_threshold[ reference_threshold == i ] ) ) > 0 :
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


