# =============================================================================
# Federal University of Rio Grande do Sul (UFRGS)
# Connectionist Artificial Intelligence Laboratory (LIAC)
# Renato de Pontes Pereira - rppereira@inf.ufrgs.br
# =============================================================================
# Copyright (c) 2013 Renato de Pontes Pereira, renato.ppontes at gmail dot com
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy 
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights 
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
# copies of the Software, and to permit persons to whom the Software is 
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in 
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# =============================================================================

import numpy as np
import copy
from ratslam._globals import *

class VisualOdometry(object):
    '''Visual Odometry Module.'''

    def __init__(self):
        '''Initializes the visual odometry module.'''
        self.old_vtrans_template = np.zeros(IMAGE_ODO_X_RANGE.stop)
        self.old_vrot_template = np.zeros(IMAGE_ODO_X_RANGE.stop)
        self.odometry = [0., 0., np.pi/2]
        
        self.truncation_factor_vtrans = 16
        self.truncation_factor_vrot = 16
        self.offset_max = 0;

        
    def _create_template(self, subimg,trunc_factor):
        '''Compute the sum of columns in subimg and normalize it.

        :param subimg: a sub-image as a 2D numpy array.
        :return: the view template as a 1D numpy array.
        '''
        x_sums = np.sum(subimg, 0)
        
        # quantization (4 bit 160 * 16)
        x_sums = (x_sums - np.remainder(x_sums, 160*16)) /(160*16)

        x_tmp = np.zeros(x_sums.size/trunc_factor)
        
        for j in range(0,x_sums.size/trunc_factor):
            max_val = 0
            for i in range(0,trunc_factor):
                    if (x_sums[j*trunc_factor+i]>max_val):
                        max_val = x_sums[j*trunc_factor+i]
                    x_tmp[j] = max_val

        x_sums = copy.deepcopy(x_tmp)

        return x_sums

    def __call__(self, img):
        '''Execute an interation of visual odometry.

        :param img: the full gray-scaled image as a 2D numpy array.
        :return: the deslocation and rotation of the image from the previous 
                 frame as a 2D tuple of floats.
        '''
        subimg = img[IMAGE_VTRANS_Y_RANGE, IMAGE_ODO_X_RANGE]
        template = self._create_template(subimg, self.truncation_factor_vtrans)

        # slen parameter is divided by the truncation factor
        # VTRANS
        offset, diff = compare_segments_odo(
            template, 
            self.old_vtrans_template, 
            VISUAL_ODO_SHIFT_MATCH/self.truncation_factor_vtrans 
        )
        # post process
#         diff = diff / 9.765625

        diff = diff / 10
        
        vtrans = diff*VTRANS_SCALE
        
        if vtrans > 10: 
            print 'over_vtrans', vtrans
            vtrans = 0

        if (offset == 8):
            offset = 7
        if (self.offset_max < np.abs(offset)):
            self.offset_max = np.abs(offset)
            print 'max_offset', offset
        
#         print offset
        vrot = offset*(50./img.shape[1])*np.pi/180/50*FOV_DEG
        # multiplication of the truncation factor
        vrot = vrot * self.truncation_factor_vrot
# prev save
        self.old_vtrans_template = template
        
        # Update raw odometry
        self.odometry[2] += vrot 
        self.odometry[0] += vtrans*np.cos(self.odometry[2])
        self.odometry[1] += vtrans*np.sin(self.odometry[2])

        return vtrans, vrot