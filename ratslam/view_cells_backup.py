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

class ViewCell(object):
    '''A single view cell.

    A ViewCell object is used to store the information of a single view cell.
    '''
    _ID = 0

    def __init__(self, template, x_pc, y_pc, th_pc):
        '''Initialize a ViewCell.

        :param template: a 1D numpy array with the cell template.
        :param x_pc: the x position relative to the pose cell.
        :param y_pc: the y position relative to the pose cell.
        :param th_pc: the th position relative to the pose cell.
        '''
        self.id = ViewCell._ID
        self.template = template
        self.x_pc = x_pc
        self.y_pc = y_pc
        self.th_pc = th_pc
        self.decay = VT_ACTIVE_DECAY
        self.first = True
        self.exps = []

        ViewCell._ID += 1

class ViewCells(object):
    '''View Cell module.'''

    def __init__(self):
        '''Initializes the View Cell module.'''
        self.size = 0
        self.cells = []
        self.prev_cell = None
        self.truncation_factor = 16
                
        self.memory_access = 0
        self.memory_access_max = 0
        self.previous_matched = 0
        self.previous_matched_prev = 0
        
        # self.create_cell(np.zeros(561), 30, 30, 18)

    def _create_template(self, img):
        '''Compute the sum of columns in subimg and normalize it.

        :param subimg: a sub-image as a 2D numpy array.
        :return: the view template as a 1D numpy array.
        '''
        subimg = img[IMAGE_VT_Y_RANGE, IMAGE_VT_X_RANGE]
        
#         # Quantization
#         subimg = subimg - np.remainder(subimg, 8)
#         #print subimg
        
        x_sums = np.sum(subimg, 0)

        # quantization (4bit 160*16)
        x_sums = (x_sums - np.remainder(x_sums, 160*16)) /(160*16)

        x_tmp = np.zeros(x_sums.size/self.truncation_factor)
        
        for j in range(0,x_sums.size/self.truncation_factor):
            max_val = 0
            for i in range(0,self.truncation_factor):
                    if (x_sums[j*self.truncation_factor+i]>max_val):
                        max_val = x_sums[j*self.truncation_factor+i]
                    x_tmp[j] = max_val

        x_sums = copy.deepcopy(x_tmp)

        return x_sums
#         return x_sums/np.sum(x_sums, dtype=np.float32)

    def _score(self, template):
        '''Compute the similarity of a given template with all view cells.

        :param template: 1D numpy array.
        :return: 1D numpy array.
        '''
        scores = []
        for cell in self.cells:

            cell.decay -= VT_GLOBAL_DECAY
            if cell.decay < 0:
                cell.decay = 0
            # slen is divided by the truncation factor
#             _, s = compare_segments(template, cell.template, VT_SHIFT_MATCH/self.truncation_factor)
#             _, s = compare_segments(template, cell.template, 1)
            _, s = compare_segments(template, cell.template, 0)
      
            scores.append(s)

        return scores

    def create_cell(self, template, x_pc, y_pc, th_pc):
        '''Create a new View Cell and register it into the View Cell module

        :param template: 1D numpy array.
        :param x_pc: index x of the current pose cell.
        :param y_pc: index y of the current pose cell.
        :param th_pc: index th of the current pose cell.
        :return: the new View Cell.
        '''
        cell = ViewCell(template, x_pc, y_pc, th_pc)
        self.cells.append(cell)
        self.size += 1
        return cell

    def __call__(self, img, x_pc, y_pc, th_pc):
        '''Execute an interation of visual template.

        :param img: the full gray-scaled image as a 2D numpy array.
        :param x_pc: index x of the current pose cell.
        :param y_pc: index y of the current pose cell.
        :param th_pc: index th of the current pose cell.
        :return: the active view cell.
        '''
        template = self._create_template(img)
        scores = self._score(template)
        
#         if (self.size >= 1):
#             _, s = compare_segments_vt(template, self.cells[np.argmin(scores)].template, 0)

        if not self.size or np.min(scores)*template.size > VT_MATCH_THRESHOLD :
            cell = self.create_cell(template, x_pc, y_pc, th_pc)
            self.prev_cell = cell
            
            self.previous_matched = np.size(scores)
            self.memory_access += np.size(scores)
            self.memory_access_max += np.size(scores)
#             print scores
            
            return cell
        
# # First search -- NOT GOOD, origin
#         i = 0
#         j = 0
#         for scr in scores:
#             if scr*template.size <= 16.0 :
#                     cell = self.cells[i]
#                     j=1
#                     break
#             i = i+1
            
#         if (j==0):
#             i = np.argmin(scores)
#             cell = self.cells[i]
# #             print 'longer scan'
# #         else:
# #             print 'shorter scan'

# First search -- NOT GOOD
        i = 0
        j = 0
        index_score = 0
        index_scan = 0
        
#         if(self.previous_matched_prev > self.previous_matched):
#             index_scan = self.previous_matched
            
        index_scan = self.previous_matched
            
        self.previous_matched_prev = self.previous_matched
        
        for index_score in range(index_scan,np.size(scores)) + range(0,index_scan):
            if scores[index_score]*template.size <= 16.0 :
                    cell = self.cells[index_score]
                    j=1
                    self.previous_matched = index_score
                    self.memory_access += i
                    break
            i = i+1
            
        if (j==0):
            i = np.argmin(scores)
            cell = self.cells[i]
            self.previous_matched = i
            self.memory_access += np.size(scores)
#             print 'longer scan'
#         else:
#             print 'shorter scan'

        
        self.memory_access_max += np.size(scores)
        print 'memory_access', self.memory_access
        print 'memory_max', self.memory_access_max
        
            
#         print 'matched index', i
            
                
# optimal search            
#         i = np.argmin(scores)
#         cell = self.cells[i]

#         print 'The matched VT diff', scores[i]*template.size
    
        cell.decay += VT_ACTIVE_DECAY

        if self.prev_cell != cell:
            cell.first = False

        self.prev_cell = cell
        return cell
