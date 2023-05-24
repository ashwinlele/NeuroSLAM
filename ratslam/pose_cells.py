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
import itertools
from ratslam._globals import *

class PoseCells(object):
    '''Pose Cell module.'''

    def __init__(self):
        '''Initializes the Pose Cell module.'''

        # 4 -> 45 degree // 2-> 90 degree
        self.theta_resol = 2
        self.cells = np.zeros([PC_DIM_XY, PC_DIM_XY])
        self.active = a, b, c = [PC_DIM_XY/2, PC_DIM_XY/2, self.theta_resol]
        
        self.active = a, b, c = [3, 3, 3]
        
        self.cells[a, b] = 1
        
        self.max_pc = 0
        self.vtrans_acc = 0
        self.vrot_acc = 2*(np.pi/2)
        self.th_layer = np.zeros([2])

        self.counter = 0
        self.index_einj = 0

    def posecell_quantization(self):
        # incorrect mod in the float number computation
#         self.cells = self.cells - self.cells % 0.004
        # correct mod
        self.cells = (1.000/16.000)*np.floor(self.cells/(1.000/16.000))
        
        
    def compute_activity_matrix(self, xywrap, thwrap, wdim, pcw): 
        '''Compute the activation of pose cells.'''
        
        # The goal is to return an update matrix that can be added/subtracted
        # from the posecell matrix
        pca_new = np.zeros([PC_DIM_XY, PC_DIM_XY])
        
        # for nonzero posecell values  
        indices = np.nonzero(self.cells)

        for i,j in itertools.izip(*indices):
            pca_new[np.ix_(xywrap[i:i+wdim], 
                           xywrap[j:j+wdim])] += self.cells[i,j]*pcw
         
        return pca_new


    def get_pc_max(self, xywrap):
        '''Find the x, y, th center of the activity in the network.'''
        
        pc_max_cells = (1.000/16.000)*np.floor(self.cells/(1.000/16.000))
        
#         x, y, z = np.unravel_index(np.argmax(self.cells), self.cells.shape)
        x, y = np.unravel_index(np.argmax(pc_max_cells), self.cells.shape)
        th = np.round(self.vrot_acc / (np.pi/self.theta_resol))
        if (th == 2*self.theta_resol):
            th = 0

        return (x, y, th)

    def __call__(self, view_cell, vtrans, vrot):
        '''Execute an interation of pose cells.

        :param view_cell: the last most activated view cell.
        :param vtrans: the translation of the robot given by odometry.
        :param vrot: the rotation of the robot given by odometry.
        :return: a 3D-tuple with the (x, y, th) index of most active pose cell.
        '''
        self.counter += 1
            
        
        vtrans = vtrans*POSECELL_VTRANS_SCALING
        vrot = vrot*1.0
        
#         for the test 
#         vtrans = 0
#         vrot = 0
        
#         print vtrans
        self.vtrans_acc += vtrans
        self.vrot_acc += vrot
        
        if (self.vtrans_acc > 1):
            vtrans = 1
            self.vtrans_acc -= 1 
        else:
            vtrans = 0
            
        if (self.vrot_acc >= 2*np.pi):
            self.vrot_acc -= 2*np.pi
        elif (self.vrot_acc < 0):
            self.vrot_acc += 2*np.pi


        # if this isn't a new vt then add the energy at its associated posecell
        # location
        if not view_cell.first:
            print 'The counter is', self.counter    
    
#             if (int(np.round(view_cell.x_pc)) == PC_DIM_XY):
#                 act_x = 0
#             else:
#                 act_x = int(np.round(view_cell.x_pc))
                
#             if (int(np.round(view_cell.y_pc)) == PC_DIM_XY):
#                 act_y = 0
#             else:
#                 act_y = int(np.round(view_cell.y_pc))  
            act_x = view_cell.x_pc
            act_y = view_cell.y_pc

            
            if (self.vrot_acc - view_cell.th_pc * (np.pi/self.theta_resol) > np.pi):
                self.vrot_acc = 0.5 * self.vrot_acc + 0.5 * (view_cell.th_pc * (np.pi/self.theta_resol) + 2*np.pi)
            elif (self.vrot_acc - view_cell.th_pc * (np.pi/self.theta_resol) < -np.pi):
                self.vrot_acc = 0.5 * self.vrot_acc + 0.5 * (view_cell.th_pc * (np.pi/self.theta_resol) - 2*np.pi)        
            else:
                self.vrot_acc = 0.5 * self.vrot_acc + 0.5 * view_cell.th_pc * (np.pi/self.theta_resol)
                
            if (self.vrot_acc >= 2*np.pi):
                self.vrot_acc -= 2*np.pi
            elif (self.vrot_acc < 0):
                self.vrot_acc += 2*np.pi

            print 'The angle is reset to', self.vrot_acc*180/np.pi            
            
            print 'Energy injection at', [act_x, act_y]
#             self.cells[self.cells < 0.003*8] = 0
#             self.cells[self.cells >= 0.003*8] -= 0.003*8
            
#             self.cells[self.cells < 0.003*32] = 0
#             self.cells[self.cells >= 0.003*32] -= 0.003*32

# 0.3 origin
            if (self.counter == self.index_einj + 1):
                self.cells[self.cells < 0.2] = 0
                self.cells[self.cells >= 0.2] -= 0.2
            else:
                self.cells[self.cells < 0.2] = 0
                self.cells[self.cells >= 0.2] -= 0.2
            

            self.index_einj = self.counter
            
            self.cells[act_x, act_y] = 1       

            

        #===============================
#         if (self.counter%4 == 0):
#             self.cells[self.cells < 0.2] = 0
#             self.cells[self.cells >= 0.2] -= 0.2
#             self.cells[0, 0] = 1
#         elif (self.counter == 13 or self.counter == 14 or self.counter == 15 or self.counter == 18):
#             self.cells[self.cells < 0.2] = 0
#             self.cells[self.cells >= 0.2] -= 0.2
#             self.cells[0, 0] = 1

#         if (self.counter == 8 or self.counter == 9 or self.counter == 10 or self.counter == 11):
#             self.cells[self.cells < 0.2] = 0
#             self.cells[self.cells >= 0.2] -= 0.2
#             self.cells[5, 3] = 1
#         if (self.counter == 16 or self.counter == 17 or self.counter == 18 ):
#             self.cells[self.cells < 0.2] = 0
#             self.cells[self.cells >= 0.2] -= 0.2
#             self.cells[0, 0] = 1

            
        self.posecell_quantization()
       
            
        # local excitation and inhibition
        self.cells = self.compute_activity_matrix(PC_E_XY_WRAP, 
                                                  PC_E_TH_WRAP, 
                                                  PC_W_E_DIM, 
                                                  PC_W_EXCITE)
        
#         self.posecell_quantization()
        
        # local global inhibition - PC_gi = PC_li elements - inhibition
#         self.cells[self.cells < 0.003*4] = 0
#         self.cells[self.cells >= 0.003*4] -= 0.003*4
        
        self.cells[self.cells < 0.012*4] = 0
        self.cells[self.cells >= 0.012*4] -= 0.012*4
        
#         print self.cells
        
        # energy preservation
#         self.cells[self.cells >= 0.012*4] += 0.096 *3
        self.cells[self.cells >= 1.0/32.0] += 0.35

#         self.posecell_quantization()


        if (np.max(self.cells) > self.max_pc):
            self.max_pc = np.max(self.cells)
            print 'The counter is', self.counter
            print 'The maximum pc value is', self.max_pc
            
        if (np.max(self.cells) < 0.001):
            print 'The counter is', self.counter
            print 'The energy of the pose cell diminished'

        # Path Integration
        # vtrans affects xy direction
        # shift in each th given by the th
        
        if (vtrans == 1):
            for dir_pc in xrange(PC_DIM_TH): 
                self.th_layer[0] += np.cos(np.round(self.vrot_acc / (np.pi/self.theta_resol)) * (np.pi/self.theta_resol))
                self.th_layer[1] += np.sin(np.round(self.vrot_acc / (np.pi/self.theta_resol)) * (np.pi/self.theta_resol))
                
                # y direction (cos from the original code)
                if self.th_layer[0] >= 1:
                    # y increase
                    self.cells[:,:] = np.roll(self.cells[:,:], 1, 1)
                    self.th_layer[0] -= 1
                elif self.th_layer[0] <= -1:
                    # y decrease
                    self.cells[:,:] = np.roll(self.cells[:,:], -1, 1)
                    self.th_layer[0] += 1
                
                # x direction (sin from the original code)
                if self.th_layer[1] >= 1:
                    # x increase
                    self.cells[:,:] = np.roll(self.cells[:,:], 1, 0)
                    self.th_layer[1] -= 1
                elif self.th_layer[1] <= -1:
                    # x decrease
                    self.cells[:,:] = np.roll(self.cells[:,:], -1, 0)
                    self.th_layer[1] += 1

#         self.posecell_quantization()
        
        # test
        self.active = self.get_pc_max(PC_AVG_XY_WRAP)
        
        max_index = np.argwhere(self.cells == np.amax(self.cells))
        x = max_index.flatten().tolist()[0::2]
        y = max_index.flatten().tolist()[1::2]

        if (len(x) > 2):
            print 'counter', self.counter
            print 'multiple peak # >2'
            print x,y 
            
        cell_print = np.swapaxes(self.cells,1,0)

        index_write = 0
        # Write the array to disk
        with open('results\\Posecell\\'+'Posecell' + str(self.counter) + '.txt', 'w') as outfile:
            outfile.write('# Array shape: {0}\n'.format(cell_print.shape))
            outfile.write('PC position : {0}\n'.format(self.active))
            np.savetxt(outfile, cell_print, fmt='%.6f')
            
        ####

        
        return self.active
    