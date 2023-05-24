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

'''
This is a full Ratslam implementation in python. This implementation is based 
on Milford's original implementation [1]_ in matlab, and Christine Lee's python 
implementation [2]_. The original data movies can also be found in [1]_.

This file is the only dependent of OpenCV, which is used to open and convert 
the movie files. Thus, you can change only this file to use other media API.

.. [1] https://wiki.qut.edu.au/display/cyphy/RatSLAM+MATLAB
.. [2] https://github.com/coxlab/ratslam-python
'''

import cv2
import numpy as np
from matplotlib import pyplot as plot
import matplotlib.image as mpimg
from matplotlib import rc,rcParams

from matplotlib.font_manager import FontProperties
import mpl_toolkits.mplot3d.axes3d as p3

import ratslam

from ratslam._globals import *

import time

if __name__ == '__main__':
    # Change this line to open other movies
    data = r'D:\Python_workspace\Klaus_short_retake10.avi'
    print 'The name of the dataset is', data
    video = cv2.VideoCapture(data)
    if video.isOpened():
        print 'The data set is successfully opened'
    else:
        print 'Fail to open the dataset'
        
 
    slam = ratslam.Ratslam()
    
    print 'The value of PC_VT_INJECT_ENERGY is', PC_VT_INJECT_ENERGY
    print 'The value of PC_DIM_XY is', PC_DIM_XY
    print 'The vaule of PC_DIM_TH is', PC_DIM_TH
    print 'The value of PC_W_E&I_DIM is', PC_W_E_DIM

    print 'The vaule of EXP_DELTA_PC_THRESHOLD is', EXP_DELTA_PC_THRESHOLD
    print 'The value of VT_MATCH_THRESHOLD is', VT_MATCH_THRESHOLD 
    print 'The value of FOV_DEG is', FOV_DEG
    print 'The value of VTRANS_SCALE is', VTRANS_SCALE

    print 'The quantization on img is 8'      
    print 'The simulation is started at', time.strftime("%m/%d/%y %H:%M:%S")
       
    exp_shape_prev = 0
        
    start_time = time.time() 
    
    loop = 0
    _, frame = video.read()
    while True:
        loop += 1

        # RUN A RATSLAM ITERATION ==================================
        _, frame = video.read()
        if frame is None: 
            print 'none' 
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        slam.digest(img)
        # ==========================================================

#         # test output
#         if (loop >= 1700) and (loop < 1800):
#             xs = []
#             ys = []
#             for exp in slam.experience_map.exps:
#                 xs.append(exp.x_m)
#                 ys.append(exp.y_m)

#             with open('results\\map2\\'+'map' +'%05d'%loop + '.txt', 'w') as outfile:
#                 outfile.write('{0}\n'.format(xs))
#                 outfile.write('{0}\n'.format(ys))     

    # exp map recording
        xs = []
        ys = []
        for exp in slam.experience_map.exps:
            xs.append(exp.x_m)
            ys.append(exp.y_m)
        with open('results\\map\\'+'map' +'%05d'%loop + '.txt', 'w') as outfile:
            outfile.write('[')
            outfile.write('{0}\n'.format(xs))
            outfile.write('{0}\n'.format(ys))
            outfile.write(']')

        


        # view cell count        
        with open('results\\Viewcell\\'+'viewcell_count' +'.txt', 'a+') as outfile:
            outfile.write('{0}\t'.format(loop))
            outfile.write('{0}\n'.format(len(slam.view_cells.cells)))     
        
        # Plot each 100 frames
        if loop%50 != 0:
            continue
        
        print '-----------------------------------------------------------'
        print 'The current number of the iteration is', loop
        print("%s seconds are elapsed per 100 frames." %(time.time() - start_time))
        print 'The number of units of the VT is', len(slam.view_cells.cells)
        print 'The number of units of the experience map is', len(slam.experience_map.exps)
        print time.strftime("%m/%d/%y %H:%M:%S")
        print '-----------------------------------------------------------'
        
        start_time = time.time() 
        start_time_plot = start_time
        # PLOT THE CURRENT RESULTS =================================
        b, g, r = cv2.split(frame)
        rgb_frame = cv2.merge([r, g, b])

        plot.clf()

        # RAW IMAGE -------------------
        ax = plot.subplot(2, 2, 1)
        plot.title('Raw input image', fontname='Arial', fontweight='bold')
        plot.imshow(rgb_frame)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        # -----------------------------

        # RAW ODOMETRY ----------------
        ax = plot.subplot(2, 2, 2)
        plot.title('Raw odometry', fontname='Arial', fontweight='bold')
        plot.axis([-1000, 1000, -1000, 1000])
        plot.xticks(fontname = "Arial", fontweight='bold')
        plot.yticks(fontname = "Arial", fontweight='bold')
        plot.plot(slam.odometry[0], slam.odometry[1])
        plot.plot(slam.odometry[0][-1], slam.odometry[1][-1], 'ko')
        #------------------------------

        # POSE CELL ACTIVATION --------
#         ax = plot.subplot(2, 2, 3)
#         ax.set_axis_off()
#         plot.title('POSE CELL ACTIVATION')        
#         img_pc = mpimg.imread(r'D:\Python_workspace\ratslam-Klaus_XY55_2d_v2_quant2_Forum\MATLAB\Posecell_colormap\Posecell_colormap_cropped'+str(loop/100)+'.png')
#         plot.imshow(img_pc)
        
        ax = plot.subplot(2, 2, 3, projection='3d')     
        plot.title('Pose-cell activation', fontname='Arial', fontweight='bold')     
        x, y, th = slam.pc
        ax.plot(x[-50:-1], y[-50:-1], 'x')
        ax.plot3D([0, PC_DIM_XY], [y[-1], y[-1]], [th[-1], th[-1]], 'K')
        ax.plot3D([x[-1], x[-1]], [0, PC_DIM_XY], [th[-1], th[-1]], 'K')
        ax.plot3D([x[-1], x[-1]], [y[-1], y[-1]], [0, 4], 'K')
        ax.plot3D([x[-1]], [y[-1]], [th[-1]], 'mo')

        
        ax.grid()
        ax.axis([0, PC_DIM_XY, PC_DIM_XY, 0]);
        ax.set_zlim(0, 4)
        plot.xticks(fontname = "Arial", fontweight='bold')
        plot.yticks(fontname = "Arial", fontweight='bold')
        # -----------------------------

        # EXPERIENCE MAP --------------
        plot.subplot(2, 2, 4)
        plot.title('Experience map', fontname='Arial', fontweight='bold')
        xs = []
        ys = []
        for exp in slam.experience_map.exps:
            xs.append(exp.x_m)
            ys.append(exp.y_m)
        
        exp_shape_current = np.prod(np.shape(xs))
        plot.axis([-1000, 1000, -1000, 1000])
        plot.xticks(fontname = "Arial", fontweight='bold')
        plot.yticks(fontname = "Arial", fontweight='bold')
        plot.plot(xs, ys, 'b')
        plot.plot(xs[(exp_shape_prev-exp_shape_current):-1], ys[(exp_shape_prev-exp_shape_current):-1], 'r')
#         plot.plot(xs[exp_shape_prev-exp_shape_current], ys[exp_shape_prev-exp_shape_current], 'ks')
        plot.plot(slam.experience_map.current_exp.x_m,
                  slam.experience_map.current_exp.y_m, 'ko')
        
        exp_shape_prev = np.prod(np.shape(xs))
        
        # -----------------------------

        plot.tight_layout()
        plot.savefig('results\\' + '%05d.jpg'%loop, dpi=300)
        plot.pause(0.1)
        # ==========================================================
        
        print '-----------------------------------------------------------'
        print 'test fig'
        print '-----------------------------------------------------------'
        
        
        # EXPERIENCE MAP_test --------------
        plot.figure(figsize=(6, 6))
        plot.title('EXPERIENCE MAP')
        xs = []
        ys = []
        for exp in slam.experience_map.exps:
            xs.append(exp.x_m)
            ys.append(exp.y_m)

        plot.axis([-2000, 2000, -2000, 2000])
        plot.plot(xs, ys, 'b')        
        
#         with open('results\\map\\'+'map' +'%05d'%loop + '.txt', 'w') as outfile:
#             outfile.write('{0}\n'.format(xs))
#             outfile.write('{0}\n'.format(ys))
#             np.savetxt(outfile, cell_print, fmt='%.6f')
        
        
        # -----------------------------
        
        plot.tight_layout()
        plot.savefig('results\\' + '%05d_test.jpg'%loop, dpi=300)
        plot.pause(0.1)
        # ==========================================================
       
        

    print 'DONE!'
    print 'n_ templates:', len(slam.view_cells.cells)
    print 'n_ experiences:', len(slam.experience_map.exps)
    plot.show()