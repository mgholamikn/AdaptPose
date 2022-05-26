"""
Functions to visualize human poses
"""
import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D

def show3DposePair(realt3d, faket3d, ax, lcolor="#3498db", rcolor="#e74c3c", add_labels=True,
                   gt=True, pred=False):  # blue, orange
  """
  Visualize a 3d skeleton pair

  Args
  channels: 96x1 vector. The pose to plot.
  ax: matplotlib 3d axis to draw on
  lcolor: color for left part of the body
  rcolor: color for right part of the body
  add_labels: whether to add coordinate labels
  Returns
  Nothing. Draws on ax.
  """
  #   assert channels.size == len(data_utils.H36M_NAMES)*3, "channels should have 96 entries, it has %d instead" % channels.size
  realt3d = np.reshape(realt3d, (16, -1))
  faket3d = np.reshape(faket3d, (16, -1))

  I = np.array([0, 1, 2, 0, 4, 5, 0, 7, 8, 8, 10, 11, 8, 13, 14])  # start points
  J = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])  # end points
  LR = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)

  for idx, vals in enumerate([realt3d, faket3d]):
    # Make connection matrix
    for i in np.arange(len(I)):
      x, y, z = [np.array([vals[I[i], j], vals[J[i], j]]) for j in range(3)]
      if idx == 0:
        ax.plot(x, z, -y, lw=2, c='k')
      #        ax.plot(x,y, z,  lw=2, c='k')

      elif idx == 1:
        ax.plot(x, z, -y, lw=2, c='r')
      #        ax.plot(x,y, z,  lw=2, c='r')

      else:
        #        ax.plot(x,z, -y,  lw=2, c=lcolor if LR[i] else rcolor)
        ax.plot(x, y, z, lw=2, c=lcolor if LR[i] else rcolor)

  RADIUS = 1  # space around the subject
  xroot, yroot, zroot = vals[0,0], vals[0,1], vals[0,2]
  ax.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
  ax.set_ylim3d([-RADIUS+zroot, RADIUS+zroot])
  ax.set_zlim3d([-RADIUS-yroot, RADIUS-yroot])

  if add_labels:
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_zlabel("-y")

  # Get rid of the ticks and tick labels
  #  ax.set_xticks([])
  #  ax.set_yticks([])
  #  ax.set_zticks([])
  #
  #  ax.get_xaxis().set_ticklabels([])
  #  ax.get_yaxis().set_ticklabels([])
  #  ax.set_zticklabels([])
  #     ax.set_aspect('equal')

  # Get rid of the panes (actually, make them white)
  white = (1.0, 1.0, 1.0, 0.0)
  ax.w_xaxis.set_pane_color(white)
  ax.w_yaxis.set_pane_color(white)
  # Keep z pane

  # Get rid of the lines in 3d
  ax.w_xaxis.line.set_color(white)
  ax.w_yaxis.line.set_color(white)
  ax.w_zaxis.line.set_color(white)


def show3Dpose(channels, ax, lcolor="#3498db", rcolor="#e74c3c", add_labels=True,
               gt=False,pred=False): # blue, orange
    """
    Visualize a 3d skeleton

    Args
    channels: 96x1 vector. The pose to plot.
    ax: matplotlib 3d axis to draw on
    lcolor: color for left part of the body
    rcolor: color for right part of the body
    add_labels: whether to add coordinate labels
    Returns
    Nothing. Draws on ax.
    """

    #   assert channels.size == len(data_utils.H36M_NAMES)*3, "channels should have 96 entries, it has %d instead" % channels.size
    vals = np.reshape( channels, (16, -1) )

    I  = np.array([0,1,2,0,4,5,0,7,8,8,10,11,8,13,14]) # start points
    J  = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]) # end points
    LR = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)

    # Make connection matrix
    for i in np.arange( len(I) ):
        x, y, z = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(3)]
        if gt:
            ax.plot(x,z, -y,  lw=2, c='k')
        #        ax.plot(x,y, z,  lw=2, c='k')

        elif pred:
            ax.plot(x,z, -y,  lw=2, c='r')
        #        ax.plot(x,y, z,  lw=2, c='r')

        else:
        #        ax.plot(x,z, -y,  lw=2, c=lcolor if LR[i] else rcolor)
            ax.plot(x, z, -y,  lw=2, c=lcolor if LR[i] else rcolor)

    RADIUS = 1 # space around the subject
    xroot, yroot, zroot = vals[0,0], vals[0,1], vals[0,2]
    ax.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
    ax.set_ylim3d([-RADIUS+zroot, RADIUS+zroot])
    ax.set_zlim3d([-RADIUS-yroot, RADIUS-yroot])


    if add_labels:
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.set_zlabel("-y")

    # Get rid of the ticks and tick labels
    #  ax.set_xticks([])
    #  ax.set_yticks([])
    #  ax.set_zticks([])
    #
    #  ax.get_xaxis().set_ticklabels([])
    #  ax.get_yaxis().set_ticklabels([])
    #  ax.set_zticklabels([])
#     ax.set_aspect('equal')

    # Get rid of the panes (actually, make them white)
    white = (1.0, 1.0, 1.0, 0.0)
    ax.w_xaxis.set_pane_color(white)
    ax.w_yaxis.set_pane_color(white)
    # Keep z pane

    # Get rid of the lines in 3d
    ax.w_xaxis.line.set_color(white)
    ax.w_yaxis.line.set_color(white)
    ax.w_zaxis.line.set_color(white)

def show2Dpose(channels, ax, lcolor="#3498db", rcolor="#e74c3c", add_labels=True):
  """
  Visualize a 2d skeleton

  Args
  channels: 64x1 vector. The pose to plot.
  ax: matplotlib axis to draw on
  lcolor: color for left part of the body
  rcolor: color for right part of the body
  add_labels: whether to add coordinate labels
  Returns
  Nothing. Draws on ax.
  """
  vals = np.reshape(channels, (-1, 2))
  # plt.plot(vals[:,0], vals[:,1], 'ro')
  I = np.array([0, 1, 2, 0, 4, 5, 0, 7, 8, 8, 10, 11, 8, 13, 14])  # start points
  J = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])  # end points
  LR = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)

  # Make connection matrix
  for i in np.arange(len(I)):
    x, y = [np.array([vals[I[i], j], vals[J[i], j]]) for j in range(2)]
    #         print('x',x)
    #         print(y)
    ax.plot(x, -y, lw=2, c=lcolor if LR[i] else rcolor)

  # Get rid of the ticks
  #  ax.set_xticks([])
  #  ax.set_yticks([])
  #
  #  # Get rid of tick labels
  #  ax.get_xaxis().set_ticklabels([])
  #  ax.get_yaxis().set_ticklabels([])

  RADIUS = 1  # space around the subject
  xroot, yroot = vals[0, 0], vals[0, 1]
  #     ax.set_xlim([-RADIUS+xroot, RADIUS+xroot])
  #     ax.set_ylim([-RADIUS+yroot, RADIUS+yroot])

  ax.set_xlim([-1, 1])
  ax.set_ylim([-1, 1])

  if add_labels:
    ax.set_xlabel("x")
    ax.set_ylabel("-y")

  ax.set_aspect('equal')


##############################
# wrap for simple usage
##############################
def wrap_show3d_pose(vals3d):
    fig3d = plt.figure()
    # ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax3d = Axes3D(fig3d)
    show3Dpose(vals3d, ax3d)
    plt.show()


def wrap_show2d_pose(vals2d):
    ax2d = plt.axes()
    show2Dpose(vals2d, ax2d)
    plt.show()
  


def plot_16j(poses, frame_colors=None,frame_legend=None,show_animation=False):
    import matplotlib as mpl
    mpl.use('Qt5Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.animation as anim

    from mpl_toolkits.mplot3d import axes3d, Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if not show_animation:
        plot_idx = 1
      
        frames = np.linspace(start=0, stop=poses.shape[0]-1, num=poses.shape[0]).astype(int)
 
        for i in frames:
            
            # ax = fig.add_subplot(1, poses.shape[0], plot_idx, projection='3d')
            
            x = poses[i,:,0]
            y = poses[i,:,1]
            z = poses[i,:,2]
            bones=[[0,1],[1,2],[2,3],[0,4],[4,5],[5,6],[0,7],[7,8],[8,9],[8,10],[11,10],[12,11],[8,13],[14,13],[15,14]]
            if frame_colors is None:
              colors=['r','r','r','b','b','b','g','g','g','b','b','b','r','r','r']
            else:
              colors=frame_colors[i]*15
            if frame_legend is None:
              frame_legend=['pred']
            for kk in range(len(bones)):
              if kk>0:
                frame_legend[i]=None
              ax.plot(poses[i,bones[kk],0],poses[i,bones[kk],1],poses[i,bones[kk],2],colors[kk],label=frame_legend[i])

            # Create cubic bounding box to simulate equal aspect ratio
            max_range = np.array([x.max() - x.min(),y.max() - y.min(), z.max() - z.min()] ).max()
            Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (x.max() + x.min())
            Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (y.max() + y.min())
            Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (z.max() + z.min())

            for xb, yb, zb in zip(Xb, Yb, Zb):
                ax.plot([xb], [yb], [zb], 'w')

            
              
            # ax.axis('equal')
            ax.axis('auto')
            # ax.axis('on')
            ax.set_title('frame = ' +str(i))
            ax.view_init(elev=10, azim=90)
            plot_idx += 1
        
        if frame_legend is not None:
          plt.legend()

        # plt.grid(True,'r')
        plt.show()

    return




def plot_16j_2d(poses, show_animation=False):
    import matplotlib as mpl
    mpl.use('Qt5Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.animation as anim

    from mpl_toolkits.mplot3d import axes3d, Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111)
    if not show_animation:
        plot_idx = 1
      
        frames = np.linspace(start=0, stop=poses.shape[0]-1, num=poses.shape[0]).astype(int)
 
        for i in frames:
            ax = fig.add_subplot(1, poses.shape[0], plot_idx)
            
            x = poses[i,:,0]
            y = poses[i,:,1]
            


            ax.plot(poses[i,[0,1],0], poses[i,[0,1],1], 'b')
            ax.plot(poses[i,[1,2],0], poses[i,[1,2],1], 'b')
            ax.plot(poses[i,[2,3],0], poses[i,[2,3],1], 'b')
            ax.plot(poses[i,[0,4],0], poses[i,[0,4],1], 'r')
            ax.plot(poses[i,[4,5],0], poses[i,[4,5],1], 'r')
            ax.plot(poses[i,[5,6],0], poses[i,[5,6],1], 'r')
            ax.plot(poses[i,[0,7],0], poses[i,[0,7],1], 'g')
            ax.plot(poses[i,[7,8],0], poses[i,[7,8],1], 'g')
            ax.plot(poses[i,[8,9],0], poses[i,[8,9],1], 'g')
            ax.plot(poses[i,[8,10],0], poses[i,[8,10],1],'b')
            ax.plot(poses[i,[11,10],0], poses[i,[11,10],1],'b')
            ax.plot(poses[i,[12,11],0], poses[i,[12,11],1],'b')
            ax.plot(poses[i,[8,13],0], poses[i,[8,13],1],'r')
            ax.plot(poses[i,[14,13],0], poses[i,[14,13],1],'r')
            ax.plot(poses[i,[15,14],0], poses[i,[15,14],1],'r')

            # Create cubic bounding box to simulate equal aspect ratio
            max_range = np.array([x.max() - x.min(), y.max() - y.min()]).max()
            Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (x.max() + x.min())
            Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (y.max() + y.min())

            for xb, yb in zip(Xb, Yb):
                ax.plot([xb], [yb], 'w')

            # ax.axis('equal')
            ax.axis('auto')
            ax.axis('on')
            ax.set_title('frame = ' +str(i))
            # ax.view_init(elev=10, azim=90)
            plot_idx += 1


        # plt.grid(True,'r')
        plt.show()

    return