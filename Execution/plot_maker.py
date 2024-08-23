import matplotlib.pyplot as plt
import numpy as np
from simulate_trajectories_model import *
#from simulate_trajectories_model_2 import *
from matplotlib.gridspec import GridSpec
from model_fit_max_data  import *
molecule = BaF
from model.molecules import BaF

'''
In this script I write all the functions that generate all the plots of interest related to laser cooling.
'''
'''
plot_all plots the trajectories of all the molecules simulated, as well as the position vs velocity.
'''
def plot_all(xls, zls, vxls):
    nn=len(xls)
    for n in range(nn):
        plt.plot(zls[n],xls[n],"b")
    plt.xlabel('z_axis')
    plt.ylabel('x_axis')
    plt.show()

    plt.clf()
    for n in range(nn):
        plt.plot(zls[n],xls[n],"b")
        plt.xlim(L[0]+L[1]+L[2],L[0]+L[1]+L[2]+L[3])
    plt.xlabel('z_axis')
    plt.ylabel('x_axis')
    plt.show()
    plt.clf()
    for n in range(nn):
        plt.plot(zls[n],vxls[n],"b")
    plt.xlabel('z_axis')
    plt.ylabel('vx_axis')
    plt.show()

    plt.clf()
'''
plot_side_by_side is made to plot the trajectories of two simulations side by side (originally laser cooling vs no laser
cooling.
'''
def plot_side_by_side(x1, z1, x2, z2):
    nn=len(x1)
    fig,axs=plt.subplots(nrows=1,ncols=2)
    for n in range(nn):
        axs[0].plot(z1[n],x1[n],'b')
        axs[1].plot(z2[n],x2[n],'r')
        axs[0].set_ylim([-0.3,0.3])
        axs[1].set_ylim([-0.3,0.3])
    axs[0].set_xlabel('z_Axis')
    axs[1].set_xlabel('z_Axis')
    axs[0].set_ylabel('x_Axis')
    axs[1].set_ylabel('x_Axis')
    axs[0].set_title('Cooling')
    axs[1].set_title('No cooling')
    fig.tight_layout()

    plt.show()
    plt.clf()

    fig1,axs1=plt.subplots(nrows=1,ncols=2)
    for n in range(nn):
        axs1[0].plot(z1[n],x1[n],'b')
        axs1[1].plot(z2[n],x2[n],'r')
        axs1[0].set_xlim([0,L[0]+L[1]+L[2]+L[3]])
        axs1[1].set_xlim([0,L[0]+L[1]+L[2]+L[3]])
        axs1[0].set_ylim([-0.3,0.3])
        axs1[1].set_ylim([-0.3,0.3])

    axs1[0].set_xlabel('z_Axis')
    axs1[1].set_xlabel('z_Axis')
    axs1[0].set_ylabel('x_Axis')
    axs1[1].set_ylabel('x_Axis')
    axs1[0].set_title('Cooling')
    axs1[1].set_title('No cooling')
    fig1.tight_layout()

    plt.show()
    plt.clf()

def compare_lc_nolc(x1, x2, z1, z2):
    difx=0
    difz=0
    for i in range(len(z1)):
        difx+=np.linalg.norm(np.array(x2[i])-np.array(x1[i]))
        difz+=np.linalg.norm(np.array(z2[i])-np.array(z1[i]))
        
    return difx,difz
'''
plot_heatmap plots the distribution of a simulation. Its the image we originally want
'''

def plot_heatmap(xcooling, ycooling, zcooling):
    print(len(xcooling),len(ycooling),len(zcooling))
    #x_edges = np.linspace(min(xcooling), max(xcooling), num=1000)
    #y_edges = np.linspace(min(ycooling), max(ycooling), num=1000)
    x_edges=np.linspace(-0.03,0.03,100)
    y_edges=np.linspace(-0.03,0.03,100)
    H, xedges, yedges = np.histogram2d(xcooling, ycooling, bins=[x_edges, y_edges])

    fig=plt.figure()
    axs=fig.add_subplot(111)

    pos=plt.imshow(H.T, origin='lower', aspect='auto', cmap='inferno',
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    plt.colorbar(label='number of counts')
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    plt.title('2D Heatmap')
    plt.show()
'''
plot_2_heatmap is used to plot two heatmaps side by side. ideally lc vs no-lc but it can be other simulations
'''
def plot_2_heatmap(xcooling1, ycooling1, xcooling2, ycooling2, ff, s0, det, hexvolt, Ldetection):

    x_edges=np.linspace(-25,25,150)
    y_edges=np.linspace(-12.5,12.5,150)
    H1, xedges1, yedges1 = np.histogram2d(np.array(xcooling1)*1e3, np.array(ycooling1)*1e3, bins=[x_edges, y_edges])
    H2, xedges2, yedges2 = np.histogram2d(np.array(xcooling2)*1e3, np.array(ycooling2)*1e3, bins=[x_edges, y_edges])


    fig,(ax1,ax2)=plt.subplots(nrows=2,ncols=1,sharex=True)
    fig.set_size_inches(10,10)
    cax1=ax1.imshow(H1.T, origin='lower', aspect='auto', cmap='sunset',
                    extent=[xedges1[0], xedges1[-1], yedges1[0], yedges1[-1]])
    title=('Fudgefactor: '+str(ff)+', Saturation: '+str(s0)+', Detuning: '+str(det)+' Gamma'+
           ', Hexapole Voltage: '+str(hexvolt)+' V, Ldetection: '+str(Ldetection)+' m')
    ax1.set_title('Cooling')
    #ax1.set_xlabel('x axis (mm)')
    ax1.set_ylabel('y axis (mm)')
    #ax1.annotate(title,xy = (1.0, -0.2),xycoords='axes fraction', ha='right', va='center', fontsize=10)
    fig.colorbar(cax1,ax=ax1)
    cax2=ax2.imshow(H2.T, origin = 'lower', aspect = 'auto', cmap = 'sunset',
                    extent = [xedges2[0], xedges2[-1], yedges2[0], yedges2[-1]])
    ax2.set_title('No cooling')
    ax2.set_xlabel('x axis (mm)')
    ax2.set_ylabel('y axis (mm)')
    fig.colorbar(cax2, ax=ax2)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.suptitle(title, x=0.45, fontsize=16)

    #fig.suptitle(title)
    #plt.tight_layout()
    plt.show()
'''
plot_4_heatmaps plots 4 heatmaps. Ideally two laser cooling and two no-lc for two distances
Since it can work for different simulations it needs the argument legends to be a list of strings (the legend for each
simulation.
plot circle plots a circle of radius 5 mm to see how collimated the beam is
'''

def plot_4_heatmap(x1, y1, x2, y2, x3, y3, x4, y4, ff, s0, det, hexvolt, Ldetection1, Ldetection2,
                   legends, plot_circle = False, save_plot  = False):

    x_edges=np.linspace(-25,25,150)
    y_edges=np.linspace(-12.5,12.5,75)
    H1, xedges1, yedges1 = np.histogram2d(np.array(x1)*1e3, np.array(y1)*1e3, bins=[x_edges, y_edges])
    H2, xedges2, yedges2 = np.histogram2d(np.array(x2)*1e3, np.array(y2)*1e3, bins=[x_edges, y_edges])
    H3, xedges3, yedges3 = np.histogram2d(np.array(x3)*1e3, np.array(y3)*1e3, bins=[x_edges, y_edges])
    H4, xedges4, yedges4 = np.histogram2d(np.array(x4)*1e3, np.array(y4)*1e3, bins=[x_edges, y_edges])
    a=1
    label_font=20
    ticks_font=12
    Ldetection1=Ldetection1*1e3
    Ldetection2=Ldetection2*1e3
    fig,axs=plt.subplots(nrows=2,ncols=2,sharex=True,sharey=True)
    fig.set_size_inches(16,6)
    
    cax1=axs[0,0].imshow(H1.T, origin = 'lower', aspect = 'auto', cmap = 'sunset',
                         extent = [xedges1[0], xedges1[-1], yedges1[0], yedges1[-1]])
    axs[0,0].set_title(legends[2])#'z= '+'%.2f'%Ldetection1+' mm',fontsize=20)
    #ax1.set_xlabel('x axis (mm)')
    axs[0,0].set_ylabel('y (mm)',fontsize=label_font-3)
    axs[0,0].text(-70,0,legends[0],fontsize=label_font+5)
    cbar1=fig.colorbar(cax1,ax=axs[0,0],shrink=a, aspect=20*a)
    cbar1.ax.tick_params(labelsize=ticks_font-5)
    axs[0,0].set_box_aspect(0.55)
    axs[0,0].tick_params(axis='y', labelsize=ticks_font)

    cax2=axs[1,0].imshow(H2.T, origin = 'lower', aspect = 'auto', cmap = 'sunset',
                         extent = [xedges2[0], xedges2[-1], yedges2[0], yedges2[-1]])
   # axs[1,0].set_title('No cooling, z; '+'%.2f'%Ldetection1+'m')
    axs[1,0].set_xlabel('x (mm)',fontsize=label_font-3)
    axs[1,0].set_ylabel('y (mm)',fontsize=label_font-3)
    axs[1,0].text(-70,0,legends[1],fontsize=label_font+5)
    cbar2=fig.colorbar(cax1,ax=axs[1,0],shrink=a, aspect=20*a)
    cbar2.ax.tick_params(labelsize=ticks_font-5)
    axs[1,0].set_box_aspect(0.55)
    axs[1,0].tick_params(axis='x', labelsize=ticks_font)
    axs[1,0].tick_params(axis='y', labelsize=ticks_font)

    cax3=axs[0,1].imshow(H3.T, origin = 'lower', aspect = 'auto', cmap = 'sunset',
                         extent = [xedges3[0], xedges3[-1], yedges3[0], yedges3[-1]])
    axs[0,1].set_title(legends[3])#'z= '+'%.2f'%3500+' mm',fontsize=20)
    #axs[0,1].set_xlabel('x axis (mm)')
    #axs[0,1].set_ylabel('y axis (mm)')
    cbar3=fig.colorbar(cax1,ax=axs[0,1],shrink=a, aspect=20*a)
    cbar3.ax.tick_params(labelsize=ticks_font-5)
    axs[0,1].set_box_aspect(0.55)
    axs[0,1].tick_params(axis='y', labelsize=ticks_font)


    cax4 = axs[1,1].imshow(H4.T, origin = 'lower', aspect = 'auto', cmap = 'sunset',
                         extent = [xedges4[0], xedges4[-1], yedges4[0], yedges4[-1]])
    #axs[1,1].set_title('No cooling, z: '+'%.2f'%Ldetection2+'m')
    axs[1,1].set_xlabel('x (mm)',fontsize=label_font-3)
    axs[1,1].tick_params(axis='x', labelsize=ticks_font)
    axs[1,1].tick_params(axis='y', labelsize=ticks_font)
    #axs[1,1].set_ylabel('y axis (mm)')
    cbar4=fig.colorbar(cax1,ax=axs[1,1],shrink=a, aspect=20*a)
    cbar4.ax.tick_params(labelsize=ticks_font-5)
    axs[1,1].set_box_aspect(0.55)
    
    if plot_circle==False:
        save_title='../4_heatmap/'+'Heatmap_'+'%.2f_%.2f_%.2f_%.2f'%(ff,s0,det,hexvolt)+'.png'
    else:
        theta=np.linspace(0,2*pi,1001)
        x=5*np.cos(theta)
        y=5*np.sin(theta)
        axs[0,0].plot(x,y,'w')
        axs[1,0].plot(x,y,'w')
        axs[0,1].plot(x,y,'w')
        axs[1,1].plot(x,y,'w')
        save_title='../4_heatmap/'+'%.2f_%.2f_%.2f_%.2f'%(ff,s0,det,hexvolt)+'_circle'+'.png'
    
    r=5
    points_in_c1=points_in_circle(np.array(x1)*1e3,np.array(y1)*1e3,r)
    points_in_c2=points_in_circle(np.array(x2)*1e3,np.array(y2)*1e3,r)
    points_in_c3=points_in_circle(np.array(x3)*1e3,np.array(y3)*1e3,r)
    points_in_c4=points_in_circle(np.array(x4)*1e3,np.array(y4)*1e3,r)

    axs[0,0].text(-23,10,'# in circle = '+str(points_in_c1),fontsize = 10,color = 'black',
                  bbox = dict(facecolor='white', alpha = 0.8))
    axs[1,0].text(-23,10,'# in circle = '+str(points_in_c2),fontsize = 10,color = 'black',
                  bbox = dict(facecolor = 'white', alpha = 0.8))
    axs[0,1].text(-23,10,'# in circle = '+str(points_in_c3),fontsize = 10,color = 'black',
                  bbox = dict(facecolor = 'white', alpha = 0.8))
    axs[1,1].text(-23,10,'# in circle = '+str(points_in_c4),fontsize = 10,color = 'black',
                  bbox = dict(facecolor = 'white', alpha = 0.8))
    amax=min(ascat(np.linspace(-10,10,101),ff,s0,det))


    plt.tight_layout(rect=[0, 0, 1, 0.95])
    #fig.suptitle(title, x=.1, fontsize=18)
    title='F = '+'%.2f'%ff+', sat = '+'%.2f'%s0+', $\\Delta$ = '+'%.2f'%det+' $\\Gamma$'+'\n $V_{Hex}$= '+'%.2f'%hexvolt+' V'
    title_acc='$a_{peak}$ = '+'%.2f'%amax+' $ m/s^2$'
    axs[0,0].text(-75,12,title,fontsize=label_font)
    axs[0,0].text(-75,9,title_acc,fontsize=label_font-3)


    #fig.suptitle(title)
    if save_plot==False:
        plt.show()
    else:
        plt.savefig(save_title,dpi=200)
'''
plot_hist2 plots the histogram of velocities for two different simulations
'''

def plot_hist2(velr_1, velr_2, ff, s0, det, hexvolt, Ldetection1, bins = 60,save_plot = False):
    fig = plt.figure(figsize=(10, 6))
    gs = GridSpec(2, 2, width_ratios=[1, 2])
    ax_text1 = fig.add_subplot(gs[0, 0])
    ax_text1.axis('off')  # Turn off the axis
    text1 = "2D LC on"
    amax=min(ascat(np.linspace(-10,10,101),ff,s0,det))
    title=('F = '+'%.2f'%ff+', sat = '+'%.2f'%s0+', $\\Delta$ = '
           +'%.2f'%det+' $\\Gamma$'+'\n $V_{Hex}$= '+'%.2f'%hexvolt+' V')
    ax_text1.text(0.5, 0.5, text1, va='center', ha='center', fontsize=20)
    ax_text1.text(0.5,1,title,va='center', ha='center', fontsize=15)
    ax_text2 = fig.add_subplot(gs[1, 0])
    ax_text2.axis('off')  # Turn off the axis
    text2 = "2D LC off"  
    ax_text2.text(0.5, 0.5, text2, va='center', ha='center', fontsize=20)
    # Add the first graph to the second column, first row
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.hist(velr_1,bins=bins)  # Example plot
    #ax1.set_xlabel('velocities (m/s)')
    ax1.set_ylabel('# counts')
    ax1.set_title('z = '+str(Ldetection1))
    #ax1.set_title('First Graph')
    # Add the second graph to the second column, second row
    ax2 = fig.add_subplot(gs[1, 1],sharex=ax1)
    ax2.hist(velr_2,bins=bins)  # Example plot
    ax2.set_xlabel('velocities (m/s)')
    ax2.set_ylabel('# counts')
    plt.setp(ax1.get_xticklabels(), visible=False)

    # Adjust layout to prevent overlap
    plt.tight_layout()
    save_title='../LC_histograms/'+'Histogram_'+'%.2f_%.2f_%.2f_%.2f%.2f'%(ff,s0,det,hexvolt,Ldetection1)+'.png'
    #save_title='histpgram_6hetmap'
    if save_plot == False:
        plt.show()
    else:
        plt.savefig(save_title, dpi=200)

def plot_6_hist_vel(velr_lst, ff, s0, det, hexvolt, labels = True,bins = 60):
        n=len(velr_lst)
        fig = plt.figure(figsize=(6, 6))  # int((n/2)*2.3333)))
        gs = GridSpec(3, 2)#, width_ratios=[1, 1])
        counter=0
        axes={}
        legends=['2D-LC ON \n Hexapole ON','2D-LC OFF \n Hexapole ON','2D-LC OFF \n Hexapole OFF']
        for i in range(2):
            for j in range(3):
                ax=fig.add_subplot(gs[j,i])
                if i==0:
                    ax.axis('off')
                    ax.text(0.5, 0.5, legends[j], va='center', ha='center', fontsize=15)
                    if j==0 and labels:
                        title = ('F = ' + '%.2f' % ff + ', sat = ' + '%.2f' % s0 +
                                 '\n $\\Delta$ = ' + '%.2f' % det + ' $\\Gamma$'
                                 + '\n $V_{Hex}$= ' + '%.2f' % hexvolt + ' V')
                        ax.text(0.5,1,title,va='center', ha='center', fontsize=12)
                else:
                    ax.hist(velr_lst[counter],bins=bins)
                    counter+=1
                ax.set_box_aspect(0.55)
                axes[j,i]=ax
                if i == 1:
                    ax.set_ylabel('# counts')

                if j== 2:
                    ax.set_xlabel('velocities (m/s)')


        plt.tight_layout()
        plt.show()

def read_camera_simulated_data(filename, _directory=''):
    data = np.load(file=filename)
    # self.data[_filename]=np.load(file=_filename)
    #self.data[filename] = np.load(file=filename
    return data
'''
plot_6_heatmap works the same as 4_heatmap but it also adds ads another row of heatmaps for hexapole off
'''
def plot_6_heatmap(x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, ff, s0, det, hexvolt,
                   Ldetection1, Ldetection2, plot_circle = False, save_plot = False, extra_title = ''):
    x_edges = np.linspace(-25, 25, 128)
    y_edges = np.linspace(-12.5, 12.5, 128)
    H1, xedges1, yedges1 = np.histogram2d(np.array(x1) * 1e3, np.array(y1) * 1e3, bins=[x_edges, y_edges])
    H2, xedges2, yedges2 = np.histogram2d(np.array(x2) * 1e3, np.array(y2) * 1e3, bins=[x_edges, y_edges])
    H3, xedges3, yedges3 = np.histogram2d(np.array(x3) * 1e3, np.array(y3) * 1e3, bins=[x_edges, y_edges])
    H4, xedges4, yedges4 = np.histogram2d(np.array(x4) * 1e3, np.array(y4) * 1e3, bins=[x_edges, y_edges])
    H5, xedges5, yedges5 = np.histogram2d(np.array(x5) * 1e3, np.array(y5) * 1e3, bins=[x_edges, y_edges])
    H6, xedges6, yedges6 = np.histogram2d(np.array(x6) * 1e3, np.array(y6) * 1e3, bins=[x_edges, y_edges])

    Heatmaps=[H1,H2,H3,H4,H5,H6]
    a = 1
    label_font = 20
    ticks_font = 12
    Ldetection1 = Ldetection1 * 1e3
    Ldetection2 = Ldetection2 * 1e3
    fig, axs = plt.subplots(nrows=3, ncols=2, sharex=True, sharey=True)
    fig.set_size_inches(18, 9)
    cax1 = axs[0, 0].imshow(Heatmaps[0].T, origin='lower', aspect='auto', cmap='sunset',
                            extent=[xedges1[0], xedges1[-1], yedges1[0], yedges1[-1]])
    axs[0, 0].set_title('z= ' + '%.2f' % Ldetection1 + ' mm', fontsize=20)
    axs[0, 0].set_ylabel('y (mm)', fontsize=label_font - 3)
    axs[0, 0].text(-70, 0, '2D-LC on \n Hexapole on', fontsize=label_font + 2)
    cbar1 = fig.colorbar(cax1, ax=axs[0, 0], shrink=a, aspect=20 * a)
    cbar1.ax.tick_params(labelsize=ticks_font - 5)
    axs[0, 0].set_box_aspect(0.55)
    axs[0, 0].tick_params(axis='y', labelsize=ticks_font)

    cax2 = axs[1, 0].imshow(Heatmaps[1].T, origin='lower', aspect='auto', cmap='sunset',
                            extent=[xedges2[0], xedges2[-1], yedges2[0], yedges2[-1]])

    axs[1, 0].set_ylabel('y (mm)', fontsize=label_font - 3)
    axs[1, 0].text(-70, 0, '2D-LC off \n Hexapole on', fontsize=label_font + 2)
    cbar2 = fig.colorbar(cax1, ax=axs[1, 0], shrink=a, aspect=20 * a)
    cbar2.ax.tick_params(labelsize=ticks_font - 5)
    axs[1, 0].set_box_aspect(0.55)
    axs[1, 0].tick_params(axis='x', labelsize=ticks_font)
    axs[1, 0].tick_params(axis='y', labelsize=ticks_font)

    cax3 = axs[0, 1].imshow(Heatmaps[2].T, origin='lower', aspect='auto', cmap='sunset',
                            extent=[xedges3[0], xedges3[-1], yedges3[0], yedges3[-1]])
    axs[0, 1].set_title('z= ' + '%.2f' % 3500 + ' mm', fontsize=20)

    cbar3 = fig.colorbar(cax1, ax=axs[0, 1], shrink=a, aspect=20 * a)
    cbar3.ax.tick_params(labelsize=ticks_font - 5)
    axs[0, 1].set_box_aspect(0.55)
    axs[0, 1].tick_params(axis='y', labelsize=ticks_font)

    cax4 = axs[1, 1].imshow(Heatmaps[3].T, origin='lower', aspect='auto', cmap='sunset',
                            extent=[xedges4[0], xedges4[-1], yedges4[0], yedges4[-1]])

    axs[1, 1].tick_params(axis='x', labelsize=ticks_font)
    axs[1, 1].tick_params(axis='y', labelsize=ticks_font)
    cbar4 = fig.colorbar(cax1, ax=axs[1, 1], shrink=a, aspect=20 * a)
    cbar4.ax.tick_params(labelsize=ticks_font - 5)
    axs[1, 1].set_box_aspect(0.55)

    cax5 = axs[2, 0].imshow(Heatmaps[4].T, origin='lower', aspect='auto', cmap='sunset',
                            extent=[xedges5[0], xedges5[-1], yedges5[0], yedges5[-1]])
    axs[2, 0].set_xlabel('x (mm)', fontsize=label_font - 3)
    axs[2, 0].set_ylabel('y (mm)', fontsize=label_font - 3)
    axs[2, 0].text(-70, 0, '(2D-LC \n Hexapole) off ', fontsize=label_font + 2)
    cbar5 = fig.colorbar(cax1, ax=axs[2, 0], shrink=a, aspect=20 * a)
    cbar5.ax.tick_params(labelsize=ticks_font - 5)
    axs[2, 0].set_box_aspect(0.55)
    axs[2, 0].tick_params(axis='x', labelsize=ticks_font)
    axs[2, 0].tick_params(axis='y', labelsize=ticks_font)

    cax6 = axs[2, 1].imshow(Heatmaps[5].T, origin='lower', aspect='auto', cmap='sunset',
                            extent=[xedges6[0], xedges6[-1], yedges6[0], yedges6[-1]])
    axs[2, 1].set_xlabel('x (mm)', fontsize=label_font - 3)
    cbar6 = fig.colorbar(cax1, ax=axs[2, 1], shrink=a, aspect=20 * a)
    cbar6.ax.tick_params(labelsize=ticks_font - 5)
    axs[2, 1].set_box_aspect(0.55)
    axs[2, 1].tick_params(axis='x', labelsize=ticks_font)
    axs[2, 1].tick_params(axis='y', labelsize=ticks_font)
    if extra_title !='':
        fig.suptitle(extra_title, fontsize=label_font)
    if plot_circle == False:
         save_title = '../6_heatmap/' + 'Heatmap_' + '%.2f_%.2f_%.2f_%.2f' % (ff, s0, det, hexvolt) + '.png'
    else:
        theta = np.linspace(0, 2 * pi, 1001)
        x = 5 * np.cos(theta)
        y = 5 * np.sin(theta)
        axs[0,0].plot(x, y, 'w')
        axs[0,1].plot(x, y, 'w')
        axs[1,0].plot(x, y, 'w')
        axs[1,1].plot(x, y, 'w')
        axs[2,0].plot(x, y, 'w')
        axs[2,1].plot(x, y, 'w')
        save_title = '../6_heatmap/'+ 'Heatmap_Gauss3' + '%.2f_%.2f_%.2f_%.2f' % (ff, s0, det, hexvolt) + '_circle' + '.png'

    r = 5
    points_in_c1 = points_in_circle(np.array(x1) * 1e3, np.array(y1) * 1e3, r)
    points_in_c2 = points_in_circle(np.array(x2) * 1e3, np.array(y2) * 1e3, r)
    points_in_c3 = points_in_circle(np.array(x3) * 1e3, np.array(y3) * 1e3, r)
    points_in_c4 = points_in_circle(np.array(x4) * 1e3, np.array(y4) * 1e3, r)
    points_in_c5 = points_in_circle(np.array(x5) * 1e3, np.array(y5) * 1e3, r)
    points_in_c6 = points_in_circle(np.array(x6) * 1e3, np.array(y6) * 1e3, r)
    if points_in_c6==0: points_in_c6=1

    x = -23
    y = 9
    axs[0,0].text(x, y, '# in circle= ' + str(points_in_c1), fontsize=10, color='black',
             bbox=dict(facecolor='white', alpha=0.8))
    axs[1,0].text(x, y, '# in circle= ' + str(points_in_c2), fontsize=10, color='black',
             bbox=dict(facecolor='white', alpha=0.8))

    axs[0,1].text(x, y, '# in circle= ' + str(points_in_c3), fontsize=10, color='black',
             bbox=dict(facecolor='white', alpha=0.8))
    axs[0, 1].text(x+20, y, 'Ratio b)/f) = ' + '%.2f'%(points_in_c3/points_in_c6), fontsize=10, color='black',
                   bbox=dict(facecolor='white', alpha=0.8))
    axs[1,1].text(x, y, '# in circle= ' + str(points_in_c4), fontsize=10, color='black',
             bbox=dict(facecolor='white', alpha=0.8))
    axs[2,0].text(x, y, '# in circle= ' + str(points_in_c5), fontsize=10, color='black',
             bbox=dict(facecolor='white', alpha=0.8))
    axs[2,1].text(x, y, '# in circle= ' + str(points_in_c6), fontsize=10, color='black',
             bbox=dict(facecolor='white', alpha=0.8))
    amax = min(ascat(np.linspace(-10, 10, 101), ff, s0, det))

    axs[0,0].text(x,y-20,'a)',color='black',bbox=dict(facecolor='white',alpha=0.8))
    axs[0,1].text(x,y-20,'b)',color='black',bbox=dict(facecolor='white',alpha=0.8))
    axs[1,0].text(x,y-20,'c)',color='black',bbox=dict(facecolor='white',alpha=0.8))
    axs[1,1].text(x,y-20,'d)',color='black',bbox=dict(facecolor='white',alpha=0.8))
    axs[2,0].text(x,y-20,'e)',color='black',bbox=dict(facecolor='white',alpha=0.8))
    axs[2,1].text(x,y-20,'f)',color='black',bbox=dict(facecolor='white',alpha=0.8))


    plt.tight_layout(rect=[0, 0, 1, 0.95])
    title = 'F = ' + '%.2f' % ff + ', sat = ' + '%.2f' % s0 + ', $\\Delta$ = ' + '%.2f' % det + ' $\\Gamma$' + '\n $V_{Hex}$= ' + '%.2f' % hexvolt + ' V'
    title_acc = '$a_{peak}$ = ' + '%.2f' % amax + ' $ m/s^2$'
    axs[0, 0].text(-75, 12, title, fontsize=label_font-3)
    axs[0, 0].text(-75, 9, title_acc, fontsize=label_font - 3)
    if save_plot == False:
        plt.show()
    else:
        plt.savefig(save_title, dpi=200)

def plot_8_heatmap(x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6,x7, y7, x8, y8, ff, s0, det, hexvolt,
                   Ldetection1, Ldetection2, plot_circle = False, save_plot = True, extra_title = ''):
    x_edges = np.linspace(-25, 25, 120)
    y_edges = np.linspace(-12.5, 12.5, 120)
    H1, xedges1, yedges1 = np.histogram2d(np.array(x1) * 1e3, np.array(y1) * 1e3, bins=[x_edges, y_edges])
    H2, xedges2, yedges2 = np.histogram2d(np.array(x2) * 1e3, np.array(y2) * 1e3, bins=[x_edges, y_edges])
    H3, xedges3, yedges3 = np.histogram2d(np.array(x3) * 1e3, np.array(y3) * 1e3, bins=[x_edges, y_edges])
    H4, xedges4, yedges4 = np.histogram2d(np.array(x4) * 1e3, np.array(y4) * 1e3, bins=[x_edges, y_edges])
    H5, xedges5, yedges5 = np.histogram2d(np.array(x5) * 1e3, np.array(y5) * 1e3, bins=[x_edges, y_edges])
    H6, xedges6, yedges6 = np.histogram2d(np.array(x6) * 1e3, np.array(y6) * 1e3, bins=[x_edges, y_edges])
    H7, xedges7, yedges7 = np.histogram2d(np.array(x7) * 1e3, np.array(y7) * 1e3, bins=[x_edges, y_edges])
    H8, xedges8, yedges8 = np.histogram2d(np.array(x8) * 1e3, np.array(y8) * 1e3, bins=[x_edges, y_edges])

    Heatmaps=[H1,H2,H3,H4,H5,H6]
    a = 1
    label_font = 20
    ticks_font = 12
    Ldetection1 = Ldetection1 * 1e3
    Ldetection2 = Ldetection2 * 1e3
    fig, axs = plt.subplots(nrows=4, ncols=2, sharex=True, sharey=True)
    fig.set_size_inches(18, 12)
    cax1 = axs[0, 0].imshow(Heatmaps[0].T, origin='lower', aspect='auto', cmap='sunset',
                            extent=[xedges1[0], xedges1[-1], yedges1[0], yedges1[-1]])
    axs[0, 0].set_title('z= ' + '%.2f' % Ldetection1 + ' mm', fontsize=20)
    axs[0, 0].set_ylabel('y (mm)', fontsize=label_font - 3)
    axs[0, 0].text(-70, 0, '2D-LC on \n Hexapole on', fontsize=label_font + 2)
    cbar1 = fig.colorbar(cax1, ax=axs[0, 0], shrink=a, aspect=20 * a)
    cbar1.ax.tick_params(labelsize=ticks_font - 5)
    axs[0, 0].set_box_aspect(0.55)
    axs[0, 0].tick_params(axis='y', labelsize=ticks_font)

    cax2 = axs[1, 0].imshow(Heatmaps[1].T, origin='lower', aspect='auto', cmap='sunset',
                            extent=[xedges2[0], xedges2[-1], yedges2[0], yedges2[-1]])
    axs[1, 0].set_ylabel('y (mm)', fontsize=label_font - 3)
    axs[1, 0].text(-70, 0, '2D-LC off \n Hexapole on', fontsize=label_font + 2)
    cbar2 = fig.colorbar(cax1, ax=axs[1, 0], shrink=a, aspect=20 * a)
    cbar2.ax.tick_params(labelsize=ticks_font - 5)
    axs[1, 0].set_box_aspect(0.55)
    axs[1, 0].tick_params(axis='x', labelsize=ticks_font)
    axs[1, 0].tick_params(axis='y', labelsize=ticks_font)

    cax3 = axs[0, 1].imshow(Heatmaps[2].T, origin='lower', aspect='auto', cmap='sunset',
                            extent=[xedges3[0], xedges3[-1], yedges3[0], yedges3[-1]])
    axs[0, 1].set_title('z= ' + '%.2f' % 3500 + ' mm', fontsize=20)
    cbar3 = fig.colorbar(cax1, ax=axs[0, 1], shrink=a, aspect=20 * a)
    cbar3.ax.tick_params(labelsize=ticks_font - 5)
    axs[0, 1].set_box_aspect(0.55)
    axs[0, 1].tick_params(axis='y', labelsize=ticks_font)

    cax4 = axs[1, 1].imshow(Heatmaps[3].T, origin='lower', aspect='auto', cmap='sunset',
                            extent=[xedges4[0], xedges4[-1], yedges4[0], yedges4[-1]])
    axs[1, 1].tick_params(axis='x', labelsize=ticks_font)
    axs[1, 1].tick_params(axis='y', labelsize=ticks_font)
    cbar4 = fig.colorbar(cax1, ax=axs[1, 1], shrink=a, aspect=20 * a)
    cbar4.ax.tick_params(labelsize=ticks_font - 5)
    axs[1, 1].set_box_aspect(0.55)
    cax5 = axs[2, 0].imshow(Heatmaps[4].T, origin='lower', aspect='auto', cmap='sunset',
                            extent=[xedges5[0], xedges5[-1], yedges5[0], yedges5[-1]])
    axs[3, 0].set_xlabel('x (mm)', fontsize=label_font - 3)
    axs[2, 0].set_ylabel('y (mm)', fontsize=label_font - 3)
    axs[2, 0].text(-70, 0, '(2D-LC \n Hexapole) off ', fontsize=label_font + 2)
    cbar5 = fig.colorbar(cax1, ax=axs[2, 0], shrink=a, aspect=20 * a)
    cbar5.ax.tick_params(labelsize=ticks_font - 5)
    axs[2, 0].set_box_aspect(0.55)
    axs[2, 0].tick_params(axis='x', labelsize=ticks_font)
    axs[2, 0].tick_params(axis='y', labelsize=ticks_font)

    cax6 = axs[2, 1].imshow(Heatmaps[5].T, origin='lower', aspect='auto', cmap='sunset',
                            extent=[xedges6[0], xedges6[-1], yedges6[0], yedges6[-1]])
    axs[3, 1].set_xlabel('x (mm)', fontsize=label_font - 3)
    cbar6 = fig.colorbar(cax1, ax=axs[2, 1], shrink=a, aspect=20 * a)
    cbar6.ax.tick_params(labelsize=ticks_font - 5)
    axs[2, 1].set_box_aspect(0.55)
    axs[2, 1].tick_params(axis='x', labelsize=ticks_font)
    axs[2, 1].tick_params(axis='y', labelsize=ticks_font)

    cax7 = axs[3, 0].imshow(H7.T, origin='lower', aspect='auto', cmap='sunset',
                            extent=[xedges1[0], xedges1[-1], yedges1[0], yedges1[-1]])
    axs[3, 0].set_ylabel('y (mm)', fontsize=label_font - 3)
    axs[3, 0].text(-70, 0, '2D-LH on \n Hexapole on', fontsize=label_font + 2)
    cbar7 = fig.colorbar(cax1, ax=axs[3, 0], shrink=a, aspect=20 * a)
    cbar7.ax.tick_params(labelsize=ticks_font - 5)
    axs[3, 0].set_box_aspect(0.55)
    axs[3, 0].tick_params(axis='y', labelsize=ticks_font)

    cax8 = axs[3, 1].imshow(H8.T, origin='lower', aspect='auto', cmap='sunset',
                            extent=[xedges2[0], xedges2[-1], yedges2[0], yedges2[-1]])
    axs[3, 1].set_ylabel('y (mm)', fontsize=label_font - 3)
    cbar8 = fig.colorbar(cax1, ax=axs[3, 1], shrink=a, aspect=20 * a)
    cbar8.ax.tick_params(labelsize=ticks_font - 5)
    axs[3, 1].set_box_aspect(0.55)
    axs[3, 1].tick_params(axis='x', labelsize=ticks_font)
    axs[3, 1].tick_params(axis='y', labelsize=ticks_font)
    if extra_title !='':
        fig.suptitle(extra_title, fontsize=label_font)
    if plot_circle == False:
         save_title = '../6_heatmap/' + 'Heatmap_' + '%.2f_%.2f_%.2f_%.2f' % (ff, s0, det, hexvolt) + '.png'
    else:
        theta = np.linspace(0, 2 * pi, 1001)
        x = 5 * np.cos(theta)
        y = 5 * np.sin(theta)
        axs[0,0].plot(x, y, 'w')
        axs[0,1].plot(x, y, 'w')
        axs[1,0].plot(x, y, 'w')
        axs[1,1].plot(x, y, 'w')
        axs[2,0].plot(x, y, 'w')
        axs[2,1].plot(x, y, 'w')
        axs[3,0].plot(x, y, 'w')
        axs[3,1].plot(x, y, 'w')
        save_title = '../6_heatmap/'+ 'Heatmap_Gauss3' + '%.2f_%.2f_%.2f_%.2f' % (ff, s0, det, hexvolt) + '_circle' + '.png'
    r = 5
    points_in_c1 = points_in_circle(np.array(x1) * 1e3, np.array(y1) * 1e3, r)
    points_in_c2 = points_in_circle(np.array(x2) * 1e3, np.array(y2) * 1e3, r)
    points_in_c3 = points_in_circle(np.array(x3) * 1e3, np.array(y3) * 1e3, r)
    points_in_c4 = points_in_circle(np.array(x4) * 1e3, np.array(y4) * 1e3, r)
    points_in_c5 = points_in_circle(np.array(x5) * 1e3, np.array(y5) * 1e3, r)
    points_in_c6 = points_in_circle(np.array(x6) * 1e3, np.array(y6) * 1e3, r)
    points_in_c7 = points_in_circle(np.array(x7) * 1e3, np.array(y7) * 1e3, r)
    points_in_c8 = points_in_circle(np.array(x8) * 1e3, np.array(y8) * 1e3, r)
    if points_in_c6==0: points_in_c6=1

    x = -23
    y = 9
    axs[0,0].text(x, y, '# in circle= ' + str(points_in_c1), fontsize=10, color='black',
             bbox=dict(facecolor='white', alpha=0.8))
    axs[1,0].text(x, y, '# in circle= ' + str(points_in_c2), fontsize=10, color='black',
             bbox=dict(facecolor='white', alpha=0.8))

    axs[0,1].text(x, y, '# in circle= ' + str(points_in_c3), fontsize=10, color='black',
             bbox=dict(facecolor='white', alpha=0.8))
    axs[0, 1].text(x+20, y, 'Ratio b)/f) = ' + '%.2f'%(points_in_c3/points_in_c6), fontsize=10, color='black',
                   bbox=dict(facecolor='white', alpha=0.8))
    axs[1,1].text(x, y, '# in circle= ' + str(points_in_c4), fontsize=10, color='black',
             bbox=dict(facecolor='white', alpha=0.8))
    axs[2,0].text(x, y, '# in circle= ' + str(points_in_c5), fontsize=10, color='black',
             bbox=dict(facecolor='white', alpha=0.8))
    axs[2,1].text(x, y, '# in circle= ' + str(points_in_c6), fontsize=10, color='black',
             bbox=dict(facecolor='white', alpha=0.8))
    axs[3, 0].text(x, y, '# in circle= ' + str(points_in_c7), fontsize=10, color='black',
                   bbox=dict(facecolor='white', alpha=0.8))
    axs[3, 1].text(x, y, '# in circle= ' + str(points_in_c8), fontsize=10, color='black',
                   bbox=dict(facecolor='white', alpha=0.8))
    amax = min(ascat(np.linspace(-10, 10, 101), ff, s0, det))

    axs[0,0].text(x,y-20,'a)',color='black',bbox=dict(facecolor='white',alpha=0.8))
    axs[0,1].text(x,y-20,'b)',color='black',bbox=dict(facecolor='white',alpha=0.8))
    axs[1,0].text(x,y-20,'c)',color='black',bbox=dict(facecolor='white',alpha=0.8))
    axs[1,1].text(x,y-20,'d)',color='black',bbox=dict(facecolor='white',alpha=0.8))
    axs[2,0].text(x,y-20,'e)',color='black',bbox=dict(facecolor='white',alpha=0.8))
    axs[2,1].text(x,y-20,'f)',color='black',bbox=dict(facecolor='white',alpha=0.8))
    axs[3,0].text(x,y-20,'g)',color='black',bbox=dict(facecolor='white',alpha=0.8))
    axs[3,1].text(x,y-20,'h)',color='black',bbox=dict(facecolor='white',alpha=0.8))


    plt.tight_layout(rect=[0, 0, 1, 0.95])
    title = 'F = ' + '%.2f' % ff + ', sat = ' + '%.2f' % s0 + ', $\\Delta$ = ' + '%.2f' % det + ' $\\Gamma$' + '\n $V_{Hex}$= ' + '%.2f' % hexvolt + ' V'
    title_acc = '$a_{peak}$ = ' + '%.2f' % amax + ' $ m/s^2$'
    axs[0, 0].text(-75, 12, title, fontsize=label_font-3)
    axs[0, 0].text(-75, 9, title_acc, fontsize=label_font -3)
    if save_plot == False:
        plt.show()
    else:
        plt.savefig('8_heatmap_test', dpi=200)

'''
accleration_curve_pylcp_plot takes as argument the detunning and saturation of a pylcp file and looks for it in all the
files Max generated. It also needs the ff,s0 and det that would fit the given acceleration curve.
It plots both curves to compare
'''

def acceleration_curve_pylcp_plot(det_pylcp,sat_pylcp,ff,s0,det,save_plot=False):
    velocity_sim, ascat_sim = load_forces('obe', det_pylcp, sat_pylcp, molecule, additional_title='2.0_45',
                                          velocity_in_y=False, directory='data_grid')
    acc_fit=ascat(velocity_sim,ff,s0,det)
    amax=min(acc_fit)
    title = ('F = ' + '%.2f' % ff + ', sat = ' + '%.2f' % s0 + ', $\\Delta$ = ' + '%.2f' % det + ' $\\Gamma$'
             )
    title_acc = '$a_{peak}$ = ' + '%.2f' % amax + ' $ m/s^2$'
    title_pylcp='sat_pylcp = ' + '%.2f' % sat_pylcp + '\n $\\Delta$_pylcp = ' + '%.2f' % det_pylcp + ' $\\Gamma$'

    fig = plt.figure(figsize=(10, 6))
    gs = GridSpec(1, 2)

    ax_text1 = fig.add_subplot(gs[0, 0])
    ax_text1.axis('off')  # Turn off the axis
    ax_text1.text(0.5, 0.5, title_pylcp, va='center', ha='center', fontsize=15)
    ax_text1.text(0.5, 1, title, va='center', ha='center', fontsize=15)
    ax_text1.text(0.5, 0, title_acc, va='center', ha='center', fontsize=15)

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(velocity_sim,ascat_sim,'ob')
    ax1.plot(velocity_sim,acc_fit,'r')
    ax1.set_ylabel('acceleration $(m/s^2)$')
    ax1.set_xlabel('velocity (m/s)')


    plt.plot(velocity_sim,ascat_sim,'ob')
    plt.plot(velocity_sim,acc_fit,'r')
    plt.xlabel('velocity (m/s)')
    plt.ylabel('acceleration $(m/s^2)$')
    plt.legend(['pylcp curve','fitted curved'])
    title='acceleration_curve'+'%.2f%.2f'%(det_pylcp,sat_pylcp)+'.png'
    save_title = '../Acceleration_curves/' + 'Acc_' + '%.2f_%.2f_%.2f_' % (ff, s0, det) + '.png'
    if save_plot==False:
        plt.show()
    else:
        plt.savefig(save_title,dpi=2000)

def acceleration_curve_plot(ffs,sats,detuns,sats_pylcp,title,import_pylcp=True):
    n=len(sats)
    velocity_sim1, ascat_sim1 = load_forces('obe', -0.5, 5, molecule, additional_title='2.0_45',
                                          velocity_in_y=False, directory='data_grid')

    #x=np.linspace(0,10,1001)
    if import_pylcp:
        velocity_sim1, ascat_sim1 = load_forces('obe', detuns[0], sats[0], molecule, additional_title='2.0_45',
                                                velocity_in_y=False, directory='data_grid')
        velocity_sim2, ascat_sim2 = load_forces('obe', detuns[-1], sats[-1], molecule, additional_title='2.0_45',
                                                velocity_in_y=False, directory='data_grid')
        for i in range(len(ffs)):
            if i == 0:
                plt.scatter(velocity_sim1,ascat_sim1)
            elif i == len(ffs)-1:
                plt.scatter(velocity_sim2,ascat_sim2)
            else:
                plt.plot(velocity_sim1,ascat(velocity_sim1,ffs[i],sats[i],detuns[i]))
    else:
        for i in range(len(ffs)):
            plt.plot(velocity_sim1, ascat(velocity_sim1, ffs[i], sats[i], detuns[i]))

    plt.xlabel('Velocities (m/s)')
    plt.ylabel('Acceleration $(m/s^2)$')
    plt.legend(sats_pylcp)
    plt.title(title)
    plt.show()



'''
plots the trajectories of the molecules
'''

def plot_trajectories(x1,z1,x2,z2,x3,z3,ff,s0,det,hexvolt):
    nn = len(x1)
    fig = plt.figure(figsize=(10, 9))
    gs = GridSpec(3, 2, width_ratios=[1, 1])
    ax_text1 = fig.add_subplot(gs[0, 0])
    ax_text1.axis('off')  # Turn off the axis
    text1 = "2D LC on"
    amax = min(ascat(np.linspace(-10, 10, 101), ff, s0, det))
    title = 'F = ' + '%.2f' % ff + ', sat = ' + '%.2f' % s0 + ', $\\Delta$ = ' + '%.2f' % det + ' $\\Gamma$' + '\n $V_{Hex}$= ' + '%.2f' % hexvolt + ' V'
    ax_text1.text(0.5, 0.5, text1, va='center', ha='center', fontsize=20)
    ax_text1.text(0.5, 1, title, va='center', ha='center', fontsize=15)
    ax_text2 = fig.add_subplot(gs[1, 0])
    ax_text2.axis('off')  # Turn off the axis
    text2 = "2D LC off"
    ax_text2.text(0.5, 0.5, text2, va='center', ha='center', fontsize=20)
    ax_text3 = fig.add_subplot(gs[2, 0])
    ax_text3.axis('off')  # Turn off the axis
    text3 = "(2D-LC and Hexapole) off"
    ax_text3.text(0.5, 0.5, text3, va='center', ha='center', fontsize=20)
    # Add the first graph to the second column, first row
    ax1 = fig.add_subplot(gs[0, 1])
    for n in range(nn):
        z=np.array(z1[n])*1e3
        x=np.array(x1[n])*1e3
        ax1.plot(z, x, 'b')
    ax1.set_xlabel('z_Axis (mm)')
    ax1.set_ylabel('x_Axis (mm)')

    ax2 = fig.add_subplot(gs[1, 1], sharex=ax1)
    for n in range(nn):
        z=np.array(z2[n])*1e3
        x=np.array(x2[n])*1e3
        ax2.plot(z, x, 'b')
       # ax2.set_ylim([-0.3, 0.3])
    ax2.set_xlabel('z_Axis (mm)')
    ax2.set_ylabel('x_Axis (mm)')

    ax3 = fig.add_subplot(gs[2, 1], sharex=ax1)
    for n in range(nn):
        z=np.array(z3[n])*1e3
        x=np.array(x3[n])*1e3
        ax3.plot(z, x, 'b')
        #ax3.set_ylim([-0.3, 0.3])
    ax3.set_xlabel('z_Axis (mm)')
    ax3.set_ylabel('x_Axis (mm)')

    plt.tight_layout()
    plt.show()
'''
It works to plot n_heatmaps but it would just use Annos code to plot n heatmaps
'''

def plot_n_heatmaps(xs,ys,ffs,s0s,dets,hexvolts,hfs_bool,plot_titles,plot_circle):
    n=len(xs)
    fig = plt.figure(figsize=(10,8))# int((n/2)*2.3333)))
    gs = GridSpec(int(n/2), 2, width_ratios=[1, 1])
    x_edges = np.linspace(-25, 25, 128)
    y_edges = np.linspace(-12.5, 12.5, 128)


    counter=0
    a=0.8
    x = -23
    y = 9
    axes = {}
    Heatmaps=[]
    for i in range(2):
        for j in range((int(n/2))):
            Xs=xs[counter]
            Ys=ys[counter]
            H, xedges, yedges = np.histogram2d(np.array(Xs) * 1e3, np.array(Ys) * 1e3, bins=[x_edges, y_edges])
            Heatmaps.append(H.T)


            counter+=1
    #vmin=min([H.min() for H in Heatmaps])
    #vmax=max([H.max() for H in Heatmaps])
    #print(vmin,vmax)
    counter=0
    for i in range(2):
        for j in range((int(n/2))):
            Xs = xs[counter]
            Ys = ys[counter]
            H, xedges, yedges = np.histogram2d(np.array(Xs) * 1e3, np.array(Ys) * 1e3, bins=[x_edges, y_edges])
            H_shape=list(H.shape)
            vmin=H.min()
            vmax=H.max()
            for k in range(H_shape[0]):
                for l in range(H_shape[1]):
                    if H[k,l]>0.9*vmax:
                        H[k,l]=vmax*0.9

            ax = fig.add_subplot(gs[j, i])
            axes[j, i] = ax
            cax = ax.imshow(H.T,vmin=vmin,vmax=vmax*0.9, origin='lower', aspect='auto', cmap='sunset',extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
            #title='F = '+'%.2f'%ffs[counter]+', s = '+'%.2f' % s0s[counter]+', det = '+'%.2f'%-dets[counter]+ ', $V_{hex} = $'+'%.2f'%hexvolts[counter]+' V'
            #ax.set_title(title)
            if i==0:
                ax.set_ylabel('y (mm)')
                if plot_titles:
                    title = ('F = ' + '%.2f' % ffs[counter] + ', s = ' + '%.2f' % s0s[counter] +
                             ', det = ' + '%.2f' % dets[counter] + ', $V_{hex} = $' + '%.2f' % hexvolts[counter]
                             + ' V'+', hfs = '+str(hfs_bool[counter]))
                    ax.set_title(title)
            else:
                if plot_titles:
                    title = ('F = ' + '%.2f' % ffs[counter] + ', s = ' + '%.2f' % s0s[counter] + ', det = '
                             + '%.2f' % dets[counter] + ', $V_{hex} = $' + '%.2f' % hexvolts[counter]
                             + ' V'+', hfs = '+str(hfs_bool[counter]))
                    ax.set_title(title)

            if j==int(n/2-1):
                ax.set_xlabel('x (mm)')

            ax.set_box_aspect(0.55)

            fig.colorbar(cax, ax=ax, shrink=a, aspect=20 * a)
            r = 5
            points_in_c1 = points_in_circle(np.array(Xs) * 1e3, np.array(Ys) * 1e3, r)
            ax.text(x, y, '# in circle= ' + str(points_in_c1), fontsize=10, color='black',
                           bbox=dict(facecolor='white', alpha=0.8))
            ax.text(x, y - 20, '%.2f'%(hexvolts[counter]*1e-3)+' kV', color='black', bbox=dict(facecolor='white', alpha=0.8))
            counter += 1
            if plot_circle == False:
                continue
            else:
                theta = np.linspace(0, 2 * pi, 1001)
                x = 5 * np.cos(theta)
                y = 5 * np.sin(theta)
                ax.plot(x, y, 'w')
    plt.tight_layout()
    plt.show()

def plot_mol_vs_voltage(Voltagehex, num_molecules,title = '', r=[5,2.5],make_title = False,
                        save_plot = False,normalize=True):
    fig,ax1=plt.subplots()
    fig.set_size_inches(8,6)
    ax1.set_xlabel('Voltage [V]')
    ax1.set_ylabel('Number of Molecules')
    ax1.tick_params(axis='y')
    if normalize:
        for i in range(len(num_molecules)):
            num_molecules[i] = np.array(num_molecules[i])/max(num_molecules[i])
            ax1.scatter(Voltagehex, num_molecules[i],marker = 'o')
    else:
        for i in range(len(num_molecules)):
            num_molecules[i] = np.array(num_molecules[i])
            ax1.scatter(Voltagehex, num_molecules[i],marker = 'o')
    ax1.legend(['r = '+"%.2f"%r[0]+' mm','r = '+"%.2f"%r[1]+' mm'])
    if make_title != False:
        plt.title(title)
    fig.tight_layout()
    if save_plot == False:
        plt.show()
    else:

        plt.savefig('../voltage_vs_num_molecules/'+make_title+'.png')

'''
    a = 0.7
    label_font = 18
    ticks_font = 8
    Ldetection1 = Ldetection1 * 1e3
    Ldetection2 = Ldetection2 * 1e3
    fig = plt.figure(figsize=(10, 6))
    gs = GridSpec(3, 3, width_ratios=[1, 2, 2])
    ax_text1 = fig.add_subplot(gs[0, 0])
    ax_text1.axis('off')  # Turn off the axis
    text1 = "2D LC on"
    amax = min(ascat(np.linspace(-10, 10, 101), ff, s0, det))
    title='F = '+'%.2f'%ff+', sat = '+'%.2f'%s0+', $\\Delta$ = '+'%.2f'%-det+' $\\Gamma$'+'\n $V_{Hex}$= '+'%.2f'%hexvolt+' V'
    ax_text1.text(0.5, 0.5, text1, va='center', ha='center', fontsize=20)
    ax_text1.text(0.5, 1, title, va='center', ha='center', fontsize=15)
    ax_text2 = fig.add_subplot(gs[1, 0])
    ax_text2.axis('off')  # Turn off the axis
    text2 = "2D LC off"
    ax_text2.text(0.5, 0.5, text2, va='center', ha='center', fontsize=20)
    ax_text3 = fig.add_subplot(gs[2, 0])
    ax_text3.axis('off')  # Turn off the axis
    text3 = "Hexapole off"
    ax_text3.text(0.5, 0.5, text3, va='center', ha='center', fontsize=20)


    ax1 = fig.add_subplot(gs[0, 1])
    cax1 = ax1.imshow(H1.T, origin='lower', aspect='auto', cmap='sunset',
                            extent=[xedges1[0], xedges1[-1], yedges1[0], yedges1[-1]])
    ax1.set_title('z= ' + '%.2f' % Ldetection1 + ' mm', fontsize=20)
    # ax1.set_xlabel('x axis (mm)')
    ax1.set_ylabel('y (mm)', fontsize=label_font - 3)
    cbar1 = fig.colorbar(cax1, ax=ax1, shrink=a, aspect=20 * a)
    cbar1.ax.tick_params(labelsize=ticks_font - 5)
    ax1.set_box_aspect(0.55)
    ax1.tick_params(axis='y', labelsize=ticks_font)

    ax2 = fig.add_subplot(gs[1, 1])
    cax2 = ax2.imshow(H2.T, origin='lower', aspect='auto', cmap='sunset',
                      extent=[xedges2[0], xedges2[-1], yedges2[0], yedges2[-1]])
    #ax2.set_xlabel('x axis (mm)')
    ax2.set_ylabel('y (mm)', fontsize=label_font - 3)
    cbar2 = fig.colorbar(cax2, ax=ax2, shrink=a, aspect=20 * a)
    cbar2.ax.tick_params(labelsize=ticks_font - 5)
    ax2.set_box_aspect(0.55)
    ax2.tick_params(axis='y', labelsize=ticks_font)

    ax3=fig.add_subplot(gs[0, 2])
    cax3 = ax3.imshow(H3.T, origin='lower', aspect='auto', cmap='sunset',
                      extent=[xedges3[0], xedges3[-1], yedges3[0], yedges3[-1]])
    ax3.set_title('z= ' + '%.2f' % Ldetection2 + ' mm', fontsize=20)
    # ax1.set_xlabel('x axis (mm)')
    #ax3.set_ylabel('y (mm)', fontsize=label_font - 3)
    cbar3 = fig.colorbar(cax3, ax=ax3, shrink=a, aspect=20 * a)
    cbar3.ax.tick_params(labelsize=ticks_font - 5)
    ax3.set_box_aspect(0.55)
    ax3.tick_params(axis='y', labelsize=ticks_font)

    ax4=fig.add_subplot(gs[1, 2])
    cax4 = ax4.imshow(H4.T, origin='lower', aspect='auto', cmap='sunset',
                      extent=[xedges4[0], xedges4[-1], yedges4[0], yedges4[-1]])
    #ax4.set_title('z= ' + '%.2f' % Ldetection1 + ' mm', fontsize=20)
    # ax1.set_xlabel('x axis (mm)')
    #ax4.set_ylabel('y (mm)', fontsize=label_font - 3)
    cbar4 = fig.colorbar(cax4, ax=ax4, shrink=a, aspect=20 * a)
    cbar4.ax.tick_params(labelsize=ticks_font - 5)
    ax4.set_box_aspect(0.55)
    ax4.tick_params(axis='y', labelsize=ticks_font)

    ax5=fig.add_subplot(gs[2, 1])
    cax5 = ax5.imshow(H5.T, origin='lower', aspect='auto', cmap='sunset',
                      extent=[xedges5[0], xedges5[-1], yedges5[0], yedges5[-1]])
    ax5.set_xlabel('x axis (mm)', fontsize=label_font - 3)
    ax5.set_ylabel('y axis (mm)', fontsize=label_font - 3)
    #ax5.set_ylabel('y (mm)', fontsize=label_font - 3)
    cbar5 = fig.colorbar(cax5, ax=ax5, shrink=a, aspect=20 * a)
    cbar5.ax.tick_params(labelsize=ticks_font - 5)
    ax5.set_box_aspect(0.55)
    ax5.tick_params(axis='y', labelsize=ticks_font)

    ax6=fig.add_subplot(gs[2, 2])
    cax6 = ax6.imshow(H6.T, origin='lower', aspect='auto', cmap='sunset',
                      extent=[xedges6[0], xedges6[-1], yedges6[0], yedges6[-1]])
    ax6.set_xlabel('x axis (mm)', fontsize=label_font - 3)
    cbar6 = fig.colorbar(cax6, ax=ax6, shrink=a, aspect=20 * a)
    cbar6.ax.tick_params(labelsize=ticks_font - 5)
    ax6.set_box_aspect(0.55)
    ax6.tick_params(axis='y', labelsize=ticks_font)

    if plot_circle == False:
        #save_title = '6_heatmap/' + 'Heatmap_' + '%.2f_%.2f_%.2f_%.2f' % (ff, s0, det, hexvolt) + '.png'
        save_title='figure.png'
    else:
        theta = np.linspace(0, 2 * pi, 1001)
        x = 5 * np.cos(theta)
        y = 5 * np.sin(theta)
        ax1.plot(x, y, 'w')
        ax2.plot(x, y, 'w')
        ax3.plot(x, y, 'w')
        ax4.plot(x, y, 'w')
        ax5.plot(x, y, 'w')
        ax6.plot(x, y, 'w')
        #save_title = '4_heatmap/' + '%.2f_%.2f_%.2f_%.2f' % (ff, s0, det, hexvolt) + '_circle' + '.png'
        save_title='figure.png'

    r = 5
    points_in_c1 = points_in_circle(np.array(x1) * 1e3, np.array(y1) * 1e3, r)
    points_in_c2 = points_in_circle(np.array(x2) * 1e3, np.array(y2) * 1e3, r)
    points_in_c3 = points_in_circle(np.array(x3) * 1e3, np.array(y3) * 1e3, r)
    points_in_c4 = points_in_circle(np.array(x4) * 1e3, np.array(y4) * 1e3, r)
    points_in_c5 = points_in_circle(np.array(x5) * 1e3, np.array(y5) * 1e3, r)
    points_in_c6 = points_in_circle(np.array(x6) * 1e3, np.array(y6) * 1e3, r)

    x=-23
    y=9
    ax1.text(x, y, '# in circle= ' + str(points_in_c1), fontsize=10, color='black',
                   bbox=dict(facecolor='white', alpha=0.8))
    ax2.text(x, y, '# in circle= ' + str(points_in_c2), fontsize=10, color='black',
                   bbox=dict(facecolor='white', alpha=0.8))
    ax3.text(x, y, '# in circle= ' + str(points_in_c3), fontsize=10, color='black',
                   bbox=dict(facecolor='white', alpha=0.8))
    ax4.text(x, y, '# in circle= ' + str(points_in_c4), fontsize=10, color='black',
                   bbox=dict(facecolor='white', alpha=0.8))
    ax5.text(x, y, '# in circle= ' + str(points_in_c5), fontsize=10, color='black',
             bbox=dict(facecolor='white', alpha=0.8))
    ax6.text(x, y, '# in circle= ' + str(points_in_c6), fontsize=10, color='black',
             bbox=dict(facecolor='white', alpha=0.8))
    amax = min(ascat(np.linspace(-10, 10, 101), ff, s0, det))

    plt.tight_layout()


    if plot_save == False:
        plt.show()
    else:
        plt.savefig(save_title, dpi=200)
'''




