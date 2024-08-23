import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from plot_maker import *
'''
This script is used to defined the functions of the given simulations we would like to use.
The general structure of any of this functions is to defined the initial distribution of nn molecules (given by initial)
Then we simulate their trajectories. it needs the parameters initial,nn , nj, ff, s0, det, voltage. if we want 
lasercooling we can either defined lc=True or not defined it at all. If we don't want lasercoooling we defined lc=False
To turn the hexapole i recommend to just pass a small voltage to the hexapole voltage instead of turning it off.
Finally we need the function xy_distribution that will return all the x,y,z,vx,vy coordinates of the distribution for a
given distance.
'''
'''
simulate_lc returns a heatmap of lc and lc off
'''
def simulate_lc(nn ,nj ,fudgefactor ,s00 ,detuning ,Voltagehex ,Ldet1):
    initial = [phasespaceellipse2D(xx0,dxx) for i in range(nn)]
    xlist,ylist,zlist,vx,vy,ax,ay = trajectory_simulation(initial ,nn ,nj ,ff = fudgefactor ,s0 = s00,detun = detuning,
                                                          phi0hex = Voltagehex,lc = True)
    xlist2,ylist2,zlist2,vx2,vy2,ax2,ay2=trajectory_simulation(initial ,nn ,nj ,ff =fudgefactor,s0=s00 ,detun=detuning,
                                                            phi0hex = Voltagehex, lc = False)
    xc1,yc1,zc1,vxc,vyc=xy_distribution(xlist,ylist,zlist,vx,vy,Ldet1)
    xc2,yc2,zc2,vxc2,vyc2=xy_distribution(xlist2,ylist2,zlist2,vx2,vy2,Ldet1)
    plot_2_heatmap(xc1,yc1,xc2,yc2,fudgefactor,s00,detuning,hexvolt=Voltagehex,Ldetection=Ldet1)

'''
simulate_lc_2_distance returns a heatmap of lc and lc off for two given distances
'''
def simulate_lc_2_distance(nn,nj,fudgefactor,s00,detuning,Voltagehex
                           ,Ldet1 ,Ldet2 ,plot_circle = False):
    initial = [phasespaceellipse2D(xx0,dxx) for i in range(nn)]
    xlist,ylist,zlist,vx,vy,ax,ay = trajectory_simulation(initial ,nn ,nj , ff = fudgefactor ,s0 = s00 ,detun = detuning
                                                          ,phi0hex = Voltagehex,lc = True)
    xlist2,ylist2,zlist2,vx2,vy2,ax2,ay2 = trajectory_simulation(initial ,nn ,nj ,ff = fudgefactor ,s0 = s00,
                                                                 detun = detuning, phi0hex = Voltagehex,lc = False)
    xc1,yc1,zc1,vxc1,vyc1 = xy_distribution(xlist,ylist,zlist,vx,vy,Ldet1)
    xc2,yc2,zc2,vxc2,vyc2 = xy_distribution(xlist2,ylist2,zlist2,vx2,vy2,Ldet1)
    xc3,yc3,zc3,vxc3,vyc3 = xy_distribution(xlist,ylist,zlist,vx,vy,Ldet2)
    xc4,yc4,zc4,vxc4,vyc4 = xy_distribution(xlist2,ylist2,zlist2,vx2,vy2,Ldet2)
    legends=['2D-LC on','2D-LC off','z = '+'%.2f'%Ldet1+' mm','z = '+'%.2f'%Ldet2+' mm']
    plot_4_heatmap(xc1 ,yc1 ,xc2 ,yc2 ,xc3 ,yc3 ,xc4 ,yc4 ,fudgefactor ,s00 ,detuning
                   ,hexvolt = Voltagehex ,Ldetection1 = Ldet1 ,Ldetection2 = Ldet2 ,legends = legends,
                   plot_circle = plot_circle)
'''
simulate_x_vx_dis return a scatter plot for a given situation
'''
def simulate_x_vx_dis(nn ,nj ,fudgefactor ,s00 ,detuning ,Voltagehex ,Ldet1):
    initial = [phasespaceellipse2D(xx0,dxx) for i in range(nn)]
    xlist,ylist,zlist,vx,vy,ax,ay = trajectory_simulation(initial ,nn ,nj ,ff = fudgefactor ,s0 = s00 ,detun = detuning,
                                                          phi0hex = Voltagehex, lc = True)
    xc1,yc1,zc1,vxc1,vyc1 = xy_distribution(xlist ,ylist ,zlist ,vx ,vy ,Ldet1)
    plt.scatter(xc1,vxc1)
    plt.xlabel('x distance (m)')
    plt.ylabel('velocity in x (m/s)')
    plt.show()
'''
simulate_vr_hist returns two histograms of velocities, one of laser cooling and one without lasr cooling
'''

def simulate_vr_hist(nn,nj,fudgefactor,s00,detuning,Voltagehex,Ldet1,save_plot=False):
    initial = [phasespaceellipse2D(xx0,dxx) for i in range(nn)]
    xlist,ylist,zlist,vx,vy,ax,ay = trajectory_simulation(initial, nn, nj, ff = fudgefactor, s0 = s00, detun = detuning,
                                                          phi0hex = Voltagehex, lc = True)
    xlist2,ylist2,zlist2,vx2,vy2,ax2,ay2 = trajectory_simulation(initial, nn, nj, ff = fudgefactor, s0 = s00,
                                                            detun = detuning, phi0hex = Voltagehex, lc = False)
    xc1,yc1,zc1,vxc1,vyc1 = xy_distribution(xlist, ylist, zlist, vx, vy, Ldet1)
    xc2,yc2,zc2,vxc2,vyc2 = xy_distribution(xlist2, ylist2, zlist2, vx2, vy2, Ldet1)
    velr_1 = [np.sqrt(vxc1[i]**2+vyc1[i]**2) for i in range(len(vxc1))]
    velr_2 = [np.sqrt(vxc2[i]**2+vyc2[i]**2) for i in range(len(vxc2))]
    velr_1 = [vel for vel in velr_1 if vel<2]
    velr_2 = [vel for vel in velr_2 if vel<2]
    plot_hist2(velr_1,velr_2,fudgefactor,s00,detuning,Voltagehex,Ldet1, save_plot=save_plot)


def simulate_lc_hex_off_histogram(nn, nj, fudgefactor, s00, detuning, Voltagehex, Ldet1, Ldet2,lfs=0.666):
    initial = [phasespaceellipse2D(xx0, dxx,lfs_percentage=lfs) for i in range(nn)]
    xlist, ylist, zlist, vx, vy, ax, ay = trajectory_simulation(initial, nn, nj, ff=fudgefactor, s0=s00, detun=detuning,
                                                                phi0hex=Voltagehex, lc=True)
    xlist2, ylist2, zlist2, vx2, vy2, ax2, ay2 = trajectory_simulation(initial, nn, nj, ff=fudgefactor, s0=s00,
                                                                       detun=detuning, phi0hex=Voltagehex, lc=False)
    xlist3, ylist3, zlist3, vx3, vy3, ax3, ay3 = trajectory_simulation(initial, nn, nj, ff=fudgefactor, s0=s00,
                                                                       detun=detuning, phi0hex=Voltagehex, lc=False,
                                                                       hex=False)
    xc1, yc1, zc1, vxc1, vyc1 = xy_distribution(xlist, ylist, zlist, vx, vy, Ldet1)
    xc2, yc2, zc2, vxc2, vyc2 = xy_distribution(xlist2, ylist2, zlist2, vx2, vy2, Ldet1)
    xc3, yc3, zc3, vxc3, vyc3 = xy_distribution(xlist3, ylist3, zlist3, vx3, vy3, Ldet1)
    vx_list=[vxc1,vxc2,vxc3]#,vxc4,vxc5,vxc6]
    vy_list=[vyc1,vyc2,vyc3]#,vyc4,vyc5,vyc6]
    vr_final=[]
    for i in range(len(vx_list)):
        vel = [np.sqrt(vx_list[i][j]**2+vy_list[i][j]**2) for j in range(len(vx_list[i]))]
        vel = [vel1 for vel1 in vel if vel1<2]
        vr_final.append(vel)
    plot_6_hist_vel(vr_final, fudgefactor, s00, detuning, Voltagehex, bins=100)

'''
simulate_lc_2_distance_hex_off returns 6 heatmaps. for two distances it returns the heatmaps for lc-hex onn
lc off hex on and everything off.
'''
def simulate_lc_2_distance_hex_off(nn, nj, fudgefactor, s00, detuning, Voltagehex, Ldet1, Ldet2,lfs= 0.666,
                                   plot_circle = False, save_plot = False, extra_title = '',
                                   sub_doppler = False,par_sub_doppler = np.zeros(6)):
    initial = [phasespaceellipse2D(xx0, dxx,lfs_percentage=lfs) for i in range(nn)]
    xlist, ylist, zlist, vx, vy, ax, ay = trajectory_simulation(initial, nn, nj, ff = fudgefactor, s0 = s00,
                                                                detun = detuning, phi0hex = Voltagehex, lc = True,
                                                                sub_doppler=sub_doppler,par_sub_doppler=par_sub_doppler)
    xlist2, ylist2, zlist2, vx2, vy2, ax2, ay2 = trajectory_simulation(initial, nn, nj, ff = fudgefactor, s0 = s00,
                                                                detun = detuning, phi0hex = Voltagehex, lc = False,
                                                                sub_doppler=sub_doppler,par_sub_doppler=par_sub_doppler)
    xlist3, ylist3, zlist3, vx3, vy3, ax3, ay3 = trajectory_simulation(initial, nn, nj, ff = fudgefactor, s0 = s00,
                                                                detun = detuning, phi0hex = 1.e-3, lc = False)
    xc1, yc1, zc1, vxc1, vyc1 = xy_distribution(xlist, ylist, zlist, vx, vy, Ldet1)
    xc2, yc2, zc2, vxc2, vyc2 = xy_distribution(xlist2, ylist2, zlist2, vx2, vy2, Ldet1)
    xc3, yc3, zc3, vxc3, vyc3 = xy_distribution(xlist, ylist, zlist, vx, vy, Ldet2)
    xc4, yc4, zc4, vxc4, vyc4 = xy_distribution(xlist2, ylist2, zlist2, vx2, vy2, Ldet2)
    xc5, yc5, zc5, vxc5, vyc5 = xy_distribution(xlist3, ylist3, zlist3, vx3, vy3, Ldet1)
    xc6, yc6, zc6, vxc6, vyc6 = xy_distribution(xlist3, ylist3, zlist3, vx3, vy3, Ldet2)
    plot_6_heatmap(xc1, yc1, xc2, yc2, xc3, yc3, xc4, yc4, xc5, yc5, xc6, yc6,fudgefactor,
                   s00, detuning, hexvolt = Voltagehex, Ldetection1 = Ldet1, Ldetection2 = Ldet2,
                   plot_circle = plot_circle,save_plot = save_plot,extra_title = extra_title)

def simulate_lc_2_distance_hex_off_lH(nn, nj, fudgefactor, s00, detuning, Voltagehex, Ldet1, Ldet2,lfs = 0.666,
                                   plot_circle = False, save_plot = False, extra_title = '',
                                   sub_doppler = False,par_sub_doppler = np.zeros(6)):
    initial = [phasespaceellipse2D(xx0, dxx,lfs_percentage=lfs) for i in range(int(nn))]
    xlist, ylist, zlist, vx, vy, ax, ay = trajectory_simulation(initial, nn, nj, ff = fudgefactor, s0 = s00,
                                                                detun = detuning, phi0hex = Voltagehex, lc = True,
                                                                sub_doppler=sub_doppler,par_sub_doppler=par_sub_doppler)
    xlist2, ylist2, zlist2, vx2, vy2, ax2, ay2 = trajectory_simulation(initial, nn, nj, ff = fudgefactor, s0 = s00,
                                                                detun = detuning, phi0hex = Voltagehex, lc = False,
                                                                sub_doppler=sub_doppler,par_sub_doppler=par_sub_doppler)
    xlist3, ylist3, zlist3, vx3, vy3, ax3, ay3 = trajectory_simulation(initial, nn, nj, ff = fudgefactor, s0 = s00,
                                                                detun = detuning, phi0hex = 1.e-3, lc = False)
    xlist4, ylist4, zlist4, vx4, vy4, ax4, ay4 = trajectory_simulation(initial, nn, nj, ff = fudgefactor, s0 = s00,
                                                                detun = -detuning, phi0hex = Voltagehex, lc = True)
    xc1, yc1, zc1, vxc1, vyc1 = xy_distribution(xlist, ylist, zlist, vx, vy, Ldet1)
    xc2, yc2, zc2, vxc2, vyc2 = xy_distribution(xlist2, ylist2, zlist2, vx2, vy2, Ldet1)
    xc3, yc3, zc3, vxc3, vyc3 = xy_distribution(xlist, ylist, zlist, vx, vy, Ldet2)
    xc4, yc4, zc4, vxc4, vyc4 = xy_distribution(xlist2, ylist2, zlist2, vx2, vy2, Ldet2)
    xc5, yc5, zc5, vxc5, vyc5 = xy_distribution(xlist3, ylist3, zlist3, vx3, vy3, Ldet1)
    xc6, yc6, zc6, vxc6, vyc6 = xy_distribution(xlist3, ylist3, zlist3, vx3, vy3, Ldet2)
    xc7, yc7, zc7, vxc7, vyc7 = xy_distribution(xlist4, ylist4, zlist4, vx4, vy4, Ldet1)
    xc8, yc8, zc8, vxc8, vyc8 = xy_distribution(xlist4, ylist4, zlist4, vx4, vy4, Ldet2)

    plot_8_heatmap(xc1, yc1, xc2, yc2, xc3, yc3, xc4, yc4, xc5, yc5, xc6, yc6,xc7, yc7,xc8, yc8,
                   fudgefactor, s00, detuning, hexvolt = Voltagehex, Ldetection1 = Ldet1,
                   Ldetection2 = Ldet2,plot_circle = plot_circle,save_plot = save_plot,extra_title = extra_title)

def simulate_trajectories_111(nn, nj, fudgefactor, s00, detuning, Voltagehex):
    initial = [phasespaceellipse2D(xx0, dxx) for i in range(nn)]
    xlist, ylist, zlist, vx, vy, ax, ay = trajectory_simulation(initial, nn, nj, ff = fudgefactor, s0 = s00,
                                                                detun = detuning, phi0hex = Voltagehex, lc = True)
    xlist2, ylist2, zlist2, vx2, vy2, ax2, ay2 = trajectory_simulation(initial, nn, nj, ff = fudgefactor, s0=s00,
                                                                detun = detuning, phi0hex = Voltagehex, lc = False)
    xlist3, ylist3, zlist3, vx3, vy3, ax3, ay3 = trajectory_simulation(initial, nn, nj, ff = fudgefactor, s0 = s00,
                                                                detun = detuning, phi0hex = Voltagehex, lc = False,
                                                                hex = False)
    plot_trajectories(xlist, zlist, xlist2, zlist2, xlist3, zlist3, fudgefactor, s00, detuning, Voltagehex)

'''
simulate_n_lc returns n heatmaps although I would recommend to just use Annos program to plots this
'''
def simulate_n_lc(nn, nj, fudgefactor, s00, detuning, Voltagehex, Ldet1, lc_bool, hex_bool, lfs,
                                                plot_titles = False, plot_circle = False):
    n=len(Voltagehex)
    xs=[]
    ys=[]
    for i in range(n):
        if Voltagehex[i]==0.0:
            initial = [phasespaceellipse2D(xx0, dxx,lfs_percentage=lfs[i]) for _ in range(nn)]
            Voltagehex[i]=0.01
            x,y,z,vx,vy,ax,ay = trajectory_simulation(initial,nn,nj, ff = fudgefactor[i], s0 = s00[i],
                                                      detun = detuning[i], phi0hex = Voltagehex[i], lc = lc_bool[i],
                                                      hex = hex_bool[i])
            xf,yf,zf,vxf,vyf = xy_distribution(x,y,z,vx,vy,Ldet1)
            xs.append(xf)
            ys.append(yf)
        else:
            initial = [phasespaceellipse2D(xx0, dxx,lfs_percentage=lfs[i]) for _ in range(nn)]
            x, y, z, vx, vy, ax, ay = trajectory_simulation(initial, nn, nj, ff = fudgefactor[i], s0 = s00[i],
                                                            detun = detuning[i], phi0hex = Voltagehex[i],
                                                            lc = lc_bool[i], hex = hex_bool[i])
            xf, yf, zf, vxf, vyf = xy_distribution(x, y, z, vx, vy, Ldet1)
            xs.append(xf)
            ys.append(yf)
    plot_n_heatmaps(xs, ys, fudgefactor, s00, detuning, Voltagehex,lfs[i],plot_titles = plot_titles,
                    plot_circle = plot_circle)

'''
From this point forward all the functions are to simulate a specific situation. and can be edited in which ever way needed.
'''
def simulate_number_mol_vs_hexvoltage(nn, nj, fudgefactor, s00, detuning, Voltagehex, Ldet1, lc_bool, lfs=0.666):
    n=len(Voltagehex)
    print(n)
    num_mol1=[]
    num_mol2=[]
    r=5
    for i in range(n):
        initial = [phasespaceellipse2D(xx0, dxx,lfs_percentage = lfs) for _ in range(nn)]
        x, y, z, vx, vy, ax, ay = trajectory_simulation(initial,nn,nj,ff = fudgefactor[i],s0 = s00[i],
                                                        detun = detuning[i], phi0hex = Voltagehex[i],lc = lc_bool[i])
        x1,y1,z1,vx1,vy1 = xy_distribution(x,y,z,vx,vy,Ldet1)
        num_mol1.append(points_in_circle(np.array(x1) * 1e3, np.array(y1) * 1e3, r))
    r = 2.5
    for i in range(n):
        initial = [phasespaceellipse2D(xx0, dxx,lfs_percentage = lfs) for _ in range(nn)]
        x, y, z, vx, vy, ax, ay = trajectory_simulation(initial, nn, nj, ff = fudgefactor[i], s0 = s00[i],
                                                        detun = detuning[i], phi0hex = Voltagehex[i], lc = lc_bool[i])
        x1, y1, z1, vx1, vy1 = xy_distribution(x, y, z, vx, vy, Ldet1)
        num_mol2.append(points_in_circle(np.array(x1) * 1e3, np.array(y1) * 1e3, r))
    plot_mol_vs_voltage(Voltagehex, num_molecules=[num_mol1,num_mol2])


def simulate_y_cooling(nn, nj, fudgefactor, s00, detuning, Voltagehex, Ldet1, Ldet2, plot_circle = False,
                       save_plot = False):
    initial = [phasespaceellipse2D(xx0, dxx) for i in range(nn)]
    xlist, ylist, zlist, vx, vy, ax, ay = trajectory_simulation(initial, nn, nj, ff = fudgefactor, s0 = s00,
                                            detun = detuning, phi0hex = Voltagehex, lc = True, ycooling = False)
    xlist2, ylist2, zlist2, vx2, vy2, ax2, ay2 = trajectory_simulation(initial, nn, nj, ff=fudgefactor, s0=s00,
                                            detun = detuning, phi0hex = Voltagehex, lc = False, ycooling = False)
    xlist3, ylist3, zlist3, vx3, vy3, ax3, ay3 = trajectory_simulation(initial, nn, nj, ff=fudgefactor, s0=s00,
                                            detun = detuning, phi0hex = 0.001, lc = True, ycooling = False, hex = True)
    xlist4, ylist4, zlist4, vx4, vy4, ax4, ay4 = trajectory_simulation(initial, nn, nj, ff=fudgefactor, s0=s00,
                                            detun = detuning, phi0hex = 0.001, lc = False, ycooling = False)
    xc1, yc1, zc1, vxc1, vyc1 = xy_distribution(xlist, ylist, zlist, vx, vy, Ldet1)
    xc2, yc2, zc2, vxc2, vyc2 = xy_distribution(xlist2, ylist2, zlist2, vx2, vy2, Ldet1)
    xc3, yc3, zc3, vxc3, vyc3 = xy_distribution(xlist3, ylist3, zlist3, vx3, vy3, Ldet1)
    xc4, yc4, zc4, vxc4, vyc4 = xy_distribution(xlist4, ylist4, zlist4, vx4, vy4, Ldet1)
    legends=['2D-LC on','2D-LC off','Heapole on','Heapole off']
    plot_4_heatmap(xc1, yc1, xc2, yc2, xc3, yc3, xc4, yc4, fudgefactor, s00, detuning,
                   hexvolt = Voltagehex, Ldetection1 = Ldet1, Ldetection2 = Ldet2, legends = legends,
                   plot_circle = plot_circle, save_plot = save_plot)
def simulate_num_counts_lc_nolc(nn, nj, fudgefactor, s00, detuning, Voltagehex, Ldet1):
    initial = [phasespaceellipse2D(xx0, dxx) for _ in range(nn)]
    radius = np.linspace(0,10,100)
    x1,y1,z1,vx1,vy1,ax1,ay1 = trajectory_simulation(initial, nn, nj, ff = fudgefactor, s0 = s00,detun = detuning,
                                                      phi0hex = Voltagehex, lc = True)
    x2,y2,z2,vx2,vy2,ax2,ay2 = trajectory_simulation(initial,nn,nj,ff = fudgefactor, s0 = s00,detun = detuning,
                                                      phi0hex = Voltagehex, lc = False)
    xc1,yc1,zc1,vxc1,vyc1 = xy_distribution(x1,y1,z1,vx1,vy1,Ldet1)
    xc2,yc2,zc2,vxc2,vyc2 = xy_distribution(x2,y2,z2,vx2,vy2,Ldet1)
    ratio = np.zeros(len(radius))
    for i in range(len(radius)):
        a = points_in_circle(np.array(xc1)*1e3,np.array(yc1)*1e3,radius[i])
        b = points_in_circle(np.array(xc2)*1e3,np.array(yc2)*1e3,radius[i])
        if b == 0:
            b = 1
            ratio[i] = a/b
        else:
            ratio[i] = a/b
    #print(ratio)
    #print(radius)
    plt.plot(radius,ratio,'-o')
    plt.xlabel('Radius (mm)')
    plt.ylabel('Ratio # counts (lc/no_lc)')
    plt.show()

def simulate_num_molecules_detuning(nn,nj,fudgefactor, s00, detuning, Voltagehex, Ldet1, Ldet2,
                                    normalize=True, make_title = False, title=''):
    n=len(detuning)
    radius=[0.3,1,2.5,5]
    counts = [np.zeros(n) for _ in range(len(radius))]
    initial = [phasespaceellipse2D(xx0, dxx) for _ in range(nn)]
    for j in range(len(radius)):
        for i in range(n):
            x, y, z, vx, vy, ax,ay =trajectory_simulation(initial, nn, nj, ff = fudgefactor, s0 = s00
                                                          ,detun=detuning[i],phi0hex = Voltagehex)
            xc1,yc1,zc1,vxc1,vyc1 = xy_distribution(x,y,z,vx,vy,Ldet1)
            counts[j][i]=points_in_circle(np.array(xc1)*1e3,np.array(yc1)*1e3,radius[j])
    plot_mol_vs_voltage(detuning,counts,r=radius,normalize=normalize,title=title,make_title=make_title)


def simulate_and_save_data_n(nn, nj, fudgefactor, s00, detuning, Voltagehex, Ldet1, Ldet2):
    initial = [phasespaceellipse2D(xx0, dxx) for _ in range(nn)]
    n=len(Voltagehex)
    x_edges = np.linspace(-25, 25, 129)
    y_edges = np.linspace(-12.5, 12.5, 65)
    Heatmaps=[]
    for i in range(n):
        x,y,z,vx,vy,ax,ay = trajectory_simulation(initial, nn, nj, ff = fudgefactor, s0 = s00,
                                                  detun = detuning,phi0hex = Voltagehex[i],lc=False)
        xf,yf,zf,vxf,vyf = xy_distribution(x,y,z,vx,vy,Ldet1)
        #export_phase_space('voltage'+'%.2f'%Voltagehex[i],_t=[],_x=xf,_y=yf,_z=zf)

        H1, xedges1, yedges1 = np.histogram2d(np.array(xf) * 1e3, np.array(yf) * 1e3, bins=[x_edges, y_edges])
        title = 'voltage_with_outliers_' + '%.2f' % Voltagehex[i] + '.npy'
        print(H1.shape)
        np.save(title, H1)
'''
simulate_and_save_data is used to make files that contain the x and y coordinates of the nn particles simulated.
It is used to work with the PlotManager and DataManager scripts. There there are functions that read this types of files
and can do the analysis that is already been used in the experiment. 
'''
def simulate_and_save_data(nn,nj,fudgefactor, s00, detuning, Voltagehex, Ldet1,lc=True,title='',directory=''):
    initial = [phasespaceellipse2D(xx0, dxx) for _ in range(nn)]
    x, y, z, vx, vy, ax, ay = trajectory_simulation(initial, nn, nj, ff = fudgefactor,s0=s00,detun=detuning,phi0hex = Voltagehex,lc=lc)
    xf,yf,zf,vxf,vyf=xy_distribution(x,y,z,vx,vy,Ldet1)
    if directory=='':
        np.save(title, (xf,yf))
    else:
        np.save(directory+title,(xf,yf))

'''  
        Heatmaps.append(H1)
        print(i)
    vmin = min([H.min() for H in Heatmaps])
    vmax = max([H.max() for H in Heatmaps])

    for (i,H1) in enumerate(Heatmaps):
        H_shape = list(H1.shape)
        for k in range(H_shape[0]):
            for l in range(H_shape[1]):
                if H1[k, l] > 0.9 * vmax:
                    H1[k, l] = vmax * 0.9
                else:
                    H1[k, l] = H1[k, l] + 4
        title='voltage_'+'%.2f'%Voltagehex[i]+'.npy'
        print(H1.shape)
        np.save(title,H1)

'''


























