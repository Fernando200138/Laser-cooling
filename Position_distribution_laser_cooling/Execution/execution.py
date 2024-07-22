import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from plot_maker import *

def simulate_lc(nn,nj,fudgefactor,s00,detuning,Voltagehex,Ldet1):
    initial=[phasespaceellipse2D(xx0,dxx) for i in range(nn)]
    xlist,ylist,zlist,vx,vy,ax,ay=trajectory_simulation(
    initial,nn,nj,ff=fudgefactor,s0=s00,detun=detuning,phi0hex=Voltagehex,lc=True)
    xlist2,ylist2,zlist2,vx2,vy2,ax2,ay2=trajectory_simulation(
    initial,nn,nj,ff=fudgefactor,s0=s00,detun=detuning,phi0hex=Voltagehex,lc=False)

    xc1,yc1,zc1,vxc,vyc=xy_distribution(xlist,ylist,zlist,vx,vy,Ldet1)
    xc2,yc2,zc2,vxc2,vyc2=xy_distribution(xlist2,ylist2,zlist2,vx2,vy2,Ldet1)
    plot_2_heatmap(xc1,yc1,zc1,xc2,yc2,zc2,fudgefactor,s00,detuning,hexvolt=Voltagehex,Ldetection=Ldet1)


def simulate_lc_2_distance(nn,nj,fudgefactor,s00,detuning,Voltagehex,Ldet1,Ldet2,plot_circle=False):
    initial=[phasespaceellipse2D(xx0,dxx) for i in range(nn)]
    xlist,ylist,zlist,vx,vy,ax,ay=trajectory_simulation(initial,nn,nj,ff=fudgefactor,s0=s00,detun=detuning,phi0hex=Voltagehex,lc=True)
    xlist2,ylist2,zlist2,vx2,vy2,ax2,ay2=trajectory_simulation(initial,nn,nj,ff=fudgefactor,s0=s00,detun=detuning,phi0hex=Voltagehex,lc=False)
    
    xc1,yc1,zc1,vxc1,vyc1=xy_distribution(xlist,ylist,zlist,vx,vy,Ldet1)
    xc2,yc2,zc2,vxc2,vyc2=xy_distribution(xlist2,ylist2,zlist2,vx2,vy2,Ldet1)
    xc3,yc3,zc3,vxc3,vyc3=xy_distribution(xlist,ylist,zlist,vx,vy,Ldet2)
    xc4,yc4,zc4,vxc4,vyc4=xy_distribution(xlist2,ylist2,zlist2,vx2,vy2,Ldet2)
    plot_4_heatmap(xc1,yc1,zc1,xc2,yc2,zc2,xc3,yc3,zc3,xc4,yc4,zc4,fudgefactor,s00,detuning,hexvolt=Voltagehex,Ldetection1=Ldet1,Ldetection2=Ldet2,plot_circle=plot_circle)
        
def simulate_x_vx_dis(nn,nj,fudgefactor,s00,detuning,Voltagehex,Ldet1,Ldet2,plot_circle=False):
    initial=[phasespaceellipse2D(xx0,dxx) for i in range(nn)]
    xlist,ylist,zlist,vx,vy,ax,ay=trajectory_simulation(initial,nn,nj,ff=fudgefactor,s0=s00,detun=detuning,phi0hex=Voltagehex,lc=True)
    xlist2,ylist2,zlist2,vx2,vy2,ax2,ay2=trajectory_simulation(initial,nn,nj,ff=fudgefactor,s0=s00,detun=detuning,phi0hex=Voltagehex,lc=False)
    print(len(xlist),len(ylist),len(zlist),len(vx),len(vy),len(ax),len(ay))
    xc1,yc1,zc1,vxc1,vyc1=xy_distribution(xlist,ylist,zlist,vx,vy,Ldet1)
    xc2,yc2,zc2,vxc2,vyc2=xy_distribution(xlist2,ylist2,zlist2,vx2,vy2,Ldet1)
    plt.scatter(xc1,vxc1)
    plt.xlabel('x distance (m)')
    plt.ylabel('velocity in x (m/s)')
    plt.show()

def simulate_vr_hist(nn,nj,fudgefactor,s00,detuning,Voltagehex,Ldet1,Ldet2,save_plot=False):
    initial=[phasespaceellipse2D(xx0,dxx) for i in range(nn)]
    xlist,ylist,zlist,vx,vy,ax,ay=trajectory_simulation(initial,nn,nj,ff=fudgefactor,s0=s00,detun=detuning,phi0hex=Voltagehex,lc=True)
    xlist2,ylist2,zlist2,vx2,vy2,ax2,ay2=trajectory_simulation(initial,nn,nj,ff=fudgefactor,s0=s00,detun=detuning,phi0hex=Voltagehex,lc=False)
    xc1,yc1,zc1,vxc1,vyc1=xy_distribution(xlist,ylist,zlist,vx,vy,Ldet1)
    xc2,yc2,zc2,vxc2,vyc2=xy_distribution(xlist2,ylist2,zlist2,vx2,vy2,Ldet1)

    velr_1=[np.sqrt(vxc1[i]**2+vyc1[i]**2) for i in range(len(vxc1))]
    velr_2=[np.sqrt(vxc2[i]**2+vyc2[i]**2) for i in range(len(vxc2))]
    velr_1=[vel for vel in velr_1 if vel<2]
    velr_2=[vel for vel in velr_2 if vel<2]

    plot_hist2(velr_1,velr_2,fudgefactor,s00,detuning,Voltagehex,Ldet1,plot_safe=save_plot)

def simulate_lc_hex_off_histogram(nn, nj, fudgefactor, s00, detuning, Voltagehex, Ldet1, Ldet2,hfs_bool=True, plot_circle=False,plot_safe=False,extra_title=''):
    initial = [phasespaceellipse2D(xx0, dxx,hfs=hfs_bool) for i in range(nn)]
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
    xc4, yc4, zc4, vxc4, vyc4 = xy_distribution(xlist, ylist, zlist, vx, vy, Ldet2)
    xc5, yc5, zc5, vxc5, vyc5 = xy_distribution(xlist2, ylist2, zlist2, vx2, vy2, Ldet2)
    xc6, yc6, zc6, vxc6, vyc6 = xy_distribution(xlist3, ylist3, zlist3, vx3, vy3, Ldet2)
    vx_list=[vxc1,vxc2,vxc3]#,vxc4,vxc5,vxc6]
    vy_list=[vyc1,vyc2,vyc3]#,vyc4,vyc5,vyc6]

    vr_final=[]
    for i in range(len(vx_list)):
        vel=[np.sqrt(vx_list[i][j]**2+vy_list[i][j]**2) for j in range(len(vx_list[i]))]
        vel=[vel1 for vel1 in vel if vel1<2]
        vr_final.append(vel)
    plot_6_hist_vel(vr_final, fudgefactor, s00, detuning, Voltagehex, Ldet1, Ldet2, labels=True, bins=100, plot_safe=False)


def simulate_lc_2_distance_hex_off(nn, nj, fudgefactor, s00, detuning, Voltagehex, Ldet1, Ldet2,hfs_bool=False, plot_circle=False,plot_safe=False,extra_title=''):
    initial = [phasespaceellipse2D(xx0, dxx,hfs=hfs_bool) for i in range(nn)]
    xlist, ylist, zlist, vx, vy, ax, ay = trajectory_simulation(initial, nn, nj, ff=fudgefactor, s0=s00, detun=detuning,
                                                                phi0hex=Voltagehex, lc=True)
    xlist2, ylist2, zlist2, vx2, vy2, ax2, ay2 = trajectory_simulation(initial, nn, nj, ff=fudgefactor, s0=s00,
                                                                       detun=detuning, phi0hex=Voltagehex, lc=False)
    xlist3, ylist3, zlist3, vx3, vy3, ax3, ay3 = trajectory_simulation(initial, nn, nj, ff=fudgefactor, s0=s00,
                                                                       detun=detuning, phi0hex=Voltagehex, lc=False,hex=False)
    xc1, yc1, zc1, vxc1, vyc1 = xy_distribution(xlist, ylist, zlist, vx, vy, Ldet1)
    xc2, yc2, zc2, vxc2, vyc2 = xy_distribution(xlist2, ylist2, zlist2, vx2, vy2, Ldet1)
    xc3, yc3, zc3, vxc3, vyc3 = xy_distribution(xlist, ylist, zlist, vx, vy, Ldet2)
    xc4, yc4, zc4, vxc4, vyc4 = xy_distribution(xlist2, ylist2, zlist2, vx2, vy2, Ldet2)
    xc5, yc5, zc5, vxc5, vyc5 = xy_distribution(xlist3, ylist3, zlist3, vx3, vy3, Ldet1)
    xc6, yc6, zc6, vxc6, vyc6 = xy_distribution(xlist3, ylist3, zlist3, vx3, vy3, Ldet2)
    plot_6_heatmap(xc1, yc1, zc1, xc2, yc2, zc2, xc3, yc3, zc3, xc4, yc4, zc4,xc5,yc5,zc5,xc6,yc6,zc6, fudgefactor, s00, detuning,
                   hexvolt=Voltagehex, Ldetection1=Ldet1, Ldetection2=Ldet2, plot_circle=plot_circle,plot_safe=plot_safe,extra_title=extra_title)

def simulate_trajectories_111(nn, nj, fudgefactor, s00, detuning, Voltagehex, Ldet1, Ldet2, plot_circle=False,plot_safe=False):
    initial = [phasespaceellipse2D(xx0, dxx) for i in range(nn)]
    xlist, ylist, zlist, vx, vy, ax, ay = trajectory_simulation(initial, nn, nj, ff=fudgefactor, s0=s00, detun=detuning,
                                                                phi0hex=Voltagehex, lc=True)
    xlist2, ylist2, zlist2, vx2, vy2, ax2, ay2 = trajectory_simulation(initial, nn, nj, ff=fudgefactor, s0=s00,
                                                                       detun=detuning, phi0hex=Voltagehex, lc=False)
    xlist3, ylist3, zlist3, vx3, vy3, ax3, ay3 = trajectory_simulation(initial, nn, nj, ff=fudgefactor, s0=s00,
                                                                       detun=detuning, phi0hex=Voltagehex, lc=False,hex=False)
    plot_trajectories(xlist, zlist, xlist2, zlist2, xlist3, zlist3, fudgefactor, s00, detuning, Voltagehex)

def simulate_n_lc(nn, nj, fudgefactor, s00, detuning, Voltagehex, Ldet1, Ldet2,lc_bool,hex_bool,hfs_bool,plot_titles=False, plot_circle=False,plot_safe=False):
    n=len(Voltagehex)
    xs=[]
    ys=[]
    for i in range(n):
        if Voltagehex[i]==0.0:
            initial = [phasespaceellipse2D(xx0, dxx,hfs_bool[i]) for _ in range(nn)]
            Voltagehex[i]=0.01
            x,y,z,vx,vy,ax,ay=trajectory_simulation(initial,nn,nj,ff=fudgefactor[i],s0=s00[i],detun=detuning[i],phi0hex=Voltagehex[i],lc=lc_bool[i],hex=hex_bool[i])
            xf,yf,zf,vxf,vyf=xy_distribution(x,y,z,vx,vy,Ldet1)
            xs.append(xf)
            ys.append(yf)
        else:
            initial = [phasespaceellipse2D(xx0, dxx,hfs=hfs_bool[i]) for _ in range(nn)]
            x, y, z, vx, vy, ax, ay = trajectory_simulation(initial, nn, nj, ff=fudgefactor[i], s0=s00[i],detun=detuning[i], phi0hex=Voltagehex[i],lc=lc_bool[i],hex=hex_bool[i])
            xf, yf, zf, vxf, vyf = xy_distribution(x, y, z, vx, vy, Ldet1)
            xs.append(xf)
            ys.append(yf)



    plot_n_heatmaps(xs, ys, fudgefactor, s00, detuning, Voltagehex,hfs_bool,plot_titles=plot_titles, plot_circle=plot_circle)


def simulate_number_mol_vs_hexvoltage(nn, nj, fudgefactor, s00, detuning, Voltagehex, Ldet1, Ldet2,lc_bool,hex_bool,hfs_bool,hfs_percent,plot_titles=False, plot_circle=False,plot_safe=False):
    n=len(Voltagehex)
    num_mol1=[]
    num_mol2=[]
    r=5
    for i in range(n):
        initial = [phasespaceellipse2D(xx0, dxx,hfs_percentage=hfs_percent, hfs=hfs_bool[i]) for _ in range(nn)]
        x,y,z,vx,vy,ax,ay=trajectory_simulation(initial,nn,nj,ff=fudgefactor[i],s0=s00[i],detun=detuning[i],phi0hex=Voltagehex[i],lc=lc_bool[i],hex=hex_bool[i])
        x1,y1,z1,vx1,vy1=xy_distribution(x,y,z,vx,vy,Ldet1)
        num_mol1.append(points_in_circle(np.array(x1) * 1e3, np.array(y1) * 1e3, r))
    r = 2.5
    for i in range(n):
        initial = [phasespaceellipse2D(xx0, dxx,hfs_percentage=hfs_percent, hfs=hfs_bool[i]) for _ in range(nn)]
        x, y, z, vx, vy, ax, ay = trajectory_simulation(initial, nn, nj, ff=fudgefactor[i], s0=s00[i],
                                                        detun=detuning[i], phi0hex=Voltagehex[i], lc=lc_bool[i],
                                                        hex=hex_bool[i])
        x1, y1, z1, vx1, vy1 = xy_distribution(x, y, z, vx, vy, Ldet1)
        num_mol2.append(points_in_circle(np.array(x1) * 1e3, np.array(y1) * 1e3, r))
    plot_mol_vs_voltage(Voltagehex, num_mol1, num_mol2,hfs_percent, r=[5,2.5])


def simulate_y_cooling(nn, nj, fudgefactor, s00, detuning, Voltagehex, Ldet1, Ldet2,plot_titles=False, plot_circle=False,plot_safe=False):
    initial = [phasespaceellipse2D(xx0, dxx) for i in range(nn)]
    xlist, ylist, zlist, vx, vy, ax, ay = trajectory_simulation(initial, nn, nj, ff=fudgefactor, s0=s00, detun=detuning,
                                                                phi0hex=Voltagehex, lc=True,ycooling=False)
    xlist2, ylist2, zlist2, vx2, vy2, ax2, ay2 = trajectory_simulation(initial, nn, nj, ff=fudgefactor, s0=s00, detun=detuning,
                                                                phi0hex=Voltagehex, lc=False, ycooling=False)

    xlist3, ylist3, zlist3, vx3, vy3, ax3, ay3 = trajectory_simulation(initial, nn, nj, ff=fudgefactor, s0=s00,
                                                                       detun=detuning,
                                                                       phi0hex=0.001, lc=True, ycooling=False,hex=True)
    xlist4, ylist4, zlist4, vx4, vy4, ax4, ay4 = trajectory_simulation(initial, nn, nj, ff=fudgefactor, s0=s00,
                                                                       detun=detuning,
                                                                       phi0hex=0.001, lc=False, ycooling=False)

    xc1, yc1, zc1, vxc1, vyc1 = xy_distribution(xlist, ylist, zlist, vx, vy, Ldet1)
    xc2, yc2, zc2, vxc2, vyc2 = xy_distribution(xlist2, ylist2, zlist2, vx2, vy2, Ldet1)
    xc3, yc3, zc3, vxc3, vyc3 = xy_distribution(xlist3, ylist3, zlist3, vx3, vy3, Ldet1)
    xc4, yc4, zc4, vxc4, vyc4 = xy_distribution(xlist4, ylist4, zlist4, vx4, vy4, Ldet1)

    plot_4_heatmap(xc1, yc1, zc1, xc2, yc2, zc2, xc3, yc3, zc3, xc4, yc4, zc4, fudgefactor, s00, detuning,
                   hexvolt=Voltagehex, Ldetection1=Ldet1, Ldetection2=Ldet2, plot_circle=plot_circle,plot_safe=plot_safe)
