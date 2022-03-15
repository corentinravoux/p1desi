# coding: utf-8
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn as sns
import numpy as np
import glob
import fitsio
zlist=[h.read_header()['MEANZ'] for f in glob.glob('Pk1D-*.fits.gz') for h in fitsio.FITS(f)[1:]]
z_qsolist=[h.read_header()['Z'] for f in glob.glob('Pk1D-*.fits.gz') for h in fitsio.FITS(f)[1:]]
skylist=np.loadtxt('../../../list_veto_line_Pk1D.txt',usecols=[1,2])
z_skylist=(skylist[:,1]+skylist[:,0])/2/1216-1
g=sns.jointplot(z_qsolist,zlist,marker='.',label='spectral chunks')
g.set_axis_labels('$z_{qso}$','$z_{chunk}$')
for i,z in enumerate(np.arange(2.1,5.3,0.2)):
    g.ax_joint.axhline(z,color='0.5',zorder=-1,label='' if i>0 else 'bin boundaries')
for i,z in enumerate(z_skylist):
    g.ax_joint.axhline(z,color='r',zorder=-2,alpha=0.5,label='' if i>0 else 'sky line centers')
g.ax_joint.legend()
g.savefig('joint_zqso_zpk_sky_dist.pdf')
