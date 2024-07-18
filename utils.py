import numpy as np
import Pk_library as PKL
import MAS_library as MASL
import readgadget
import h5py
import matplotlib.pylab as plt

def get_delta(pos, grid, W=False, mass=1, BoxSize=1000.0, MAS='CIC', verbose=True):
    """
    W: weights corresponding to each particle position. (i.e. W=False for equal weighting to each particle; W=True, mass=mass for mass weighting)
    
    """
    
    # define 3D density field
    delta = np.zeros((grid,grid,grid), dtype=np.float32)
    
    # construct 3D density field
    if W:
        MASL.MA(pos, delta, BoxSize, MAS, W=mass, verbose=verbose)
    else:
        MASL.MA(pos, delta, BoxSize, MAS, verbose=verbose)
          
    
    # at this point, delta contains the effective number of particles in each voxel
    # now compute overdensity and density constrast
    delta /= np.mean(delta, dtype=np.float64)
    delta -= 1.0
    
    return delta


def compute_Pk(delta, BoxSize=1000.0, MAS='CIC', axis=0, verbose=True):    

    # compute power spectrum
    Pk = PKL.Pk(delta, BoxSize, axis=axis, MAS=MAS, threads=1, verbose=verbose)
    
    # 3D P(k)
    k       = Pk.k3D
    Pk0     = Pk.Pk[:,0] #monopole
    Pk2     = Pk.Pk[:,1] #quadrupole
    Pk4     = Pk.Pk[:,2] #hexadecapole

    return k, Pk0, Pk2, Pk4

def compute_Pk_from_snapshot(snapshot, ptype = [1], grid = 512, verbose=True):
    
    # read header
    header   = readgadget.header(snapshot)
    BoxSize  = header.boxsize/1e3  #Mpc/h
    
    # read positions, velocities and IDs of the particles
    pos = readgadget.read_block(snapshot, "POS ", ptype)/1e3 #positions in Mpc/h
    mass = readgadget.read_block(snapshot, "MASS", ptype)*1e10
    
    pos = pos.astype(np.float32)
    delta = get_delta(pos, grid=grid, BoxSize=BoxSize, W=mass, MAS='CIC', verbose=verbose)
    k, Pk0, Pk2, Pk4 = compute_Pk(delta, BoxSize=BoxSize, MAS='CIC', axis=0, verbose=verbose)
    
    return k, Pk0, Pk2, Pk4



def get_subhalo_pos_from_halo_catalog(snapdir, snapnum):
    pos_halo=np.array([]).reshape(0,3)

    for i in range(16):
        f_snap = snapdir+'groups_'+str(snapnum).zfill(3)+'/fof_subhalo_tab_'+str(snapnum).zfill(3)+'.'+str(i)+'.hdf5'
        data = h5py.File(f_snap, 'r')
        pos = data['Subhalo']['SubhaloPos'][()]
        pos_halo=np.vstack([pos_halo,pos])
        
    return pos_halo

def get_subhalo_mass_from_halo_catalog(snapdir, snapnum):
    mass_halo=np.array([])

    for i in range(16):
        f_snap = snapdir+'groups_'+str(snapnum).zfill(3)+'/fof_subhalo_tab_'+str(snapnum).zfill(3)+'.'+str(i)+'.hdf5'
        data = h5py.File(f_snap, 'r')
        mass = data['Subhalo']['SubhaloMass'][()] 
        mass_halo=np.hstack([mass_halo,mass])
        
    return mass_halo

def compute_halo_Pk_from_halo_catalog(snapdir, snapnum, grid = 512, threshold=0, W=False, verbose=True):
    """
    threshold: mass limit; 0:all
    W: weights corresponding to each particle position. (i.e. W=False for equal weighting to each particle; W=True, mass=mass for mass weighting)
    
    """
    
    f_snap = snapdir+'/groups_'+str(snapnum).zfill(3)+'/fof_subhalo_tab_'+str(snapnum).zfill(3)+'.0.hdf5'
    data = h5py.File(f_snap, 'r')
    BoxSize = data['Header'].attrs[u'BoxSize']/1e3 #Mpc/h

    pos = get_subhalo_pos_from_halo_catalog(snapdir, snapnum).astype(np.float32)/1e3 #Mpc/h 
    mass = get_subhalo_mass_from_halo_catalog(snapdir, snapnum).astype(np.float32)*1e10     #Msun/h 
    
    if threshold:
        ind = np.where(mass>=threshold)[0]
        mass = mass[ind]
        pos = pos[ind]

    delta = get_delta(pos, grid=grid, W=W, mass=mass, BoxSize=BoxSize, MAS='CIC', verbose=verbose)
    k, Pk0, Pk2, Pk4 = compute_Pk(delta, BoxSize=BoxSize, MAS='CIC', axis=0, verbose=verbose)
    
    return k, Pk0, Pk2, Pk4


def plot_halo_Pk(snaptype,snapnum, kmax=None, threshold=0, W=False, verbose=True):
    
    """
    snaptype={'EQ':"equil", 'LC':"local", 'OR_CMB':"ortho-CMB", 'OR_LSS':"ortho-LSS"}
    """
    snaptype_dict={'EQ':"{equil}", 'LC':"{local}", 'OR_CMB':"{ortho-CMB}", 'OR_LSS':"{ortho-LSS}"}
    
    a=np.loadtxt("../times_extended.txt")[snapnum]
    z=1/a-1
    
    snapdir = "/home/jovyan/L501P/1P_"+snaptype+"_0_50/" #folder hosting the catalogue
    k, Pk0_G, _, _ = compute_halo_Pk_from_halo_catalog(snapdir, snapnum, threshold=threshold, W=W, verbose=verbose)
    
    if kmax!=None:
        ind = np.where(k<=kmax)[0]
        Pk0_G = Pk0_G[ind]
    
    figure, axis = plt.subplots(1,2,figsize=(15,5))
    fNL_dict={'200':200, '100':100, '50':50, '0':0, 'n50':-50, 'n100':-100, 'n200':-200}
    for fNL in fNL_dict:
        snapdir = "/home/jovyan/L501P/1P_"+snaptype+"_"+fNL+"_50/" #folder hosting the catalogue
        k, Pk0, _, _ = compute_halo_Pk_from_halo_catalog(snapdir, snapnum, threshold=threshold, W=W, verbose=verbose)
        
        
        if kmax!=None:
            ind = np.where(k<=kmax)[0]
            k = k[ind]
            Pk0 = Pk0[ind]
        
        axis[0].loglog(k, Pk0, label="$f_{}^{}$ = {}".format("{NL}", snaptype_dict[snaptype],fNL_dict[fNL]))
        axis[1].semilogx(k, (Pk0-Pk0_G)/Pk0_G, label="$f_{}^{}$ = {}".format("{NL}", snaptype_dict[snaptype],fNL_dict[fNL]))
    
    axis[0].legend()
    axis[1].legend()
    
    axis[0].set_title("z={}".format(round(z,1)))
    axis[0].set_xlabel("k")
    axis[0].set_ylabel("P(k)")
    axis[1].set_title("z={}".format(round(z,1)))
    axis[1].set_xlabel("k")
    axis[1].set_ylabel(r"$\frac{P_{NG}(k)-P_{G}(k)}{P_{G}(k)}$")