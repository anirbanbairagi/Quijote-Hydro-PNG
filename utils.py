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
        
    return pos_halo/1e3  #Mpc/h

def get_subhalo_mass_from_halo_catalog(snapdir, snapnum):
    mass_halo=np.array([])

    for i in range(16):
        f_snap = snapdir+'groups_'+str(snapnum).zfill(3)+'/fof_subhalo_tab_'+str(snapnum).zfill(3)+'.'+str(i)+'.hdf5'
        data = h5py.File(f_snap, 'r')
        mass = data['Subhalo']['SubhaloMass'][()] 
        mass_halo=np.hstack([mass_halo,mass])
        
    return mass_halo*1e10  #Msun/h  

def compute_halo_Pk_from_halo_catalog(snapdir, snapnum, grid = 512, threshold=0, W=False, verbose=True):
    """
    threshold: mass limit; 0:all
    W: weights corresponding to each particle position. (i.e. W=False for equal weighting to each particle; W=True, mass=mass for mass weighting)
    
    """
    
    f_snap = snapdir+'groups_'+str(snapnum).zfill(3)+'/fof_subhalo_tab_'+str(snapnum).zfill(3)+'.0.hdf5'
    data = h5py.File(f_snap, 'r')
    BoxSize = data['Header'].attrs[u'BoxSize']/1e3 #Mpc/h

    pos = get_subhalo_pos_from_halo_catalog(snapdir, snapnum).astype(np.float32)    #Mpc/h 
    mass = get_subhalo_mass_from_halo_catalog(snapdir, snapnum).astype(np.float32)  #Msun/h 
    
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
    if snaptype=='LC':
        fNL_dict={'200':200, '0':0, 'n200':-200}
    else:
        fNL_dict={'200':200, 'n200':-200}
        # fNL_dict={'200':200, '100':100, '50':50, '0':0, 'n50':-50, 'n100':-100, 'n200':-200}
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
    

    
def stellar_to_halo_mass_relation(snapdir, snapnum=90, binning='log', num_bins=50):
    """
    binning : {'log', 'linear'}
    """
    
    # Reading and storing Stellar and Halo masses from all the catalogs in array
    M_star=np.array([])
    M_halo=np.array([])

    for i in range(16):
        # catalog name
        catalog = snapdir+'groups_'+str(snapnum).zfill(3)+'/fof_subhalo_tab_'+str(snapnum).zfill(3)+'.'+str(i)+'.hdf5'

        # open the catalogue
        f = h5py.File(catalog, 'r')

        # read the positions and stellar masses of the subhalos/galaxies
        Mass = f['Group/GroupMassType'][:,4]*1e10 #stellar masses in Msun/h
        Mass_halo = f['Group/GroupMass'][:]*1e10 #stellar masses in Msun/h

        M_star=np.hstack([M_star,Mass])
        M_halo=np.hstack([M_halo,Mass_halo])

        # close file
        f.close()
        
    
    # Creating Halo mass bins
    if binning=='log':
        bins = np.logspace(np.log10(M_halo.min()), np.log10(M_halo.max()),num_bins+1) # Halo mass bins
    elif binning=='linear':
        bins = np.linspace(M_halo.min(), M_halo.max(),num_bins+1) # Halo mass bins

    index = np.digitize(M_halo, bins) # digitize starts with 1
    # np.bincount(index) starts with bincount from index=0, thus remove the first one 
    if index.max()==num_bins+1:
        Stellar_mass_avg = np.bincount(index, M_star, minlength=num_bins+2)[1:-1]/np.bincount(index)[1:-1]
        Halo_mass_bins = (bins[1:]+bins[:-1])/2                                 # mean pos of Halo mass bins
    else:
        Stellar_mass_avg = np.bincount(index, M_star, minlength=index.max()+1)[1:]/np.bincount(index)[1:] 
        Halo_mass_bins = (bins[1:(index.max()+1)]+bins[:index.max()])/2         # mean pos of Halo mass bins

    # std calculation
    # Stellar_mass_var = np.bincount(index, (M_star-Stellar_mass_avg[index])**2, minlength=num_bins)/np.bincount(index)
    # Stellar_mass_std = np.sqrt(Stellar_mass_var)
    
    # Remove indices where there is no halo in some mass bin 
    Halo_mass_bins = np.delete(Halo_mass_bins,np.argwhere(np.isnan(Stellar_mass_avg)))
    # Stellar_mass_std = np.delete(Stellar_mass_std,np.argwhere(np.isnan(Stellar_mass_avg)))
    Stellar_mass_avg = np.delete(Stellar_mass_avg,np.argwhere(np.isnan(Stellar_mass_avg)))
    
    return Halo_mass_bins, Stellar_mass_avg, M_halo, M_star

def plot_stellar_to_halo_mass(datadir, snaptype, snapnum=90, binning='log', num_bins=50):
    
    """
    snaptype : {'EQ':"equil", 'LC':"local", 'OR_CMB':"ortho-CMB", 'OR_LSS':"ortho-LSS"}
    
    binning : {'log', 'linear'}
    """
    snaptype_dict={'EQ':"{equil}", 'LC':"{local}", 'OR_CMB':"{ortho-CMB}", 'OR_LSS':"{ortho-LSS}"}
    
    a=np.loadtxt("../times_extended.txt")[snapnum]
    z=1/a-1   
    
    if snaptype=='LC':
        fNL_dict={'200':200, '0':0, 'n200':-200}
    else:
        fNL_dict={'200':200, 'n200':-200}
        # fNL_dict={'200':200, '100':100, '50':50, '0':0, 'n50':-50, 'n100':-100, 'n200':-200}
    for fNL in fNL_dict:
        if fNL=='0':
            snapdir = datadir+"1P_LC_0_50/" #folder hosting the catalogue
        else:    
            snapdir = datadir+"1P_"+snaptype+"_"+fNL+"_50/" #folder hosting the catalogue
        Halo_mass_bins, Stellar_mass_avg, _, _ = stellar_to_halo_mass_relation(snapdir, snapnum=snapnum, binning=binning, num_bins=num_bins) 
        
        
        plt.loglog(Halo_mass_bins, Stellar_mass_avg, label="$f_{}^{}$ = {}".format("{NL}", snaptype_dict[snaptype],fNL_dict[fNL]))
        
    plt.legend()
    
    plt.title("z={}".format(round(z,1)))
    plt.xlabel("Halo mass")
    plt.ylabel("Stellar mass")
    
    
    
def SFR_to_stellar_mass_relation(snapdir, snapnum=90, binning='log', num_bins=50):
    """
    binning : {'log', 'linear'}
    """
    
    # Reading and storing Stellar and Halo masses from all the catalogs in array
    M_star=np.array([])
    M_subhalo=np.array([])
    SFR_g=np.array([])

    for i in range(16):
        # catalog name
        catalog = snapdir+'groups_'+str(snapnum).zfill(3)+'/fof_subhalo_tab_'+str(snapnum).zfill(3)+'.'+str(i)+'.hdf5'

        # open the catalogue
        f = h5py.File(catalog, 'r')

        # read the positions and stellar masses of the subhalos/galaxies
        Mass = f['Subhalo/SubhaloMassType'][:,4]*1e10 #stellar masses in Msun/h
        Mass_subhalo = f['Subhalo/SubhaloMass'][:]*1e10 #stellar masses in Msun/h
        SFR = f['Subhalo']['SubhaloSFR'][()]
    
        M_star=np.hstack([M_star,Mass])
        M_subhalo=np.hstack([M_subhalo,Mass_subhalo])
        SFR_g=np.hstack([SFR_g,SFR])
        
        # close file
        f.close()
        
    ind=np.where(M_star==0)[0]
    M_star = np.delete(M_star, ind)
    SFR_g = np.delete(SFR_g, ind)
    
    # Creating Stellar mass bins
    if binning=='log':
        bins = np.logspace(np.log10(M_star.min()), np.log10(M_star.max()),num_bins+1) # Halo mass bins
    elif binning=='linear':
        bins = np.linspace(M_star.min(), M_star.max(),num_bins+1) # Halo mass bins

    index = np.digitize(M_star, bins) # digitize starts with 1
    # np.bincount(index) starts with bincount from index=0, thus remove the first one 
    if index.max()==num_bins+1:
        SFR_avg = np.bincount(index, SFR_g, minlength=num_bins+2)[1:-1]/np.bincount(index)[1:-1]
        Stellar_mass_bins = (bins[1:]+bins[:-1])/2                           # mean pos of Stellar mass bins
    else:
        SFR_avg = np.bincount(index, SFR_g, minlength=index.max()+1)[1:]/np.bincount(index)[1:] 
        Stellar_mass_bins = (bins[1:(index.max()+1)]+bins[:index.max()])/2   # mean pos of Stellar mass bins

     
    # Remove indices where there is no star in some mass bin 
    Stellar_mass_bins = np.delete(Stellar_mass_bins,np.argwhere(np.isnan(SFR_avg)))
    SFR_avg = np.delete(SFR_avg,np.argwhere(np.isnan(SFR_avg)))
    
    return Stellar_mass_bins, SFR_avg, M_star, SFR_g


def specific_SFR_to_stellar_mass_relation(snapdir, snapnum=90, binning='log', num_bins=50):
    """
    binning : {'log', 'linear'}
    """
    
    # Reading and storing Stellar and Halo masses from all the catalogs in array
    M_star=np.array([])
    M_subhalo=np.array([])
    SFR_g=np.array([])

    for i in range(16):
        # catalog name
        catalog = snapdir+'groups_'+str(snapnum).zfill(3)+'/fof_subhalo_tab_'+str(snapnum).zfill(3)+'.'+str(i)+'.hdf5'

        # open the catalogue
        f = h5py.File(catalog, 'r')

        # read the positions and stellar masses of the subhalos/galaxies
        Mass = f['Subhalo/SubhaloMassType'][:,4]*1e10 #stellar masses in Msun/h
        Mass_subhalo = f['Subhalo/SubhaloMass'][:]*1e10 #stellar masses in Msun/h
        SFR = f['Subhalo']['SubhaloSFR'][()]
    
        M_star=np.hstack([M_star,Mass])
        M_subhalo=np.hstack([M_subhalo,Mass_subhalo])
        SFR_g=np.hstack([SFR_g,SFR])
        
        # close file
        f.close()
        
    ind=np.where(M_star==0)[0]
    M_star = np.delete(M_star, ind)
    SFR_g = np.delete(SFR_g, ind)
    sSFR = SFR_g/M_star
    
    # Creating Stellar mass bins
    if binning=='log':
        bins = np.logspace(np.log10(M_star.min()), np.log10(M_star.max()),num_bins+1) # Halo mass bins
    elif binning=='linear':
        bins = np.linspace(M_star.min(), M_star.max(),num_bins+1) # Halo mass bins

    index = np.digitize(M_star, bins) # digitize starts with 1
    # np.bincount(index) starts with bincount from index=0, thus remove the first one
    if index.max()==num_bins+1:
        sSFR_avg = np.bincount(index, sSFR, minlength=num_bins+2)[1:-1]/np.bincount(index)[1:-1]
        Stellar_mass_bins = (bins[1:]+bins[:-1])/2    # mean pos of Stellar mass bins
    else:
        sSFR_avg = np.bincount(index, sSFR, minlength=index.max()+1)[1:]/np.bincount(index)[1:] 
        Stellar_mass_bins = (bins[1:(index.max()+1)]+bins[:index.max()])/2   # mean pos of Stellar mass bins

     
    # Remove indices where there is no star in some mass bin 
    Stellar_mass_bins = np.delete(Stellar_mass_bins,np.argwhere(np.isnan(sSFR_avg)))
    sSFR_avg = np.delete(sSFR_avg,np.argwhere(np.isnan(sSFR_avg)))
    
    return Stellar_mass_bins, sSFR_avg, M_star, sSFR

def plot_hist(datadir, snaptype, snapnum=90, parameter='', xlabel='', threshold=0, num_bins=100, log=False, histtype='step'):
    
    """
    snaptype : {'EQ':"equil", 'LC':"local", 'OR_CMB':"ortho-CMB", 'OR_LSS':"ortho-LSS"}
    
    binning : {'log', 'linear'}
    """
    snaptype_dict={'EQ':"{equil}", 'LC':"{local}", 'OR_CMB':"{ortho-CMB}", 'OR_LSS':"{ortho-LSS}"}
    
    a=np.loadtxt("../times_extended.txt")[snapnum]
    z=1/a-1   
    
       
    if snaptype=='LC':
        fNL_dict={'200':200, '0':0, 'n200':-200}
    else:
        fNL_dict={'200':200, 'n200':-200}
        
    for fNL in fNL_dict:
        if fNL=='0':
            snapdir = datadir+"1P_LC_0_50/" #folder hosting the catalogue
        else:    
            snapdir = datadir+"1P_"+snaptype+"_"+fNL+"_50/" #folder hosting the catalogue
        
        param_g=np.array([])
        for i in range(16):
            # catalog name
            catalog = snapdir+'groups_'+str(snapnum).zfill(3)+'/fof_subhalo_tab_'+str(snapnum).zfill(3)+'.'+str(i)+'.hdf5'

            # open the catalogue
            f = h5py.File(catalog, 'r')
            param = f[parameter][()]         
            param_g=np.hstack([param_g,param])
            
            # close file
            f.close()
        
        if threshold:
            ind = np.where(param_g>=threshold)[0]
            param_g = param_g[ind]
        
        if log==True:
            bins = np.logspace(np.log10(param_g.min()), np.log10(param_g.max()),num_bins+1)
        else:
            bins = np.linspace(param_g.min(), param_g.max(),num_bins+1)
      
        y, _ = np.histogram(param_g, bins)
        # y, x, _ = plt.hist(param_g, bins=bins, log=log, histtype=histtype, label="$f_{}^{}$ = {}".format("{NL}", snaptype_dict[snaptype],fNL_dict[fNL]))
        
        plt.plot((bins[1:]+bins[:-1])/2, y, marker='+', label="$f_{}^{}$ = {}".format("{NL}", snaptype_dict[snaptype],fNL_dict[fNL]))
        
    plt.legend()
    
    plt.title("z={}".format(round(z,1)))
    if log==True:
        plt.loglog()
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    

def plot_hist_all(datadir, snaptype_list=['EQ', 'LC', 'OR_CMB', 'OR_LSS'], snapnum_list=[2, 18, 32, 90], parameter='', xlabel='', threshold=0, num_bins=100, log=False, histtype='step'):
    
    """
    snaptype : {'EQ':"equil", 'LC':"local", 'OR_CMB':"ortho-CMB", 'OR_LSS':"ortho-LSS"}
    
    binning : {'log', 'linear'}
    """
    snaptype_dict={'EQ':"{equil}", 'LC':"{local}", 'OR_CMB':"{ortho-CMB}", 'OR_LSS':"{ortho-LSS}"}
    
    figure, axis = plt.subplots(len(snapnum_list)//2,2,figsize=(15,5*len(snapnum_list)//2))
    for ind, snapnum in enumerate(snapnum_list):
        a=np.loadtxt("../times_extended.txt")[snapnum]
        z=1/a-1
        
        for snaptype in snaptype_list:
            if snaptype=='LC':
                fNL_dict={'200':200, '0':0, 'n200':-200}
            else:
                fNL_dict={'200':200, 'n200':-200}

            for fNL in fNL_dict:
                if fNL=='0':
                    snapdir = datadir+"1P_LC_0_50/" #folder hosting the catalogue
                else:    
                    snapdir = datadir+"1P_"+snaptype+"_"+fNL+"_50/" #folder hosting the catalogue

                param_g=np.array([])
                for i in range(16):
                    # catalog name
                    catalog = snapdir+'groups_'+str(snapnum).zfill(3)+'/fof_subhalo_tab_'+str(snapnum).zfill(3)+'.'+str(i)+'.hdf5'

                    # open the catalogue
                    f = h5py.File(catalog, 'r')
                    param = f[parameter][()]         
                    param_g=np.hstack([param_g,param])

                    # close file
                    f.close()

                if threshold:
                    ind = np.where(param_g>=threshold)[0]
                    param_g = param_g[ind]

                if log==True:
                    bins = np.logspace(np.log10(param_g.min()), np.log10(param_g.max()),num_bins+1)
                else:
                    bins = np.linspace(param_g.min(), param_g.max(),num_bins+1)
                    
                axis[ind//2, ind%2].hist(param_g, bins=bins, log=log, histtype=histtype, label="$f_{}^{}$ = {}".format("{NL}", snaptype_dict[snaptype],fNL_dict[fNL]));

            

        axis[ind//2, ind%2].set_title("z={}".format(round(z,1)))
        if log==True: 
            axis[ind//2, ind%2].set_xscale('log')
        axis[ind//2, ind%2].set_xlabel(xlabel)
        axis[ind//2, ind%2].set_ylabel("Frequency")

    axis[ind//2, ind%2].legend()
    
            
            
def any_relation(snapdir, snapnum=90, parameters=['Subhalo/SubhaloMass','Subhalo/SubhaloGasMetallicity'], binning='log', num_bins=50):
    """
    binning : {'log', 'linear'}
    """
    
    # Reading and storing Stellar and Halo masses from all the catalogs in array
    param_1=np.array([])
    param_2=np.array([])

    for i in range(16):
        # catalog name
        catalog = snapdir+'groups_'+str(snapnum).zfill(3)+'/fof_subhalo_tab_'+str(snapnum).zfill(3)+'.'+str(i)+'.hdf5'

        # open the catalogue
        f = h5py.File(catalog, 'r')

        
        param1 = f[parameters[0]][:]
        param2 = f[parameters[1]][:]

        param_1=np.hstack([param_1,param1])
        param_2=np.hstack([param_2, param2])

        # close file
        f.close()
        
    
    # Creating Halo mass bins
    if binning=='log':
        print(param_1.min(), param_1.max())
        bins = np.logspace(np.log10(param_1.min()), np.log10(param_1.max()),num_bins+1) # Halo mass bins
    elif binning=='linear':
        bins = np.linspace(param_1.min(), param_1.max(),num_bins+1) # Halo mass bins

    index = np.digitize(param_1, bins) # digitize starts with 1
    # np.bincount(index) starts with bincount from index=0, thus remove the first one 
    if index.max()==num_bins+1:
        param_2_avg = np.bincount(index, param_2, minlength=num_bins+2)[1:-1]/np.bincount(index)[1:-1]
        bins = (bins[1:]+bins[:-1])/2                                 # mean pos of Halo mass bins
    else:
        param_2_avg = np.bincount(index, param_2, minlength=index.max()+1)[1:]/np.bincount(index)[1:] 
        bins = (bins[1:(index.max()+1)]+bins[:index.max()])/2         # mean pos of Halo mass bins

    
    # Remove indices where there is no halo in some mass bin 
    bins = np.delete(bins,np.argwhere(np.isnan(param_2_avg)))
    param_2_avg = np.delete(param_2_avg,np.argwhere(np.isnan(param_2_avg)))
    
    return bins, param_2_avg, param_1, param_2


def plot_any_relation_all(datadir, snaptype_list=['EQ', 'LC', 'OR_CMB', 'OR_LSS'], snapnum_list=[2, 18, 32, 90], parameters=[], label=[], binning='log', bins=100):
    
    """
    snaptype : {'EQ':"equil", 'LC':"local", 'OR_CMB':"ortho-CMB", 'OR_LSS':"ortho-LSS"}
    
    binning : {'log', 'linear'}
    """
    snaptype_dict={'EQ':"{equil}", 'LC':"{local}", 'OR_CMB':"{ortho-CMB}", 'OR_LSS':"{ortho-LSS}"}
    
    figure, axis = plt.subplots(len(snapnum_list)//2,2,figsize=(15,5*len(snapnum_list)//2))
    for ind, snapnum in enumerate(snapnum_list):
        a=np.loadtxt("../times_extended.txt")[snapnum]
        z=1/a-1
        
        for snaptype in snaptype_list:
            if snaptype=='LC':
                fNL_dict={'200':200, '0':0, 'n200':-200}
            else:
                fNL_dict={'200':200, 'n200':-200}

            for fNL in fNL_dict:
                if fNL=='0':
                    snapdir = datadir+"1P_LC_0_50/" #folder hosting the catalogue
                else:    
                    snapdir = datadir+"1P_"+snaptype+"_"+fNL+"_50/" #folder hosting the catalogue

                bins, param_2_avg, _, _ = any_relation(snapdir, snapnum, parameters, binning, bins)
                
                plt.loglog(bins, param_2_avg, label="$f_{}^{}$ = {}".format("{NL}", snaptype_dict[snaptype],fNL_dict[fNL]))

        axis[ind//2, ind%2].set_title("z={}".format(round(z,1)))
        axis[ind//2, ind%2].set_xlabel(label[0])
        axis[ind//2, ind%2].set_ylabel(label[0])

    axis[ind//2, ind%2].legend()