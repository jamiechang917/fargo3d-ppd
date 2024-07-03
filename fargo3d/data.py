import os
import numpy as np
import astropy.units as u
import astropy.constants as const

class Data():
    def __init__(self, path:str, snap:int, n_dust:int, unit_length:u.Unit, unit_mass:u.Unit):
        self.path = path
        self.snap = snap
        self.n_dust = n_dust
        self._check_files()
        self.params = self._get_params()
        self.units = self._get_units(unit_length, unit_mass)
        self.coords = self._get_coords(ghost_cells=3) # 'x': azimuthal, 'y': radial, 'z': coaltitude
        self.mesh   = self._get_mesh()
        self.fluids = self._get_fluids()

    # private methods
    def _check_files(self) -> None:
        """
        Check the required files in the directory and change the path to the directory
        """
        # check if dir exists
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Directory {self.path} not found")
        # check domain_x.dat, domain_y.dat, domain_z.dat
        files = ["domain_x.dat", "domain_y.dat", "domain_z.dat", "variables.par"]
        for file in files:
            if not os.path.exists(os.path.join(self.path, file)):
                raise FileNotFoundError(f"File {file} not found in {self.path}")
        os.chdir(self.path)
        return

    def _get_params(self) -> dict:
        """
        Read variables.par and planet0.dat file and return the parameters as a dictionary
        Return: dict
        """
        params = {}
        with open("variables.par", "r") as f:
            for l in f.readlines():
                l = l.split("\t")
                params[l[0]] = l[1].strip()
        with open("planet0.dat", "r") as f:
            keys = ["index", "x", "y", "z", "vx", "vy", "vz", "mass", "date", "frame_rotation_rate"]
            planet_params = {key: [] for key in keys}
            for l in f.readlines():
                l = l.split("\t")
                for i, v in enumerate(keys):
                    if i == 0: # index
                        planet_params[v].append(int(l[i]))
                    else:
                        planet_params[v].append(float(l[i]))
            params["planet0"] = planet_params
        return params
    
    def _get_units(self, unit_length:u.Unit, unit_mass:u.Unit) -> dict:
        """
        Calculate the units
        """
        units = {}
        units["unit_mass"] = unit_mass.to(u.M_sun)
        units["unit_length"] = unit_length.to(u.au)
        units["unit_time"] = np.sqrt((unit_length**3 / (const.G * unit_mass))).to(u.s)
        units["unit_rho"] = (units["unit_mass"] / units["unit_length"]**3).to(u.g / u.cm**3)
        units["unit_sigma"] = (units["unit_mass"] / units["unit_length"]**2).to(u.g / u.cm**2)
        units["unit_velocity"] = (units["unit_length"] / units["unit_time"]).to(u.m / u.s)
        units["unit_time_onesnap"] = (units["unit_time"] * int(self.params["NINTERM"]) * float(self.params["DT"])).to(u.yr)
        return units

    def _get_coords(self, ghost_cells=3) -> dict:
        """
        Read domain files in the current directory and return the grids
        ghost_cells: int, number of ghost cells (default=3)
        Return: dict, keys: x, y, z; values: numpy array
        """
        domains = {'x': [], 'y': [], 'z': []} # azimuthal, radial, coaltitude
        for i in domains.keys():
            with open(f"domain_{i}.dat", "r") as f:
                for l in f.readlines():
                    domains[i].append(float(l))
            if i in {'y', 'z'}:
                domains[i] = domains[i][ghost_cells:-ghost_cells]
            domains[i] = np.array(domains[i])
        # convert coords to center of cells
        coords = {'sph_phi': domains['x'], 'sph_r': domains['y'], 'sph_theta': domains['z']}
        coords['sph_phi'] = ((coords['sph_phi'][1:] + coords['sph_phi'][:-1]) / 2) * u.rad
        coords['sph_r'] = ((coords['sph_r'][1:] + coords['sph_r'][:-1]) / 2) * self.units["unit_length"]
        coords['sph_theta'] = ((coords['sph_theta'][1:] + coords['sph_theta'][:-1]) / 2) * u.rad
        coords['domain_x'] = domains['x'] * u.rad
        coords['domain_y'] = domains['y'] * self.units["unit_length"]
        coords['domain_z'] = domains['z'] * u.rad
        return coords

    def _get_mesh(self) -> dict:
        """
        Calculate the mesh grid
        Returns: dict, keys: spherical, cylindrical; values: dict, keys: phi, r, theta/z; values: numpy array
        """
        # check coordinates
        coordinates = self.params["COORDINATES"]
        if coordinates != "spherical":
            raise NotImplementedError("Only the simulation data using spherical coordinates are supported")        
        SPH_THETA, SPH_R, SPH_PHI = np.meshgrid(self.coords['sph_theta'], self.coords['sph_r'], self.coords['sph_phi'], indexing='ij')
        CYL_PHI, CLY_R, CLY_Z = SPH_PHI, SPH_R*np.sin(SPH_THETA), SPH_R*np.cos(SPH_THETA)
        mesh = {"sph_phi": SPH_PHI, "sph_r": SPH_R, "sph_theta": SPH_THETA, "cyl_phi": CYL_PHI, "cyl_r": CLY_R, "cyl_z": CLY_Z}
        return mesh

    def _get_fluids(self) -> dict:
        """
        """
        NX, NY, NZ = int(self.params["NX"]), int(self.params["NY"]), int(self.params["NZ"]) # azimuthal, radial, coaltitude
        fluids_density, fluids_velocity = {}, {}
        # density
        fluids_density["gas"] = np.fromfile(f"gasdens{self.snap}.dat").reshape(NZ, NY, NX) * self.units["unit_rho"]
        for i in range(1, self.n_dust+1):
            fluids_density[f"dust{i}"] = np.fromfile(f"dust{i}dens{self.snap}.dat").reshape(NZ, NY, NX) * self.units["unit_rho"]
        # velocity
        fluids_velocity["gas"] = {
            'x': np.fromfile(f"gasvx{self.snap}.dat").reshape(NZ, NY, NX) * self.units["unit_velocity"],
            'y': np.fromfile(f"gasvy{self.snap}.dat").reshape(NZ, NY, NX) * self.units["unit_velocity"],
            'z': np.fromfile(f"gasvz{self.snap}.dat").reshape(NZ, NY, NX) * self.units["unit_velocity"]
        }
        for i in range(1, self.n_dust+1):
            fluids_velocity[f"dust{i}"] = {
                'x': np.fromfile(f"dust{i}vx{self.snap}.dat").reshape(NZ, NY, NX) * self.units["unit_velocity"],
                'y': np.fromfile(f"dust{i}vy{self.snap}.dat").reshape(NZ, NY, NX) * self.units["unit_velocity"],
                'z': np.fromfile(f"dust{i}vz{self.snap}.dat").reshape(NZ, NY, NX) * self.units["unit_velocity"]
            }
        # ======================================== #
        flaring_index = float(self.params["FLARINGINDEX"])
        aspect_ratio = float(self.params["ASPECTRATIO"])
        sigmaslope = float(self.params["SIGMASLOPE"])
        xi = sigmaslope + 1.0 + flaring_index
        beta = 1.0 - 2*flaring_index

        omega = self.params["planet0"]["frame_rotation_rate"][self.snap] / self.units["unit_time"]
        h = aspect_ratio * pow((self.mesh["sph_r"] / self.units["unit_length"]), flaring_index)
        
        star_mass = self.units["unit_mass"]
        vkep = np.sqrt(const.G * star_mass / self.mesh['sph_r']).to('m/s') # keplerian velocity
        

        for fluid in fluids_velocity.keys():
            fluids_velocity[fluid]["sph_phi"]   = (fluids_velocity[fluid]["x"] + self.mesh["sph_r"] * omega * np.sin(self.mesh["sph_theta"])) / np.sqrt(pow(np.sin(self.mesh["sph_theta"]), -2 * flaring_index) - (beta+xi)*h*h)
            fluids_velocity[fluid]["sph_r"]     = fluids_velocity[fluid]["y"]
            fluids_velocity[fluid]["sph_theta"] = fluids_velocity[fluid]["z"]
            fluids_velocity[fluid]["cyl_phi"]   = fluids_velocity[fluid]["sph_phi"]
            fluids_velocity[fluid]["cyl_r"]     = fluids_velocity[fluid]["sph_r"] * np.sin(self.mesh["sph_theta"]) + fluids_velocity[fluid]["sph_theta"] * np.cos(self.mesh["sph_theta"])
            fluids_velocity[fluid]["cyl_z"]     = - fluids_velocity[fluid]["sph_theta"] * np.sin(self.mesh["sph_theta"]) + fluids_velocity[fluid]["sph_r"] * np.cos(self.mesh["sph_theta"])
            fluids_velocity[fluid]["vkep"]      = vkep
        return {'density': fluids_density, 'velocity': fluids_velocity}
    
    # public methods
    def get_velocity_profiles(self, r_size:int, zr_min:float, zr_max:float,
                              phi_min=-np.pi, phi_max=np.pi, fluid='gas', coordinate='sph') -> dict:
        """
        Get radial velocity profile of the fluid
        r_size:     int, number of radial bins (number of points in the r direction)
        zr_min:     float, minimum z/r
        zr_max:     float, maximum z/r
        phi_min:    float, minimum phi (default=-pi)
        phi_max:    float, maximum phi (default=pi)
        fluid:      str, fluid name (default='gas')
        coordinate: str, coordinate system (default='sph', options: 'sph', 'cyl')
        """
        if coordinate not in {'sph', 'cyl'}:
            raise ValueError("coordinate should be either 'sph' or 'cyl'")
        if fluid not in self.fluids['density'].keys():
            raise ValueError(f"fluid {fluid} not found")
        if phi_min < -np.pi or phi_max > np.pi:
            raise ValueError("phi_min, phi_max should be within [-pi, pi]")
        
        zr_mesh = self.mesh['cyl_z'] / self.mesh['cyl_r'] # z/r mesh
        mask = (zr_mesh >= zr_min) & (zr_mesh <= zr_max) & (self.mesh['cyl_phi'].value >= phi_min) & (self.mesh['cyl_phi'].value <= phi_max)
    
        # get r-vr, r-vtheta, r-vphi profiles
        if coordinate == 'sph':
            r_masked        = self.mesh['sph_r'][mask]
            vphi_masked     = self.fluids['velocity'][fluid]['sph_phi'][mask]
            dvphi_masked    = (self.fluids['velocity'][fluid]['sph_phi'] - self.fluids['velocity'][fluid]['vkep'])[mask]
            vr_masked       = self.fluids['velocity'][fluid]['sph_r'][mask]
            vtheta_masked   = self.fluids['velocity'][fluid]['sph_theta'][mask]
            r, vphi, dvphi, vr, vtheta, r_err, vphi_err, dvphi_err, vr_err, vtheta_err = np.zeros(r_size), np.zeros(r_size), np.zeros(r_size), np.zeros(r_size), np.zeros(r_size), np.zeros(r_size), np.zeros(r_size), np.zeros(r_size), np.zeros(r_size), np.zeros(r_size)
            r_bounds = np.linspace(np.log10(r_masked.to('au').value).min(), np.log10(r_masked.to('au').value).max(), r_size+1, endpoint=True)
            r_bounds = 10**r_bounds * u.au
            for i in range(r_size):                
                mask_r = (r_masked >= r_bounds[i]) & (r_masked < r_bounds[i+1])
                # if all the values are masked, skip
                if mask_r.sum() == 0:
                    raise ValueError(f"Oversampling! Please decrease 'r_size' or check the mask condition")
                r[i] = np.mean(r_masked[mask_r].to('au').value)
                vphi[i] = np.mean(vphi_masked[mask_r].to('m/s').value)
                dvphi[i] = np.mean(dvphi_masked[mask_r].to('m/s').value)
                vr[i] = np.mean(vr_masked[mask_r].to('m/s').value)
                vtheta[i] = np.mean(vtheta_masked[mask_r].to('m/s').value)
                r_err[i] = np.std(r_masked[mask_r].to('au').value)
                vphi_err[i] = np.std(vphi_masked[mask_r].to('m/s').value)
                dvphi_err[i] = np.std(dvphi_masked[mask_r].to('m/s').value)
                vr_err[i] = np.std(vr_masked[mask_r].to('m/s').value)
                vtheta_err[i] = np.std(vtheta_masked[mask_r].to('m/s').value)
            return {'r': r * u.au, 'vphi': vphi * u.m/u.s, 'dvphi': dvphi * u.m/u.s, 'vr': vr * u.m/u.s, 'vtheta': vtheta * u.m/u.s,
                    'r_err': r_err * u.au, 'vphi_err': vphi_err * u.m/u.s, 'dvphi_err': dvphi_err * u.m/u.s, 'vr_err': vr_err * u.m/u.s, 'vtheta_err': vtheta_err * u.m/u.s}
        else:
            r_masked        = self.mesh['cyl_r'][mask]
            vphi_masked     = self.fluids['velocity'][fluid]['cyl_phi'][mask]
            dvphi_masked    = (self.fluids['velocity'][fluid]['cyl_phi'] - self.fluids['velocity'][fluid]['vkep'])[mask]
            vr_masked       = self.fluids['velocity'][fluid]['cyl_r'][mask]
            vz_masked       = self.fluids['velocity'][fluid]['cyl_z'][mask]
            r, vphi, dvphi, vr, vz, r_err, vphi_err, dvphi_err, vr_err, vz_err = np.zeros(r_size), np.zeros(r_size), np.zeros(r_size), np.zeros(r_size), np.zeros(r_size), np.zeros(r_size), np.zeros(r_size), np.zeros(r_size), np.zeros(r_size), np.zeros(r_size)
            r_bounds = np.linspace(np.log10(r_masked.to('au').value).min(), np.log10(r_masked.to('au').value).max(), r_size+1, endpoint=True)
            r_bounds = 10**r_bounds * u.au
            for i in range(r_size):
                mask_r = (r_masked >= r_bounds[i]) & (r_masked < r_bounds[i+1])
                # if all the values are masked, skip
                if mask_r.sum() == 0:
                    raise ValueError(f"Oversampling! Please decrease 'r_size' or check the mask condition")
                r[i] = np.mean(r_masked[mask_r].to('au').value)
                vphi[i] = np.mean(vphi_masked[mask_r].to('m/s').value) 
                dvphi[i] = np.mean(dvphi_masked[mask_r].to('m/s').value)
                vr[i] = np.mean(vr_masked[mask_r].to('m/s').value) 
                vz[i] = np.mean(vz_masked[mask_r].to('m/s').value) 
                r_err[i] = np.std(r_masked[mask_r].to('au').value)
                vphi_err[i] = np.std(vphi_masked[mask_r].to('m/s').value)
                dvphi_err[i] = np.std(dvphi_masked[mask_r].to('m/s').value)
                vr_err[i] = np.std(vr_masked[mask_r].to('m/s').value)
                vz_err[i] = np.std(vz_masked[mask_r].to('m/s').value)
            return {'r': r * u.au, 'vphi': vphi * u.m/u.s, 'dvphi': dvphi * u.m/u.s, 'vr': vr * u.m/u.s, 'vz': vz * u.m/u.s,
                    'r_err': r_err * u.au, 'vphi_err': vphi_err * u.m/u.s, 'dvphi_err': dvphi_err * u.m/u.s, 'vr_err': vr_err * u.m/u.s, 'vz_err': vz_err * u.m/u.s}

if __name__ == '__main__':
    path = r"C:\Users\jamiechang917\Desktop\fargo3d-vis\data"
    data = Data(path, 1000, 1, 100.0*u.au, 1.0*u.M_sun)
    # planet_phi_index = np.abs(data.coords['sph_phi'] - data.params['planet0']['y'][data.snap]*u.rad).argmin()
    
    import matplotlib.pyplot as plt

    profiles = data.get_velocity_profiles(254, 0.0, 0.004, 
                                        #   -0.01, 0.01,
                                          fluid='gas', coordinate='cyl')

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(profiles['r'], profiles['dvphi'], label=r'$v_{phi}$')
    ax.fill_between(profiles['r'].value, (profiles['dvphi']-profiles['dvphi_err']).value, (profiles['dvphi']+profiles['dvphi_err']).value, alpha=0.3)
    ax.errorbar(profiles['r'], profiles['dvphi'], yerr=profiles['dvphi_err'], xerr=profiles['r_err'], alpha=0.5, color='gray', fmt='none')
    
    ax.axhline(0, color='k', linestyle='--')
    ax.axvline(100, color='k', linestyle='--')
    plt.show()