import os
import astropy.constants as const
import radmc3dPy as r3d

rt_params = {
    'dpc': 100.0,         # distance in parsecs
    'widthkms': 6.0,      # km/s
    'linenlam': 120,      # number of channels
    'nu0': 219.560 * 1e9, # Hz
    
}

if __name__ == '__main__':
    os.chdir("../rt_output")
    im = r3d.image.readImage()
    bandwidthmhz = (rt_params["nu0"] * 1e-6) * (2 * rt_params['widthkms'] / const.c.to('km/s').value / (rt_params['linenlam'] - 1))
    im.writeFits(fname="lines.fits", dpc=rt_params['dpc'],
                 casa=True, coord='00h00m00s +00d00m00s', 
                 nu0=rt_params['nu0'], bandwidthmhz=bandwidthmhz)