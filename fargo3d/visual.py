from fargo3d.data import Data
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.animation import FuncAnimation

def plot_velocity_profiles(data:Data, r_size:int, zr_min:float, zr_max:float,
                           phi_min=-np.pi, phi_max=np.pi, fluid='gas', coordinate='cyl'):
    if coordinate != 'cyl':
        raise NotImplementedError('Only cylindrical coordinates are supported')
    
    profiles = data.get_velocity_profiles(r_size=r_size, zr_min=zr_min, zr_max=zr_max, 
                                          phi_min=phi_min, phi_max=phi_max, fluid=fluid, coordinate=coordinate)
    
    planet_r = data.params["planet0"]["x"][data.snap] * data.units["unit_length"]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, dpi=150)
    plt.subplots_adjust(hspace=0)
    for i in range(len(axes)):
        if i == 0:
            axes[i].plot(profiles['r'], profiles['dvphi'], label=r'$\delta v_{\phi}$')
            axes[i].fill_between(profiles['r'].value, (profiles['dvphi'] - profiles['dvphi_err']).value, (profiles['dvphi'] + profiles['dvphi_err']).value, alpha=0.5)            
            axes[i].set_ylabel(r'$\mathbf{\delta v_{\phi}}$ [m/s]', fontsize=16, fontweight='bold')
        elif i == 1:
            axes[i].plot(profiles['r'], profiles['vr'], label=r'$\delta v_{r}$')
            axes[i].fill_between(profiles['r'].value, (profiles['vr'] - profiles['vr_err']).value, (profiles['vr'] + profiles['vr_err']).value, alpha=0.5)
            axes[i].set_ylabel(r'$\mathbf{\delta v_{r}}$ [m/s]', fontsize=16, fontweight='bold')
        elif i == 2:
            axes[i].plot(profiles['r'], profiles['vz'], label=r'$\delta v_{z}$')
            axes[i].fill_between(profiles['r'].value, (profiles['vz'] - profiles['vz_err']).value, (profiles['vz'] + profiles['vz_err']).value, alpha=0.5)
            axes[i].set_ylabel(r'$\mathbf{\delta v_{z}}$ [m/s]', fontsize=16, fontweight='bold')
            axes[i].set_xlabel('r [AU]', fontsize=16, fontweight='bold')
        axes[i].axhline(0, color='silver', linestyle='--', alpha=0.5)
        axes[i].axvline(planet_r.value, color='silver', linestyle='--', alpha=0.5, label='planet')
        axes[i].set_xlim(profiles['r'].min().value, profiles['r'].max().value)
        axes[i].set_ylim(-max(np.abs(axes[i].get_ylim())), max(np.abs(axes[i].get_ylim()))) # symmetric y-axis
        axes[i].grid(True, which='major', axis='y', color='silver', linestyle='--', alpha=0.3)

        # beautify
        axes[i].xaxis.set_major_locator(MultipleLocator(50))
        axes[i].xaxis.set_minor_locator(MultipleLocator(10))
        axes[i].yaxis.set_major_locator(MultipleLocator(20))
        axes[i].yaxis.set_minor_locator(MultipleLocator(5))
        axes[i].tick_params(axis='both', direction='in', which='both', top=True, right=True, width=2)
        for spine in axes[i].spines.values():
            spine.set_linewidth(2)
        axes[i].yaxis.set_label_coords(-0.05, 0.5) # align ylabels
        

        # custom
        axes[0].yaxis.set_major_locator(MultipleLocator(100))
        axes[0].yaxis.set_minor_locator(MultipleLocator(20))
        axes[1].set_ylim(-50, 50)
        axes[2].set_ylim(-50, 50)
    plt.show(fig)
    return

def plot_velocity_map(data:Data, r_bin:int, z_bin:int, scale:float,
                      phi_min=-np.pi, phi_max=np.pi,
                      r_min=None, r_max=None,
                      z_min=None, z_max=None,
                      fluid='gas', coordinate='cyl',
                      animation=False, animation_kwargs={}):
    """
    Plot the 2D velocity map in the r-z plane.
    data: Data object
    r_bin: int, the bin size in r direction for the quiver plot
    z_bin: int, the bin size in z direction for the quiver plot
    scale: float, the scale of the quiver plot, bigger scale produces smaller arrows
    phi_min: float, the minimum phi angle when averaging the velocity field
    phi_max: float, the maximum phi angle when averaging the velocity field
    r_min: float, the minimum r value for the plot
    r_max: float, the maximum r value for the plot
    z_min: float, the minimum z value for the plot
    z_max: float, the maximum z value for the plot
    fluid: str, the fluid type to plot
    coordinate: str, the coordinate system to plot (only 'cyl' is supported now)
    """
    if coordinate != 'cyl':
        raise NotImplementedError('Only cylindrical coordinates are supported')
    r_mesh, z_mesh = data.mesh["cyl_r"], data.mesh["cyl_z"]
    vr_mesh, vz_mesh = data.fluids["velocity"][fluid]["cyl_r"], data.fluids["velocity"][fluid]["cyl_z"]
    phi_mask = (data.coords["sph_phi"].value >= phi_min) & (data.coords["sph_phi"].value <= phi_max)
    r_mesh_masked, z_mesh_masked, vr_mesh_masked, vz_mesh_masked = r_mesh[:, :, phi_mask], z_mesh[:, :, phi_mask], vr_mesh[:, :, phi_mask], vz_mesh[:, :, phi_mask]
    r_mesh_flatten, z_mesh_flatten, vr_mesh_flatten, vz_mesh_flatten = np.mean(r_mesh_masked, axis=2), np.mean(z_mesh_masked, axis=2), np.mean(vr_mesh_masked, axis=2), np.mean(vz_mesh_masked, axis=2)
    # normalize vr, vz
    vrz_mesh_flatten = np.sqrt(vr_mesh_flatten**2 + vz_mesh_flatten**2)
    vr_mesh_norm = vr_mesh_flatten / vrz_mesh_flatten
    vz_mesh_norm = vz_mesh_flatten / vrz_mesh_flatten

    density_mesh = data.fluids["density"][fluid]
    density_mesh_masked = density_mesh[:, :, phi_mask]
    density_mesh_flatten = np.mean(density_mesh_masked, axis=2)

    fig, ax = plt.subplots(1, 1, figsize=(12, 4), dpi=150)
    pcm = ax.pcolormesh(r_mesh_flatten.value, z_mesh_flatten.value, vz_mesh_flatten.value, cmap='coolwarm', shading='auto', vmax=20, vmin=-20)
    cbar = fig.colorbar(pcm, ax=ax, orientation='vertical', pad=0.0, extend='both', label=r'$\delta v_{z}$ [m/s]')
    # ax.pcolormesh(r_mesh_flatten.value, z_mesh_flatten.value,  np.log10(density_mesh_flatten.value), cmap='jet', shading='auto', vmin=-17, vmax=-16)
    quiver = ax.quiver(
              r_mesh_flatten.value[::z_bin, ::r_bin], z_mesh_flatten.value[::z_bin, ::r_bin],
              vr_mesh_norm.value[::z_bin, ::r_bin],   vz_mesh_norm.value[::z_bin, ::r_bin],
              vrz_mesh_flatten.value[::z_bin, ::r_bin],
              scale=scale, scale_units='inches', alpha=0.8, zorder=100, cmap='rainbow', clim=(0, 100))
    # cbar = fig.colorbar(quiver, ax=ax, orientation='vertical', pad=0.0, extend='both', label=r'$\delta v_{r}$ [m/s]')

    
    # time
    time = data.snap * data.units["unit_time_onesnap"].to('yr').value
    time_text = ax.text(0.02, 0.95, f'{time/1e6:.3f} Myr', transform=ax.transAxes, fontsize=12, 
                        fontweight='bold', color='black', ha='left', va='top', 
                        bbox=dict(facecolor='white', alpha=0.5, edgecolor='black', linewidth=1.5))

    # z/r
    for zr in [0.1, 0.2, 0.3, 0.4, 0.5]:
        r = np.linspace(0.0, r_mesh_flatten.max().value, 100)
        z = r * zr
        ax.plot(r, z, color='gray', linestyle='-', alpha=0.5)

    ax.set_xlabel('r [AU]', fontsize=16, fontweight='bold')
    ax.set_ylabel('z [AU]', fontsize=16, fontweight='bold')

    # beautify
    ax.xaxis.set_major_locator(MultipleLocator(50))
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    ax.tick_params(axis='both', direction='in', which='both', top=True, right=True, width=2)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.yaxis.set_label_coords(-0.05, 0.5) # align ylabels

    if r_min is not None and r_max is not None:
        ax.set_xlim(r_min, r_max)
    else:
        ax.set_xlim(0, r_mesh_flatten.max().value)
    if z_min is not None and z_max is not None:
        ax.set_ylim(z_min, z_max)
    else:
        ax.set_ylim(0.0, z_mesh_flatten.max().value)

    if animation:
        data_kwargs = {"path": data.path, "n_dust": data.n_dust, "unit_length": data.units["unit_length"], "unit_mass": data.units["unit_mass"]}
        # progress bar
        pbar = tqdm(total=animation_kwargs["snap_end"] - animation_kwargs["snap_start"] + 1, desc='Plotting velocity map')
        # update function
        def update(frame):
            plt.clf()
            fig, ax = plt.subplots(1, 1, figsize=(12, 4), dpi=150)
            data = Data(snap=frame, **data_kwargs)
            vr_mesh, vz_mesh = data.fluids["velocity"][fluid]["cyl_r"], data.fluids["velocity"][fluid]["cyl_z"]
            vr_mesh_masked, vz_mesh_masked = vr_mesh[:, :, phi_mask], vz_mesh[:, :, phi_mask]
            vr_mesh_flatten, vz_mesh_flatten = np.mean(vr_mesh_masked, axis=2), np.mean(vz_mesh_masked, axis=2)
            vrz_mesh_flatten = np.sqrt(vr_mesh_flatten**2 + vz_mesh_flatten**2)
            vr_mesh_norm = vr_mesh_flatten / vrz_mesh_flatten
            vz_mesh_norm = vz_mesh_flatten / vrz_mesh_flatten

            # pcm.set_array(vz_mesh_flatten.ravel())
            # quiver.set_UVC(vr_mesh_norm.value[::z_bin, ::r_bin], vz_mesh_norm.value[::z_bin, ::r_bin])
            # time = data.snap * data.units["unit_time_onesnap"].to('yr').value
            # time_text.set_text(f'{time/1e6:.3f} Myr')

            pcm = ax.pcolormesh(r_mesh_flatten.value, z_mesh_flatten.value, vz_mesh_flatten.value, cmap='coolwarm', shading='auto', vmax=20, vmin=-20)
            cbar = fig.colorbar(pcm, ax=ax, orientation='vertical', pad=0.0, extend='both', label=r'$\delta v_{z}$ [m/s]')
            quiver = ax.quiver(
                                r_mesh_flatten.value[::z_bin, ::r_bin], z_mesh_flatten.value[::z_bin, ::r_bin],
                                vr_mesh_norm.value[::z_bin, ::r_bin],   vz_mesh_norm.value[::z_bin, ::r_bin],
                                vrz_mesh_flatten.value[::z_bin, ::r_bin],
                                scale=scale, scale_units='inches', alpha=0.8, zorder=100, cmap='rainbow', clim=(0, 100))
            time = data.snap * data.units["unit_time_onesnap"].to('yr').value
            time_text = ax.text(0.02, 0.95, f'{time/1e6:.3f} Myr', transform=ax.transAxes, fontsize=12, 
                                fontweight='bold', color='black', ha='left', va='top', 
                                bbox=dict(facecolor='white', alpha=0.5, edgecolor='black', linewidth=1.5))
            for zr in [0.1, 0.2, 0.3, 0.4, 0.5]:
                r = np.linspace(0.0, r_mesh_flatten.max().value, 100)
                z = r * zr
                ax.plot(r, z, color='gray', linestyle='-', alpha=0.5)
            ax.set_xlabel('r [AU]', fontsize=16, fontweight='bold')
            ax.set_ylabel('z [AU]', fontsize=16, fontweight='bold')

            # beautify
            ax.xaxis.set_major_locator(MultipleLocator(50))
            ax.xaxis.set_minor_locator(MultipleLocator(10))
            ax.yaxis.set_major_locator(MultipleLocator(20))
            ax.yaxis.set_minor_locator(MultipleLocator(5))
            ax.tick_params(axis='both', direction='in', which='both', top=True, right=True, width=2)
            for spine in ax.spines.values():
                spine.set_linewidth(2)
            ax.yaxis.set_label_coords(-0.05, 0.5) # align ylabels

            if r_min is not None and r_max is not None:
                ax.set_xlim(r_min, r_max)
            else:
                ax.set_xlim(0, r_mesh_flatten.max().value)
            if z_min is not None and z_max is not None:
                ax.set_ylim(z_min, z_max)
            else:
                ax.set_ylim(0.0, z_mesh_flatten.max().value)



            pbar.update(1)
        anim = FuncAnimation(fig, update, frames=np.arange(animation_kwargs["snap_start"], animation_kwargs["snap_end"]+1), interval=animation_kwargs["interval"])
        anim.save('velocity_map_vrvz.mp4', writer='ffmpeg', dpi=300)
        pbar.close()
    else:
        plt.show(fig)
    return
