import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter
import numpy as np
from matplotlib.animation import FuncAnimation

from lenspack.utils import bin2d
from lenspack.image.inversion import ks93
from lenspack.peaks import find_peaks2d

def plot_convergence(convergence, boundaries, config, output_name):
    """
    Make plot of convergence map and save to file using information passed
    in run configuration file. 

    Arguments
        convergence: XXX raw convergence map XXX
        boundaries: XXX RA/Dec axis limits for plot, set in XXX
        config: overall run configuration file

    """

    # Embiggen font sizes, tick marks, etc.
    fontsize = 15
    plt.rcParams.update({'axes.linewidth': 1.3})
    plt.rcParams.update({'xtick.labelsize': fontsize})
    plt.rcParams.update({'ytick.labelsize': fontsize})
    plt.rcParams.update({'xtick.major.size': 8})
    plt.rcParams.update({'xtick.major.width': 1.3})
    plt.rcParams.update({'xtick.minor.visible': True})
    plt.rcParams.update({'xtick.minor.width': 1.})
    plt.rcParams.update({'xtick.minor.size': 6})
    plt.rcParams.update({'xtick.direction': 'in'})
    plt.rcParams.update({'ytick.major.width': 1.3})
    plt.rcParams.update({'ytick.major.size': 8})
    plt.rcParams.update({'ytick.minor.visible': True})
    plt.rcParams.update({'ytick.minor.width': 1.})
    plt.rcParams.update({'ytick.minor.size':6})
    plt.rcParams.update({'ytick.direction':'in'})
    plt.rcParams.update({'axes.labelsize': fontsize})
    plt.rcParams.update({'axes.titlesize': fontsize})

    
    # Apply Gaussian filter -- is this the right place to do it?
    # We are planning on implementing other filters at some point, right?
    filtered_convergence = gaussian_filter(convergence, config['gaussian_kernel'])

    # Make the plot!
    fig, ax = plt.subplots(
        nrows=1, ncols=1, figsize=config['figsize'], tight_layout=True
    )
    
    # Create an aspect ratio that accounts for the curvature of the sky (pixels are not square)
    # TODO: make this more robust. Currently, this takes an average declination (middle of image)
    # and linearly scales the aspect ratio based on that 'middle' declination.
    aspect_ratio = np.cos(np.deg2rad((boundaries['dec_max'] + boundaries['dec_min']) / 2))
    
    im = ax.imshow(
        filtered_convergence[:, ::-1], 
        cmap=config['cmap'],
        vmax=config['vmax'], 
        vmin=config['vmin'],
        extent=[boundaries['ra_max'], 
                    boundaries['ra_min'], 
                    boundaries['dec_min'], 
                    boundaries['dec_max']],
        origin='lower', # Sets the origin to bottom left to match the RA/DEC convention
        aspect=(1/aspect_ratio)
    )  

    ax.set_xlabel(config['xlabel'])
    ax.set_ylabel(config['ylabel'])
    ax.set_title(config['plot_title'])

    # Is there a better way to force something to be a boolean?
    if config['gridlines'] == True:
        ax.grid(color='black')

    # Add colorbar; turn off minor axes first
    plt.rcParams.update({'ytick.minor.visible': False})
    plt.rcParams.update({'xtick.minor.visible': False})

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.07)
    fig.colorbar(im, cax=cax)

    # Save to file and exit, redoing tight_layout b/c sometimes figure gets cut off 
    fig.tight_layout() 
    fig.savefig(output_name)
    print(f"Convergence map saved as PNG file: {output_name}")
    plt.close(fig)

def plot_convergence_v2(convergence, boundaries, config, output_name="Converenge map", center_cl=None, smoothing=None, vmax=None, vmin=None, title=None, threshold = None):
    """
    Make plot of convergence map and save to file using information passed
    in run configuration file. 

    Arguments
        convergence: XXX raw convergence map XXX
        boundaries: XXX RA/Dec axis limits for plot, set in XXX
        config: overall run configuration file

    """

    # Embiggen font sizes, tick marks, etc.
    fontsize = 15
    plt.rcParams.update({'axes.linewidth': 1.3})
    plt.rcParams.update({'xtick.labelsize': fontsize})
    plt.rcParams.update({'ytick.labelsize': fontsize})
    plt.rcParams.update({'xtick.major.size': 8})
    plt.rcParams.update({'xtick.major.width': 1.3})
    plt.rcParams.update({'xtick.minor.visible': True})
    plt.rcParams.update({'xtick.minor.width': 1.})
    plt.rcParams.update({'xtick.minor.size': 6})
    plt.rcParams.update({'xtick.direction': 'in'})
    plt.rcParams.update({'ytick.major.width': 1.3})
    plt.rcParams.update({'ytick.major.size': 8})
    plt.rcParams.update({'ytick.minor.visible': True})
    plt.rcParams.update({'ytick.minor.width': 1.})
    plt.rcParams.update({'ytick.minor.size':6})
    plt.rcParams.update({'ytick.direction':'in'})
    plt.rcParams.update({'axes.labelsize': fontsize})
    plt.rcParams.update({'axes.titlesize': fontsize})

    
    # Apply Gaussian filter -- is this the right place to do it?
    # We are planning on implementing other filters at some point, right?
    #filtered_convergence = gaussian_filter(convergence, config['gaussian_kernel'])
    
    if smoothing is not None:
        filtered_convergence = gaussian_filter(convergence, smoothing)
    else:
        filtered_convergence = convergence
        
    if threshold is not None:
        y, x, h = find_peaks2d(filtered_convergence, threshold=threshold, include_border=False)
    else: # empty lists
        x, y, h = [], [], []

    # Make the plot!
    fig, ax = plt.subplots(
        nrows=1, ncols=1, figsize=config['figsize'], tight_layout=True
    )
    
    if vmax is None:
        vmax = config['vmax']
    if vmin is None:
        vmin = config['vmin']
    if title is None:
        title = config['plot_title']
    
    im = ax.imshow(
        filtered_convergence[:, ::-1], 
        cmap=config['cmap'],
        vmax=vmax, 
        vmin=vmin,
        extent=[boundaries['ra_max'], 
                    boundaries['ra_min'], 
                    boundaries['dec_min'], 
                    boundaries['dec_max']],
        origin='lower' # Sets the origin to bottom left to match the RA/DEC convention
    )
    
    if center_cl is not None:
        ra_c, dec_c = center_cl["ra_c"], center_cl["dec_c"]
        ax.plot(ra_c, dec_c, 'wx', markersize=10)
        
    # convert x,y to ra,dec
    ra_peak, dec_peak = [], []
    for i in range(len(x)):
        ra_peak.append(boundaries['ra_min'] + (x[i]+0.5) * (boundaries['ra_max'] - boundaries['ra_min']) / filtered_convergence.shape[1])
        dec_peak.append(boundaries['dec_min'] + (y[i]+0.5) * (boundaries['dec_max'] - boundaries['dec_min']) / filtered_convergence.shape[0])
    
    ax.scatter(ra_peak, dec_peak, s=100, facecolors='none', edgecolors='g', linewidth=1.5)
      
    ax.set_xlabel(config['xlabel'])
    ax.set_ylabel(config['ylabel'])
    ax.set_title(title)

    # Is there a better way to force something to be a boolean?
    if config['gridlines'] == True:
        ax.grid(color='black')

    # Add colorbar; turn off minor axes first
    plt.rcParams.update({'ytick.minor.visible': False})
    plt.rcParams.update({'xtick.minor.visible': False})

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.07)
    fig.colorbar(im, cax=cax)

    # Save to file and exit, redoing tight_layout b/c sometimes figure gets cut off 
    fig.tight_layout() 
    fig.savefig(output_name)
    print(f"Convergence map saved as PNG file: {output_name}")
    plt.close(fig)
    
def plot_convergence_v3(convergence, boundaries, config, output_name="Converenge map", center_cl=None, smoothing=None, vmax=None, vmin=None, title=None, threshold = None):
    """
    Make plot of convergence map and save to file using information passed
    in run configuration file. 

    Arguments
        convergence: XXX raw convergence map XXX
        boundaries: XXX RA/Dec axis limits for plot, set in XXX
        config: overall run configuration file

    """

    # Embiggen font sizes, tick marks, etc.
    fontsize = 15
    plt.rcParams.update({'axes.linewidth': 1.3})
    plt.rcParams.update({'xtick.labelsize': fontsize})
    plt.rcParams.update({'ytick.labelsize': fontsize})
    plt.rcParams.update({'xtick.major.size': 8})
    plt.rcParams.update({'xtick.major.width': 1.3})
    plt.rcParams.update({'xtick.minor.visible': True})
    plt.rcParams.update({'xtick.minor.width': 1.})
    plt.rcParams.update({'xtick.minor.size': 6})
    plt.rcParams.update({'xtick.direction': 'in'})
    plt.rcParams.update({'ytick.major.width': 1.3})
    plt.rcParams.update({'ytick.major.size': 8})
    plt.rcParams.update({'ytick.minor.visible': True})
    plt.rcParams.update({'ytick.minor.width': 1.})
    plt.rcParams.update({'ytick.minor.size':6})
    plt.rcParams.update({'ytick.direction':'in'})
    plt.rcParams.update({'axes.labelsize': fontsize})
    plt.rcParams.update({'axes.titlesize': fontsize})

    
    # Apply Gaussian filter -- is this the right place to do it?
    # We are planning on implementing other filters at some point, right?
    #filtered_convergence = gaussian_filter(convergence, config['gaussian_kernel'])
    
    if smoothing is not None:
        filtered_convergence = gaussian_filter(convergence, smoothing)
    else:
        filtered_convergence = convergence
        
    if threshold is not None:
        y, x, h = find_peaks2d(filtered_convergence, threshold=threshold, include_border=False)
    else: # empty lists
        x, y, h = [], [], []

    # Make the plot!
    fig, ax = plt.subplots(
        nrows=1, ncols=1, figsize=config['figsize'], tight_layout=True
    )

    
    im = ax.imshow(
        filtered_convergence[:, ::-1], 
        cmap=config['cmap'],
        vmax=vmax, 
        vmin=vmin,
        extent=[boundaries['ra_max'], 
                    boundaries['ra_min'], 
                    boundaries['dec_min'], 
                    boundaries['dec_max']],
        origin='lower' # Sets the origin to bottom left to match the RA/DEC convention
    )
    
    if center_cl is not None:
        ra_c, dec_c = center_cl["ra_center"], center_cl["dec_center"]
        ax.plot(ra_c, dec_c, 'wx', markersize=10)
        
    # convert x,y to ra,dec
    ra_peak, dec_peak = [], []
    for i in range(len(x)):
        ra_peak.append(boundaries['ra_min'] + (x[i]+0.5) * (boundaries['ra_max'] - boundaries['ra_min']) / filtered_convergence.shape[1])
        dec_peak.append(boundaries['dec_min'] + (y[i]+0.5) * (boundaries['dec_max'] - boundaries['dec_min']) / filtered_convergence.shape[0])
    
    ax.scatter(ra_peak, dec_peak, s=100, facecolors='none', edgecolors='g', linewidth=1.5)
      
    ax.set_xlabel(config['xlabel'])
    ax.set_ylabel(config['ylabel'])
    ax.set_title(title)

    # Is there a better way to force something to be a boolean?
    if config['gridlines'] == True:
        ax.grid(color='black')

    # Add colorbar; turn off minor axes first
    plt.rcParams.update({'ytick.minor.visible': False})
    plt.rcParams.update({'xtick.minor.visible': False})

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.07)
    fig.colorbar(im, cax=cax)

    # Save to file and exit, redoing tight_layout b/c sometimes figure gets cut off 
    fig.tight_layout() 
    plt.show()
    #fig.savefig(config['output_path'])
    plt.close(fig)

def plot_convergence_v4(convergence, scaled_boundaries, true_boundaries, config,  output_name="Converenge map", center_cl=None, smoothing=None, invert_map=True, vmax=None, vmin=None, title=None, threshold = None, con_peaks=None):
    """
    Make plot of convergence map and save to file using information passed
    in run configuration file. 

    Arguments
        convergence: XXX raw convergence map XXX
        boundaries: XXX RA/Dec axis limits for plot, set in XXX
        config: overall run configuration file

    """

    # Embiggen font sizes, tick marks, etc.
    fontsize = 15
    plt.rcParams.update({'axes.linewidth': 1.3})
    plt.rcParams.update({'xtick.labelsize': fontsize})
    plt.rcParams.update({'ytick.labelsize': fontsize})
    plt.rcParams.update({'xtick.major.size': 8})
    plt.rcParams.update({'xtick.major.width': 1.3})
    plt.rcParams.update({'xtick.minor.visible': True})
    plt.rcParams.update({'xtick.minor.width': 1.})
    plt.rcParams.update({'xtick.minor.size': 6})
    plt.rcParams.update({'xtick.direction': 'in'})
    plt.rcParams.update({'ytick.major.width': 1.3})
    plt.rcParams.update({'ytick.major.size': 8})
    plt.rcParams.update({'ytick.minor.visible': True})
    plt.rcParams.update({'ytick.minor.width': 1.})
    plt.rcParams.update({'ytick.minor.size':6})
    plt.rcParams.update({'ytick.direction':'in'})
    plt.rcParams.update({'axes.labelsize': fontsize})
    plt.rcParams.update({'axes.titlesize': fontsize})

    
    # Apply Gaussian filter -- is this the right place to do it?
    # We are planning on implementing other filters at some point, right?
    #filtered_convergence = gaussian_filter(convergence, config['gaussian_kernel'])
    
    if smoothing is not None:
        filtered_convergence = gaussian_filter(convergence, smoothing)
    else:
        filtered_convergence = convergence
    
    # Determine the central 50% area
    ny, nx = filtered_convergence.shape
    x_start, x_end = nx // 4, 3 * nx // 4
    y_start, y_end = ny // 4, 3 * ny // 4
   
    peaks = find_peaks2d(filtered_convergence[:,::-1], threshold=threshold, include_border=False, ordered=False) if threshold is not None else ([], [], [])
    
    # Find peaks which are in the central 50% area
    filtered_indices = [i for i in range(len(peaks[0])) if y_start <= peaks[0][i] < y_end and x_start <= peaks[1][i] < x_end]
    peaks = ([peaks[0][i] for i in filtered_indices], [peaks[1][i] for i in filtered_indices], [peaks[2][i] for i in filtered_indices])
    
    # find the center of the peaks by adding 0.5 with every pixel
    peaks = ([x+0.5 for x in peaks[0]], [y+0.5 for y in peaks[1]], peaks[2])
    
    print(f"Number of peaks: {len(peaks[0])}")
    
    if invert_map:
        #peaks = find_peaks2d(filtered_convergence, threshold=threshold, include_border=False) if threshold is not None else ([], [], [])
        xcr = []
        for x in peaks[1]:
            xcr.append(filtered_convergence.shape[1] - x)
        peaks = (peaks[0], xcr, peaks[2])
    else:
#        peaks = find_peaks2d(filtered_convergence[:,::-1], threshold=threshold, include_border=False) if threshold is not None else ([], [], [])
        peaks = ([x for x in peaks[0]], [y-1.0 for y in peaks[1]], peaks[2])
    ra_peaks = [scaled_boundaries['ra_min'] + (x) * (scaled_boundaries['ra_max'] - scaled_boundaries['ra_min']) / filtered_convergence.shape[1] for x in peaks[1]]
    dec_peaks = [scaled_boundaries['dec_min'] + (y) * (scaled_boundaries['dec_max'] - scaled_boundaries['dec_min']) / filtered_convergence.shape[0] for y in peaks[0]]        
    if invert_map:
        filtered_convergence = filtered_convergence[:, ::-1]
        
    # Find peaks of convergence


    # Make the plot!
    fig, ax = plt.subplots(
        nrows=1, ncols=1, figsize=config['figsize'], tight_layout=True
    )
    if threshold is not None:
        for ra, dec, peak_value in zip(ra_peaks, dec_peaks, peaks[2]):
            ax.scatter(ra, dec, s=50, facecolors='none', edgecolors='g', linewidth=1.5, label='Convergence Peak = %f' % peak_value)
        # To avoid multiple identical legend entries, we can add a single legend entry manually
        ax.legend(['Convergence Peaks'])
    extent = [scaled_boundaries['ra_max'], 
                scaled_boundaries['ra_min'], 
                scaled_boundaries['dec_min'], 
                scaled_boundaries['dec_max']]
    
    #extent = [scaled_boundaries['ra_min'], scaled_boundaries['ra_max'], scaled_boundaries['dec_min'], scaled_boundaries['dec_max']]
    
    im = ax.imshow(
        filtered_convergence, 
        cmap=config['cmap'],
        vmax=vmax, 
        vmin=vmin,
        extent=extent,
        origin='lower' # Sets the origin to bottom left to match the RA/DEC convention
    )
    
    # Mark cluster center if specified
    cluster_center = center_cl
    ra_center = None
    dec_center = None
    
    if cluster_center == 'auto':
        ra_center = (scaled_boundaries['ra_max'] + scaled_boundaries['ra_min']) / 2
        dec_center = (scaled_boundaries['dec_max'] + scaled_boundaries['dec_min']) / 2
    elif isinstance(cluster_center, dict):
        ra_center = cluster_center['ra_center']
        dec_center = cluster_center['dec_center']
    elif cluster_center is not None:
        print("Unrecognized cluster_center format, skipping marker.")
        ra_center = dec_center = None

    if ra_center is not None:
        if not invert_map:
            ra_center =  (scaled_boundaries['ra_max'] - ra_center) + scaled_boundaries['ra_min']
        ax.scatter(ra_center, dec_center, marker='x', color='lime', s=75, label='Nominal Cluster Center')
        #ax.axhline(y=dec_center, color='w', linestyle='--')
        #ax.axvline(x=ra_center, color='w', linestyle='--')
        




    # Determine nice step sizes based on the range
    ra_range = true_boundaries['ra_max'] - true_boundaries['ra_min']
    dec_range = true_boundaries['dec_max'] - true_boundaries['dec_min']

    # Choose step size (0.01, 0.05, 0.1, 0.25, 0.5) based on range size
    possible_steps = np.array([0.01, 0.05, 0.1, 0.2, 0.5])
    ra_step = possible_steps[np.abs(ra_range/5 - possible_steps).argmin()]
    dec_step = possible_steps[np.abs(dec_range/5 - possible_steps).argmin()]

    # Generate ticks
    x_ticks = np.arange(np.ceil(true_boundaries['ra_min']/ra_step)*ra_step,
                        np.floor(true_boundaries['ra_max']/ra_step)*ra_step + ra_step/2,
                        ra_step)
    y_ticks = np.arange(np.ceil(true_boundaries['dec_min']/dec_step)*dec_step,
                        np.floor(true_boundaries['dec_max']/dec_step)*dec_step + dec_step/2,
                        dec_step)

    # Convert to scaled coordinates
    scaled_x_ticks = np.interp(x_ticks, 
                            [true_boundaries['ra_min'], true_boundaries['ra_max']], 
                            [scaled_boundaries['ra_min'], scaled_boundaries['ra_max']])
    scaled_y_ticks = np.interp(y_ticks, 
                            [true_boundaries['dec_min'], true_boundaries['dec_max']], 
                            [scaled_boundaries['dec_min'], scaled_boundaries['dec_max']])

    # Set the ticks
    ax.set_xticks(scaled_x_ticks)
    ax.set_yticks(scaled_y_ticks)
    ax.set_xticklabels([f"{x:.2f}" for x in x_ticks])
    ax.set_yticklabels([f"{y:.2f}" for y in y_ticks])
      
    ax.set_xlabel(config['xlabel'])
    ax.set_ylabel(config['ylabel'])
    ax.set_title(title)
    ax.legend(loc='upper left')

    # Is there a better way to force something to be a boolean?
    if config['gridlines'] == True:
        ax.grid(color='black')

    # Add colorbar; turn off minor axes first
    plt.rcParams.update({'ytick.minor.visible': False})
    plt.rcParams.update({'xtick.minor.visible': False})

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.07)
    fig.colorbar(im, cax=cax)

    # Save to file and exit, redoing tight_layout b/c sometimes figure gets cut off 
    fig.tight_layout() 
    #plt.show(block=True)
    #plt.show()
    fig.savefig(config['output_path'])
    plt.close(fig)
    return ra_peaks, dec_peaks, peaks[2]

def plot_animation(convergence, boundaries, config, output_name='animation.mp4', center_cl=None, smoothing=False, num_frames=50, fps=5):
    """
    Make plot of convergence map and save to file using information passed
    in run configuration file. 

    Arguments
        convergence: XXX raw convergence map XXX
        boundaries: XXX RA/Dec axis limits for plot, set in XXX
        config: overall run configuration file

    """

    # Embiggen font sizes, tick marks, etc.
    fontsize = 15
    plt.rcParams.update({'axes.linewidth': 1.3})
    plt.rcParams.update({'xtick.labelsize': fontsize})
    plt.rcParams.update({'ytick.labelsize': fontsize})
    plt.rcParams.update({'xtick.major.size': 8})
    plt.rcParams.update({'xtick.major.width': 1.3})
    plt.rcParams.update({'xtick.minor.visible': True})
    plt.rcParams.update({'xtick.minor.width': 1.})
    plt.rcParams.update({'xtick.minor.size': 6})
    plt.rcParams.update({'xtick.direction': 'in'})
    plt.rcParams.update({'ytick.major.width': 1.3})
    plt.rcParams.update({'ytick.major.size': 8})
    plt.rcParams.update({'ytick.minor.visible': True})
    plt.rcParams.update({'ytick.minor.width': 1.})
    plt.rcParams.update({'ytick.minor.size':6})
    plt.rcParams.update({'ytick.direction':'in'})
    plt.rcParams.update({'axes.labelsize': fontsize})
    plt.rcParams.update({'axes.titlesize': fontsize})

    
    # Apply Gaussian filter -- is this the right place to do it?
    # We are planning on implementing other filters at some point, right?
    #filtered_convergence = gaussian_filter(convergence, config['gaussian_kernel'])
    # Set the number of frames to 20 or the total length of kappa_e_stack
    #num_frames = min(50, len(convergence))
    
    # Make the plot!
    fig, ax = plt.subplots(
        nrows=1, ncols=1, figsize=config['figsize'], tight_layout=True
    )
        
    def update(frame):
        
        if smoothing:
            filtered_convergence = gaussian_filter(convergence[frame], config['gaussian_kernel'])
        else:
            filtered_convergence = convergence[frame]
            
        im = ax.imshow(
            filtered_convergence[:, ::-1], 
            cmap=config['cmap'],
            vmax=1.5, 
            vmin=-1.5,
            extent=[boundaries['ra_max'], 
                    boundaries['ra_min'], 
                    boundaries['dec_min'], 
                    boundaries['dec_max']],
            origin='lower' # Sets the origin to bottom left to match the RA/DEC convention
        )  
        if center_cl is not None:
            ra_c, dec_c = center_cl["ra_c"], center_cl["dec_c"]
            ax.plot(ra_c, dec_c, 'wx', markersize=10)

        ax.set_xlabel(config['xlabel'])
        ax.set_ylabel(config['ylabel'])
        #ax.set_title(config['plot_title'])

        # Is there a better way to force something to be a boolean?
        if config['gridlines'] == True:
            ax.grid(color='black')

        # Add colorbar; turn off minor axes first
        plt.rcParams.update({'ytick.minor.visible': False})
        plt.rcParams.update({'xtick.minor.visible': False})

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.07)
        fig.colorbar(im, cax=cax)

        # Save to file and exit, redoing tight_layout b/c sometimes figure gets cut off 
        fig.tight_layout() 
        #plt.show()
        #fig.savefig(config['output_path'])
        #plt.close(fig)
        # Create the animation for the first 20 frames
    ani = FuncAnimation(fig, update, frames=num_frames, repeat=False)
    #ani.save('kappa_e_stack_animation_fixed.gif', writer='imagemagick', fps=2)
    #ani.save('kappa_e_stack_animation_smoothed.mp4', writer='ffmpeg', fps=5)
    ani.save(output_name, writer='ffmpeg', fps=fps)
    print(f"Convergence map saved as MP4 file: {output_name}")
    plt.close(fig)