from era5_plot_params import Params
p = Params()
p.init()

from siphon.catalog import TDSCatalog
from siphon.http_util import session_manager

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndimage
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap
from matplotlib.colors import BoundaryNorm
import matplotlib.gridspec as gridspec

import metpy.calc as mpcalc
import metpy.plots as mpplt
from metpy.units import units

import datetime

# Set the requested date
rd = datetime.datetime.strptime(p.opt['tstring'],'%Y-%m-%d %H:%M:%S')

# What date string
yyyymm = rd.strftime('%Y%m')
yyyymmdd = rd.strftime('%Y%m%d')

# Set RDA credentials
session_manager.set_session_options(auth=p.opt['creds'])

# The dataset catalog
cat = TDSCatalog('https://rda.ucar.edu/thredds/catalog/files/g/ds633.0/e5.oper.an.pl/'+yyyymm+'/catalog.xml')

# Get all of the datasets in the catalog
files = cat.datasets

# Turn this list of files into a list
allfiles = list(files)

# Loop through the files and save the ones we want to load
casefiles = [i for i in allfiles if yyyymmdd in  i]

# Find the indexes in the list of files we want to load
indexes = [allfiles.index(f) for f in casefiles]

# Load using list comprehension, creating list of xarray dataset objects
singlesets = [files[i].remote_access(use_xarray=True) for i in indexes]

# Combine all of the datasets (all files into a single dataset)
ds = xr.combine_by_coords(singlesets)
print(ds)

# Subset the dataset. We want all levels, at a specific time, and reduce lat/lon
ds = ds.sel(time=rd,latitude=slice(60,15),longitude=slice(230,300))

# Coordinate reference system
crs = ccrs.LambertConformal(central_longitude=-100.0, central_latitude=45.0)

# Set colors for RH shading
rh_colors = [(139/255, 255/255, 9/255),
             (34/255, 255/255, 8/255),
             (26/255, 198/255, 4/255)]
rhcmap = ListedColormap(rh_colors)

# Create a normalization of RH values into the colormap
norm = BoundaryNorm([70,80,90],rhcmap.N)

# Combine 1D latitude and longitudes into a 2D grid of locations
lon_2d, lat_2d = np.meshgrid(ds['longitude'], ds['latitude'])

# Smooth the height data
heights_500 = ndimage.gaussian_filter(mpcalc.geopotential_to_height(ds['Z'].sel(level=500.0)), sigma=1.5, order=0)
heights_700 = ndimage.gaussian_filter(mpcalc.geopotential_to_height(ds['Z'].sel(level=700.0)), sigma=1.5, order=0)
heights_850 = ndimage.gaussian_filter(mpcalc.geopotential_to_height(ds['Z'].sel(level=850.0)), sigma=1.5, order=0)
heights_925 = ndimage.gaussian_filter(mpcalc.geopotential_to_height(ds['Z'].sel(level=925.0)), sigma=1.5, order=0)

# Contour levels for heights
h5lev = np.arange(4800.0,5800.0,30.0)
h6lev = np.arange(3500.0,4500.0,30.0)
h7lev = np.arange(2500.0,3500.0,30.0)
h85lev = np.arange(1000.0,2000.0,30.0)
h92lev = np.arange(0.0,1000.0,30.0)

# Set a new figure window
fig = plt.figure(1, figsize=(22, 15))

# Use gridspec
gs = gridspec.GridSpec(nrows=2,ncols=2,height_ratios=[1,1],hspace=0.03,wspace=0.03)

# Setup axis
def axis_setup(ax):
  ax.set_extent(p.opt['zoom'],ccrs.PlateCarree())
  ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.5)
  ax.add_feature(cfeature.STATES, linewidth=0.5)
  ax.add_feature(cfeature.BORDERS, linewidth=0.5)
  return ax

# UPPER LEFT PANEL- 500 hPa height/temp/RH
ax1 = plt.subplot(gs[0,0],projection=crs)
axis_setup(ax1)
clev = 500.0
cf1 = ax1.contourf(lon_2d, lat_2d, ds['R'].sel(level=500.0),
                         extend='max', cmap=rhcmap, norm=plt.Normalize(70,110), levels=[70,80,90],transform=ccrs.PlateCarree())
c1 = ax1.contour(lon_2d, lat_2d, heights_500, h5lev, colors='black', linewidths=2,
                       transform=ccrs.PlateCarree())
c1b = ax1.contour(lon_2d, lat_2d, ds['T'].sel(level=clev)-273.15, levels=[-30,-25,-20,-15,-10,-5],
                        colors='cyan', linewidths=1, linestyles='dashed', transform=ccrs.PlateCarree())
c1c = ax1.contour(lon_2d, lat_2d, ds['T'].sel(level=clev)-273.15, levels=[0],
                        colors='blue', linewidths=2, linestyles='solid', transform=ccrs.PlateCarree())
c1d = ax1.contour(lon_2d, lat_2d, ds['T'].sel(level=clev)-273.15, levels=[5],
                        colors='orange', linewidths=1, linestyles='solid', transform=ccrs.PlateCarree())
c1e = ax1.contour(lon_2d, lat_2d, ds['T'].sel(level=clev)-273.15, levels=[10,15,20,25,30],
                        colors='red', linewidths=1, linestyles='solid', transform=ccrs.PlateCarree())
#ax1.clabel(c1, fontsize=10, inline=1, inline_spacing=1, fmt='%i', rightside_up=True)
ax1.set_title('500-hPa RH/T and Heights', fontsize=16)
cb1 = plt.colorbar(cf1, ax=ax1, orientation='horizontal', shrink=0.74, pad=0.05)
cb1.set_label('%', size='x-large')

# UPPER RIGHT PANEL- 700 hPa height/temp/RH
ax2 = plt.subplot(gs[0,1],projection=crs)
axis_setup(ax2)
clev = 700.0
cf2 = ax2.contourf(lon_2d, lat_2d, ds['R'].sel(level=700.0), norm=plt.Normalize(70,110), levels=[70,80,90],
                         extend='max', cmap=rhcmap, transform=ccrs.PlateCarree())
c2 = ax2.contour(lon_2d, lat_2d, heights_700, h7lev, colors='black', linewidths=2,
                       transform=ccrs.PlateCarree())
c2b = ax2.contour(lon_2d, lat_2d, ds['T'].sel(level=700.0)-273.15, levels=[-30,-25,-20,-15,-10,-5],
                        colors='cyan', linewidths=1, linestyles='dashed', transform=ccrs.PlateCarree())
c2c = ax2.contour(lon_2d, lat_2d, ds['T'].sel(level=700.0)-273.15, levels=[0],
                        colors='blue', linewidths=2, linestyles='solid', transform=ccrs.PlateCarree())
c2d = ax2.contour(lon_2d, lat_2d, ds['T'].sel(level=700.0)-273.15, levels=[5],
                        colors='orange', linewidths=1, linestyles='solid', transform=ccrs.PlateCarree())
c2e = ax2.contour(lon_2d, lat_2d, ds['T'].sel(level=700.0)-273.15, levels=[10,15,20,25,30],
                        colors='red', linewidths=1, linestyles='solid', transform=ccrs.PlateCarree())
#ax2.clabel(c2, fontsize=10, inline=1, inline_spacing=1, fmt='%i', rightside_up=True)
ax2.set_title('700-hPa RH/T and Heights', fontsize=16)
cb2 = plt.colorbar(cf2, ax=ax2, orientation='horizontal', shrink=0.74, pad=0.05)
cb2.set_label('%', size='x-large')

# LOWER LEFT PANEL- 850 hPa height/temp/RH
ax3 = plt.subplot(gs[1,0],projection=crs)
axis_setup(ax3)
clev = 850.0
cf3 = ax3.contourf(lon_2d, lat_2d, ds['R'].sel(level=clev), norm=plt.Normalize(70,110), levels=[70,80,90],
                         extend='max', cmap=rhcmap, transform=ccrs.PlateCarree())
c3 = ax3.contour(lon_2d, lat_2d, heights_850, h85lev, colors='black', linewidths=2,
                       transform=ccrs.PlateCarree())
c3b = ax3.contour(lon_2d, lat_2d, ds['T'].sel(level=clev)-273.15, levels=[-30,-25,-20,-15,-10,-5],
                        colors='cyan', linewidths=1, linestyles='dashed', transform=ccrs.PlateCarree())
c3c = ax3.contour(lon_2d, lat_2d, ds['T'].sel(level=clev)-273.15, levels=[0],
                        colors='blue', linewidths=2, linestyles='solid', transform=ccrs.PlateCarree())
c3d = ax3.contour(lon_2d, lat_2d, ds['T'].sel(level=clev)-273.15, levels=[5],
                        colors='orange', linewidths=1, linestyles='solid', transform=ccrs.PlateCarree())
c3e = ax3.contour(lon_2d, lat_2d, ds['T'].sel(level=clev)-273.15, levels=[10,15,20,25,30],
                        colors='red', linewidths=1, linestyles='solid', transform=ccrs.PlateCarree())
#ax3.clabel(c3, fontsize=10, inline=1, inline_spacing=1, fmt='%i', rightside_up=True)
ax3.set_title('850-hPa RH/T and Heights', fontsize=16)
cb3 = plt.colorbar(cf3, ax=ax3, orientation='horizontal', shrink=0.74, pad=0.05)
cb3.set_label('%', size='x-large')

# LOWER RIGHT PANEL- 925 hPa height/temp/RH
ax4 = plt.subplot(gs[1,1],projection=crs)
axis_setup(ax4)
clev = 925.0
cf4 = ax4.contourf(lon_2d, lat_2d, ds['R'].sel(level=clev), norm=plt.Normalize(70,110), levels=[70,80,90],
                         extend='max', cmap=rhcmap, transform=ccrs.PlateCarree())
c4 = ax4.contour(lon_2d, lat_2d, heights_925, h92lev, colors='black', linewidths=2,
                       transform=ccrs.PlateCarree())
c4b = ax4.contour(lon_2d, lat_2d, ds['T'].sel(level=clev)-273.15, levels=[-30,-25,-20,-15,-10,-5],
                        colors='cyan', linewidths=1, linestyles='dashed', transform=ccrs.PlateCarree())
c4c = ax4.contour(lon_2d, lat_2d, ds['T'].sel(level=clev)-273.15, levels=[0],
                        colors='blue', linewidths=2, linestyles='solid', transform=ccrs.PlateCarree())
c4d = ax4.contour(lon_2d, lat_2d, ds['T'].sel(level=clev)-273.15, levels=[5],
                        colors='orange', linewidths=1, linestyles='solid', transform=ccrs.PlateCarree())
c4e = ax4.contour(lon_2d, lat_2d, ds['T'].sel(level=clev)-273.15, levels=[10,15,20,25,30],
                        colors='red', linewidths=1, linestyles='solid', transform=ccrs.PlateCarree())
#ax4.clabel(c4, fontsize=10, inline=1, inline_spacing=1, fmt='%i', rightside_up=True)
ax4.set_title('925-hPa RH/T and Heights', fontsize=16)
cb4 = plt.colorbar(cf4, ax=ax4, orientation='horizontal', shrink=0.74, pad=0.05)
cb4.set_label('%', size='x-large')

# Set figure title
fig.suptitle(rd.strftime('%d %B %Y %H:%MZ')+' F%02d' % (int(p.opt['fnum'])), fontsize=24)

# Save figure
plt.savefig('rh_'+rd.strftime('%Y%m%d%H')+'_'+'%02d' % (int(p.opt['fnum']))+'.png')
