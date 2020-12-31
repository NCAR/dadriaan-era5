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
import matplotlib.gridspec as gridspec

import metpy.calc as mpcalc
import metpy.plots as mpplt
from metpy.units import units

import datetime, os

# Set the requested date
rd = datetime.datetime.strptime(p.opt['tstring'],'%Y-%m-%d %H:%M:%S')

# What date string
yyyymm = rd.strftime('%Y%m')
yyyymmdd = rd.strftime('%Y%m%d')
hhmmss = rd.strftime('%H%M%S')
fn = int(p.opt['fnum'])

# File strings
f3d = '%s/flight%02d/%s_%s_F%02d_3D.nc' % (p.opt['input_dir'],fn,yyyymmdd,hhmmss,fn)
f2d = '%s/flight%02d/%s_%s_F%02d_2D.nc' % (p.opt['input_dir'],fn,yyyymmdd,hhmmss,fn)

if not os.path.exists(f3d):

  print("\nUSING RDA\n")

  # What 3D product strings
  prod3d = ['_u.','_v.','_z.','_t.','_p.','_q.','_r.']

  # Set RDA credentials
  session_manager.set_session_options(auth=tuple([p.opt['user'],p.opt['auth']]))

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

  # Trim down files further based on product
  li = []
  for cf in indexes:
    for p3 in prod3d:
      if p3 in files[cf].name and '.nc' in files[cf].name:
        li.append(cf)

  # Load using list comprehension, creating list of xarray dataset objects
  singlesets = [files[i].remote_access(use_xarray=True) for i in li]

  # Combine all of the datasets (all files into a single dataset)
  ds = xr.combine_by_coords(singlesets,combine_attrs="drop")
  print(ds)

  # Subset the dataset. We want all levels, at a specific time, and reduce lat/lon
  ds = ds.sel(time=rd,latitude=slice(60,15),longitude=slice(230,300))
  
  ds.to_netcdf(f3d)
else:
  print("\nLOADING LOCAL\n")
  ds = xr.open_dataset(f3d)

# Coordinate reference system
crs = ccrs.LambertConformal(central_longitude=-100.0, central_latitude=45.0)

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

# Combine 1D latitude and longitudes into a 2D grid of locations
lon_2d, lat_2d = np.meshgrid(ds['longitude'], ds['latitude'])

# Calculate the grid deltas for frontogenesis calculation
dx, dy = mpcalc.lat_lon_grid_deltas(lon_2d, lat_2d)

# Smooth the height data
#heights_500 = ndimage.gaussian_filter(mpcalc.geopotential_to_height(ds['Z'].sel(level=500.0)), sigma=1.5, order=0)
heights_600 = ndimage.gaussian_filter(mpcalc.geopotential_to_height(ds['Z'].sel(level=600.0)), sigma=1.5, order=0)
heights_700 = ndimage.gaussian_filter(mpcalc.geopotential_to_height(ds['Z'].sel(level=700.0)), sigma=1.5, order=0)
heights_850 = ndimage.gaussian_filter(mpcalc.geopotential_to_height(ds['Z'].sel(level=850.0)), sigma=1.5, order=0)
heights_925 = ndimage.gaussian_filter(mpcalc.geopotential_to_height(ds['Z'].sel(level=925.0)), sigma=1.5, order=0)

# Contour levels for heights
h5lev = np.arange(4800.0,6800.0,40.0)
h6lev = np.arange(3500.0,4500.0,30.0)
h7lev = np.arange(2500.0,3500.0,30.0)
h85lev = np.arange(1000.0,2000.0,30.0)
h92lev = np.arange(0.0,2000.0,30.0)

# Compute 700 hPa frontogenesis
# First compute potential temperature, then compute frontogenesis
theta_600 = mpcalc.potential_temperature(600.0*units.hPa,ds['T'].sel(level=600.0))
theta_700 = mpcalc.potential_temperature(700.0*units.hPa,ds['T'].sel(level=700.0))
theta_850 = mpcalc.potential_temperature(850.0*units.hPa,ds['T'].sel(level=850.0))
theta_925 = mpcalc.potential_temperature(925.0*units.hPa,ds['T'].sel(level=925.0))
front_600 = mpcalc.frontogenesis(theta_600,ds['U'].sel(level=600.0),ds['V'].sel(level=600.0),dx,dy)
front_700 = mpcalc.frontogenesis(theta_700,ds['U'].sel(level=700.0),ds['V'].sel(level=700.0),dx,dy)
front_850 = mpcalc.frontogenesis(theta_850,ds['U'].sel(level=850.0),ds['V'].sel(level=850.0),dx,dy)
front_925 = mpcalc.frontogenesis(theta_925,ds['U'].sel(level=925.0),ds['V'].sel(level=925.0),dx,dy)

# A conversion factor to get frontogensis units of K per 100 km per 3 h
convert_to_per_100km_3h = 1000*100*3600*3

# UPPER LEFT PANEL- 600 hPa height/frontogenesis
ax1 = plt.subplot(gs[0,0],projection=crs)
axis_setup(ax1)
cf1 = ax1.contourf(lon_2d, lat_2d, front_600*convert_to_per_100km_3h, np.arange(-8, 8.5, 0.5), extend='both', cmap='bwr'
                        , transform=ccrs.PlateCarree())
c1 = ax1.contour(lon_2d, lat_2d, heights_600, h6lev, colors='black', linewidths=2,
                       transform=ccrs.PlateCarree())
#ax1.clabel(c1, fontsize=10, inline=1, inline_spacing=1, fmt='%i', rightside_up=True)
ax1.set_title('600-hPa Pettersen Fgen and Heights', fontsize=16)
cb1 = fig.colorbar(cf1, ax=ax1, orientation='horizontal', shrink=0.74, pad=.05)
cb1.set_label('degK/100km/3h', size='x-large')

# UPPER RIGHT PANEL- 700 hPa height/frontogenesis
ax2 = plt.subplot(gs[0,1],projection=crs)
axis_setup(ax2)
cf2 = ax2.contourf(lon_2d, lat_2d, front_700*convert_to_per_100km_3h, np.arange(-8, 8.5, 0.5), extend='both', cmap='bwr'
                        , transform=ccrs.PlateCarree())
c2 = ax2.contour(lon_2d, lat_2d, heights_700, h7lev, colors='black', linewidths=2,
                       transform=ccrs.PlateCarree())
#ax2.clabel(c1, fontsize=10, inline=1, inline_spacing=1, fmt='%i', rightside_up=True)
ax2.set_title('700-hPa Pettersen Fgen and Heights', fontsize=16)
cb2 = fig.colorbar(cf2, ax=ax2, orientation='horizontal', shrink=0.74, pad=.05)
cb2.set_label('degK/100km/3h', size='x-large')

# LOWER LEFT PANEL- 850 hPa height/frontogenesis
ax3 = plt.subplot(gs[1,0],projection=crs)
axis_setup(ax3)
cf3 = ax3.contourf(lon_2d, lat_2d, front_850*convert_to_per_100km_3h, np.arange(-8, 8.5, 0.5), extend='both', cmap='bwr'
                        , transform=ccrs.PlateCarree())
c3 = ax3.contour(lon_2d, lat_2d, heights_850, h85lev, colors='black', linewidths=2,
                       transform=ccrs.PlateCarree())
#ax3.clabel(c1, fontsize=10, inline=1, inline_spacing=1, fmt='%i', rightside_up=True)
ax3.set_title('850-hPa Pettersen Fgen and Heights', fontsize=16)
cb3 = fig.colorbar(cf3, ax=ax3, orientation='horizontal', shrink=0.74, pad=.05)
cb3.set_label('degK/100km/3h', size='x-large')

# LOWER RIGHT PANEL- 925 hPa height/frontogenesis
ax4 = plt.subplot(gs[1,1],projection=crs)
axis_setup(ax4)
cf4 = ax4.contourf(lon_2d, lat_2d, front_925*convert_to_per_100km_3h, np.arange(-8, 8.5, 0.5), extend='both', cmap='bwr'
                        , transform=ccrs.PlateCarree())
c4 = ax4.contour(lon_2d, lat_2d, heights_925, h92lev, colors='black', linewidths=2,
                       transform=ccrs.PlateCarree())
#ax4.clabel(c1, fontsize=10, inline=1, inline_spacing=1, fmt='%i', rightside_up=True)
ax4.set_title('925-hPa Pettersen Fgen and Heights', fontsize=16)
cb4 = fig.colorbar(cf4, ax=ax4, orientation='horizontal', shrink=0.74, pad=.05)
cb4.set_label('degK/100km/3h', size='x-large')

# Set figure title
fig.suptitle(rd.strftime('%d %B %Y %H:%MZ')+' F%02d' % (fn), fontsize=24)

# Save figure
plt.savefig('fgen_'+rd.strftime('%Y%m%d%H')+'_'+'%02d' % (fn)+'.png')
