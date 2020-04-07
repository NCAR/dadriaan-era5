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

import datetime

# Set the requested date
rd = datetime.datetime.strptime(p.opt['tstring'],'%Y-%m-%d %H:%M:%S')

# What date string
yyyymm = rd.strftime('%Y%m')
yyyymmdd = rd.strftime('%Y%m%d')

# What 3D product strings
prod3d = ['_u.','_v.','_z.','_t.','_q.','_r.']

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

# Trim down files further based on product
li = []
for cf in indexes:
  for p3 in prod3d:
    if p3 in files[cf].name:
      li.append(cf)

# Load using list comprehension, creating list of xarray dataset objects
singlesets = [files[i].remote_access(use_xarray=True) for i in li]

# Combine all of the datasets (all files into a single dataset)
ds = xr.combine_by_coords(singlesets)
print(ds)

# Subset the dataset. We want all levels, at a specific time, and reduce lat/lon
ds = ds.sel(time=rd,latitude=slice(60,15),longitude=slice(230,300))

# Coordinate reference system
crs = ccrs.LambertConformal(central_longitude=-100.0, central_latitude=45.0)

# Combine 1D latitude and longitudes into a 2D grid of locations
lon_2d, lat_2d = np.meshgrid(ds['longitude'], ds['latitude'])

# Calculate the grid deltas for frontogenesis calculation
dx, dy = mpcalc.lat_lon_grid_deltas(lon_2d, lat_2d)

# Isentropic levels to compute
isentlevs = [290.0, 292.0, 294.0, 296.0]*units.kelvin

# Isentropic coordinates
iso_anx = mpcalc.isentropic_interpolation(isentlevs,ds.level,ds['T'],ds['Q'],ds['R'],ds['U'],ds['V'],ds['Z'],temperature_out=True)

# Pull out the components
isop, isot, isoq, isor, isou, isov, isoz = iso_anx
isou.ito('kt')
isov.ito('kt')

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

# Contour levels for pressure
plev = np.arange(0,1000.0,50.0)
rlev = np.arange(0,120.0,10.0)

# UPPER LEFT PANEL- 290K pressure/wind/RH/mixr
il = 0
ax1 = plt.subplot(gs[0,0],projection=crs)
axis_setup(ax1)
cf1 = ax1.contourf(lon_2d, lat_2d, isor[il,:,:], cmap=plt.cm.gist_earth_r, transform=ccrs.PlateCarree(), levels=rlev)
c1a = ax1.contour(lon_2d, lat_2d, isop[il,:,:], plev, colors='black', linewidths=1,
                       transform=ccrs.PlateCarree())
c1b = ax1.contour(lon_2d,lat_2d,isoq[il,:,:],colors='lightgreen',linewidths=1,linestyles='--',transform=ccrs.PlateCarree())
ax1.set_title('290K Pressure/Wind/RH/mixr', fontsize=16)
ax1.barbs(lon_2d, lat_2d, isou[il,:,:].m,isov[il,:,:].m, length=6,
         regrid_shape=10, pivot='middle', transform=ccrs.PlateCarree())
ax1.clabel(c1a, fontsize=10, inline=1, inline_spacing=7,
          fmt='%i', rightside_up=True, use_clabeltext=True, colors='white')
cb1 = fig.colorbar(cf1, ax=ax1, orientation='horizontal', shrink=0.74, pad=0.05)
cb1.set_label('RH (%)', size='x-large')

# UPPER RIGHT PANEL- 292K pressure/wind/RH/mixr
il = 1
ax2 = plt.subplot(gs[0,1],projection=crs)
axis_setup(ax2)
cf2 = ax2.contourf(lon_2d, lat_2d, isor[il,:,:], cmap=plt.cm.gist_earth_r, transform=ccrs.PlateCarree(), levels=rlev)
c2a = ax2.contour(lon_2d, lat_2d, isop[il,:,:], plev, colors='black', linewidths=1,
                       transform=ccrs.PlateCarree())
c2b = ax2.contour(lon_2d,lat_2d,isoq[il,:,:],colors='lightgreen',linewidths=1,linestyles='--',transform=ccrs.PlateCarree())
ax2.set_title('292K Pressure/Wind/RH/mixr', fontsize=16)
ax2.barbs(lon_2d, lat_2d, isou[il,:,:].m,isov[il,:,:].m, length=6,
         regrid_shape=10, pivot='middle', transform=ccrs.PlateCarree())
ax2.clabel(c2a, fontsize=10, inline=1, inline_spacing=7,
          fmt='%i', rightside_up=True, use_clabeltext=True, colors='white')
cb2 = fig.colorbar(cf2, ax=ax2, orientation='horizontal', shrink=0.74, pad=0.05)
cb2.set_label('RH (%)', size='x-large')

# LOWER LEFT PANEL- 294K pressure/wind/RH/mixr
il = 2
ax3 = plt.subplot(gs[1,0],projection=crs)
axis_setup(ax3)
cf3 = ax3.contourf(lon_2d, lat_2d, isor[il,:,:], cmap=plt.cm.gist_earth_r, transform=ccrs.PlateCarree(), levels=rlev)
c3a = ax3.contour(lon_2d, lat_2d, isop[il,:,:], plev, colors='black', linewidths=1,
                       transform=ccrs.PlateCarree())
c3b = ax3.contour(lon_2d,lat_2d,isoq[il,:,:],colors='lightgreen',linewidths=1,linestyles='--',transform=ccrs.PlateCarree())
ax3.set_title('294K Pressure/Wind/RH/mixr', fontsize=16)
ax3.barbs(lon_2d, lat_2d, isou[il,:,:].m,isov[il,:,:].m, length=6,
         regrid_shape=10, pivot='middle', transform=ccrs.PlateCarree())
ax3.clabel(c3a, fontsize=10, inline=1, inline_spacing=7,
          fmt='%i', rightside_up=True, use_clabeltext=True, colors='white')
cb3 = fig.colorbar(cf3, ax=ax3, orientation='horizontal', shrink=0.74, pad=0.05)
cb3.set_label('RH (%)', size='x-large')

# LOWER RIGHT PANEL- 296K pressure/wind/RH/mixr
il = 3
ax4 = plt.subplot(gs[1,1],projection=crs)
axis_setup(ax4)
cf4 = ax4.contourf(lon_2d, lat_2d, isor[il,:,:], cmap=plt.cm.gist_earth_r, transform=ccrs.PlateCarree(), levels=rlev)
c4a = ax4.contour(lon_2d, lat_2d, isop[il,:,:], plev, colors='black', linewidths=1,
                       transform=ccrs.PlateCarree())
c4b = ax4.contour(lon_2d,lat_2d,isoq[il,:,:],colors='lightgreen',linewidths=1,linestyles='--',transform=ccrs.PlateCarree())
ax4.set_title('296K Pressure/Wind/RH/mixr', fontsize=16)
ax4.barbs(lon_2d, lat_2d, isou[il,:,:].m,isov[il,:,:].m, length=6,
         regrid_shape=10, pivot='middle', transform=ccrs.PlateCarree())
ax4.clabel(c4a, fontsize=10, inline=1, inline_spacing=7,
          fmt='%i', rightside_up=True, use_clabeltext=True, colors='white')
cb4 = fig.colorbar(cf4, ax=ax4, orientation='horizontal', shrink=0.74, pad=0.05)
cb4.set_label('RH (%)', size='x-large')

# Set figure title
fig.suptitle(rd.strftime('%d %B %Y %H:%MZ')+' F%02d' % (int(p.opt['fnum'])), fontsize=24)

# Save figure
plt.savefig('iso_'+rd.strftime('%Y%m%d%H')+'_'+'%02d' % (int(p.opt['fnum']))+'.png')
