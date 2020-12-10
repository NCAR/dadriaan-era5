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
hhmmss = rd.strftime('%H%M%S')
fn = int(p.opt['fnum'])

# File strings
f3d = 'tadv_%s_%s_F%02d_3D.nc' % (yyyymmdd,hhmmss,fn)
f2d = 'tadv_%s_%s_F%02d_2D.nc' % (yyyymmdd,hhmmss,fn)

if not os.path.exists(f3d) and not os.path.exists(f2d):

  print("\nUSING RDA\n")

  # What 3D product strings
  prod3d = ['_u.','_v.','_z.','_t.']

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

# Calculate the grid deltas for advection calculation
dx, dy = mpcalc.lat_lon_grid_deltas(lon_2d, lat_2d)

# Smooth the height data
#heights_500 = ndimage.gaussian_filter(mpcalc.geopotential_to_height(ds['Z'].sel(level=500.0)), sigma=1.5, order=0)
heights_600 = ndimage.gaussian_filter(mpcalc.geopotential_to_height(ds['Z'].sel(level=600.0)), sigma=1.5, order=0)
heights_700 = ndimage.gaussian_filter(mpcalc.geopotential_to_height(ds['Z'].sel(level=700.0)), sigma=1.5, order=0)
heights_850 = ndimage.gaussian_filter(mpcalc.geopotential_to_height(ds['Z'].sel(level=850.0)), sigma=1.5, order=0)
heights_925 = ndimage.gaussian_filter(mpcalc.geopotential_to_height(ds['Z'].sel(level=925.0)), sigma=1.5, order=0)

# Contour levels for heights
h5lev = np.arange(4800.0,5800.0,30.0)
h6lev = np.arange(3500.0,4500.0,30.0)
h7lev = np.arange(2500.0,3500.0,30.0)
h85lev = np.arange(1000.0,2000.0,30.0)
h92lev = np.arange(0.0,1000.0,30.0)

# Smooth the wind data
#uwnd_500 = mpcalc.smooth_n_point(ds['U'].metpy.sel(level=500.0).squeeze(), 9)
#uwnd_600 = mpcalc.smooth_n_point(ds['U'].metpy.sel(level=600.0).squeeze(), 9)
#uwnd_700 = mpcalc.smooth_n_point(ds['U'].metpy.sel(level=700.0).squeeze(), 9)
#uwnd_850 = mpcalc.smooth_n_point(ds['U'].metpy.sel(level=850.0).squeeze(), 9)
#uwnd_925 = mpcalc.smooth_n_point(ds['U'].metpy.sel(level=925.0).squeeze(), 9)
#vwnd_500 = mpcalc.smooth_n_point(ds['V'].metpy.sel(level=500.0).squeeze(), 9)
#vwnd_600 = mpcalc.smooth_n_point(ds['V'].metpy.sel(level=600.0).squeeze(), 9)
#vwnd_700 = mpcalc.smooth_n_point(ds['V'].metpy.sel(level=700.0).squeeze(), 9)
#vwnd_850 = mpcalc.smooth_n_point(ds['V'].metpy.sel(level=850.0).squeeze(), 9)
#vwnd_925 = mpcalc.smooth_n_point(ds['V'].metpy.sel(level=925.0).squeeze(), 9)

#uwind_500 = ds['U'].metpy.sel(level=500.0).metpy.unit_array
#vwind_500 = ds['V'].metpy.sel(level=500.0).metpy.unit_array
uwind_600 = ds['U'].metpy.sel(level=600.0).metpy.unit_array
vwind_600 = ds['V'].metpy.sel(level=600.0).metpy.unit_array
uwind_700 = ds['U'].metpy.sel(level=700.0).metpy.unit_array
vwind_700 = ds['V'].metpy.sel(level=700.0).metpy.unit_array
uwind_850 = ds['U'].metpy.sel(level=850.0).metpy.unit_array
vwind_850 = ds['V'].metpy.sel(level=850.0).metpy.unit_array
uwind_925 = ds['U'].metpy.sel(level=925.0).metpy.unit_array
vwind_925 = ds['V'].metpy.sel(level=925.0).metpy.unit_array

# Smooth the temperature data
#tmpk_500 = mpcalc.smooth_n_point(ds['T'].metpy.sel(level=500.0).squeeze(), 9).to('degC')
#tmpk_600 = mpcalc.smooth_n_point(ds['T'].metpy.sel(level=600.0).metpy.unit_array.squeeze(), 9).to('degC')
#tmpk_700 = mpcalc.smooth_n_point(ds['T'].metpy.sel(level=700.0).metpy.unit_array.squeeze(), 9).to('degC')
#tmpk_850 = mpcalc.smooth_n_point(ds['T'].metpy.sel(level=850.0).metpy.unit_array.squeeze(), 9).to('degC')
#tmpk_925 = mpcalc.smooth_n_point(ds['T'].metpy.sel(level=925.0).metpy.unit_array.squeeze(), 9).to('degC')

#tmpk_500 = ds['T'].metpy.sel(level=600.0).metpy.unit_array
tmpk_600 = ds['T'].metpy.sel(level=600.0).metpy.unit_array
tmpk_700 = ds['T'].metpy.sel(level=600.0).metpy.unit_array
tmpk_850 = ds['T'].metpy.sel(level=600.0).metpy.unit_array
tmpk_925 = ds['T'].metpy.sel(level=600.0).metpy.unit_array

# Compute advection
#tadv_500 = mpcalc.advection(tmpk_500, (uwind_500, vwind_500),(dx, dy)).to_base_units()
tadv_600 = mpcalc.advection(tmpk_600, [uwind_600,vwind_600],(dx, dy))
tadv_700 = mpcalc.advection(tmpk_700, [uwind_700, vwind_700],(dx, dy))
tadv_850 = mpcalc.advection(tmpk_850, [uwind_850, vwind_850],(dx, dy))
tadv_925 = mpcalc.advection(tmpk_925, [uwind_925, vwind_925],(dx, dy))

# Smooth advection
#tadv_500 = ndimage.gaussian_filter(tadv_500,sigma=3,order=0)*units('K/sec')
tadv_600 = ndimage.gaussian_filter(tadv_600,sigma=3,order=0)*units('K/sec')
tadv_700 = ndimage.gaussian_filter(tadv_700,sigma=3,order=0)*units('K/sec')
tadv_850 = ndimage.gaussian_filter(tadv_850,sigma=3,order=0)*units('K/sec')
tadv_925 = ndimage.gaussian_filter(tadv_925,sigma=3,order=0)*units('K/sec')

# UPPER LEFT PANEL- 600 hPa height/tadv
cint = np.arange(-8,9)
ax1 = plt.subplot(gs[0,0],projection=crs)
axis_setup(ax1)
cf1 = ax1.contourf(lon_2d, lat_2d, 3.0*tadv_600.to(units('delta_degC/hour')), cint[cint!=0], extend='both', cmap='bwr'
                        , transform=ccrs.PlateCarree())
c1 = ax1.contour(lon_2d, lat_2d, heights_600, h6lev, colors='black', linewidths=2,
                       transform=ccrs.PlateCarree())
ax1.barbs(lon_2d, lat_2d, uwind_600.to(units.knots).m,vwind_600.to(units.knots).m, length=6,
         regrid_shape=10, pivot='middle', transform=ccrs.PlateCarree())
#ax1.clabel(c1, fontsize=10, inline=1, inline_spacing=1, fmt='%i', rightside_up=True)
ax1.set_title('600-hPa Temp Advection and Heights', fontsize=16)
cb1 = fig.colorbar(cf1, ax=ax1, orientation='horizontal', shrink=0.74, pad=.05)
cb1.set_label('degC/3h', size='x-large')

# UPPER RIGHT PANEL- 700 hPa height/tadv
ax2 = plt.subplot(gs[0,1],projection=crs)
axis_setup(ax2)
cf2 = ax2.contourf(lon_2d, lat_2d, 3.0*tadv_700.to(units('delta_degC/hour')), cint[cint!=0], extend='both', cmap='bwr'
                        , transform=ccrs.PlateCarree())
c2 = ax2.contour(lon_2d, lat_2d, heights_700, h7lev, colors='black', linewidths=2,
                       transform=ccrs.PlateCarree())
ax2.barbs(lon_2d, lat_2d, uwind_700.to(units.knots).m,vwind_700.to(units.knots).m, length=6,
         regrid_shape=10, pivot='middle', transform=ccrs.PlateCarree())
#ax1.clabel(c2, fontsize=10, inline=1, inline_spacing=1, fmt='%i', rightside_up=True)
ax2.set_title('700-hPa Temp Advection and Heights', fontsize=16)
cb2 = fig.colorbar(cf2, ax=ax2, orientation='horizontal', shrink=0.74, pad=.05)
cb2.set_label('degC/3h', size='x-large')

# LOWER LEFT PANEL- 850 hPa height/tadv
ax3 = plt.subplot(gs[1,0],projection=crs)
axis_setup(ax3)
cf3 = ax3.contourf(lon_2d, lat_2d, 3.0*tadv_850.to(units('delta_degC/hour')), cint[cint!=0], extend='both', cmap='bwr'
                        , transform=ccrs.PlateCarree())
c3 = ax3.contour(lon_2d, lat_2d, heights_850, h85lev, colors='black', linewidths=2,
                       transform=ccrs.PlateCarree())
ax3.barbs(lon_2d, lat_2d, uwind_850.to(units.knots).m,vwind_850.to(units.knots).m, length=6,
         regrid_shape=10, pivot='middle', transform=ccrs.PlateCarree())
#ax3.clabel(c3, fontsize=10, inline=1, inline_spacing=1, fmt='%i', rightside_up=True)
ax3.set_title('850-hPa Temp Advection and Heights', fontsize=16)
cb3 = fig.colorbar(cf3, ax=ax3, orientation='horizontal', shrink=0.74, pad=.05)
cb3.set_label('degC/3h', size='x-large')

# LOWER RIGHT PANEL- 925 hPa height/frontogenesis
ax4 = plt.subplot(gs[1,1],projection=crs)
axis_setup(ax4)
cf4 = ax4.contourf(lon_2d, lat_2d, 3.0*tadv_925.to(units('delta_degC/hour')), cint[cint!=0], extend='both', cmap='bwr'
                        , transform=ccrs.PlateCarree())
c4 = ax4.contour(lon_2d, lat_2d, heights_925, h92lev, colors='black', linewidths=2,
                       transform=ccrs.PlateCarree())
ax4.barbs(lon_2d, lat_2d, uwind_925.to(units.knots).m,vwind_925.to(units.knots).m, length=6,
         regrid_shape=10, pivot='middle', transform=ccrs.PlateCarree())
#ax4.clabel(c4, fontsize=10, inline=1, inline_spacing=1, fmt='%i', rightside_up=True)
ax4.set_title('925-hPa Temp Advection and Heights', fontsize=16)
cb4 = fig.colorbar(cf4, ax=ax4, orientation='horizontal', shrink=0.74, pad=.05)
cb4.set_label('degC/3h', size='x-large')

# Set figure title
fig.suptitle(rd.strftime('%d %B %Y %H:%MZ')+' F%02d' % (fn), fontsize=24)

# Save figure
plt.savefig('tadv_'+rd.strftime('%Y%m%d%H')+'_'+'%02d' % (fn)+'.png')
