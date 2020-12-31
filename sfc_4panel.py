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
import matplotlib.gridspec as gridspec

from matplotlib.path import Path
import matplotlib.patches as patches

import cmocean

import metpy.calc as mpcalc
import metpy.plots as mpplt
from metpy.units import units

import datetime, os

# Find the correct Convair position file
#acpos = "/home/dadriaan/projects/icicle/data/convair/planet/position/FINAL/ICICLE_Flight_%02d_position.csv" % (int(p.opt['fnum']))

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

if not os.path.exists(f3d) or not os.path.exists(f2d):

  print("\nUSING RDA\n")

  # What 3D product strings
  prod3d = ['_u.','_v.','_z.','_t.','_p.','_q.','_r.']
  prod2d = ['_10u.','_10v.','_msl.','_2t.']

  # Set RDA credentials
  session_manager.set_session_options(auth=tuple([p.opt['user'],p.opt['auth']]))

  # The dataset catalog
  cat = TDSCatalog('https://rda.ucar.edu/thredds/catalog/files/g/ds633.0/e5.oper.an.pl/'+yyyymm+'/catalog.xml')
  catsfc = TDSCatalog('https://rda.ucar.edu/thredds/catalog/files/g/ds633.0/e5.oper.an.sfc/'+yyyymm+'/catalog.xml')

  # Get all of the datasets in the catalog
  files = cat.datasets
  sfiles = catsfc.datasets

  # Turn this list of files into a list
  allfiles = list(files)
  allsfc = list(sfiles)

  # Loop through the files and save the ones we want to load
  casefiles = [i for i in allfiles if yyyymmdd in  i]

  # Find the indexes in the list of files we want to load
  indexes = [allfiles.index(f) for f in casefiles]
  sindexes = [allsfc.index(f) for f in allsfc]

  # Trim down files further based on product
  li = []
  for cf in indexes:
    for p3 in prod3d:
      if p3 in files[cf].name and '.nc' in files[cf].name:
        li.append(cf)
  lsi = []
  for sf in sindexes:
    for p2 in prod2d:
      if p2 in sfiles[sf].name and '.nc' in files[cf].name:
        lsi.append(sf)

  # Load using list comprehension, creating list of xarray dataset objects
  singlesets = [files[i].remote_access(use_xarray=True) for i in li]
  singlesfc = [sfiles[i].remote_access(use_xarray=True) for i in lsi]

  # Combine all of the datasets (all files into a single dataset)
  ds = xr.combine_by_coords(singlesets,combine_attrs="drop")
  sds = xr.combine_by_coords(singlesfc,combine_attrs="drop")
  print(sds)

  # Trim down the surface data to what we want
  sds = sds[['MSL','VAR_10U','VAR_10V','VAR_2T']].sel(time=rd,latitude=slice(60,15),longitude=slice(230,300))

  # Subset the dataset. We want all levels, at a specific time, and reduce lat/lon
  ds = ds.sel(time=rd,latitude=slice(60,15),longitude=slice(230,300))

  ds.to_netcdf(f3d)
  sds.to_netcdf(f2d)
else:
  print("\nLOADING LOCAL\n")
  ds = xr.open_dataset(f3d)
  sds = xr.open_dataset(f2d)

# Coordinate reference system
crs = ccrs.LambertConformal(central_longitude=-100.0, central_latitude=45.0)

# Set colors for RH shading
rh_colors = [(139/255, 255/255, 9/255),
             (34/255, 255/255, 8/255),
             (26/255, 198/255, 4/255)]
rhcmap = ListedColormap(rh_colors,N=3)

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

# Create 3D P
ds['P'] = xr.DataArray(ds.level.values,dims=['level'],coords={'level':ds.level.values},attrs={'units':'hectopascals'}).broadcast_like(ds['T'])

# Compute Td from Q
ds['TD'] = xr.DataArray(mpcalc.dewpoint_from_specific_humidity(ds['P'],ds['T'],ds['Q']),dims=['level','latitude','longitude'],coords=ds.coords,attrs={'units':'degC'})

# Compute theta-E 850
thetaE_850 = mpcalc.equivalent_potential_temperature(ds['P'].sel(level=850.0),ds['T'].sel(level=850.0),ds['TD'].sel(level=850.0))

temps_925 = ndimage.gaussian_filter(ds['T'].sel(level=925.0),sigma=1.5,order=0)*units('kelvin')

# Smooth the height data
heights_250 = ndimage.gaussian_filter(mpcalc.geopotential_to_height(ds['Z'].sel(level=250.0)), sigma=1.5, order=0)
heights_300 = ndimage.gaussian_filter(mpcalc.geopotential_to_height(ds['Z'].sel(level=300.0)), sigma=1.5, order=0)
heights_500 = ndimage.gaussian_filter(mpcalc.geopotential_to_height(ds['Z'].sel(level=500.0)), sigma=1.5, order=0)
heights_700 = ndimage.gaussian_filter(mpcalc.geopotential_to_height(ds['Z'].sel(level=700.0)), sigma=1.5, order=0)
heights_850 = ndimage.gaussian_filter(mpcalc.geopotential_to_height(ds['Z'].sel(level=850.0)), sigma=1.5, order=0)
heights_925 = ndimage.gaussian_filter(mpcalc.geopotential_to_height(ds['Z'].sel(level=925.0)), sigma=1.5, order=0)
heights_1000 = ndimage.gaussian_filter(mpcalc.geopotential_to_height(ds['Z'].sel(level=1000.0)), sigma=1.5, order=0)
thick = heights_500-heights_1000

# Smooth surface pressure data
smooth_MSL = ndimage.gaussian_filter(sds['MSL'], sigma=1.5, order=0)

# Smooth the wind data
uwnd_250 = ndimage.gaussian_filter(ds['U'].sel(level=250.0), sigma=3.0) * units('m/s')
vwnd_250 = ndimage.gaussian_filter(ds['V'].sel(level=250.0), sigma=3.0) * units('m/s')
uwnd_300 = ndimage.gaussian_filter(ds['U'].sel(level=300.0), sigma=3.0) * units('m/s')
vwnd_300 = ndimage.gaussian_filter(ds['V'].sel(level=300.0), sigma=3.0) * units('m/s')
uwnd_700 = ndimage.gaussian_filter(ds['U'].sel(level=700.0), sigma=3.0) * units('m/s')
vwnd_700 = ndimage.gaussian_filter(ds['V'].sel(level=700.0), sigma=3.0) * units('m/s')
uwnd_850 = ndimage.gaussian_filter(ds['U'].sel(level=850.0), sigma=3.0) * units('m/s')
vwnd_850 = ndimage.gaussian_filter(ds['V'].sel(level=850.0), sigma=3.0) * units('m/s')
uwnd_925 = ndimage.gaussian_filter(ds['U'].sel(level=925.0), sigma=3.0) * units('m/s')
vwnd_925 = ndimage.gaussian_filter(ds['V'].sel(level=925.0), sigma=3.0) * units('m/s')
uwnd_10 = ndimage.gaussian_filter(sds['VAR_10U'],sigma=3.0)*units('m/s')
vwnd_10 = ndimage.gaussian_filter(sds['VAR_10V'],sigma=3.0)*units('m/s')

# Compute the 300mb wind speeds
winds_250 = mpcalc.wind_speed(uwnd_250,vwnd_250).to(units.knots)
winds_300 = mpcalc.wind_speed(uwnd_300,vwnd_300).to(units.knots)
winds_700 = mpcalc.wind_speed(uwnd_700,vwnd_700).to(units.knots)
winds_850 = mpcalc.wind_speed(uwnd_850,vwnd_850).to(units.knots)
winds_925 = mpcalc.wind_speed(uwnd_925,vwnd_925).to(units.knots)
winds_10m = mpcalc.wind_speed(uwnd_10,vwnd_10).to(units.knots)

# Contour levels for heights
h25lev = np.arange(9000.0,14000.0,100.0)
h3lev = np.arange(8500.0,10000.0,30.0)
h5lev = np.arange(4800.0,6800.0,40.0)
h6lev = np.arange(3500.0,4500.0,30.0)
h7lev = np.arange(2500.0,3500.0,30.0)
h85lev = np.arange(1000.0,2000.0,30.0)
h92lev = np.arange(0.0,2000.0,30.0)

# Stuff for Convair path)
verts = [(-100.0,45.0),(-90.0,45.0)]
codes = [Path.MOVETO,Path.LINETO]
path = Path(verts,codes)
patch = patches.PathPatch(path,lw=4,color="red",transform=ccrs.PlateCarree())

# UPPER LEFT PANEL- MSLP/2mT
ax4 = plt.subplot(gs[0,0],projection=crs)
axis_setup(ax4)
cf4 = ax4.contourf(lon_2d, lat_2d, sds['VAR_2T']-273.15,cmap='coolwarm', levels=np.arange(-40,40,2),transform=ccrs.PlateCarree())
c4a = ax4.contour(lon_2d, lat_2d, smooth_MSL/100.0, colors='black', linewidths=1, levels=np.arange(888,1056,2),
                       transform=ccrs.PlateCarree())
#c4b = ax4.contour(lon_2d,lat_2d,winds_10m.m,colors='black',linewidths=0.5,linestyles='dashed',levels=np.arange(10,50,5),transform=ccrs.PlateCarree())
ax4.barbs(lon_2d, lat_2d, uwnd_10.to(units.knots).m,vwnd_10.to(units.knots).m, length=6,
         regrid_shape=10, pivot='middle', transform=ccrs.PlateCarree())
ax4.clabel(c4a, fontsize=10, inline=1, inline_spacing=7, fmt='%i', rightside_up=True, use_clabeltext=True, colors='white')
ax4.set_title('MSLP/2mT', fontsize=16)
cb4 = fig.colorbar(cf4, ax=ax4, orientation='horizontal', shrink=0.74, pad=0.01)
cb4.set_label('degC', size='x-large')

# UPPER RIGHT PANEL- 250 hPa winds/mslp/thickness
ax2 = plt.subplot(gs[0,1],projection=crs)
axis_setup(ax2)
cf2 = ax2.contourf(lon_2d, lat_2d, winds_250.m,cmap='cool',transform=ccrs.PlateCarree(),levels=np.arange(80,240,20))
c2a = ax2.contour(lon_2d, lat_2d, smooth_MSL/100.0, colors='black', linewidths=1, levels=np.arange(888,1056,2),
                       transform=ccrs.PlateCarree())
c2b = ax2.contour(lon_2d, lat_2d, thick, colors='gray', linewidths=1, linestyles='dashed', transform=ccrs.PlateCarree(), levels=np.arange(4000,6000,50))
c2c = ax2.contour(lon_2d, lat_2d, thick, colors='blue', linewidths=2, linestyles='dashed', transform=ccrs.PlateCarree(), levels=[5400.])
#ax2.barbs(lon_2d, lat_2d, uwnd_250.to(units.knots).m,vwnd_250.to(units.knots).m, length=6,
#         regrid_shape=10, pivot='middle', transform=ccrs.PlateCarree())
#ax2.clabel(c2a, fontsize=10, inline=1, inline_spacing=1, fmt='%i', rightside_up=True)
ax2.set_title('250 hPa winds, 1000-500hPa thickness, MSLP', fontsize=16)
cb2 = fig.colorbar(cf2, ax=ax2, orientation='horizontal', shrink=0.74, pad=0.01)
cb2.set_label('kts', size='x-large')

# LOWER LEFT PANEL- 850 hPa theta-E/heights
# LOWER LEFT PANEL- 10m winds/MSLP
ax1 = plt.subplot(gs[1,0],projection=crs)
axis_setup(ax1)
#cf1 = ax1.contourf(lon_2d, lat_2d, winds_10m.m, cmap='cool', levels=np.arange(10,50,2), transform=ccrs.PlateCarree())
#c1a = ax1.contour(lon_2d, lat_2d, smooth_MSL/100.0, colors='black', linewidths=1, transform=ccrs.PlateCarree(), levels=np.arange(888,1056,2))
#ax1.set_title('10m winds/MSLP', fontsize=16)
#cb1 = fig.colorbar(cf1, ax=ax1, orientation='horizontal', shrink=0.74, pad=0.01)
#cb1.set_label('kts', size='x-large')

#cf1 = ax1.contourf(lon_2d, lat_2d, thetaE_850, cmap='jet', levels=np.arange(250,350,5), transform=ccrs.PlateCarree())
#c1a = ax1.contour(lon_2d, lat_2d, heights_850, levels=h85lev, linewidths=1, transform=ccrs.PlateCarree(),colors='black')
#ax1.barbs(lon_2d, lat_2d, uwnd_850.to(units.knots).m,vwnd_850.to(units.knots).m, length=6,
#         regrid_shape=10, pivot='middle', transform=ccrs.PlateCarree())
#ax1.set_title('850 hPa theta-E, heights', fontsize=16)
#cb1 = fig.colorbar(cf1, ax=ax1, orientation='horizontal', shrink=0.74, pad=0.01)
#cb1.set_label('K', size='x-large')

cf1 = ax1.contourf(lon_2d, lat_2d, winds_925.m,cmap='cool',transform=ccrs.PlateCarree(),levels=np.arange(10,50,5))
c1a = ax1.contour(lon_2d, lat_2d, heights_925, h92lev, colors='black', linewidths=2, transform=ccrs.PlateCarree())
c1b = ax1.contour(lon_2d,lat_2d,temps_925.to('degC'),levels=np.arange(1,20,5), colors='red',transform=ccrs.PlateCarree(),linewidths=1,linestyles='dashed')
c1c = ax1.contour(lon_2d,lat_2d,temps_925.to('degC'),levels=np.arange(-20,0,5), colors='blue',transform=ccrs.PlateCarree(),linewidths=1,linestyles='dashed')
ax1.barbs(lon_2d, lat_2d, uwnd_925.to(units.knots).m,vwnd_925.to(units.knots).m, length=6,
         regrid_shape=10, pivot='middle', transform=ccrs.PlateCarree())
ax1.set_title('925-hPa Winds/Heights/Temps', fontsize=16)
cb1 = fig.colorbar(cf1, ax=ax1, orientation='horizontal', shrink=0.74, pad=0.01)
cb1.set_label('kts', size='x-large')

# LOWER RIGHT PANEL- 10m streamlines
ax3 = plt.subplot(gs[1,1],projection=crs)
axis_setup(ax3)
sp3 = ax3.streamplot(lon_2d, lat_2d, uwnd_10, vwnd_10, density=5, transform=ccrs.PlateCarree(), color=np.asarray(winds_10m))
ax3.set_title('10m Streamlines', fontsize=16)
cb3 = fig.colorbar(sp3.lines, ax=ax3, orientation='horizontal', shrink=0.74, pad=0.01)
cb3.set_label('kts', size='x-large')

# Set figure title
fig.suptitle(rd.strftime('%d %B %Y %H:%MZ')+' F%02d' % (fn), fontsize=24)

# Save figure
plt.savefig('sfc_'+rd.strftime('%Y%m%d%H')+'_'+'%02d' % (fn)+'.png')
