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
f3d = 'uppr_%s_%s_F%02d_3D.nc' % (yyyymmdd,hhmmss,fn)
f2d = 'uppr_%s_%s_F%02d_2D.nc' % (yyyymmdd,hhmmss,fn)

if not os.path.exists(f3d) and not os.path.exists(f2d):

  print("\nUSING RDA\n")

  # What 3D product strings
  prod3d = ['_u.','_v.','_z.','_t.','_r.']
  prod2d = ['_10u.','_10v.','_msl.','_2t.']

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

  # Subset the dataset. We want all levels, at a specific time, and reduce lat/lon
  ds = ds.sel(time=rd,latitude=slice(60,15),longitude=slice(230,300))

  ds.to_netcdf(f3d)
else:
  print("\nLOADING LOCAL\n")
  ds = xr.open_dataset(f3d)

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

# Compute 500mb heights
Z500 = mpcalc.geopotential_to_height(ds['Z'])

# Get geopotential height where RH >80% and T>=-23C (RHwater)
#zmask = Z500.where(((ds['R']>80.0) & ((ds['T']>=250.15) & (ds['T']<=273.15))))
#zmax = zmask.max(dim='level',skipna=True)
#zmin = zmask.min(dim='level',skipna=True)
#print(zmax-zmin)
#colmaxRH = ds['R'].where((ds['T']>=250.15) & (ds['T']<=273.15)).max(dim='level',skipna=True)

# Smooth the height data
heights_500 = ndimage.gaussian_filter(mpcalc.geopotential_to_height(ds['Z'].sel(level=500.0)), sigma=1.5, order=0)
heights_250 = ndimage.gaussian_filter(mpcalc.geopotential_to_height(ds['Z'].sel(level=250.0)), sigma=1.5, order=0)

temps_500 = ndimage.gaussian_filter(ds['T'].sel(level=500.0),sigma=1.5,order=0)*units('kelvin')

# Compute the grid delta
dx, dy = mpcalc.lat_lon_grid_deltas(ds['longitude'],ds['latitude'])

# Compute coriolis force
f = mpcalc.coriolis_parameter(np.deg2rad(lat_2d)).to(units('1/sec'))

# Get the wind data
uwnd_500 = ds['U'].sel(level=500.0)
vwnd_500 = ds['V'].sel(level=500.0)

# Compute vorticity
avor_500 = mpcalc.absolute_vorticity(uwnd_500,vwnd_500,dx=dx,dy=dy,latitude=ds['latitude'])
# Smooth vorticity
avor_500 = ndimage.gaussian_filter(avor_500, sigma=3, order=0) * units('1/s')
# Compute the vorticity advection
vort_adv_500 = mpcalc.advection(avor_500, uwnd_500, vwnd_500,dx=dx,dy=dy) * 1e9

# Smooth the wind data
uwnd_500 = ndimage.gaussian_filter(ds['U'].sel(level=500.0), sigma=3.0) * units('m/s')
vwnd_500 = ndimage.gaussian_filter(ds['V'].sel(level=500.0), sigma=3.0) * units('m/s')
uwnd_250 = ndimage.gaussian_filter(ds['U'].sel(level=250.0), sigma=3.0) * units('m/s')
vwnd_250 = ndimage.gaussian_filter(ds['V'].sel(level=250.0), sigma=3.0) * units('m/s')

# Compute the 250mb wind speeds
winds_500 = mpcalc.wind_speed(uwnd_500,vwnd_500).to(units.knots)
winds_250 = mpcalc.wind_speed(uwnd_250,vwnd_250).to(units.knots)

# Compute divergence at 250 hPa
div_250 = mpcalc.divergence(uwnd_250,vwnd_250,dx=dx,dy=dy)

# Contour levels for heights
h25lev = np.arange(9000.0,14000.0,100.0)
h5lev = np.arange(4800.0,6800.0,40.0)

# Absolute Vorticity colors
# Use two different colormaps from matplotlib and combine into one color set
clevs_500_avor = list(range(-8, 1, 1))+list(range(8, 46, 1))
colors1 = plt.cm.YlOrRd(np.linspace(0, 1, 48))
colors2 = plt.cm.BuPu(np.linspace(0.5, 0.75, 8))
colors = np.vstack((colors2, (1, 1, 1, 1), colors1))

# Stuff for Convair path
verts = [(-100.0,45.0),(-90.0,45.0)]
codes = [Path.MOVETO,Path.LINETO]
path = Path(verts,codes)
patch = patches.PathPatch(path,lw=4,color="red",transform=ccrs.PlateCarree())

# UPPER RIGHT PANEL- 500 hPa heights/vort_adv
ax1 = plt.subplot(gs[0,1],projection=crs)
axis_setup(ax1)

# Plot Height Contours
c1a = ax1.contour(lon_2d, lat_2d, heights_500, h5lev, colors='black', linewidths=2.0,
                linestyles='solid', transform=ccrs.PlateCarree())
#plt.clabel(c1a, fontsize=10, inline=1, inline_spacing=10, fmt='%i',
#           rightside_up=True, use_clabeltext=True)

# Plot Absolute Vorticity Contours
clevvort500 = np.arange(-9, 50, 5)
#c1b = ax1.contour(lon_2d, lat_2d, avor_500*10**5, clevvort500, colors='grey',
#                 linewidths=1.25, linestyles='dashed', transform=ccrs.PlateCarree())
#plt.clabel(c1b, fontsize=10, inline=1, inline_spacing=10, fmt='%i',
#           rightside_up=True, use_clabeltext=True)

# Plot Colorfill of Vorticity Advection
clev_avoradv = np.arange(-30, 31, 5)
cf1 = ax1.contourf(lon_2d, lat_2d, vort_adv_500.m, clev_avoradv[clev_avoradv != 0], extend='both',
                 cmap='bwr', transform=ccrs.PlateCarree())

cb = plt.colorbar(cf1, ax=ax1, orientation='horizontal', extendrect='True', ticks=clev_avoradv, shrink=0.74, pad=0.01)
#cb = fig.colorbar(cf,ax=ax1,orientation='horizontal',extendrect='True',ticks=clev_avoradv,shrink=0.74,pad=0.01)
cb.set_label(r'$1/s^2$', size='large')

# Testing RH stuff
#cf1 = ax1.contourf(lon_2d,lat_2d,zmax-zmin,levels=np.arange(0,2000,100),cmap='cool',transform=ccrs.PlateCarree())
#cb = fig.colorbar(cf1,ax=ax1,orientation='horizontal',extendrect='True',shrink=0.74,pad=0.01)
#cf1 = ax1.contourf(lon_2d,lat_2d,colmaxRH,levels=np.arange(50.0,105.0,5.0),cmap='cool',transform=ccrs.PlateCarree())
#cb = fig.colorbar(cf1,ax=ax1,orientation='horizontal',extendrect='True',shrink=0.74,pad=0.01)

# Plot Wind Barbs
ax1.barbs(lon_2d, lat_2d, uwnd_500.to(units.knots).m, vwnd_500.to(units.knots).m, length=6, regrid_shape=10,
         pivot='middle', transform=ccrs.PlateCarree())
ax1.set_title('500 hPa absolute vorticity advection/heights',fontsize=16)

# UPPER LEFT PANEL- 500 hPa heights/abs vort/advection
ax2 = plt.subplot(gs[0,0],projection=crs)
axis_setup(ax2)

# Plot absolute vorticity values (multiplying by 10^5 to scale appropriately)
cf2 = ax2.contourf(lon_2d, lat_2d, avor_500*1e5, clevs_500_avor, colors=colors, extend='max',
                 transform=ccrs.PlateCarree())
cb = plt.colorbar(cf2, ax=ax2, orientation='horizontal', shrink=0.74, pad=0.01, extendrect=True)
cb.set_label('Abs. Vorticity ($s^{-1}$)',size='large')

# Plot 500-hPa Geopotential Heights in meters
c2a = ax2.contour(lon_2d, lat_2d, heights_500, h5lev, colors='black', transform=ccrs.PlateCarree())
#plt.clabel(c2a, fmt='%d')

# Plot vort advection if we want
#c2b = ax2.contour(lon_2d, lat_2d, vort_adv_500.m, clev_avoradv[clev_avoradv > 0], linewidths=1, linestyles='dashed', colors='black', transform=ccrs.PlateCarree())
#c2b = ax2.contour(lon_2d, lat_2d, vort_adv_500.m, clev_avoradv[clev_avoradv < 0], linewidths=1, linestyles='dashed', colors='gray', transform=ccrs.PlateCarree())

# Set up a 2D slice to reduce the number of wind barbs plotted (every 20th)
#wind_slice = (slice(None, None, 20), slice(None, None, 20))
ax2.barbs(lon_2d, lat_2d, uwnd_500.to(units.knots).m, vwnd_500.to(units.knots).m, length=6, regrid_shape=10,
         pivot='middle', transform=ccrs.PlateCarree())
ax2.set_title('500 hPa absolute vorticity/heights',fontsize=16)

# Plot two titles, one on right and left side
#plt.title('500-hPa NAM Geopotential Heights (m)'
#          ' and Wind Barbs (kt)', loc='left')
#plt.title('Valid Time: {}'.format(vtime), loc='right')

# LOWER LEFT PANEL- 500 hPa winds/heights
ax3 = plt.subplot(gs[1,0],projection=crs)
axis_setup(ax3)

cf3 = ax3.contourf(lon_2d, lat_2d, winds_500,cmap='cool',transform=ccrs.PlateCarree(),levels=np.arange(40,100,10))
c3a = ax3.contour(lon_2d,lat_2d,temps_500.to('degC'),levels=np.arange(-40,0,2), colors='blue',transform=ccrs.PlateCarree(),linewidths=1,linestyles='dashed')
c3b = ax3.contour(lon_2d,lat_2d,temps_500.to('degC'),levels=np.arange(1,10,2), colors='red',transform=ccrs.PlateCarree(),linewidths=1,linestyles='dashed')
c3c = ax3.contour(lon_2d, lat_2d, heights_500, h5lev, colors='black', linewidths=2,
                       transform=ccrs.PlateCarree())
#c3b = ax3.contour(lon_2d,lat_2d,winds_500,linewidths=1,linestyles='dashed',transform=ccrs.PlateCarree(),levels=np.arange(40,100,10))
#ax3.barbs(lon_2d, lat_2d, uwnd_500.to(units.knots).m,vwnd_500.to(units.knots).m, length=6,
#         regrid_shape=10, pivot='middle', transform=ccrs.PlateCarree())
#ax3.clabel(c3a, fontsize=10, inline=1, inline_spacing=1, fmt='%i', rightside_up=True)
ax3.set_title('500-hPa Winds/Heights/Temp', fontsize=16)
cb3 = fig.colorbar(cf3, ax=ax3, orientation='horizontal', shrink=0.74, pad=0.01)
cb3.set_label('kts', size='x-large')

# LOWER RIGHT PANEL- 250 hPa winds/heights/divergence
ax4 = plt.subplot(gs[1,1],projection=crs)
axis_setup(ax4)

cf4 = ax4.contourf(lon_2d, lat_2d, winds_250.to(units.knots).m,cmap='cool',transform=ccrs.PlateCarree(),levels=np.arange(80,240,20))
c4a = ax4.contour(lon_2d, lat_2d, heights_250, h25lev, colors='black', linewidths=2,
                       transform=ccrs.PlateCarree())
#clevs_div = np.arange(-15, 16, 2)
clevs_div = np.arange(2, 16, 2)
c4b = ax4.contour(lon_2d, lat_2d, div_250*1e5, clevs_div, linewidths=1, linestyles='dashed', transform=ccrs.PlateCarree(),colors='red')
#ax4.barbs(lon_2d, lat_2d, uwnd_250.to(units.knots).m,vwnd_250.to(units.knots).m, length=6,
#         regrid_shape=10, pivot='middle', transform=ccrs.PlateCarree())
#ax4.clabel(c4a, fontsize=10, inline=1, inline_spacing=1, fmt='%i', rightside_up=True)
ax4.set_title('250-hPa Winds/Heights/Divergence', fontsize=16)
#ax1.add_patch(patch)
cb4 = fig.colorbar(cf4, ax=ax4, orientation='horizontal', shrink=0.74, pad=0.01)
cb4.set_label('kts', size='x-large')

# Set figure title
fig.suptitle(rd.strftime('%d %B %Y %H:%MZ')+' F%02d' % (fn), fontsize=24)

# Save figure
plt.savefig('uppr_'+rd.strftime('%Y%m%d%H')+'_'+'%02d' % (fn)+'.png')
