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

import datetime, math

# Find the correct Convair position file
#acpos = "/home/dadriaan/projects/icicle/data/convair/planet/position/FINAL/ICICLE_Flight_%02d_position.csv" % (int(p.opt['fnum']))

# Set the requested date
rd = datetime.datetime.strptime(p.opt['tstring'],'%Y-%m-%d %H:%M:%S')

# What date string
yyyymm = rd.strftime('%Y%m')
yyyymmdd = rd.strftime('%Y%m%d')

# What 3D product strings
prod3d = ['_u.','_v.','_z.','_t.','_p.','_q.']

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
#ds = xr.merge(singlesets)

# Subset the dataset. We want all levels, at a specific time, and reduce lat/lon
ds = ds.sel(time=rd,latitude=slice(60,15),longitude=slice(230,300))

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
dimval = list(ds.dims.values())
p3d = np.empty([dimval[1],dimval[0],dimval[2]])
for y in np.arange(0,dimval[0],1):
  for x in np.arange(0,dimval[2],1):
    #p3d[:,y,x] = ds.level.values[::-1]
    p3d[:,y,x] = ds.level.values
ds['P'] = xr.DataArray(p3d,dims=['level','latitude','longitude'],coords=ds.coords,attrs={'units':'hectopascals'})

# Have a spot to reset RH to 100 if > 100?

# Compute Td from Q
ds['TD'] = xr.DataArray(mpcalc.dewpoint_from_specific_humidity(ds['P'],ds['T'],ds['Q']),dims=['level','latitude','longitude'],coords=ds.coords,attrs={'units':'degC'})
pbot = 1000.0*units('hPa')
ptop = 100.0*units('hPa')

# Pull out the pressure slices for IVT and PW calcs 
p_ivt = ds['P'].isel(latitude=0,longitude=0).reindex(level=ds.level[::-1]).sel(level=slice(1000.0,300.0))*100.0*units('pascals')
#p_ivt = ds['P'].isel(latitude=0,longitude=0).sel(level=slice(300.0,1000.0))*100.0*units('pascals')
#p_pw = ds['P'].isel(latitude=0,longitude=0).reindex(level=ds.level[::-1]).sel(level=slice(1000.0,100.0))*100.0*units('pascals')

# Load the dataset into memory. This will prevent further operations from taking inordinate amount of time
print("LOADING DATA TO MEMORY...")
t1 = datetime.datetime.now()
ds.load()
print(datetime.datetime.now()-t1)

# Loop over x and y and compute the PW
pw = np.empty([dimval[0],dimval[2]])
ivt = np.empty([dimval[0],dimval[2]])
for x in range(0,dimval[0],1):
  for y in range(0,dimval[2],1):

    #print("=====IVT=====")
    q = ds['Q'].isel(latitude=x,longitude=y).reindex(level=ds.level[::-1]).sel(level=slice(1000.0,300.0))
    u = ds['U'].isel(latitude=x,longitude=y).reindex(level=ds.level[::-1]).sel(level=slice(1000.0,300.0))
    v = ds['V'].isel(latitude=x,longitude=y).reindex(level=ds.level[::-1]).sel(level=slice(1000.0,300.0))
    
    termA = math.pow(np.trapz((q*u),p_ivt)/9.81,2)
    termB = math.pow(np.trapz((q*v),p_ivt)/9.81,2)
    ivt[x,y] = math.sqrt(termA+termB)

    #print("=====PW=====")
    #w = mpcalc.mixing_ratio_from_specific_humidity(ds['Q'].isel(latitude=x,longitude=y).reindex(level=ds.level[::-1]).sel(level=slice(1000.0,100.0)))
    #print(np.trapz(w.m,p_pw)/(9.81*1000.0)*-1000.0)
    #pw[x,y] = np.trapz(w.m,p_pw)/(9.81*1000.0)
    pw[x,y] = mpcalc.precipitable_water(ds['P'].isel(latitude=x,longitude=y),ds['TD'].isel(latitude=x,longitude=y),bottom=pbot,top=ptop).m
    
# Now add PW to the dataset
pwda = xr.DataArray(pw,dims=['latitude','longitude'],coords=[ds.coords['latitude'],ds.coords['longitude']],attrs={'units':'millimeters'})
ivtda = xr.DataArray(ivt,dims=['latitude','longitude'],coords=[ds.coords['latitude'],ds.coords['longitude']])

# Smooth the height data
heights_300 = ndimage.gaussian_filter(mpcalc.geopotential_to_height(ds['Z'].sel(level=300.0)), sigma=1.5, order=0)
heights_700 = ndimage.gaussian_filter(mpcalc.geopotential_to_height(ds['Z'].sel(level=700.0)), sigma=1.5, order=0)
heights_850 = ndimage.gaussian_filter(mpcalc.geopotential_to_height(ds['Z'].sel(level=850.0)), sigma=1.5, order=0)

# Smooth the wind data
uwnd_300 = ndimage.gaussian_filter(ds['U'].sel(level=300.0), sigma=3.0) * units('m/s')
vwnd_300 = ndimage.gaussian_filter(ds['V'].sel(level=300.0), sigma=3.0) * units('m/s')
uwnd_700 = ndimage.gaussian_filter(ds['U'].sel(level=700.0), sigma=3.0) * units('m/s')
vwnd_700 = ndimage.gaussian_filter(ds['V'].sel(level=700.0), sigma=3.0) * units('m/s')
uwnd_850 = ndimage.gaussian_filter(ds['U'].sel(level=850.0), sigma=3.0) * units('m/s')
vwnd_850 = ndimage.gaussian_filter(ds['V'].sel(level=850.0), sigma=3.0) * units('m/s')

# Compute the wind speeds
winds_300 = mpcalc.wind_speed(uwnd_300,vwnd_300).to(units.knots)
winds_700 = mpcalc.wind_speed(uwnd_700,vwnd_700).to(units.knots)
winds_850 = mpcalc.wind_speed(uwnd_850,vwnd_850).to(units.knots)

# Compute the wind directions
wdir_300 = mpcalc.wind_direction(uwnd_300,vwnd_300)
wdir_700 = mpcalc.wind_direction(uwnd_700,vwnd_700)
wdir_850 = mpcalc.wind_direction(uwnd_850,vwnd_700)

# Contour levels for heights
h3lev = np.arange(8500.0,10000.0,30.0)
h5lev = np.arange(4800.0,5800.0,30.0)
h6lev = np.arange(3500.0,4500.0,30.0)
h7lev = np.arange(2500.0,3500.0,30.0)
h85lev = np.arange(1000.0,2000.0,30.0)
h92lev = np.arange(0.0,1000.0,30.0)

# Stuff for Convair path
verts = [(-100.0,45.0),(-90.0,45.0)]
codes = [Path.MOVETO,Path.LINETO]
path = Path(verts,codes)
patch = patches.PathPatch(path,lw=4,color="red",transform=ccrs.PlateCarree())

# UPPER LEFT PANEL- 700 hPa height/PW
ax1 = plt.subplot(gs[0,0],projection=crs)
axis_setup(ax1)
#sp1 = ax1.streamplot(lon_2d, lat_2d, uwnd_10, vwnd_10, density=5, transform=ccrs.PlateCarree(), color=np.asarray(winds_10m))
#cf1 = ax1.contourf(lon_2d, lat_2d, winds_300,cmap='cool',transform=ccrs.PlateCarree(),levels=np.arange(50,230,20))
#c1a = ax1.contour(lon_2d, lat_2d, heights_300, h3lev, colors='black', linewidths=2,
#                       transform=ccrs.PlateCarree())
cf1 = ax1.contourf(lon_2d, lat_2d, pwda, cmap='Greens',transform=ccrs.PlateCarree(),levels=np.arange(0,100,10))
c1a = ax1.contour(lon_2d, lat_2d, heights_700, h7lev, colors='black', linewidths=2,transform=ccrs.PlateCarree())
#ax1.barbs(lon_2d, lat_2d, uwnd_300.to(units.knots).m,vwnd_300.to(units.knots).m, length=6,
#         regrid_shape=10, pivot='middle', transform=ccrs.PlateCarree())
#ax1.clabel(c1a, fontsize=10, inline=1, inline_spacing=1, fmt='%i', rightside_up=True)
#ax1.set_title('10m Streamlines (kts)', fontsize=16)
#ax1.add_patch(patch)
cb1 = fig.colorbar(cf1, ax=ax1, orientation='horizontal', shrink=0.74, pad=0.01)
cb1.set_label('mm', size='x-large')

# UPPER RIGHT PANEL- 700 hPa height/IVT
ax2 = plt.subplot(gs[0,1],projection=crs)
axis_setup(ax2)
#cf2 = ax2.contourf(lon_2d, lat_2d, winds_700,cmap='cool',transform=ccrs.PlateCarree(),levels=np.arange(20,100,10))
#c2a = ax2.contour(lon_2d, lat_2d, heights_700, h7lev, colors='black', linewidths=2,
#                       transform=ccrs.PlateCarree())
cf2 = ax2.contourf(lon_2d, lat_2d, ivt,cmap='inferno', transform=ccrs.PlateCarree(),levels=np.arange(0,1000,100))
c2a = ax2.contour(lon_2d, lat_2d, heights_700, h7lev, colors='black', linewidths=2, transform=ccrs.PlateCarree())
#ax2.barbs(lon_2d, lat_2d, uwnd_700.to(units.knots).m,vwnd_700.to(units.knots).m, length=6,
#         regrid_shape=10, pivot='middle', transform=ccrs.PlateCarree())
#ax2.clabel(c2a, fontsize=10, inline=1, inline_spacing=1, fmt='%i', rightside_up=True)
#ax2.set_title('700-hPa Winds and Heights', fontsize=16)
cb2 = fig.colorbar(cf2, ax=ax2, orientation='horizontal', shrink=0.74, pad=0.01)
cb2.set_label('kg/m/s', size='x-large')

# LOWER LEFT PANEL- 850 hPa winds/height
ax3 = plt.subplot(gs[1,0],projection=crs)
axis_setup(ax3)
#cf3 = ax3.contourf(lon_2d, lat_2d, winds_850,cmap='cool',transform=ccrs.PlateCarree(),levels=np.arange(20,100,10))
#c3a = ax3.contour(lon_2d, lat_2d, heights_850, h85lev, colors='black', linewidths=2,
#                       transform=ccrs.PlateCarree())
#ax3.barbs(lon_2d, lat_2d, uwnd_850.to(units.knots).m,vwnd_850.to(units.knots).m, length=6,
#         regrid_shape=10, pivot='middle', transform=ccrs.PlateCarree())
#ax3.clabel(c3a, fontsize=10, inline=1, inline_spacing=1, fmt='%i', rightside_up=True)
#ax3.set_title('850-hPa Winds and Heights', fontsize=16)
#cb3 = fig.colorbar(cf3, ax=ax3, orientation='horizontal', shrink=0.74, pad=0.01)
#cb3.set_label('kts', size='x-large')

# LOWER RIGHT PANEL- MSLP/2mT
ax4 = plt.subplot(gs[1,1],projection=crs)
axis_setup(ax4)
#cf4 = ax4.contourf(lon_2d, lat_2d, sds['VAR_2T']-273.15,cmap='coolwarm', levels=np.arange(-40,40,2),transform=ccrs.PlateCarree())
#c4a = ax4.contour(lon_2d, lat_2d, smooth_MSL/100.0, colors='black', linewidths=1, levels=np.arange(888,1056,2),
#                       transform=ccrs.PlateCarree())
#ax4.clabel(c4a, fontsize=10, inline=1, inline_spacing=7, fmt='%i', rightside_up=True, use_clabeltext=True, colors='white')
#ax4.set_title('MSLP/2mT', fontsize=16)
#cb4 = fig.colorbar(cf4, ax=ax4, orientation='horizontal', shrink=0.74, pad=0.01)
#cb4.set_label('degC', size='x-large')

# Set figure title
fig.suptitle(rd.strftime('%d %B %Y %H:%MZ')+' F%02d' % (int(p.opt['fnum'])), fontsize=24)

# Save figure
plt.savefig('moist_'+rd.strftime('%Y%m%d%H')+'_'+'%02d' % (int(p.opt['fnum']))+'.png')
