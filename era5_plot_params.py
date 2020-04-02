#!/usr/bin/env python

# Import ConfigMaster
from ConfigMaster import ConfigMaster

# Create a params class
class Params(ConfigMaster):
  defaultParams = """

#!/usr/bin/env python

# DEFAULT PARAMS

####### zoom #######
#
# NAME: zoom
# OPTIONS:
# TYPE: list
# FORMAT: [minLon, maxLon, minLat, maxLat]
# DEFAULT: conus
# DESCRIPTION: Control the zooming of the domain for plotting using latitude/longitude
# conus = [235.,290.,20.,55.]
# icicle = [255.,280.,35.,50.]
#
zoom = [255.,280.,35.,50.]

####### tstring #######
#
# NAME: tstring
# OPTIONS:
# TYPE: string
# FORMAT: YYYY-MM-DD HH:MM:SS
# DEFAULT: 2019-02-01 00:00:00
# DESCRIPTION: Set the time you wish to plot
#
tstring = '2019-02-01 00:00:00'

####### fnum #######
# NAME: fnum
# OPTIONS:
# TYPE: integer
# FORMAT: 
# DEFAULT: 5
# DESCRIPTION: Flight number to look for the flight path file
#
fnum = 5

####### creds #######
#
# NAME: creds
# OPTIONS:
# TYPE: 
# FORMAT: ('username','password')
# DEFAULT:
# DESCRIPTION: Username and password for THREDDS dataserver
#
creds = ('username','password')
"""
