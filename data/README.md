# Overview

Documenting some basic datasets for the initial real-data test using North American BBS abundances.

# Datasets

## North American BBS

source: USGS
contents:
* counts & abundances
* latitude/longitude
* species
* various data quality flags
* route (can be matched into "routes" to get date/time metadata, I think)
citation (DOI): Pardieck, K.L., D.J. Ziolkowski Jr., M. Lutmerding, K. Campbell and M.-A.R. Hudson. 2017. North American Breeding Bird Survey Dataset 1966 - 2016, version 2016.0. U.S. Geological Survey, Patuxent Wildlife Research Center. <[www.pwrc.usgs.gov/BBS/RawData/](www.pwrc.usgs.gov/BBS/RawData/)>; doi:10.5066/F7W0944J. via [link](https://www.pwrc.usgs.gov/BBS/help/Citations.cfm)
notes: some basic code to retrieve and do basic processing is in the [bbs-forecasting repo](https://github.com/weecology/bbs-forecasting/blob/master/R/forecast-bbs-core.R)

## NDVI (land cover)

source: GIMMS MODIS (NASA)
contents: NDVI (remotely sensed), 1981-2013
citation (DOI): (not 100% clear, but somewhere on https://glam1.gsfc.nasa.gov/doc/overview.html perhaps)
notes: code to retrieve this is in the [bbs-forecasting repo](https://github.com/weecology/bbs-forecasting/blob/master/R/get_ndvi_data.R)

## Life History
