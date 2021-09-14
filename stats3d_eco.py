#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 20:24:33 2020

@author: martinjanssens
"""

import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import gc
import sys
sys.path.insert(1, '/home/janssens/scripts/pp3d/')
from functions import *

lp = '/scratch-shared/janssens/bomex100'
ds = nc.Dataset(lp+'/fielddump.001.nc')
ds1= nc.Dataset(lp+'/profiles.001.nc')
ds0= nc.Dataset(lp+'/tmser.001.nc')
ilp = np.loadtxt(lp+'/lscale.inp.001')

time  = np.ma.getdata(ds.variables['time'][:]) / 3600
zf    = np.ma.getdata(ds.variables['zt'][:]) # Cell centres (f in mhh)
zh    = np.ma.getdata(ds.variables['zm'][:]) # Cell edges (h in mhh)
xf    = np.ma.getdata(ds.variables['xt'][:]) # Cell centres (f in mhh)
xh    = np.ma.getdata(ds.variables['xm'][:]) # Cell edges (h in mhh)
yf    = np.ma.getdata(ds.variables['yt'][:]) # Cell centres (f in mhh)
yh    = np.ma.getdata(ds.variables['ym'][:]) # Cell edges (h in mhh)

time1d = np.ma.getdata(ds1.variables['time'][:])
rhobf = np.ma.getdata(ds1.variables['rhobf'][:])
rhobh = np.ma.getdata(ds1.variables['rhobh'][:])

dx = np.diff(xf)[0]
dy = np.diff(yf)[0] # Assumes uniform horizontal spacing
dzh = np.diff(zf)[0] # FIXME only valid in lower part of domain

delta = (dx*dy*np.diff(zh))**(1./3)

# Larger-scale subsidence
wfls = ilp[:,3]

#%% Dry/moist regions

itmin = 0#23
itmax = 1
di    = 1
izmin = 0
izmax = 80
store = False

klp = 4

plttime = np.arange(itmin, itmax, di)
zflim = zf[izmin:izmax]

qtpf_moist_time = np.zeros((plttime.size,izmax-izmin))
qtpf_dry_time = np.zeros((plttime.size,izmax-izmin))
qtpf_prod_moist_time = np.zeros((plttime.size,izmax-izmin-2))
qtpf_prod_dry_time = np.zeros((plttime.size,izmax-izmin-2))
qtpf_prod_moist_wex_time = np.zeros((plttime.size,izmax-izmin-2))
qtpf_prod_dry_wex_time = np.zeros((plttime.size,izmax-izmin-2))
qtpf_vdiv_moist_time = np.zeros((plttime.size,izmax-izmin-2))
qtpf_vdiv_dry_time = np.zeros((plttime.size,izmax-izmin-2))
qtpf_hdiv_moist_time = np.zeros((plttime.size,izmax-izmin-2))
qtpf_hdiv_dry_time = np.zeros((plttime.size,izmax-izmin-2))
qtpf_subs_moist_time = np.zeros((plttime.size,izmax-izmin-2))
qtpf_subs_dry_time = np.zeros((plttime.size,izmax-izmin-2))
qtpf_diff_moist_time = np.zeros((plttime.size,izmax-izmin-4))
qtpf_diff_dry_time = np.zeros((plttime.size,izmax-izmin-4))

thlvpf_moist_time = np.zeros((plttime.size,izmax-izmin))
thlvpf_dry_time = np.zeros((plttime.size,izmax-izmin))
thlvpf_prod_moist_time = np.zeros((plttime.size,izmax-izmin-2))
thlvpf_prod_dry_time = np.zeros((plttime.size,izmax-izmin-2))
thlvpf_vdiv_moist_time = np.zeros((plttime.size,izmax-izmin-2))
thlvpf_vdiv_dry_time = np.zeros((plttime.size,izmax-izmin-2))
thlvpf_hdiv_moist_time = np.zeros((plttime.size,izmax-izmin-2))
thlvpf_hdiv_dry_time = np.zeros((plttime.size,izmax-izmin-2))
thlvpf_subs_moist_time = np.zeros((plttime.size,izmax-izmin-2))
thlvpf_subs_dry_time = np.zeros((plttime.size,izmax-izmin-2))
thlvpf_diff_moist_time = np.zeros((plttime.size,izmax-izmin-4))
thlvpf_diff_dry_time = np.zeros((plttime.size,izmax-izmin-4))

thlpf_moist_time = np.zeros((plttime.size,izmax-izmin))
thlpf_dry_time = np.zeros((plttime.size,izmax-izmin))

wff_moist_time = np.zeros((plttime.size,izmax-izmin))
wff_dry_time = np.zeros((plttime.size,izmax-izmin))

qlpf_moist_time = np.zeros((plttime.size,izmax-izmin))
qlpf_dry_time = np.zeros((plttime.size,izmax-izmin))

wthlpf_moist_time = np.zeros((plttime.size,izmax-izmin))
wthlpf_dry_time = np.zeros((plttime.size,izmax-izmin))

wqtpf_moist_time = np.zeros((plttime.size,izmax-izmin))
wqtpf_dry_time = np.zeros((plttime.size,izmax-izmin))

wqlpf_moist_time = np.zeros((plttime.size,izmax-izmin))
wqlpf_dry_time = np.zeros((plttime.size,izmax-izmin))

wthlvpf_av_time = np.zeros((plttime.size,izmax-izmin))
wthlvpf_moist_time = np.zeros((plttime.size,izmax-izmin))
wthlvpf_dry_time = np.zeros((plttime.size,izmax-izmin))

# Mask for low-[ass filtering
circ_mask = np.zeros((xf.size,xf.size))
rad = getRad(circ_mask)
circ_mask[rad<=klp] = 1

for i in range(len(plttime)):
    it1d = np.argmin(np.abs(time1d/3600 - time[plttime[i]]))
    
    # 1D fields
    rhobfi = rhobf[it1d,izmin:izmax]
    rhobhi = rhobh[it1d,izmin:izmax]
    
    # 3D fields
    qt  = np.ma.getdata(ds.variables['qt'][plttime[i],izmin:izmax,:,:])
    wh = np.ma.getdata(ds.variables['w'][plttime[i],izmin:izmax+1,:,:])
    thlp =  np.ma.getdata(ds.variables['thl'][plttime[i],izmin:izmax,:,:])
    qlp = np.ma.getdata(ds.variables['ql'][plttime[i],izmin:izmax,:,:])
    u = np.ma.getdata(ds.variables['u'][plttime[i],izmin:izmax,:,:])
    v = np.ma.getdata(ds.variables['v'][plttime[i],izmin:izmax,:,:])
    # e12 = np.ma.getdata(ds.variables['e12'][plttime[i],izmin:izmax,:,:])

    # Slab averaging
    thl_av = np.mean(thlp,axis=(1,2))
    qt_av = np.mean(qt,axis=(1,2))   
    ql_av = np.mean(qlp,axis=(1,2))
    thlv_av = thl_av*(1+0.608*qt_av)
    
    # -> need presf for thv_av, taken from nearest 1d data time and half-level
    presh  = np.ma.getdata(ds1.variables['presh'][it1d,izmin:izmax])
    presf  = (presh[1:]+presh[:-1])*0.5
    exnf   = (presf/1e5)**(Rd/cp)
    thv_av = (thl_av[:-1] + (Lv/cp)*ql_av[:-1]/exnf)*(1.+(Rv/Rd-1)*qt_av[:-1] -Rv/Rd*ql_av[:-1])
    
    # Eddy diffusivities
    # dthvdz = compute_dthvdz(thlp, qt, qlp, exnf, dzh)
    # _,ekhp = compute_ek(e12, dthvdz, thv_av, delta[izmin:izmax-1])
    # ekh_av = np.mean(ekhp,axis=(1,2))
    # ekhp = ekhp - ekh_av[:,np.newaxis,np.newaxis]
    # del e12
    # del dthvdz
    
    # Perturbations
    qtp = qt - qt_av[:,np.newaxis,np.newaxis]
    twp = vint(qt,rhobfi,zflim,tmin=i, tmax=i+1)
    del qt
    
    gc.collect()
    
    wf = (wh[1:,:,:] + wh[:-1,:,:])*0.5
    wh = wh[:-1,:,:]

    thlp = thlp - thl_av[:,np.newaxis,np.newaxis]
    qlp = qlp - ql_av[:,np.newaxis,np.newaxis]
    
    # Slab average resolved fluxes
    wqt_av = np.mean(wf*qtp,axis=(1,2))
    wthl_av = np.mean(wf*thlp,axis=(1,2))
    # wql_av = np.mean(wf*qlp,axis=(1,2))
 
    # Low-pass filter (and identify high-pass filtered remainder)    
    qtpf = lowPass(qtp, circ_mask)
    qtpp = qtp - qtpf
    del qtp
    
    whf = lowPass(wh, circ_mask)
    whp = wh - whf
    del wh
    
    wff = lowPass(wf, circ_mask)
    wfp = wf - wff
    del wf
            
    thlpf = lowPass(thlp, circ_mask)
    thlpp = thlp - thlpf
    del thlp
    
    qlpf = lowPass(qlp, circ_mask)
    qlpp = qlp - qlpf
    del qlp

    gc.collect()
    

    # Moist/dry averaging, over the filtered scales
    twp = lowPass(twp, circ_mask)
    mask_moist = np.zeros(twp.shape)
    mask_moist[twp - np.mean(twp) > 0] = 1
    mask_dry = 1 - mask_moist
    
    thlp_moist = mean_mask(thlpf,mask_moist)
    qtp_moist = mean_mask(qtpf,mask_moist)
    w_moist = mean_mask(wff,mask_moist)
    # w_moist_h = mean_mask(whf,mask_moist)
    qlp_moist = mean_mask(qlpf,mask_moist)
    
    thlp_dry = mean_mask(thlpf,mask_dry)
    qtp_dry = mean_mask(qtpf,mask_dry)
    w_dry = mean_mask(wff,mask_dry)
    qlp_dry = mean_mask(qlpf,mask_dry)

    thlpf_moist_time[i,:] = thlp_moist
    thlpf_dry_time[i,:] = thlp_dry

    qtpf_moist_time[i,:] = qtp_moist
    qtpf_dry_time[i,:] = qtp_dry

    thlvpf_moist_time[i,:] = thlp_moist + 0.608*thl_av*qtp_moist
    thlvpf_dry_time[i,:] = thlp_dry + 0.608*thl_av*qtp_dry

    wff_moist_time[i,:] = w_moist
    wff_dry_time[i,:] = w_dry
    
    qlpf_moist_time[i,:] = qlp_moist
    qlpf_dry_time[i,:] = qlp_dry

    ## Fluxes 
    # FIXME no sgs here yet!!
    wthlpf = lowPass((wff+wfp)*(thlpf+thlpp), circ_mask)
    wqtpf = lowPass((wff+wfp)*(qtpf+qtpp), circ_mask)
    wqlpf = lowPass((wff+wfp)*(qlpf+qtpp), circ_mask)

    wthlpf_av = np.mean(wthlpf,axis=(1,2))
    wthlpf_moist = mean_mask(wthlpf, mask_moist)
    wthlpf_dry = mean_mask(wthlpf, mask_dry)
    
    wqtpf_av = np.mean(wqtpf,axis=(1,2))
    wqtpf_moist = mean_mask(wqtpf, mask_moist)
    wqtpf_dry = mean_mask(wqtpf, mask_dry)
    
    wqlpf_moist = mean_mask(wqlpf, mask_moist)
    wqlpf_dry = mean_mask(wqlpf, mask_dry)

    wthlvpf_av = wthlpf_av + 0.608*thl_av*wqtpf_av
    wthlvpf_moist = wthlpf_moist + 0.608*thl_av*wqtpf_moist
    wthlvpf_dry = wthlpf_dry + 0.608*thl_av*wqtpf_dry
    
    
    wthlpf_moist_time[i,:] = wthlpf_moist
    wthlpf_dry_time[i,:] = wthlpf_dry
    
    wqtpf_moist_time[i,:] = wqtpf_moist
    wqtpf_dry_time[i,:] = wqtpf_dry
    
    wqlpf_moist_time[i,:] = wqlpf_moist
    wqlpf_dry_time[i,:] = wqlpf_dry
    
    wthlvpf_av_time[i,:] = wthlvpf_av
    wthlvpf_moist_time[i,:] = wthlvpf_moist
    wthlvpf_dry_time[i,:] = wthlvpf_dry
    
    del wthlpf
    del wqtpf
    del wqlpf
    gc.collect()
        
    ## BUDGET TERMS
    
    # Gradient production
    
    # Mean gradients
    Gamma_thl = (thl_av[1:] - thl_av[:-1])/dzh
    Gamma_qt = (qt_av[1:] - qt_av[:-1])/dzh
    # Gamma_ql = (ql_av[1:] - ql_av[:-1])/dzh
    Gamma_thlv = (thlv_av[1:] - thlv_av[:-1])/dzh
    
    Gamma_thl_f = (Gamma_thl[1:] + Gamma_thl[:-1])*0.5
    Gamma_qt_f = (Gamma_qt[1:] + Gamma_qt[:-1])*0.5
    # Gamma_ql_f = (Gamma_ql[1:] + Gamma_ql[:-1])*0.5
    Gamma_thlv_f = (Gamma_thlv[1:] + Gamma_thlv[:-1])*0.5
    
    # thlv production
    thlvpf_prod_moist = w_moist[1:-1]*Gamma_thlv_f
    thlvpf_prod_dry = w_dry[1:-1]*Gamma_thlv_f
    
    thlvpf_prod_moist_time[i,:] = thlvpf_prod_moist
    thlvpf_prod_dry_time[i,:] = thlvpf_prod_dry
    
    # Moisture production term with actual w'
    qtpf_prod_wex_moist = w_moist[1:-1]*Gamma_qt_f
    qtpf_prod_wex_dry = w_dry[1:-1]*Gamma_qt_f

    qtpf_prod_moist_wex_time[i,:] = qtpf_prod_wex_moist
    qtpf_prod_dry_wex_time[i,:] = qtpf_prod_wex_dry

    
    # Small-scale vertical flux divergence anomaly (with second order scheme)
    
    # Slab-averaged vertical flux divergence
    div_wthl_av = np.mean(ddzwx_2nd(whp, thlpf+thlpp, dzh, rhobf=rhobfi),axis=(1,2))
    div_wqt_av = np.mean(ddzwx_2nd(whp, qtpf+qtpp, dzh, rhobf=rhobfi),axis=(1,2))
    div_wthlv_av = div_wthl_av + 0.608*thl_av[1:-1]*div_wqt_av

    # Reynolds vertical flux divergence
    div_wthl_r = lowPass(ddzwx_2nd(whp, thlpp, dzh, rhobf=rhobfi), circ_mask)

    div_wqt_r = lowPass(ddzwx_2nd(whp, qtpp, dzh, rhobf=rhobfi), circ_mask)
    div_wqt_r_moist = mean_mask(div_wqt_r,mask_moist)
    div_wqt_r_dry = mean_mask(div_wqt_r,mask_dry)
    
    qtpf_vdiv_moist_time[i,:] = div_wqt_r_moist - div_wqt_av
    qtpf_vdiv_dry_time[i,:] = div_wqt_r_dry - div_wqt_av

    div_wthlv_r = div_wthl_r + 0.608*thl_av[1:-1,np.newaxis,np.newaxis]*div_wqt_r
    div_wthlv_r_moist = mean_mask(div_wthlv_r, mask_moist)
    div_wthlv_r_dry = mean_mask(div_wthlv_r, mask_dry)
    
    thlvpf_vdiv_moist_time[i,:] = div_wthlv_r_moist - div_wthlv_av
    thlvpf_vdiv_dry_time[i,:] = div_wthlv_r_dry - div_wthlv_av
    
    del div_wthl_r
    del div_wqt_r
    del div_wthlv_r
    gc.collect()

    # Moisture instability term model (WTG for thlv and qtpf model for flux anomaly div)
    w_mod = lowPass(np.abs(whp),circ_mask)
    div_wthlvfa_mod = ddzwx_2nd(w_mod, -0.608*thl_av[:,np.newaxis,np.newaxis]*qtpf, dzh, rhobf=rhobfi)
    div_wthlvfa_mod_moist = mean_mask(div_wthlvfa_mod, mask_moist)
    div_wthlvfa_mod_dry = mean_mask(div_wthlvfa_mod, mask_dry)
    del w_mod
    del div_wthlvfa_mod
    gc.collect()

    qtpf_prod_moist = div_wthlvfa_mod_moist*Gamma_qt_f/Gamma_thlv_f
    qtpf_prod_dry = div_wthlvfa_mod_dry*Gamma_qt_f/Gamma_thlv_f
    
    # Model that just relies on the WTG
    # qtpf_prod_moist = (div_wthlv_r_moist - div_wthlv_av)*Gamma_qt_f/Gamma_thlv_f
    # qtpf_prod_dry = (div_wthlv_r_dry - div_wthlv_av)*Gamma_qt_f/Gamma_thlv_f
    
    qtpf_prod_moist_time[i,:] = qtpf_prod_moist
    qtpf_prod_dry_time[i,:] = qtpf_prod_dry

    # Horizontal thlv advection
    # Can be verified to be small compared to div_wthlv-div_wthlv_av
    div_uhthlvp = lowPass(ddxhuha_2nd(u, v, thlpf+thlpp+0.608*thl_av[:,np.newaxis,np.newaxis]*(qtpf+qtpp), dx, dy), circ_mask)
    div_uhthlvp_moist = mean_mask(div_uhthlvp,mask_moist)
    div_uhthlvp_dry = mean_mask(div_uhthlvp,mask_dry)

    thlvpf_hdiv_moist_time[i,:] = div_uhthlvp_moist[1:-1]
    thlvpf_hdiv_dry_time[i,:] = div_uhthlvp_dry[1:-1]
    
    # Horizontal moisture advection
    # intra-scale contribution largest, but entire term kept for now
    div_uhqtp = lowPass(ddxhuha_2nd(u, v, qtpf+qtpp, dx, dy), circ_mask)
    div_uhqtp_moist = mean_mask(div_uhqtp,mask_moist)
    div_uhqtp_dry = mean_mask(div_uhqtp,mask_dry)

    qtpf_hdiv_moist_time[i,:] = div_uhqtp_moist[1:-1]
    qtpf_hdiv_dry_time[i,:] = div_uhqtp_dry[1:-1]
    
    del div_uhthlvp
    del div_uhqtp
    del u
    del v
    gc.collect()

    # Subsidence warming
    wsubdthlvpdzf = lowPass(wsubdxdz(wfls[izmin:izmax],thlpf+thlpp+0.608*thl_av[:,np.newaxis,np.newaxis]*(qtpf+qtpp), dzh),circ_mask)
    wsubdthlvpdzf_moist = mean_mask(wsubdthlvpdzf,mask_moist)
    wsubdthlvpdzf_dry = mean_mask(wsubdthlvpdzf,mask_dry)
    
    thlvpf_subs_moist_time[i,:] = wsubdthlvpdzf_moist[1:]
    thlvpf_subs_dry_time[i,:] = wsubdthlvpdzf_dry[1:]
    
    # Subsidence drying
    wsubdqtpdzf = lowPass(wsubdxdz(wfls[izmin:izmax], qtpf+qtpp, dzh),circ_mask)
    wsubdqtpdzf_moist = mean_mask(wsubdqtpdzf,mask_moist)
    wsubdqtpdzf_dry = mean_mask(wsubdqtpdzf,mask_dry)
    
    qtpf_subs_moist_time[i,:] = wsubdqtpdzf_moist[1:]
    qtpf_subs_dry_time[i,:] = wsubdqtpdzf_dry[1:]
    
    del wsubdthlvpdzf
    del wsubdqtpdzf
    gc.collect()

    # SFS diffusion
    # Heat
    # diff_thlvpf = lowPass(diffeka(ekhp+ekh_av[:,np.newaxis,np.newaxis], 
    #                               thlpf+thlpp+0.608*thl_av[:,np.newaxis,np.newaxis]*(qtpf+qtpp),
    #                               dx, dy, zf, rhobfi, rhobhi)+
    #                     diffzeka(ekhp, thl_av[:,np.newaxis,np.newaxis]*(1+0.608*qt_av[:,np.newaxis,np.newaxis]),
    #                              dzh, rhobfi, rhobhi),
    #                     circ_mask)
    # diff_thlvpf_moist = mean_mask(diff_thlvpf,mask_moist)
    # diff_thlvpf_dry = mean_mask(diff_thlvpf,mask_dry)

    # thlvpf_diff_moist_time[i,:] = diff_thlvpf_moist
    # thlvpf_diff_dry_time[i,:] = diff_thlvpf_dry
    
    # # Moisture
    # diff_qtpf = lowPass(diffeka(ekhp+ekh_av[:,np.newaxis,np.newaxis], qtpf+qtpp, dx, dy, zf, rhobfi, rhobhi)+
    #                     diffzeka(ekhp, qt_av[:,np.newaxis,np.newaxis], dzh, rhobfi, rhobhi),
    #                     circ_mask)
    # diff_qtpf_moist = mean_mask(diff_qtpf,mask_moist)
    # diff_qtpf_dry = mean_mask(diff_qtpf,mask_dry)

    # qtpf_diff_moist_time[i,:] = diff_qtpf_moist
    # qtpf_diff_dry_time[i,:] = diff_qtpf_dry
if store:
    np.save(lp+'/qtpf_moist_time.npy',qtpf_moist_time)
    np.save(lp+'/qtpf_dry_time.npy',qtpf_dry_time)
    np.save(lp+'/qtpf_prod_moist_time.npy',qtpf_prod_moist_time)
    np.save(lp+'/qtpf_prod_dry_time.npy',qtpf_prod_dry_time)
    np.save(lp+'/qtpf_prod_moist_wex_time.npy',qtpf_prod_moist_wex_time)
    np.save(lp+'/qtpf_prod_dry_wex_time.npy',qtpf_prod_dry_wex_time)
    np.save(lp+'/qtpf_vdiv_moist_time.npy',qtpf_vdiv_moist_time)
    np.save(lp+'/qtpf_vdiv_dry_time.npy',qtpf_vdiv_dry_time)
    np.save(lp+'/qtpf_hdiv_moist_time.npy',qtpf_hdiv_moist_time)
    np.save(lp+'/qtpf_hdiv_dry_time.npy',qtpf_hdiv_dry_time)
    np.save(lp+'/qtpf_subs_moist_time.npy',qtpf_subs_moist_time)
    np.save(lp+'/qtpf_subs_dry_time.npy',qtpf_subs_dry_time)
    np.save(lp+'/qtpf_diff_moist_time.npy',qtpf_diff_moist_time)
    np.save(lp+'/qtpf_diff_dry_time.npy',qtpf_diff_dry_time)
    
    np.save(lp+'/thlvpf_moist_time.npy',thlvpf_moist_time)
    np.save(lp+'/thlvpf_dry_time.npy',thlvpf_dry_time)
    np.save(lp+'/thlvpf_prod_moist_time.npy',thlvpf_prod_moist_time)
    np.save(lp+'/thlvpf_prod_dry_time.npy',thlvpf_prod_dry_time)
    np.save(lp+'/thlvpf_vdiv_moist_time.npy',thlvpf_vdiv_moist_time)
    np.save(lp+'/thlvpf_vdiv_dry_time.npy',thlvpf_vdiv_dry_time)
    np.save(lp+'/thlvpf_hdiv_moist_time.npy',thlvpf_hdiv_moist_time)
    np.save(lp+'/thlvpf_hdiv_dry_time.npy',thlvpf_hdiv_dry_time)
    np.save(lp+'/thlvpf_subs_moist_time.npy',thlvpf_subs_moist_time)
    np.save(lp+'/thlvpf_subs_dry_time.npy',thlvpf_subs_dry_time)
    np.save(lp+'/thlvpf_diff_moist_time.npy',thlvpf_diff_moist_time)
    np.save(lp+'/thlvpf_diff_dry_time.npy',thlvpf_diff_dry_time)
    
    np.save(lp+'/thlpf_moist_time.npy',thlpf_moist_time)
    np.save(lp+'/thlpf_dry_time.npy',thlpf_dry_time)
    
    np.save(lp+'/wff_moist_time.npy',wff_moist_time)
    np.save(lp+'/wff_dry_time.npy',wff_dry_time)
    
    np.save(lp+'/qlpf_moist_time.npy',qlpf_moist_time) 
    np.save(lp+'/qlpf_dry_time.npy',qlpf_dry_time)
    
    np.save(lp+'/wthlpf_moist_time',wthlpf_moist_time)
    np.save(lp+'/wthlpf_dry_time',wthlpf_dry_time)
    
    np.save(lp+'/wqtpf_moist_time',wqtpf_moist_time)
    np.save(lp+'/wqtpf_dry_time',wqtpf_dry_time)
    
    np.save(lp+'/wqlpf_moist_time',wqlpf_moist_time)
    np.save(lp+'/wqlpf_dry_time',wqlpf_dry_time)
    
    np.save(lp+'/wthlvpf_av_time',wthlvpf_av_time)
    np.save(lp+'/wthlvpf_moist_time',wthlvpf_moist_time)
    np.save(lp+'/wthlvpf_dry_time',wthlvpf_dry_time)
    

