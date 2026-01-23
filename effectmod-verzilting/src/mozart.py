# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 09:55:06 2018

Library to read and write mozart files from/to pandas dataframes

NOTE: Dataframes for write should be exactly according to column order Mozart files
no checks provided...

@author: delsman
"""
import pandas as pd
import numpy as np
from datetime import datetime
from math import modf, log10

def read_mzbalance(fn, **kwargs):
    ''' lswwaterbalans.out, zoutbalans.out'''
    return read_mzfile(fn,'balance', **kwargs)

def read_mzdw(fn, **kwargs):
    ''' dw.dik'''
    return read_mzfile(fn,'dw', **kwargs)

def read_mzdwvalue(fn, **kwargs):
    ''' dwvalue.dik / dwvalue.out'''
    return read_mzfile(fn,'dwvalue', **kwargs)

def read_mzlsw(fn, **kwargs):
    ''' lsw.dik'''
    return read_mzfile(fn,'lsw', **kwargs)

def read_mzlswattr(fn, **kwargs):
    ''' lswattr.csv'''
    return pd.read_csv(fn,header=None, names=["local_surface_water_code","level","area_open_water","concentration","meteostation_code","urban_area"],**kwargs)

def read_mzlswvalue(fn, **kwargs):
    ''' lswvalue.dik, lswvalue.out'''
    return read_mzfile(fn,'lswvalue', **kwargs)

def read_mzladvalue(fn, **kwargs):
    ''' ladvalue.dik'''
    return read_mzfile(fn,'ladvalue', **kwargs)

def read_mzlswrouting(fn, **kwargs):
    ''' lswrouting.dik, lswrouting_dbc.dik'''
    return read_mzfile(fn,'lswrouting', **kwargs)

def read_mzvadvalue(fn, **kwargs):
    ''' vadvalue.dik'''
    return read_mzfile(fn,'vadvalue', **kwargs)

def read_mzvlvalue(fn, **kwargs):
    ''' vlvalue.dik'''
    return read_mzfile(fn,'vlvalue', **kwargs)

def read_mzuslsw(fn, **kwargs):
    ''' uslsw.dik'''
    return read_mzfile(fn,'uslsw', **kwargs)

def read_mzuslswdemand(fn, **kwargs):
    ''' uslswdemand.dik'''
    return read_mzfile(fn,'uslswdemand', **kwargs)

def read_mzlswusex(fn, **kwargs):
    ''' lswusex.out'''
    return read_mzfile(fn,'lswusex', **kwargs)

def read_mzweirarea(fn, **kwargs):
    ''' weirarea.dik'''
    return read_mzfile(fn,'weirarea', **kwargs)

def read_mzwaattr(fn, **kwargs):
    ''' waattr.csv'''
    return pd.read_csv(fn,header=None, names=["local_surface_water_code","weir_area_code","level"],**kwargs)

def read_mzwavalue(fn, **kwargs):
    ''' wavalue.out'''
    return read_mzfile(fn,'wavalue', **kwargs)

def write_mzbalance(fn, df, **kwargs):
    ''' lswwaterbalans.out, zoutbalans.out
        df is pandas dataframe with exact columns as file'''
    return write_mzfile(fn, df, 'balance', **kwargs)

def write_mzdw(fn, df, **kwargs):
    ''' dw.dik
        df is pandas dataframe with exact columns as file'''
    return write_mzfile(fn, df,'dw', **kwargs)

def write_mzdwvalue(fn, df, **kwargs):
    ''' dwvalue.dik / dwvalue.out
        df is pandas dataframe with exact columns as file'''
    return write_mzfile(fn, df,'dwvalue', **kwargs)

def write_mzlsw(fn, df, **kwargs):
    ''' lsw.dik
        df is pandas dataframe with exact columns as file'''
    return write_mzfile(fn, df,'lsw', **kwargs)

def write_mzlswvalue(fn, df, **kwargs):
    ''' lswvalue.dik, lswvalue.out
        df is pandas dataframe with exact columns as file'''
    return write_mzfile(fn, df,'lswvalue', **kwargs)

def write_mzladvalue(fn, df, **kwargs):
    ''' ladvalue.dik
        df is pandas dataframe with exact columns as file'''
    return write_mzfile(fn, df,'ladvalue', **kwargs)

def write_mzlswrouting(fn, df, **kwargs):
    ''' lswrouting.dik, lswrouting_dbc.dik'''
    return write_mzfile(fn, df, 'lswrouting', **kwargs)

def write_mzvadvalue(fn, df, **kwargs):
    ''' vadvalue.dik
        df is pandas dataframe with exact columns as file'''
    return write_mzfile(fn, df,'vadvalue', **kwargs)

def write_mzvlvalue(fn, df, **kwargs):
    ''' vlvalue.dik
        df is pandas dataframe with exact columns as file'''
    return write_mzfile(fn, df,'vlvalue', **kwargs)

def write_mzuslsw(fn, df, **kwargs):
    ''' uslsw.dik
        df is pandas dataframe with exact columns as file'''
    return write_mzfile(fn, df,'uslsw', **kwargs)

def write_mzuslswdemand(fn, df, **kwargs):
    ''' uslswdemand.dik
        df is pandas dataframe with exact columns as file'''
    return write_mzfile(fn, df,'uslswdemand', **kwargs)

def write_mzweirarea(fn, df, **kwargs):
    ''' weirarea.dik
        df is pandas dataframe with exact columns as file'''
    return write_mzfile(fn, df,'weirarea', **kwargs)

def write_mzwavalue(fn, df, **kwargs):
    ''' wavalue.out
        df is pandas dataframe with exact columns as file'''
    return write_mzfile(fn, df,'wavalue', **kwargs)

mzfile_description = {
    'dw': [[3,'districtwatercode'], #, uit PAWN of vrij te kiezen
           [33,'dw-name'],  # dw name lengte is vrij te kiezen,tussen dubbele quotes...
           [6,'meteostationcode'], # uit METEOSTATION
           [9,'area'], # [m2]
           [6,'depth_surface_water'], #, gemiddelde bodemligging t.o.v. N.A.P. [m]
           [12,'volume'], #, streefvolume, [m3]
           [5,'alien_water_ratio'], #, vaste waarde voor water uit netwerk, 0.0 tot 1.0 [-]
           [12,'salt_concentration'], #, vaste waarde voor water uit netwerk [g/m3]
           [12,'discharge'],
           [12,'upstream_discharge_factor']], #, max. aanvoer uit netwerk bij mozart stand-alone [m3/s]
    'dwvalue': [[3,'districtwatercode'], # uit DW
                [15,'time_start'], # (JJJJMMDD.HHMMSS)
                [15,'time_end'], # (JJJJMMDD.HHMMSS)
                [12,'concentration'], # [g/m3]
                [	6,'evaporation'], # [mm]
                [12,'total_flush_demand'], # [m3/s]
                [12,'volume'], # [m3]
                [5,'alien_water_ratio']], # , 0.0 tot 1.0 [-]
    'lsw': [[6,'local_surface_water_code'], #
            [4,'districtwatercode'], # uit DW
            [4,'districtwatercode_exchanges'], #, uit DW, district waar¬mee lsw uitwis-selt
            [30,'name'], #
            [6,'meteostationcode'], # uit METEOSTATION
            [1,'local_surface_water_code_type'], #, type gebied, V(rijaf¬wa¬terend) of P(eilbeheerst)
            [6,'depth_surface_water'], #, t.o.v. N.A.P. [m], wordt niet gebruikt
            [6,'target_level'], #, streefpeil, t.o.v. N.A.P. [m]
            [12,'volume'], #, streefvolume, [m3]
            [6,'maximum_level'], #, t.o.v. N.A.P. [m] , wordt niet gebruikt
            [8,'fraction'], #, 0.0 t/m 1.0 [-], wordt niet gebruikt
            [4,'priority'], #, wordt niet gebruikt
            [8,'alien_waterquality_standard'], #, maximale verhouding gebiedsvreemd water. 0.0 t/m 1.0 [-]
            [8,'salt_waterquality_standard'], # [g/m3]
            [4,'seepage_to_lsw'], #, factor die de verhoging aangeeft van "seepage to local_surface_water" t.o.v. "seepage to plot", wordt niet gebruikt
            [12,'salt_concentration_gw'], # in grondwater, ivm kwel naar opp. water [g/m3]
            [8,'max_discharge_capacity']], # [m3/s] (999999 if indefinite)
    'lswvalue': [[6,'local_surface_water_code'], # uit LSW_IN
                 [15,'time_start'], # (JJJJMMDD.HHMMSS)
                 [15,'time_end'], # (JJJJMMDD.HHMMSS)
                 [12,'concentration'], # [g/m3]
                 [6,'evaporation'], # [mm]
                 [6,'level'], #, t.o.v. N.A.P. [m]
                 [12,'seepage'], #, kwel (negatief) of wegzijging (positief), [m3/s]
                 [12,'discharge'], # uit LSW [m3/s]
                 [9,'area_local_surface_water'], # [m2]
                 [12,'volume'], # [m3]
                 [8,'level_rtc'], # [m]
                 [12,'volume_rtc'], # [m3]
                 [5,'alien_water_ratio']], #, 0.0 tot 1.0 [-]
    'ladvalue': [[6,'level'], #, t.o.v. N.A.P. [m]
                 [6,'local_surface_water_code'], # uit LSW_IN
                 [9,'area'], # [m2]
                 [12,'discharge']], # [m3/s]
    'lswrouting': [[6,'local_surface_water_from_code '], #, t.o.v. N.A.P. [m]
                 [6,'local_surface_water_to_code'], # uit LSW_IN
                 [1,'connection_type'], # [m2]
                 [8,'fraction']], # [m3/s]
    'vadvalue': [[6,'local_surface_water_code'], # uit LSW_IN
                 [12,'volume'], # [m3]
                 [9,'area'], # [m2]
                 [12,'discharge']], # [m3/s]
    'vlvalue': [[6,'local_surface_water_code'], # uit LSW_IN
                [6,'weir_area_code'], # uit WEIRAREA
                [12,'volume_lsw'], # [m3]
                [6,'weir_level'], #, t.o.v. NAP [m]
                [12,'weir_level_slope']], # [-], helling van het wateroppervlak
    'uslsw': [[6,'local_surface_water_code'], # uit LSW_IN
              [3,'usercode']], # uit USER
    'uslswdemand': [[6,'local_surface_water_code'], # uit LWS_IN
                    [15,'time_start'], # (JJJJMMDD.HHMMSS)
                    [3,'usercode'], # uit USER
                    [15,'time_end'], # (JJJJMMDD.HHMMSS)
                    [8,'user_groundwater_demand'], # [m3/s]
                    [8,'user_surfacewater_demand'], # [m3/s]
                    [8,'fraction'], # [-], wordt niet gebruikt
                    [4,'priority'], # [-]
                    [8,'alien_waterquality_standard'], #, maximale verhouding gebiedsvreemd water. 0.0 t/m 1.0 [-]
                    [8,'salt_waterquality_standard'], # [g/m3]
                    [8,'level_rtc']], #, peilopzet [m]
    'lswusex': [[	6,'local_surface_water_code'],# uit LSW_IN
                [3,'usercode'],# uit USER
                [15,'time_start'],# (JJJJMMDD.HHMMSS)
                [4,'priority'],# [-]
                [15,'time_end'],# (JJJJMMDD.HHMMSS)
                [5,'alien_water_ratio'],#, 0.0 tot 1.0 [-]
                [12,'demand'],# [m3/s]
                [12,'demalloc'],# [m3/s]  NIET GEBRUIKT
                [12,'allocation'],# [m3/s]
                [12,'salt_concentration'],# [g/m3]
                [8,'fraction']],#, 0.0 t/m 1.0 [-]
    'weirarea': [[6,'weir_area_code'],
                 [6,'local_surface_water_code']], # uit LSW_IN
    'wavalue': [[6,'weir_area_code'], # uit WEIRAREA_IN
                [15,'time_start'], # (JJJJMMDD.HHMMSS)
                [15,'time_end'], # (JJJJMMDD.HHMMSS)
                [6,'level'], #, t.o.v. NAP [m], peil bij stuw
                [12,'level_slope']], # [-], verhang van het wateroppervlak achter stuw
    'balance': [6,3,1,15,15]+30*[12]  # balansfiles hebben header in bestand
}

# this dict overrules format strings generated from filedescription
mzfile_writefmt = {
        'balance': '%6i %3i %1s %s %s '+' '.join(12*['% 12g'])}
        #'uslswdemand': '%6i %s %-3s %s % 8i %8.4f % 8i % 4i % 8i %8.0f %8.3f'}

def parse_mztime(s):
    return datetime.strptime(s,'%Y%m%d.000000')

def read_mzfile(fn, mztype=None, collength=None, header=None, skiprows=None, **kwargs):
    ''' workhorse function to read all mozart files'''
    global mzfile_description
    
    names=None
    parse_dates = []
    if mztype is not None:
        try:
            fdesc = mzfile_description[mztype]
            if mztype == 'balance':
                collength = fdesc
                header = 0
                if 'parse_dates' not in locals():
                    parse_dates = [3]
                skiprows = [0]
            else:
                collength = [l[0] for l in fdesc]
                names = [l[1] for l in fdesc]
                if 'parse_dates' not in locals() and 'time_start' in names:
                    parse_dates = [names.index('time_start')]
            colspecs = []
            start=0
            for l in collength:
                colspecs += [[start,start+l]]
                start += l+1
        except KeyError:
            colspecs='infer'
        
    return pd.read_fwf(fn,colspecs=colspecs,header=header,names=names,skiprows=skiprows,
                       index_col=None,date_parser=parse_mztime,parse_dates=parse_dates, **kwargs)

def create_fmtstr(desc):
    '''create default format string from file description'''
    fmt = []
    for n,name in desc:
        name = name.lower()
        if 'usercode' in name or 'type' in name:
            fmt += ['%%-%is'%n]
        elif 'code' in name or 'priority' in name:
            fmt += ['%%%ii'%n]
        elif 'time' in name:
            fmt +=['%s']  # assuming converted to string elsewhere
        else:
            fmt +=['%% %i.%ig'%(n,n-3)]
    return ' '.join(fmt)

def to_fixed_width(n,maxwidth,test=False):
    '''Function to round number to fit in width,
       change to scientific if number doesn't fit in width'''
    minus=0
    sign=1
    if str(n)[0] == '-':
        minus=1
        sign=-1
    if n:
        # widthi is breedte minstens nodig 
        widthi = abs(log10(abs(n)))
        if test: print(n)
        if test: print(widthi)
        if modf(widthi)[0] < 1e-9 and n < 1:
            #widthi = int(widthi)+2*sign
            widthi = int(widthi)+minus
        else:
            widthi = int(widthi)+1*sign+minus
        if n < 1: widthi += 2  # for 0 and dot
        if test: print(widthi)
        
        if widthi > maxwidth: #either too large, or too small for width extra one for accuracy
            assert(maxwidth>4+minus)
            fmt = '%%%i.%ie'%(maxwidth-3+minus,maxwidth-6+minus)
            if test: print(fmt)
        else:
            if test: print(maxwidth-widthi-1)
            if abs(n) > 1.:
                n = round(n,max(0,maxwidth-widthi-1))
                fmt = '%%%i.%if'%(maxwidth-minus,max(0,maxwidth-widthi-1))
            else:
                n = round(n,maxwidth-minus-2)
                fmt = '%%%i.%if'%(maxwidth-minus,maxwidth-minus-2)
            if test: print(fmt)
    else:
        fmt = '%%%i.%if'%(maxwidth-minus,maxwidth-2-minus)

    if test: print(fmt)
    sfmt = '%%%is'%maxwidth
    try:
        return sfmt%(fmt%n)
    except ValueError:
        print (n)
        print (maxwidth)
        print (widthi)
        print (fmt)
        print (sfmt)
        raise

def write_mzfile(fn, df, mztype=None, **kwargs):
    '''workhorse function to write all mozart files'''
    global mzfile_writefmt, mzfile_description
    
    df2 = df.copy()
    if mztype is not None:
        try:
            # make datecols strings from different data types
            # separately, b/c also for balance file
            timecols = [c for c in df2.columns if 'time_' in c]
            for tc in timecols:
                if isinstance(df2[tc].iloc[0],datetime):
                    df2.loc[:,tc] = df2[tc].apply(lambda x:x.strftime('%Y%m%d.000000'))
                elif isinstance(df2[tc].iloc[0],float):
                    df2.loc[:,tc] = df2[tc].apply(lambda x:'%15.6f'%x)
                elif isinstance(df2[tc].iloc[0],int):
                    df2.loc[:,tc] = df2[tc].apply(lambda x:'%8i.000000'%x)

            if mztype in mzfile_writefmt:
                fmt = mzfile_writefmt[mztype]
            else:
                #fmt = create_fmtstr(mzfile_description[mztype])
                fmt = []
                for n,col in mzfile_description[mztype]:
                    name = col.lower()
                    if 'usercode' in name or 'type' in name or 'name' in name:
                        fmt += ['%%-%is'%n]
                    elif 'code' in name or 'priority' in name:
                        fmt += ['%%%ii'%n]
                    elif 'time' in name:
                        fmt +=['%15s']  # assuming converted to string elsewhere
                    else:
                        #fmt +=['%% %i.%ig'%(n,n-3)]
                        fmt += ['%%%is'%n]
                        df2.loc[:,col] = df2[col].apply(lambda x:to_fixed_width(x,n))
            
            header = ''
            if mztype == 'balance':
                header = 'Waterbalance or Saltbalance per Local Surfacewater. All items in either m3 or kg per timestep. Positive: influx into lsw, negative: outflux out of lsw.\n'+\
                         'LSWNR  DW  T TIMESTART       TIMEEND         PRECIP       DRAINAGE_SH  DRAINAGE_DP  URBAN_RUNOFF UPSTREAM     FROM_DW      ALLOC_AGRIC  ALLOC_WM     ALLOC_FLUSH  ALLOC_FLUSHR ALLOC_PUBWAT ALLOC_INDUS  ALLOC_GRHOUS EVAPORATION  INFILTR_SH   INFILTR_DP   TODOWNSTREAM TO_DW        STORAGE_DIFF BALANCECHECK DEM_AGRIC    DEM_WM       DEM_FLUSH    DEM_FLUSHRET DEM_PUBWAT   DEM_INDUS    DEM_GRHOUSE  DEM_WMTOTAL  DEM_WM_TODW  ALLOC_WM_DW'

            return np.savetxt(fn, df2.values, fmt=fmt, header=header)

        except:
            print('Function not yet implemented for this filetype...')
            raise


