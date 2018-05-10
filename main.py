import numpy as np

from bokeh.io import curdoc
from bokeh.layouts import row, widgetbox
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import Slider, TextInput, Button
from bokeh.models.widgets.inputs import Select
from bokeh.plotting import figure
from bokeh.themes.theme import Theme

import os,sys,glob,gc
import time,datetime,calendar
import numpy as np
import pandas as pd
import cPickle as pickle
import math,re
import xgboost as xgb

tn = 3016
ata = 36
pid = 7
unit = 'days'


x_values = [100,325]
y_values = ["Actual", "Predicted"]
source = ColumnDataSource(data = dict(x=x_values, y=y_values))

p = figure(y_range = y_values, plot_height=250, title='Lifetime', toolbar_location=None, tools="")
p.hbar(y='y', right='x', height=0.9, source = source)
p.ygrid.grid_line_color = None
p.x_range.start = 0


# Set up widgets
units = Select(title="Units", value="days", options=["days", "cycles", "hours"])
aircraft = Select(title="Aircraft", value="3016", options=["3016", "3601"])
ata_code = Select(title = "ATA CODE", value = '36', options = ['36', '37'])
pid_val = Select(title = 'PID', value = '7', options = ['7', '6'])
button1 = Button(label="Actual", button_type="success")
button2 = Button(label="Predict", button_type="success")

controls = [units, aircraft, ata_code, pid_val, button1, button2]
sliders=[]
#startR = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
startRange = [0.0,2.0,0.0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1]
endRange= [-999,10000,-999,999,7,64,14,12,23,35,-999,12,999,19,6,7,4,-999,-999,-999,-999,-999,-999,348.0,1681.0,497.0,2516.0,825.0,1034.0]
for i in range(1,30):
    varname = 'varname{}'.format(i)
    slider = Slider(title = varname, value=1, start = 1, end=1000, step=1)
    controls.append(slider)
    sliders.append(slider)

def do_whatIF_Actual():
    '''
    tn = 3016
    ata = 36
    pid = 7
    unit = 'days'
    '''

    newval = {}
    for i in range(1,30):
        newval[i] = np.random.randint(1,10000)

    tn = int(aircraft.value)
    ata = int(ata_code.value)
    pid = int(pid_val.value)
    unit = units.value
    print(tn,ata,pid,unit)
    path_whatif = '/home/ec2-user/LT_MODEL/'
    pnmask_final = pd.read_json(path_whatif + 'pnmask_final.json',orient='split')
    vardic = pd.read_excel(path_whatif + 'VARIABLE_DICTIONARY.xlsx')
    df_PARTUI = pd.read_csv(path_whatif + 'PART_DICTIONARY.csv')
    df_PARTUI.insert(0,'ATA_PID',df_PARTUI[['ATA','PID']].apply(lambda x:'{}_{:03d}'.format(x[0],x[1]),axis=1))
    pn = pnmask_final.loc[pnmask_final.ATA_PID.str.contains('{}_{:03d}'.format(ata,pid))].PN.str.replace('|','_')
    pname = df_PARTUI.loc[df_PARTUI.ATA_PID=='{}_{:03d}'.format(ata,pid)].PART_NAME.values[0]

    ''' Sub-modules '''
    def xgb_pred(xte,in_model, feature_names):
        '''
        XGBoost Classifier
        '''
        xgte = xgb.DMatrix(xte.values, feature_names=feature_names)
        y_pred = in_model.predict(xgte) #pred_leaf=True/False
        return y_pred

    ''' Main function '''
    start_time = time.time()
    print ('input/{}_life_{}_selected.json'.format(pn.values[0],unit))
    # READ INPUT
    dfin = pd.read_json(path_whatif  + 'input/{}_life_{}_selected.json'.format(pn.values[0],unit),orient='split')
    varlist = pd.read_json(path_whatif + 'model/{}_life_{}/selected_variables.json'.format(pn.values[0],unit),
                           orient='split').SELECTED_VAR.tolist()
    dfin_curr = dfin.loc[(dfin['to']=='2017-11-30')&(dfin['TAIL NUMBER']==tn)].drop_duplicates().head(1)
    sn = dfin_curr.SN.values[0]

    dfin_curr = dfin_curr[varlist].copy()
    dfin = dfin[varlist].copy()

    transvar = pd.DataFrame(dfin_curr.columns.tolist(),columns=['VN'])
    for iv in range(0,len(vardic)):
        idx = transvar['VN'].str.extract(vardic.iloc[iv].REGX,expand=False)
        tmp = idx.dropna()
        if len(tmp)==0:
            continue
        if len(tmp.shape)==1:
            tmp = tmp.apply(lambda x:vardic.iloc[iv].VARIABLE_NAME.format(x).upper())
            transvar.loc[idx.notnull(),'VN'] = tmp
        if len(tmp.shape)==2:
            tmp = tmp.apply(lambda x:vardic.iloc[iv].VARIABLE_NAME.format(x[0],x[1]).upper(),axis=1)
            transvar.loc[idx.notnull().all(axis=1),'VN'] = tmp

    dfin.columns = transvar.VN.tolist()
    dfin_curr_x = dfin_curr.copy()

    changevar = {1:'DAYS SINCE LAST 72HR INSP',2:'DAYS SINCE LAST LUBRICATION',3:'DAYS SINCE LAST NAV UPDATE',
                 4:'DAYS SINCE LAST SEMI PREP INSP',5:'NUM 72HR INSP PAST 15D',6:'NUM 72HR INSP PAST 180D',
                 7:'NUM 72HR INSP PAST 30D',8:'NUM 72HR INSP PAST 360D',9:'NUM 72HR INSP PAST 60D',
                 10:'NUM 72HR INSP PAST 90D',11:'NUM LUBRICATION PAST 15D',12:'NUM LUBRICATION PAST 180D',
                 13:'NUM LUBRICATION PAST 30D',14:'NUM LUBRICATION PAST 360D',15:'NUM LUBRICATION PAST 60D',
                 16:'NUM LUBRICATION PAST 90D',17:'NUM PART REPLACEMENT',18:'NUM SEMI PREP INSP PAST 15D',
                 19:'NUM SEMI PREP INSP PAST 180D',20:'NUM SEMI PREP INSP PAST 30D',21:'NUM SEMI PREP INSP PAST 360D',
                 22:'NUM SEMI PREP INSP PAST 60D',23:'NUM SEMI PREP INSP PAST 90D',24:'UNSCHED MX PAST 15D GLOBAL',
                 25:'UNSCHED MX PAST 180D GLOBAL',26:'UNSCHED MX PAST 30D GLOBAL',27:'UNSCHED MX PAST 360D GLOBAL',
                 28:'UNSCHED MX PAST 60D GLOBAL',29:'UNSCHED MX PAST 90D GLOBAL'}
    changelim = {}
    for k in changevar.keys():
        try:
            changelim[k] = [dfin[changevar[k]].min(),dfin[changevar[k]].max()]
        except:
            tmp = ''

    #make the predictions
    pred_new = 0.0
    pred_org = 0.0
    mlist = glob.glob('LT_MODEL/model/{}_life_{}/XGB.model.bID*.dat'.format(pn.values[0],unit))
    for k in changevar.keys():
        tmp = transvar.loc[transvar.VN==changevar[k]]
        if len(tmp)==0:
            continue
        vcol = dfin_curr_x.columns[tmp.index[0]]
        if newval[k]<changelim[k][0]:
            dfin_curr_x[vcol] = changelim[k][0]
        elif newval[k]>changelim[k][1]:
            dfin_curr_x[vcol] = changelim[k][1]
        else:
            dfin_curr_x[vcol] = newval[k]

    for model_f in mlist:
        loaded_model = pickle.load(open(model_f,'rb'))
        pred_new += xgb_pred(dfin_curr_x,loaded_model,varlist)
        pred_org += xgb_pred(dfin_curr,loaded_model,varlist)

    print('PART_NAME: {}'.format(pname))
    print('ATA_PID: {}_{:03d}, SERIAL_NUMBER: {}'.format(ata,pid,sn))
    for k in changevar.keys():
        tmp = transvar.loc[transvar.VN==changevar[k]]
        if len(tmp)==0:
            print('varname{}[no change]: {}'.format(k,changevar[k]))
            sliders[k-1].start = -1
            sliders[k-1].value = 1
            sliders[k-1].end = 1
            continue
        vcol = dfin_curr_x.columns[tmp.index[0]]
        vmn,vmx = changelim[k][0],changelim[k][1]
        org_val = dfin_curr[vcol].values[0]
        print('Varname{}[range {}~{}]: {} from {} to {}'.format(k,vmn,vmx,changevar[k],org_val,newval[k]))
        sliders[k-1].value = org_val

    print('Results in lifetime {:1.0f} -> {:1.0f} {}'.format((pred_org/len(mlist))[0],(pred_new/len(mlist))[0],unit))
    print('ELAPSED TIME {:1.0f} min'.format((time.time()-start_time)/60))
    
    x_values[0] = (pred_new/len(mlist))[0]
    source.data = dict(x=x_values, y=y_values)
    
    
def do_whatIF_Predict():

    newval = {}
    for i in range(1,30):
        newval[i] = np.random.randint(1,10000)

    tn = int(aircraft.value)
    ata = int(ata_code.value)
    pid = int(pid_val.value)
    unit = units.value
    print(tn,ata,pid,unit)
    path_whatif = '/home/ec2-user/LT_MODEL/'
    pnmask_final = pd.read_json(path_whatif + 'pnmask_final.json',orient='split')
    vardic = pd.read_excel(path_whatif + 'VARIABLE_DICTIONARY.xlsx')
    df_PARTUI = pd.read_csv(path_whatif + 'PART_DICTIONARY.csv')
    df_PARTUI.insert(0,'ATA_PID',df_PARTUI[['ATA','PID']].apply(lambda x:'{}_{:03d}'.format(x[0],x[1]),axis=1))
    pn = pnmask_final.loc[pnmask_final.ATA_PID.str.contains('{}_{:03d}'.format(ata,pid))].PN.str.replace('|','_')
    pname = df_PARTUI.loc[df_PARTUI.ATA_PID=='{}_{:03d}'.format(ata,pid)].PART_NAME.values[0]

    def xgb_pred(xte,in_model, feature_names):
        xgte = xgb.DMatrix(xte.values, feature_names=feature_names)
        y_pred = in_model.predict(xgte) #pred_leaf=True/False
        return y_pred

    start_time = time.time()
    print ('input/{}_life_{}_selected.json'.format(pn.values[0],unit))
    # READ INPUT
    dfin = pd.read_json(path_whatif  + 'input/{}_life_{}_selected.json'.format(pn.values[0],unit),orient='split')
    varlist = pd.read_json(path_whatif + 'model/{}_life_{}/selected_variables.json'.format(pn.values[0],unit),
                           orient='split').SELECTED_VAR.tolist()
    dfin_curr = dfin.loc[(dfin['to']=='2017-11-30')&(dfin['TAIL NUMBER']==tn)].drop_duplicates().head(1)
    sn = dfin_curr.SN.values[0]

    dfin_curr = dfin_curr[varlist].copy()
    dfin = dfin[varlist].copy()

    transvar = pd.DataFrame(dfin_curr.columns.tolist(),columns=['VN'])
    for iv in range(0,len(vardic)):
        idx = transvar['VN'].str.extract(vardic.iloc[iv].REGX,expand=False)
        tmp = idx.dropna()
        if len(tmp)==0:
            continue
        if len(tmp.shape)==1:
            tmp = tmp.apply(lambda x:vardic.iloc[iv].VARIABLE_NAME.format(x).upper())
            transvar.loc[idx.notnull(),'VN'] = tmp
        if len(tmp.shape)==2:
            tmp = tmp.apply(lambda x:vardic.iloc[iv].VARIABLE_NAME.format(x[0],x[1]).upper(),axis=1)
            transvar.loc[idx.notnull().all(axis=1),'VN'] = tmp

    dfin.columns = transvar.VN.tolist()
    dfin_curr_x = dfin_curr.copy()

    changevar = {1:'DAYS SINCE LAST 72HR INSP',2:'DAYS SINCE LAST LUBRICATION',3:'DAYS SINCE LAST NAV UPDATE',
                 4:'DAYS SINCE LAST SEMI PREP INSP',5:'NUM 72HR INSP PAST 15D',6:'NUM 72HR INSP PAST 180D',
                 7:'NUM 72HR INSP PAST 30D',8:'NUM 72HR INSP PAST 360D',9:'NUM 72HR INSP PAST 60D',
                 10:'NUM 72HR INSP PAST 90D',11:'NUM LUBRICATION PAST 15D',12:'NUM LUBRICATION PAST 180D',
                 13:'NUM LUBRICATION PAST 30D',14:'NUM LUBRICATION PAST 360D',15:'NUM LUBRICATION PAST 60D',
                 16:'NUM LUBRICATION PAST 90D',17:'NUM PART REPLACEMENT',18:'NUM SEMI PREP INSP PAST 15D',
                 19:'NUM SEMI PREP INSP PAST 180D',20:'NUM SEMI PREP INSP PAST 30D',21:'NUM SEMI PREP INSP PAST 360D',
                 22:'NUM SEMI PREP INSP PAST 60D',23:'NUM SEMI PREP INSP PAST 90D',24:'UNSCHED MX PAST 15D GLOBAL',
                 25:'UNSCHED MX PAST 180D GLOBAL',26:'UNSCHED MX PAST 30D GLOBAL',27:'UNSCHED MX PAST 360D GLOBAL',
                 28:'UNSCHED MX PAST 60D GLOBAL',29:'UNSCHED MX PAST 90D GLOBAL'}
    changelim = {}
    for k in changevar.keys():
        try:
            changelim[k] = [dfin[changevar[k]].min(),dfin[changevar[k]].max()]
        except:
            tmp = ''

    #make the predictions
    pred_new = 0.0
    pred_org = 0.0
    mlist = glob.glob('LT_MODEL/model/{}_life_{}/XGB.model.bID*.dat'.format(pn.values[0],unit))
    for k in changevar.keys():
        tmp = transvar.loc[transvar.VN==changevar[k]]
        if len(tmp)==0:
            continue
        vcol = dfin_curr_x.columns[tmp.index[0]]
        if newval[k]<changelim[k][0]:
            dfin_curr_x[vcol] = changelim[k][0]
        elif newval[k]>changelim[k][1]:
            dfin_curr_x[vcol] = changelim[k][1]
        else:
            dfin_curr_x[vcol] = newval[k]

    for model_f in mlist:
        loaded_model = pickle.load(open(model_f,'rb'))
        pred_new += xgb_pred(dfin_curr_x,loaded_model,varlist)
        pred_org += xgb_pred(dfin_curr,loaded_model,varlist)

    print('PART_NAME: {}'.format(pname))
    print('ATA_PID: {}_{:03d}, SERIAL_NUMBER: {}'.format(ata,pid,sn))
    for k in changevar.keys():
        tmp = transvar.loc[transvar.VN==changevar[k]]
        if len(tmp)==0:
            print('varname{}[no change]: {}'.format(k,changevar[k]))
            sliders[k-1].start = -1
            sliders[k-1].value = 1
            sliders[k-1].end = 1
            continue
        vcol = dfin_curr_x.columns[tmp.index[0]]
        vmn,vmx = changelim[k][0],changelim[k][1]
        org_val = dfin_curr[vcol].values[0]
        print('Varname{}[range {}~{}]: {} from {} to {}'.format(k,vmn,vmx,changevar[k],org_val,newval[k]))
        sliders[k-1].value = org_val

    print('Results in lifetime {:1.0f} -> {:1.0f} {}'.format((pred_org/len(mlist))[0],(pred_new/len(mlist))[0],unit))
    print('ELAPSED TIME {:1.0f} min'.format((time.time()-start_time)/60))
    
    x_values[1] = (pred_new/len(mlist))[0]
    source.data = dict(x=x_values, y=y_values)
    
    
button1.on_click(do_whatIF_Actual)
button2.on_click(do_whatIF_Predict)

# Set up layouts and add to document
inputs = widgetbox(controls)
#curdoc().theme = Theme(filename= 'theme.yaml')
curdoc().theme = Theme(filename=os.path.join(os.path.dirname(__file__), 'theme.yaml'))
curdoc().add_root(row(inputs,p, width=800))
curdoc().title = "Sliders"
