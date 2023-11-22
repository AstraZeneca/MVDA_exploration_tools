#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 11:13:32 2020

@author: mats_j
"""

import os
import numpy as np
from mpl_toolkits.mplot3d import Axes3D #Needed for 3D plotting
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import sklearn.model_selection
import bokeh.plotting as bplt



from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression

__version__ = '0.1.8'


def sumsq(A):
    squares = np.asarray(A, dtype=np.float64)**2
    return np.sum(squares, axis=0)

def reset_too_high_max_comp(max_comp, X):
    """ Should be degrees_of_freedom(X), -1 for mean centering and -1 
        for not using full set of data in cross validation sections"""
    valid = min(X.shape[0], X.shape[1])-3   
    if max_comp > valid:
        max_comp = valid
        print('Max number of components reduced to:', max_comp)
    return max_comp

def R2(M, X, y_ref0):
    trace = False
    
    y_ref = np.atleast_1d(np.squeeze(y_ref0))
    y_pred = np.atleast_1d(np.squeeze(M.predict(X)))
    
    if trace:
        print()
        print('R2 X', type(X), X.shape)
        print('R2 y_ref', type(y_ref), y_ref.shape)
        print('R2 y_pred', type(y_pred), y_pred.shape)
    """ Get the sum of the sum of squares from each y-variable """
    u = np.sum(sumsq(y_ref - y_pred)) 
    v = np.sum(sumsq(y_ref - y_ref.mean(axis=0)))
    if trace:
        print('R2 u', type(u), u.shape, u)
        print('R2 v', type(v), v.shape, v)
    R2 = 1-u/v
    return R2

def R2X_calc(SSX):
    R2X = np.zeros(len(SSX)-1)
    for comp in range(1,len(SSX)):
        R2X[comp-1] = SSX[comp-1]/SSX[0] - SSX[comp]/SSX[0]
    return R2X

def cumR2X_calc(SSX):
    cumR2X = np.zeros(len(SSX)-1)
    for comp in range(1, len(SSX)):
        cumR2X[comp-1] = 1 - SSX[comp]/SSX[0]
    return cumR2X


def PLS_R2_calc(X, Y, max_comp=10, is_UV_scale=False):
    
    if Y.ndim == 1: # make column vector
        Y = Y[:, np.newaxis]
    max_comp = reset_too_high_max_comp(max_comp, X)
    cumulative_R2_values = np.zeros((max_comp))
    for comps in range(max_comp):
        pls1comp_i = PLS_model(n_components=comps+1, scale=is_UV_scale)
        pls1comp_i.fit(X, Y)
        cumulative_R2_values[comps] = R2(pls1comp_i, X, Y)
    return cumulative_R2_values
  

def Q2(CV_pred, y_ref0):
    trace = False
    
    y_pred = np.atleast_1d(np.squeeze(CV_pred))
    y_ref = np.atleast_1d(np.squeeze(y_ref0))
    """ Get the sum of the sum of squares from each y-variable """
    PRESS = np.sum(sumsq(y_ref - y_pred)) 
    SSY = np.sum(sumsq(y_ref - y_ref.mean(axis=0)))
    if trace:
        print()
        print('Q2 CV_pred', type(CV_pred), CV_pred.ndim, CV_pred.shape)
        print('Q2 y_ref', type(y_ref), y_ref.ndim, y_ref.shape)# , y_ref)
        print('Q2 y_pred', type(y_pred), y_pred.ndim, y_pred.shape)#, y_pred)
        print('Q2 y_ref.mean(axis=0)', y_ref.mean(axis=0))
        print('Q2 y_pred.mean(axis=0)', y_pred.mean(axis=0))
        print('Q2 PRESS', PRESS, type(PRESS))
        print('Q2 SSY',SSY, type(SSY))
    Q2_calc = 1-PRESS/SSY
    if trace:
        print('Q2 Q2_calc', Q2_calc)
    return Q2_calc 
    
    
def PLS_cross_val(X, Y, max_comp=10, is_UV_scale=False, CV_sections=7, shuffle=False, random_state=None):

    max_comp = reset_too_high_max_comp(max_comp, X)
    
    if X.shape[0] < CV_sections:
        CV_sections = X.shape[0]
    
    cumulative_Q2_values = np.zeros((max_comp))

    for comps in range(max_comp):
        pls_CV_model = PLS_model(n_components=comps+1, scale=is_UV_scale)
        CV_selections = sklearn.model_selection.KFold(n_splits=CV_sections, 
                                                      shuffle=shuffle, 
                                                      random_state=random_state)
        """
        # cross_val_predict returns an array of the same size as `y` where each entry
        # is a prediction obtained by cross validation:
        """
        CV_predict = sklearn.model_selection.cross_val_predict(pls_CV_model, X, Y, cv=CV_selections)
        cumulative_Q2_values[comps] = Q2(CV_predict, Y)
#    for prn_comps in range(1,max_comp): 
#        print(prn_comps, 'Simca_Q2', Simca_Q2[prn_comps,:])
    return cumulative_Q2_values


def evalPLS_Q2(X, Y, max_comp=10, is_UV_scale=False, 
               CV_sections=7, shuffle=False, random_state=None, 
               is_plot=True, plt_fname='', plt_dir='', plt_title=''):
    trace = False
    Xa = np.asarray(X, dtype=np.float64)
    Ya = np.asarray(Y, dtype=np.float64)
    max_comp = reset_too_high_max_comp(max_comp, Xa)
    R2_calc = PLS_R2_calc(Xa, Ya, max_comp=max_comp, is_UV_scale=is_UV_scale)
    if trace:
        print('evalPLS_Q2 R2', R2_calc.shape)
        print(R2_calc)
    Q2_calc = PLS_cross_val(Xa, Ya,max_comp=max_comp, is_UV_scale=is_UV_scale, 
                            CV_sections=CV_sections, shuffle=shuffle, 
                            random_state=random_state)
    if trace:
        print('evalPLS_Q2 Q2', Q2_calc.shape)
        print(Q2_calc)

    if is_plot:
        fig1 = Fig()
        fig1.R2_Q2_bars(R2_calc, Q2_calc) #, edgecolor='black')
        fig1.legend()
        if plt_title:
            fig1.title(plt_title)
        if plt_fname:
            fig1.save(plt_dir, plt_fname)
            
    return R2_calc, Q2_calc


def mk_oneDimArray(values):
    if (values.size == 1) and (values.ndim == 2):
        valArr = np.asarray(values[0,0], dtype=np.float64)   
    elif values.ndim > 1:
        valArr = np.squeeze(np.asarray(values))
    else:
        valArr = np.asarray(values, dtype=np.float64)
    return valArr

def is_number(string):
    try:
        value = float(string)
        return True
    except ValueError:
        return False


def save_zoomable_plot(df, df_name, attrib={}, targetDir=''):
    trace = False
    txt_col_values = list(df.index)
    numeric_col_values = np.asarray([])
    try:
        numeric_col_values = np.asarray(list(map(float, txt_col_values)))
    except ValueError:
        pass
    if numeric_col_values.size:
        zplt = Fig_zoom()
        x = numeric_col_values
    else:
        zplt = Fig_zoom(x_range = txt_col_values)
        x = txt_col_values
    pen_cycle = zplt.get_pen_cycle()
    
    for pen_number, col in enumerate(df.columns):
        y = np.asarray(df[col])
        lbl = col
        color = pen_cycle[pen_number % 10]
        zplt.plot(x, y, legend_label=lbl, line_color=color)
    if attrib: 
        if 'xlabel' in attrib: 
            zplt.xlabel(attrib['xlabel'])
        if 'ylabel' in attrib:
            zplt.ylabel(attrib['ylabel'])
        if 'title' in attrib:
            zplt.title(attrib['title'])
        if 'is_inverted' in attrib:
            zplt.invert_xaxis()
        if 'legend_pos' in attrib:
            zplt.p.add_layout(zplt.p.legend[0], attrib['legend_pos']) #'above')
    if trace:
        zplt.show()
    output_fname = zplt.save(targetDir, df_name+'.ext', title_suffix='')
    print('Plot was saved at:', output_fname )
        
        

class Fig():
    def __init__(self, nrows=1, ncols=1, sharex=False, linewidth=1.0, markersize=10, plot_mode='2D', projection=None, **kwargs):
        self.nrows = nrows
        self.ncols = ncols
        self.plot_mode = plot_mode
#        plt.style.use('dark_background')
#TODO Check set_facecolor with 'dark_background'        
        if plot_mode == '3D':
            self.figure, self.axes = plt.subplots(nrows, ncols, squeeze=False, 
                                                  sharex=sharex, 
                                                  subplot_kw=dict(projection='3d'),
                                                  **kwargs)
        else: # 2D
            self.figure, self.axes = plt.subplots(nrows, ncols, squeeze=False, 
                                                  sharex=sharex, 
                                                  subplot_kw=dict(projection=projection),
                                                  **kwargs)
        self.Savefig_dpi = 300

        self.default_kwargs = {}
        self.default_kwargs['markersize'] = markersize
        self.default_kwargs['markeredgecolor'] = 'black'
        self.default_kwargs['linewidth'] = linewidth
        
    def set_default_kwargs(self, kwargs, exclude_args=[]):
        for key in self.default_kwargs:
            if not key in exclude_args:
                if (not key in kwargs):
                    kwargs[key] = self.default_kwargs[key] 
                    
    def close(self):
        plt.close(self.figure)
        
    def show(self):
        """Shows plots of all defined plot objects"""
        plt.show()
        
    def legend(self, labels='', row=0, col=0, **kwargs):
        if labels:
            self.axes[row, col].legend(labels, **kwargs)
        else:
            self.axes[row, col].legend(**kwargs)
            
    def grid(self, setting=True, row=0, col=0, **kwargs):
        self.axes[row, col].grid(setting, **kwargs)
        
    def invert_xaxis(self, row=0, col=0):
        self.axes[row, col].invert_xaxis()
            
    def invert_yaxis(self, row=0, col=0):
        self.axes[row, col].invert_yaxis()
            
    def xlabel(self, txt, row=0, col=0, **kwargs):
        self.axes[row, col].set_xlabel(txt, **kwargs)
        
    def ylabel(self, txt, row=0, col=0, **kwargs):
        self.axes[row, col].set_ylabel(txt, **kwargs) 
        
    def zlabel(self, txt, row=0, col=0, **kwargs):
        self.axes[row, col].set_zlabel(txt, **kwargs)
        
    def title(self, txt, row=0, col=0, **kwargs):
        self.axes[row, col].set_title(txt, **kwargs) 
        
    def main_title(self, txt):
        self.figure.suptitle(txt)
        
    def tight_layout(self):        
        self.figure.tight_layout()
                    
    def R2_Q2_bars(self, R2_in, Q2_in, width=0.35, row=0, col=0, **kwargs):
        self.set_default_kwargs(kwargs, exclude_args=['markersize', 'markeredgecolor'])
        R2 = np.atleast_1d(np.squeeze(np.asarray(R2_in, dtype=np.float64)))
        Q2 = np.atleast_1d(np.squeeze(np.asarray(Q2_in, dtype=np.float64)))
        bar_ix = np.arange(R2.shape[0])+1
#        cmap = plt.get_cmap("tab10")
#        R2_color = cmap(2)
#        Q2_color = cmap(0)
        viridis_colors = matplotlib.cm.viridis(np.linspace(0,1,10))
        R2_color = viridis_colors[8]
        Q2_color = viridis_colors[2]
        self.axes[row, col].bar(bar_ix-width/2, R2, width=width, color=R2_color, label='R2', **kwargs )
        if Q2.shape == R2.shape:
            self.axes[row, col].bar(bar_ix+width/2, Q2, width=width, color=Q2_color, label='Q2', **kwargs )
        self.axes[row, col].set_xlabel('PLS-component')
        self.axes[row, col].set_ylabel('cumulative fractions')
        
        
    def vert_line(self, x_positions, row=0, col=0, **kwargs):
        self.set_default_kwargs(kwargs)
        if np.isscalar(x_positions):
            self.axes[row, col].axvline(x_positions, linestyle='--', **kwargs )
        else:
            cmap = plt.get_cmap("tab10")
            for ci, x_pos in enumerate(x_positions):
                self.axes[row, col].axvline(x_pos, color=cmap(ci), linestyle='--', **kwargs )
                
    def horiz_line(self, y_positions, row=0, col=0, **kwargs):
        self.set_default_kwargs(kwargs)
        if np.isscalar(y_positions):
            self.axes[row, col].axhline(y_positions, linestyle='--', **kwargs )
        else:
            cmap = plt.get_cmap("tab10")
            for ci, x_pos in enumerate(y_positions):
                self.axes[row, col].axhline(x_pos, color=cmap(ci), linestyle='--', **kwargs )
                

    def scatter_plot(self, v1, v2=np.asarray([]), v3=np.asarray([]), row=0, col=0, pointIDs=[], fontsize:float=10.0, **kwargs):                        
        self.set_default_kwargs(kwargs)
        v1a0 = np.asarray(v1)
        v2a0 = np.asarray(v2)
        v3a0 = np.asarray(v3) 
                        
        if v3a0.size:
            v1a = np.atleast_1d(np.squeeze(np.asarray(v1a0)))
            v2a = np.atleast_1d(np.squeeze(np.asarray(v2a0)))
            v3a = np.atleast_1d(np.squeeze(np.asarray(v3a0)))
            self.axes[row, col].plot( v1a,  v2a,  v3a, 'o', **kwargs) 
        elif v2a0.size:
            v1a = np.atleast_1d(np.squeeze(np.asarray(v1a0)))
            v2a = np.atleast_1d(np.squeeze(np.asarray(v2a0)))
            self.axes[row, col].plot(v1a, v2a, 'o', **kwargs )
            txt_size = float(fontsize) 
            """
            10 is default size,
            txt_size has to be a float otherwise it has no effect in the annotation"""
            
            txt_offset = 1.2*np.ceil(kwargs['markersize']/2)
            pointIDs_lst = list(pointIDs)
            if pointIDs_lst:
                for i, txt in enumerate(pointIDs_lst):
                    self.axes[row, col].annotate(txt, (v1a[i], v2a[i]), textcoords='offset points', xytext=(txt_offset,txt_offset))
            elif (v1a.ndim == 1) and (v2a.ndim == 1) and txt_size > 0:
                for i in range(v1a.shape[0]):
                    if np.any(np.isclose(v1a[:i], v1a[i]*np.ones_like(v1a[:i]), atol=0.1)) and np.any(np.isclose(v2a[:i], v2a[i]*np.ones_like(v2a[:i]), atol=0.1)):
                        self.axes[row, col].annotate(str(i), (v1a[i], v2a[i]), textcoords='offset points', xytext=(txt_offset,-txt_offset-txt_size), fontsize=txt_size)
                    else:
                        self.axes[row, col].annotate(str(i), (v1a[i], v2a[i]), textcoords='offset points', xytext=(txt_offset,txt_offset), fontsize=txt_size) 
            else:
                """Could not find a way annotate with more than 1D vectors"""
                pass
                
        else:
            self.axes[row, col].plot(v1, 'o', **kwargs)
            
            
    def plot(self, v1, v2=np.asarray([]), v3=np.asarray([]), row=0, col=0, **kwargs):
        v1a = np.asarray(v1)
        v2a = np.asarray(v2)
        v3a = np.asarray(v3) 
        
        self.set_default_kwargs(kwargs)
        if v3a.size:
            self.axes[row, col].plot(v1a, v2a, v3a, **kwargs) 
        elif v2a.size:
            self.axes[row, col].plot(v1a, v2a, **kwargs)
        else:
            self.axes[row, col].plot(v1a, **kwargs)

            
    def bar(self, v1, v2=np.asarray([]), v3=np.asarray([]), row=0, col=0, **kwargs):
        """bar only works with x as indices"""
        v1a = np.atleast_1d(np.squeeze(np.asarray(v1))) 
        v2a = np.atleast_1d(np.squeeze(np.asarray(v2)))
        v3a = np.atleast_1d(np.squeeze(np.asarray(v3)))
        self.set_default_kwargs(kwargs, exclude_args=['markersize', 'markeredgecolor'])       
        if v3a.size:
            print('2D barplot not inplemented')
        elif v2.size:
            self.axes[row, col].bar(v1a, v2a, **kwargs)
        else:
            self.axes[row, col].bar(range(len(v1a)), v1a, **kwargs)
        if isinstance(v1a[0], str):
            for tick in self.axes[row, col].get_xticklabels():
                tick.set_rotation(45)
        


    def make_dir_if_absent(self, d):
        if not os.path.exists(d):
            os.makedirs(d)    
        
    def save(self, Output_FileDir, FileName, title_suffix='', is_main_title=True, dpi=300):
        basename_ext = os.path.basename(FileName)
        if is_main_title:
            self.main_title(basename_ext+' '+title_suffix)
        if len(os.path.splitext(basename_ext)[1]) <= 4:
            basename_no_ext = os.path.splitext(basename_ext)[0]
            ext = os.path.splitext(basename_ext)[1]
        else: # There is point in the filename but longer extensions than 4 chars are less likely 
           basename_no_ext = basename_ext
           ext = '.png'
        if Output_FileDir:
            self.make_dir_if_absent(Output_FileDir)
        fname = os.path.join(Output_FileDir, basename_no_ext+'_'+title_suffix+ext)
        self.figure.savefig( fname, dpi=dpi)

        
        
class Fig_zoom():
    def __init__(self, h=600, w=800, **kwargs):
        self.p = bplt.figure(width=w, height=h, **kwargs)
        self.p.xgrid.visible = False
        self.p.ygrid.visible = False
        self.is_title = False
        self.mpl_pen_cycle = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'] # matplotlib viridis color sequence 
        
        
    def plot(self, v1, v2, **kwargs):
        
        def squeeze_mat( a ):
            return np.squeeze( np.asarray( a ))

        is_finite = np.isfinite(v2).all()
        if not is_finite:
            print('Cannot plot as all values are not finite')
        else:
            if not isinstance(v1, list):
                if v1.ndim > 1:
                    v1 = squeeze_mat(v1)           
            if v2.ndim > 1:
                v2 = squeeze_mat(v2)
            
            if not isinstance(v1, list):                
                if v1.ndim != 1 or v2.ndim != 1:
                    errMsg = 'Input vectors needs to be one dimensional, plot will be empty'
                    print('Fig_zoom,', errMsg)
                    self.title(errMsg)
                else:
                    self.p.line(v1, v2, **kwargs)
            else:
                if v2.ndim != 1:
                    errMsg = 'Input vectors needs to be one dimensional, plot will be empty'
                    print('Fig_zoom,', errMsg)
                    self.title(errMsg)
                else:
                    self.p.line(v1, v2, **kwargs)
            


    def title(self, txt):
        self.p.title = txt
        self.is_title = True
    
    
    def xlabel(self, txt):
        self.p.xaxis.axis_label = str(txt)
        
        
    def ylabel(self, txt):
        self.p.yaxis.axis_label = str(txt)
        
      
    def invert_xaxis(self):
        self.p.x_range.flipped = True
        
    def invert_yaxis(self):
        self.p.y_range.flipped = True

        
    def grid(self, boolVal=True):
        self.p.xgrid.visible = boolVal
        self.p.ygrid.visible = boolVal

        
    def get_pen_cycle(self):
        return self.mpl_pen_cycle
        
        
    def save(self, Output_FileDir, basename_ext, title_suffix=''):
        if self.is_title:
            self.title(title_suffix +' '+self.p.title.text)
        elif title_suffix:
            self.title(title_suffix)
        basename_no_ext = os.path.splitext(basename_ext)[0]
        fname = os.path.join(Output_FileDir, basename_no_ext+'_'+title_suffix+'.html')
        output_file_name = os.path.join(Output_FileDir, fname)
        target_filename = output_file_name

        if os.path.isfile(output_file_name):
            print('A previous file with the same name was overwritten')
            
        bplt.output_file(target_filename)
        bplt.save(self.p, title=title_suffix)
        return target_filename
    
    def show(self):
        bplt.show(self.p)


def PCA_by_randomizedSVD(X, components):
    '''A faster PCA'''
    trace = True
    U, S, V = sklearn.decomposition.randomized_svd(X, components)        
    T = U*S
    P = V
    if trace:
        print()
        print('randomized SVD U S shapes', U.shape, S.shape )
        print('PCA T.shape', T.shape)
    return T, P, S


class PCA_model(PCA):
    def __init__(self, n_components=None, is_center=True, is_scale=False, force_np_type_out=False):
        super().__init__(n_components=n_components)
        
        self.n_components = n_components
        self.is_center = is_center
        self.is_scale = is_scale
        self.Xavg_ = np.asarray([])
        self.Xws_  = np.asarray([])
        self.X_model = np.asarray([])
        self.X_wkset = np.asarray([])
        self.SSX_ = np.asarray([])
        self.is_fitted = False
        self.force_np_type_out = force_np_type_out

    
    def center_scale_x(self, X, center=True, scale=False):
        """ Center X and scale if the scale parameter==True
        Returns
        -------
            X, x_mean, x_std
        """
        # center
        if center:
            x_mean = X.mean(axis=0)
            Xcs = X-x_mean
        else:
            Xcs = X
            x_mean = np.zeros(Xcs.shape[1])
        # scale
        if scale:
            x_std = Xcs.std(axis=0, ddof=1)
            x_std[x_std == 0.0] = 1.0
            Xcs /= x_std
        else:
            x_std = np.ones(Xcs.shape[1])
        return Xcs, x_mean, x_std
    

    def fit(self, X0):
        
        self.X_model = X0
        self.X_wkset, self.Xavg_, self.Xws_ = self.center_scale_x(X0, center=self.is_center, scale=self.is_scale)
        
        if self.force_np_type_out:
            super().fit(np.asarray(self.X_wkset))
        else:
            super().fit(self.X_wkset)
            
        self.SSX_ = np.asarray([])
        self.is_fitted = True
        if isinstance(self.X_model, pd.core.frame.DataFrame) and not self.force_np_type_out:
            self.is_return_pandas_type = True
        else:
            self.is_return_pandas_type = False
              
        
    def get_scores_and_obsIDs(self, scores, x_model_input, scores_symbol='t'):
        scores_columns = []
        for i in range(scores.shape[1]):
            scores_columns.append(scores_symbol+ str(i+1))
        scores_df = pd.DataFrame(scores, columns=scores_columns, index=x_model_input.index)
        return scores_df
    

    def get_model_loadings_and_varIDs(self, loadings, x_model_input, loadings_symbol='p'):
        loadings_rows = []
        for i in range(loadings.shape[0]):
            loadings_rows.append(loadings_symbol+ str(i+1))
        loadings_df = pd.DataFrame(loadings, columns=x_model_input.columns, index=loadings_rows)
        return loadings_df
    

    def not_fitted_msg(self):
        print("Don't forget to fit the model before looking for model content")

 
    @property    
    def n_latent_vars(self):
        return self.components_.shape[0]
       
    @property    
    def Xavg(self):
        return self.Xavg_ 
    
    @property    
    def Xws(self):
        return self.Xws_
    
    @property    
    def T(self):
        if self.is_fitted:
                if isinstance(self.X_wkset, np.ndarray):
                    return self.transform(self.X_wkset)
                elif isinstance(self.X_wkset, pd.core.frame.DataFrame) and self.force_np_type_out:
                    return self.transform(np.asarray(self.X_wkset))
                elif isinstance(self.X_wkset, pd.core.frame.DataFrame) and not self.force_np_type_out:
                    return self.get_scores_and_obsIDs(self.transform(self.X_wkset), self.X_wkset)
                else:
                    print('Unresolved output case in T')
        
        else:
            self.not_fitted_msg()
    
    @property    
    def P(self):
        if self.is_fitted:
            if self.is_return_pandas_type:
                return self.get_model_loadings_and_varIDs(self.components_, self.X_model)
            else:
                return self.components_ # Loading vectors as rows
        else:
            self.not_fitted_msg()

            
    @property
    def R2X(self):
        if self.is_fitted:
            R2X_arr = cumR2X_calc(self.SSX)
            if self.is_return_pandas_type:
                return pd.DataFrame(R2X_arr, columns=['R2X'], index=np.arange(len(R2X_arr), dtype=int)+1)
            else:   
                return R2X_arr
    
    
    @property    
    def SSX(self):
        if self.is_fitted:
            if not self.SSX_.size and self.X_wkset.size:
                self.SSX_ = self.get_model_SSX()
            return self.SSX_
        else:
            self.not_fitted_msg()


    def CenterAndScale_for_prediction(self, Xin, Xws, Xavg):
        """Combined center and UV scaling
           when the weights (Xws), and averages (Avg) are already defined"""
        trace = False
        observations = Xin.shape[0]
        OnesCol =  np.ones( (observations) )
        X_Cent_mat = np.outer(OnesCol, Xavg)
        if trace:
            print( 'Xin ',end=' ')
            print(Xin.shape, type(Xin))
        if trace:
            print( 'X_Cent_mat ',end=' ')
            print(X_Cent_mat.shape, type(X_Cent_mat))
        X = Xin - X_Cent_mat
        X_wgt_mat = np.outer(OnesCol, Xws)
        X = np.multiply( X, X_wgt_mat )
        return X
   

    def get_model_SSX(self):
        n_components = self.n_components
        model_SSX = np.asarray([])
        if self.X_wkset.size:
            model_SSX = np.zeros((n_components+1))
            preTreated_X = self.X_wkset
            model_SSX[0] = np.nansum(preTreated_X**2)
            for comp in range(n_components):               
                pca_num_comp = PCA_model(n_components=comp+1, is_center=self.is_center, is_scale=self.is_scale)
                pca_num_comp.fit(self.X_model)
                E = pca_num_comp.Epred(self.X_model)
                model_SSX[comp+1] = np.nansum(E**2)
        
        return model_SSX


    def get_model_pooled_SD(self, A):
        A0 = int(self.is_center)
        N, K = self.X_wkset.shape
        model_degs_of_freedom = (N-A-A0)*(K-A)
        model_pooled_SD = np.sqrt(self.SSX[A]/model_degs_of_freedom)
        return model_pooled_SD


    def Obs_residual_SD(self, E, is_normalized_residual=True, n_components=-1):
        """ Get the non-normalized observation residuals aka 'DModX'        
        Arguments:
        E:            full matrix of residuals from prediction
        
        Keyword arguments:
        is_normalized_residual: if true, the residual is relative to the 
                                model data variation
                                
        n_components: the number of components to be used in the
                      calculation, a negative number gives the default
                      number of components for the model        
        """
        if n_components < 0:
            A = self.n_components
        else:
            A = n_components
        non_normalised_obs_SDev = np.std(E, axis=1, ddof=A) 
        if is_normalized_residual:
            model_pooled_SD = self.get_model_pooled_SD(A)
            ObsResiduals_SD = non_normalised_obs_SDev/model_pooled_SD
        else:
            ObsResiduals_SD = non_normalised_obs_SDev
#        print('ObsResiduals.shape ', ObsResiduals.shape)
        return ObsResiduals_SD
            

    def Tpred(self, X_pred_set):
        if self.is_fitted:
            X_pred_wkset = self.CenterAndScale_for_prediction(X_pred_set, self.Xws, self.Xavg)
            if isinstance(X_pred_set, np.ndarray):
                return self.transform(X_pred_wkset)
            elif isinstance(X_pred_set, pd.core.frame.DataFrame) and self.force_np_type_out:
                return self.transform(np.asarray(X_pred_wkset))
            elif isinstance(X_pred_set, pd.core.frame.DataFrame) and not self.force_np_type_out:
                return self.get_scores_and_obsIDs(self.transform(X_pred_wkset), X_pred_set)
            else:
                print('Unresolved output case in Tpred')
        else:
            self.not_fitted_msg()
                        
            
    def Epred(self, X_pred_set):
        if self.is_fitted:
            X_pred_wkset = self.CenterAndScale_for_prediction(X_pred_set, self.Xws, self.Xavg)
            Epred = X_pred_wkset - self.inverse_transform(self.Tpred(X_pred_set))
            if isinstance(X_pred_set, pd.core.frame.DataFrame) and not self.force_np_type_out:
                return pd.DataFrame(Epred, columns=X_pred_set.columns, index=X_pred_set.index)
            else:
                return Epred           
        else:
            self.not_fitted_msg()

    
    def DModXpred(self, X_pred_set, is_normalized_residual=True):
        if self.is_fitted:            
            E = self.Epred( X_pred_set)
            Obs_resid = self.Obs_residual_SD(E, is_normalized_residual=is_normalized_residual)
            if isinstance(X_pred_set, pd.core.frame.DataFrame) and not self.force_np_type_out:
                return pd.DataFrame(Obs_resid, columns=['DModX'], index=X_pred_set.index)
            else:
                return Obs_resid
        else:
            self.not_fitted_msg()




class PLS_model(PLSRegression):
    
    def __init__(self, n_components=2, *, scale=False, max_iter=500, tol=1e-06, copy=True, #):
                 deflation_mode="regression", mode="A", algorithm='nipals', force_np_type_out=False):

        """  setting parameters other than these below requires a call using set_params"""
        super().__init__(
            n_components=n_components, scale=scale,
            max_iter=max_iter, tol=tol, copy=copy)
        
        self.n_components = n_components
        self.is_MeanCentered = True #Always True with standard sklearn PLSRegression
        self.scale = scale
        self.X_model = np.asarray([])
        self.Y_model = np.asarray([])
        self.SSX_ = np.asarray([])
        self.S2X_ = np.asarray([])
        self.is_fitted = False
        self.force_np_type_out = force_np_type_out
        self.io_type = {}
        
        
    @property
    def wkset_Y(self):
        model_y = self.Y_model.copy()
        return model_y
    
    @property
    def wkset_X(self):
        model_x = self.X_model.copy()
        return model_x
 
        
    def fit(self, X, Y):
        
        self.X_model = X
        self.Y_model = Y
        
        if self.force_np_type_out:
            super().fit(np.asarray(self.wkset_X), np.asarray(self.wkset_Y))
        else:
            super().fit(self.wkset_X, self.wkset_Y)
        
        self.SSX_ = np.asarray([])
        self.S2X_ = np.asarray([])
        self.is_fitted = True
        if isinstance(self.X_model, pd.core.frame.DataFrame) and not self.force_np_type_out:
            self.is_return_X_pandas_type = True
        else:
            self.is_return_X_pandas_type = False
 

    def predict(self, X):
        if self.is_fitted:
            if isinstance(X, np.ndarray):
                return super().predict(X)
            elif isinstance(X, pd.core.frame.DataFrame) and self.force_np_type_out:
                return super().predict(X.values)            
            elif isinstance(X, pd.core.frame.DataFrame) and not self.force_np_type_out:
                if isinstance(self.Y_model, pd.core.frame.DataFrame):
                    return pd.DataFrame(super().predict(X), columns=self.Y_model.columns, index=X.index)
                else:
                    return pd.DataFrame(super().predict(X), index=X.index)
            else:
                print('Unrseolved output case in predict')
        else:
            self.not_fitted_msg()
        

#    @property    
#    def n_components(self):
#        return self.n_components 
               
    def not_fitted_msg(self):
        print("Don't forget to fit the model before looking for model content")
        
            
    def get_scores_and_obsIDs(self, scores, xy_model_wkset, scores_symbol='t'):
        scores_columns = []
        for i in range(scores.shape[1]):
            scores_columns.append(scores_symbol+ str(i+1))
        scores_df = pd.DataFrame(scores, columns=scores_columns, index=xy_model_wkset.index)
        return scores_df
    

    def get_model_loadings_and_varIDs(self, loadings, xy_model_wkset, loadings_symbol='p'):
        loadings_rows = []
        for i in range(loadings.shape[0]):
            loadings_rows.append(loadings_symbol+ str(i+1))
        loadings_df = pd.DataFrame(loadings, columns=xy_model_wkset.columns, index=loadings_rows)
        return loadings_df


    @property    
    def n_latent_vars(self):
        if self.is_fitted:
            return self.components_.shape[0]
        else:
            self.not_fitted_msg()
  
    @property    
    def W(self):
        if self.is_fitted:
            if self.is_return_X_pandas_type:
                return self.get_model_loadings_and_varIDs(self.x_weights_.T, 
                                                          self.X_model, 
                                                          loadings_symbol='w')
            else:
                return self.x_weights_.T # Loading vectors as rows
        else:
            self.not_fitted_msg()
            
    
    @property    
    def P(self):       
        if self.is_fitted:
            if self.is_return_X_pandas_type:
                return self.get_model_loadings_and_varIDs(self.x_loadings_.T, self.X_model)
            else:
                return self.x_loadings_.T # Loading vectors as rows
        else:
            self.not_fitted_msg()
    

    @property    
    def C(self):
        if self.is_fitted:
            return self.y_loadings_.T # Loading vectors as rows
        else:
            self.not_fitted_msg()
       
    @property    
    def Xavg(self):
        if self.is_fitted:
            return self._x_mean
        else:
            self.not_fitted_msg()
    
    @property    
    def Xws(self):
        if self.is_fitted:
            return self._x_std
        else:
            self.not_fitted_msg()


    @property    
    def T(self):
        if self.is_fitted:
            if self.is_return_X_pandas_type:
                return self.get_scores_and_obsIDs(self.x_scores_, self.X_model)
            else:   
                return self.x_scores_
        else:
            self.not_fitted_msg()
    
    @property    
    def U(self):
        if self.is_fitted:
            return self.y_scores_ 
        else:
            self.not_fitted_msg()
    
    @property    
    def SSX(self):
        if self.is_fitted:
            if not self.SSX_.size and (self.wkset_X.size and self.wkset_Y.size):
                self.SSX_, self.S2X_ = self.get_model_SSX_S2X(self.wkset_X, self.wkset_Y, 
                                                              self.n_components, 
                                                              is_UVscaled=self.scale, 
                                                              is_MeanCentered=self.is_MeanCentered)
            return self.SSX_
        else:
            self.not_fitted_msg()
    
    @property    
    def S2X(self):
        if self.is_fitted:
            if not self.S2X_.size and (self.wkset_X.size and self.wkset_Y.size):
                self.SSX_, self.S2X_ = self.get_model_SSX_S2X(self.wkset_X, self.wkset_Y, 
                                                              self.n_components, 
                                                              is_UVscaled=self.scale, 
                                                              is_MeanCentered=self.is_MeanCentered)
            return self.S2X_
        else:
            self.not_fitted_msg()
            
    
    
    
    def center_scale_xy(self, X, Y, scale=False):
        """ Center X, Y and scale if the scale parameter==True
        Returns
        -------
            X, Y, x_mean, y_mean, x_std, y_std
        """
        if Y.ndim == 1: # make column vector
            Y = Y[:, np.newaxis]
        # center
        x_mean = X.mean(axis=0)
        X -= x_mean
        y_mean = Y.mean(axis=0)
        Y -= y_mean
        # scale
        if scale:
            x_std = X.std(axis=0, ddof=1)
            x_std[x_std == 0.0] = 1.0
            X /= x_std
            y_std = Y.std(axis=0, ddof=1)
            y_std[y_std == 0.0] = 1.0
            Y /= y_std
        else:
            x_std = np.ones(X.shape[1])
            y_std = np.ones(Y.shape[1])
        return X, Y, x_mean, y_mean, x_std, y_std
    
    
    def get_model_SSX_S2X(self, X_model, Y_model, n_components, is_UVscaled=False, is_MeanCentered=True):
        model_S2X = np.zeros((n_components+1))
        model_SSX = np.zeros((n_components+1))
        preTreated_X, preTreated_Y, x_mean, y_mean, x_std, y_std = self.center_scale_xy(X_model, Y_model, is_UVscaled)
        model_S2X[0] = np.nanvar(preTreated_X, ddof=1)
        model_SSX[0] = np.nansum(preTreated_X**2)
        A0 = int(is_MeanCentered)
        for comp in range(n_components):
            """predY, T, E = self.PLSpredict(trainingSet_X, self.Mname, selected_components= comp+1)"""
            
            pls_num_comp = PLS_model(n_components=comp+1, scale=is_UVscaled)
            pls_num_comp.fit(X_model, Y_model)
            x_scores = pls_num_comp.transform(X_model)
            X_reconstructed = pls_num_comp.inverse_transform(x_scores)
            E = X_model - X_reconstructed
                
            model_S2X[comp+1] = np.nanvar(E, ddof=comp+1+A0)
            model_SSX[comp+1] = np.nansum(E**2)
        
        return model_SSX, model_S2X
    
    def get_model_pooled_SD(self, A):
        A0 = int(self.is_MeanCentered)
        N, K = self.wkset_X.shape
        model_degs_of_freedom = (N-A-A0)*(K-A)
        model_pooled_SD = np.sqrt(self.SSX[A]/model_degs_of_freedom)
        return model_pooled_SD
    
    
    def Obs_residual_SD(self, E, is_normalized_residual=True, n_components=-1):
        """ Get the observation residuals aka 'DModX'        
        Arguments:
        E:            full matrix of residuals from prediction
        
        Keyword arguments:
        is_normalized_residual: if true, the residual is relative to the 
                                model data variation
                                
        n_components: the number of components to be used in the
                      calculation, a negative number gives the default
                      number of components for the model        
        """
        if n_components < 0:
            A = self.n_components
        else:
            A = n_components
        non_normalised_obs_SDev = np.std(E, axis=1, ddof=A) 
        if is_normalized_residual:
            model_pooled_SD = self.get_model_pooled_SD(A)
            ObsResiduals_SD = non_normalised_obs_SDev/model_pooled_SD
        else:
            ObsResiduals_SD = non_normalised_obs_SDev
#        print('ObsResiduals.shape ', ObsResiduals.shape)
        return ObsResiduals_SD
    

    def Var_residual_SD(self, E, is_normalized_residual=True, n_components=-1):
        """ Get the variable residuals        
        Arguments:
        E:            full matrix of residuals from prediction
        
        Keyword arguments:
        is_normalized_residual: If true, the residual is divided by the model SD
        n_components:           The number of components to be used in the
                                calculation, a negative number gives the default
                                number of components for the model        
        """ 
        if n_components < -0:
            A = self.n_components
        else:
            A = n_components
        non_normalised_var_SDev = np.std(E, axis=0, ddof=A) 
        if is_normalized_residual:
            model_pooled_SD = self.get_model_pooled_SD(A)
            VarResiduals_SD = non_normalised_var_SDev/model_pooled_SD
        else:
            VarResiduals_SD = non_normalised_var_SDev
        return VarResiduals_SD        
        
        
        
    def Tpred(self, X_pred_set):
        if self.is_fitted:
            if isinstance(X_pred_set, np.ndarray):
                return self.transform(X_pred_set)               
            elif isinstance(X_pred_set, pd.core.frame.DataFrame) and self.force_np_type_out:
                return self.transform(np.asarray(X_pred_set))
            elif isinstance(X_pred_set, pd.core.frame.DataFrame) and not self.force_np_type_out:
                return self.get_scores_and_obsIDs(self.transform(X_pred_set), X_pred_set)
            else:
                print('Unresolved output case in Tpred')
        else:
            self.not_fitted_msg()
    

    def Epred(self, X_pred_set):
        if self.is_fitted:
            E_predicted = X_pred_set - self.inverse_transform(self.Tpred(X_pred_set))
            if isinstance(X_pred_set, np.ndarray):
                return E_predicted
            elif isinstance(X_pred_set, pd.core.frame.DataFrame) and self.force_np_type_out:
                return np.asarray(E_predicted.values)
            elif isinstance(X_pred_set, pd.core.frame.DataFrame) and not self.force_np_type_out:
                return pd.DataFrame(E_predicted, columns=X_pred_set.columns, index=X_pred_set.index)
            else:
                print('Unresolved output case in Epred')
        else:
            self.not_fitted_msg()
            
    
    def DModXpred(self, X_pred_set, is_normalized_residual=True):
        if self.is_fitted:            
            E = self.Epred( X_pred_set)
            Obs_resid = self.Obs_residual_SD(E, is_normalized_residual=is_normalized_residual)
            if isinstance(X_pred_set, np.ndarray):
                return Obs_resid
            elif isinstance(X_pred_set, pd.core.frame.DataFrame) and self.force_np_type_out:
                return np.asarray(Obs_resid)
            elif isinstance(X_pred_set, pd.core.frame.DataFrame) and not self.force_np_type_out:
                return pd.DataFrame(Obs_resid, columns=['DModX'], index=X_pred_set.index)
            else:
                print('Unresolved output case in DModXpred')
        else:
            self.not_fitted_msg()
            
                   
    def VarResXpred(self, X_pred_set):
        if self.is_fitted:
            var_residuals = self.Var_residual_SD(np.asarray(self.Epred(X_pred_set)))
            print('VarResXpred', type(var_residuals), var_residuals.shape )
            if isinstance(X_pred_set, np.ndarray):
                return var_residuals
            elif isinstance(X_pred_set, pd.core.frame.DataFrame) and self.force_np_type_out:
                return np.asarray(var_residuals)
            elif isinstance(X_pred_set, pd.core.frame.DataFrame) and not self.force_np_type_out:
                return pd.DataFrame(var_residuals[np.newaxis,:], columns=X_pred_set.columns, index=['Var_residuals'])
            else:
                print('Unresolved output case in VarResXpred')
        else:
            self.not_fitted_msg()
        
        

    

    
class yo_PLS_model(sklearn.base.BaseEstimator):
    
    def __init__(self, n_components=2, center=True, scale=False, copy=True, verbose=0, force_np_type_out=False):
        super().__init__()
        
        self.X_model = np.asarray([])
        self.Y_model = np.asarray([])
        self.n_components = n_components
        self.is_centered = center
        self.is_scaled = scale
        self.is_fitted = False
        self.X_dataframe_info = {}
        self.Y_dataframe_info = {}        
        self.force_np_type_out = force_np_type_out
            
            
    def center_scale(self, X, scale=False):
        """ Center X and scale if the scale parameter==True
        Returns
        -------
            X, x_mean, x_std
        """
        # center
        x_mean = X.mean(axis=0)
        Xcs = X-x_mean
        # scale
        if scale:
            x_std = Xcs.std(axis=0, ddof=1)
            x_std[x_std == 0.0] = 1.0
            Xcs /= x_std
        else:
            x_std = np.ones(Xcs.shape[1])
        return Xcs, x_mean, x_std
    

                        
    def fit(self, X0, Y0):
        if isinstance(X0, pd.core.frame.DataFrame):
            self.X_dataframe_info['columns'] =  X0.columns
            self.X_dataframe_info['index'] = X0.index
        if isinstance(Y0, pd.core.frame.DataFrame):
            self.Y_dataframe_info['columns'] =  Y0.columns
            if self.X_dataframe_info: 
                self.Y_dataframe_info['index'] = self.X_dataframe_info['index']
            
        Y_arr = np.squeeze(np.asarray(Y0))
        assert Y_arr.ndim == 1, "This implementation of YO-PLS only handles one Y-variable"
       
        X, self.Xavg_, self.Xws_ = self.center_scale(X0, self.is_scaled)
        Y, self.Yavg_, self.Yws_ = self.center_scale(Y_arr[:, np.newaxis], self.is_scaled)
               
        self.X_model = np.asarray(X, dtype=np.float64)
        self.Y_model = np.asarray(Y, dtype=np.float64)       
        self.yorth_PLS1(self.n_components)
        self.is_fitted = True

        
    def not_fitted_msg(self):
        print("Don't forget to fit the model before looking for model content")
 
        
    def get_scores_and_obsIDs(self, scores, xy_dataframe_info, scores_symbol='t'):
        scores_columns = []
        for i in range(scores.shape[1]):
            scores_columns.append(scores_symbol+ str(i+1))
        scores_df = pd.DataFrame(scores, columns=scores_columns, index=xy_dataframe_info['index'])
        return scores_df
    
    
    def get_model_loadings_and_varIDs(self, loadings, xy_dataframe_info, loadings_symbol='p'):
        loadings_rows = []
        for i in range(loadings.shape[0]):
            loadings_rows.append(loadings_symbol+ str(i+1))
        loadings_df = pd.DataFrame(loadings, columns=xy_dataframe_info['columns'], index=loadings_rows)
        return loadings_df

        
    @property    
    def n_latent_vars(self):
        if self.is_fitted:
            return self.n_components
        else:
            self.not_fitted_msg()
        
    @property    
    def Xavg(self):
        if self.is_fitted:
            return self.Xavg_ 
        else:
            self.not_fitted_msg()
            
    def Yavg(self):
        if self.is_fitted:
            return self.Yavg_ 
        else:
            self.not_fitted_msg()
    
    @property    
    def Xws(self):
        if self.is_fitted:
            return self.Xws_
        else:
            self.not_fitted_msg()
    
    @property    
    def Yws(self):
        if self.is_fitted:
            return self.Yws_
        else:
            self.not_fitted_msg()

    @property    
    def T(self): 
        if self.is_fitted:
            if self.X_dataframe_info and not self.force_np_type_out:
                return self.get_scores_and_obsIDs(self.T_, self.X_dataframe_info)
            else:
                return self.T_
        else:
            self.not_fitted_msg()
    
    @property    
    def To(self): 
        if self.is_fitted:
            if self.X_dataframe_info and not self.force_np_type_out:
                return self.get_scores_and_obsIDs(self.To_, self.X_dataframe_info, scores_symbol='to')
            else:
                return self.To_
        else:
            self.not_fitted_msg()
            
    @property    
    def T_To(self):
        """
        Returns the predictive T followed by one or more ortogonal To
        """   
        if self.is_fitted:
            if self.X_dataframe_info and not self.force_np_type_out:
                T_part = self.get_scores_and_obsIDs(self.T_, self.X_dataframe_info)
                To_part = self.get_scores_and_obsIDs(self.To_, self.X_dataframe_info, scores_symbol='to')               
                return pd.concat([T_part, To_part], axis=1)
            else:
                return np.hstack((self.T_, self.To_))           
        else:
            self.not_fitted_msg()            
    
    @property    
    def P(self): 
        if self.is_fitted:
            if self.X_dataframe_info and not self.force_np_type_out:
                return self.get_model_loadings_and_varIDs(self.P_, self.X_dataframe_info)
            else:
                return self.P_
        else:
            self.not_fitted_msg()
    
    @property    
    def Po(self): 
        if self.is_fitted:
            if self.X_dataframe_info and not self.force_np_type_out:
                return self.get_model_loadings_and_varIDs(self.Po_, self.X_dataframe_info, loadings_symbol='po')
            else:
                return self.Po_
        else:
            self.not_fitted_msg()
            
    @property    
    def P_Po(self):
        """
        Returns the predictive P followed by one or more ortogonal Po
        """
        if self.is_fitted:
            if self.X_dataframe_info and not self.force_np_type_out:
                P_part = self.get_model_loadings_and_varIDs(self.P_, self.X_dataframe_info, loadings_symbol='p')
                Po_part = self.get_model_loadings_and_varIDs(self.Po_, self.X_dataframe_info, loadings_symbol='po')
                return pd.concat([P_part, Po_part]) #, ignore_index=True)
            else:
                return np.vstack((self.P_, self.Po_))
        else:
            self.not_fitted_msg()
    
    @property    
    def W(self): 
        if self.is_fitted:
            if self.X_dataframe_info and not self.force_np_type_out:
                return self.get_model_loadings_and_varIDs(self.W_, self.X_dataframe_info, loadings_symbol='w')
            else:
                return self.W_
        else:
            self.not_fitted_msg()
    
    @property    
    def Wo(self): 
        if self.is_fitted:
            if self.X_dataframe_info and not self.force_np_type_out:
                return self.get_model_loadings_and_varIDs(self.Wo_, self.X_dataframe_info, loadings_symbol='wo')
            else:
                return self.Wo_
        else:
            self.not_fitted_msg()
    
    @property    
    def C(self): 
        if self.is_fitted:
            return self.C_
        else:
            self.not_fitted_msg()
    
    @property    
    def U(self):
        if self.is_fitted:
            if self.X_dataframe_info and not self.force_np_type_out:
                return self.get_scores_and_obsIDs(self.U_, self.X_dataframe_info, scores_symbol='u')
            else:
                return self.U_
        else:
            self.not_fitted_msg()
    
    @property    
    def E(self): 
        if self.is_fitted:
            if self.X_dataframe_info and not self.force_np_type_out:
                return pd.DataFrame(self.E_, columns=self.X_dataframe_info['columns'], index=self.X_dataframe_info['index'])
            else:    
                return self.E_
        else:
            self.not_fitted_msg()
    
    @property    
    def f1_p(self): 
        if self.is_fitted:
            return self.f1_p_
        else:
            self.not_fitted_msg()
    
    @property    
    def SSX(self):
        '''Last entry is the predictive SSX'''
        if self.is_fitted:
            return self.SSX_
        else:
            self.not_fitted_msg()

    
    @property    
    def R2X_pred(self):
        if self.X_dataframe_info and not self.force_np_type_out:
            return pd.DataFrame(R2X_calc(self.SSX_)[-1], columns=['R2X_pred'], index = [1])
        else:
            return R2X_calc(self.SSX_)[-1]
    
    @property    
    def R2X_orth(self):
        R2X_orth_arr = R2X_calc(self.SSX_)[0:-1]
        if self.X_dataframe_info and not self.force_np_type_out:
            return pd.DataFrame(R2X_orth_arr, columns=['R2X_orth'], index=np.arange(len(R2X_orth_arr), dtype=int)+1)
        else:
            return R2X_orth_arr

    
    def p_trace(self, v_name, v, trace=True):
        if trace:
            if v.ndim == 2:
                print(v_name, v.shape, type(v[0,0]), type(v))
            elif v.ndim == 1:
                print(v_name, v.shape, type(v[0]), type(v))
            else:
                print('p_trace', v.ndim, 'number of ndims not handled')
    
    
    def PLS1comp(self, X, y):
        trace=False
        
        w_tmp = y.T @ X
        self.p_trace('w_tmp', w_tmp, trace)
        
        w = w_tmp/np.linalg.norm(w_tmp)
        self.p_trace('w', w, trace)
        if trace:
            print('w_len', np.linalg.norm(w))
        
        t = X @ w.T
        self.p_trace('t', t, trace)
        
        p = t.T/(t.T@t) @ X
        self.p_trace('p', p, trace)
        if trace:
            print('p_len', np.linalg.norm(p))
        
        q = y.T @ t/(t.T@t)
        if trace:
            print('q', q.shape, type(q[0,0]), q)
        
        u = y @ q/(q.T@q)
        self.p_trace('u', u, trace)
        
        E = X - t @ p
        f = y - t @ q
        
        return t, p, w, q, u, E, f
        
                             

    def yorth_PLS_loop(self, Xa, y):
        
        trace = False
        # Regular PLS part
        t, p, w, c, u, E_PLS1comp, f_PLS1comp = self.PLS1comp(Xa, y)
        
        pw_diff = p-w
        wo = pw_diff/np.linalg.norm(pw_diff)
        if trace:
            print('wo.shape', wo.shape)
        to = Xa @ wo.T
        if trace:
            print('Xa.shape, to.shape', Xa.shape, to.shape)
        po_1 = (Xa.T @ to)
        if trace:
            print('po_1.shape', po_1.shape)
        po = po_1/(to.T @ to)
        if trace:
            print('po.shape', po.shape)
                
        E = Xa - to @ po.T
        f = y
        return t, to, p, po, w, wo, c, u, E, f
    
    
    def mat_to_1Dvec(self, v):
        return np.squeeze(np.asarray(v, dtype=np.float64))
 
    
    def get_SSX(self, E):
        return np.nansum(np.asarray(E)**2)
        
            
    def yorth_PLS1(self, n_components):
        trace = False
               
        Xin = self.X_model

        if self.Y_model.ndim == 1:
            y_in = self.Y_model[:, np.newaxis]
        else:
            y_in = self.Y_model        
        
        T  = np.zeros((Xin.shape[0], 1))
        To = np.zeros((Xin.shape[0], n_components-1))
        P  = np.zeros((1, Xin.shape[1]))
        Po = np.zeros((n_components-1, Xin.shape[1]))
        W  = np.zeros((1, Xin.shape[1]))
        Wo = np.zeros((n_components-1, Xin.shape[1]))
        if y_in.ndim == 1:
            C  = np.zeros((n_components, 1))
        else:
            C  = np.zeros((n_components, y_in.shape[1]))
        U  = np.zeros((y_in.shape[0], 1 ))
        E_yorth = Xin.copy()
        y = y_in.copy()
        model_SSX = np.zeros((n_components+1))
        if trace:
            print('E_yorth[0]', E_yorth.shape, type(E_yorth))
        model_SSX[0] = self.get_SSX(E_yorth)
        
        for i in range(n_components-1):
            t_yorth, to, p_yorth, po, w_yorth, wo, c_yorth, u_yorth, E_yorth, f  = self.yorth_PLS_loop(E_yorth, y)
            if trace:
                print('E_yorth['+str(i+1)+']', E_yorth.shape, type(E_yorth))
            model_SSX[i+1] = self.get_SSX(E_yorth)
            
            To[:,i] = self.mat_to_1Dvec(to)
            Po[i,:] = self.mat_to_1Dvec(po) 
            Wo[i,:] = self.mat_to_1Dvec(wo) 
            
        t1_p, p1_p, w1_p, q1_p, u1_p, E, f1_p = self.PLS1comp(E_yorth, y)
        T[:,0] = self.mat_to_1Dvec(t1_p)
        P[0,:] = self.mat_to_1Dvec(p1_p)
        W[0,:] = self.mat_to_1Dvec(w1_p)
        C[0,:] = self.mat_to_1Dvec(q1_p)
        U[:,0] = self.mat_to_1Dvec(u1_p)
        if trace:
            print('E_pred['+str(n_components)+']', E.shape, type(E))
        model_SSX[n_components] = self.get_SSX(E) # predictive SSX
        
        self.T_   = T
        self.To_  = To
        self.P_   = P
        self.Po_  = Po
        self.W_   = W
        self.Wo_  = Wo
        self.C_   = C
        self.U_   = U
        self.E_   = E
        self.f1_p_= f1_p
        self.SSX_ = model_SSX
                            
        return T, To, P, Po, W, Wo, C, U, E, f1_p, model_SSX           

    
    