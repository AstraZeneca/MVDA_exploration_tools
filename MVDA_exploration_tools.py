#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import numpy as np
from mpl_toolkits.mplot3d import Axes3D #Needed for 3D plotting
import matplotlib
import matplotlib.pyplot as plt
import sklearn
import sklearn.model_selection



from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression

__version__ = '0.1.0'


def sumsq(A):
    squares = np.asarray(A)**2
    return np.sum(squares)

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
    y_ref = np.atleast_2d(y_ref0)
    y_pred = M.predict(X)
    if trace:
        print('R2 X', type(X), X.shape)
        print('R2 y_ref', type(y_ref), y_ref.shape)
        print('R2 y_pred', type(y_pred), y_pred.shape)
    u = sumsq(y_ref - y_pred)
    v = sumsq(y_ref - y_ref.mean())
    if trace:
        print('R2 u', type(u), u.shape, u)
        print('R2 v', type(v), v.shape, v)
    R2 = 1-u/v
    return R2


def PLS_R2_calc(X, Y, max_comp=10, is_UV_scale=False):
    
    if Y.ndim == 1: # make column vector
        Y = np.mat(Y).T    
    max_comp = reset_too_high_max_comp(max_comp, X)
    cumulative_R2_values = np.zeros((max_comp))
    for comps in range(max_comp):
        pls1comp_i = PLS_model(n_components=comps+1, scale=is_UV_scale)
        pls1comp_i.fit(X, Y)
        cumulative_R2_values[comps] = R2(pls1comp_i, X, Y)
    return cumulative_R2_values
  

def Q2(CV_pred, y_ref0):
    trace = False
    y_ref = np.atleast_2d(y_ref0)
    #y_pred = np.squeeze(CV_pred) # keep ndim=2 to match incoming y_ref
    y_pred = CV_pred
    PRESS = sumsq(y_ref - y_pred)
    SSY = sumsq(y_ref - y_ref.mean(axis=0))
    if trace:
        print('PRESS', PRESS, type(PRESS))
        print('y_ref', y_ref, type(y_ref), y_ref.ndim)
        print('y_pred', y_pred, type(y_pred), y_pred.ndim)
        print('y_ref.mean(axis=0)',y_ref.mean(axis=0))
        print('SSY',SSY, type(SSY))
    Q2_calc = 1-PRESS/SSY
    return Q2_calc 
    
    
def PLS_cross_val(X, Y, max_comp=10, is_UV_scale=False, CV_sections=7):

    max_comp = reset_too_high_max_comp(max_comp, X)
    
    if X.shape[0] < CV_sections:
        CV_sections = X.shape[0]
    
    cumulative_Q2_values = np.zeros((max_comp))

    for comps in range(max_comp):
        pls_CV_model = PLS_model(n_components=comps+1, scale=is_UV_scale)
        CV_selections = sklearn.model_selection.KFold(n_splits=CV_sections)
        """
        # cross_val_predict returns an array of the same size as `y` where each entry
        # is a prediction obtained by cross validation:
        """
        CV_predict = sklearn.model_selection.cross_val_predict(pls_CV_model, X, Y, cv=CV_selections)
        cumulative_Q2_values[comps] = Q2(CV_predict, Y)
#    for prn_comps in range(1,max_comp): 
#        print(prn_comps, 'Simca_Q2', Simca_Q2[prn_comps,:])
    return cumulative_Q2_values


def evalPLS_Q2(X, Y, max_comp=10, is_UV_scale=False, CV_sections=7, is_plot=True, plt_fname='', plt_dir=''):
    trace = False
    Xa = np.asarray(X)
    Ya = np.asarray(Y)
    max_comp = reset_too_high_max_comp(max_comp, Xa)
    R2_calc = PLS_R2_calc(Xa, Ya, max_comp=max_comp, is_UV_scale=is_UV_scale)
    if trace:
        print('R2', R2_calc.shape)
        print(R2_calc)
    Q2_calc = PLS_cross_val( Xa, Ya, max_comp=max_comp, is_UV_scale=is_UV_scale, CV_sections=CV_sections)
    if trace:
        print('Q2', Q2_calc.shape)
        print(Q2_calc)

    if is_plot:
        fig1 = Fig()
        fig1.R2_Q2_bars(R2_calc, Q2_calc) #, edgecolor='black')
        fig1.legend()
        if plt_fname:
            fig1.save(plt_dir, plt_fname)
            
    return R2_calc, Q2_calc


def mk_oneDimArray(values):
    if (values.size == 1) and (values.ndim == 2):
        valArr = np.asarray(values[0,0])   
    elif values.ndim > 1:
        valArr = np.squeeze(np.asarray(values))
    else:
        valArr = np.asarray(values)
    return valArr
        
        

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
        R2 = np.atleast_1d(np.squeeze(np.asarray(R2_in)))
        Q2 = np.atleast_1d(np.squeeze(np.asarray(Q2_in)))
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
        if isinstance(v1a[0], np.str):
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
    def __init__(self, n_components=None, scale=False):
        super().__init__(n_components=n_components)
        
        self.n_components = n_components
        self.scale = scale
        self.Xavg_ = np.asarray([])
        self.Xws_  = np.asarray([])
        self.X_model = None   

    
    def center_scale_x(self, X, scale=False):
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
    

    def fit(self, X0):

        X, self.Xavg_, self.Xws_ = self.center_scale_x(X0, self.scale)
        self.X_model = X

        super().fit(self.wkset_X)
        

        
    @property    
    def Xavg(self):
        return self.Xavg_ 
    
    @property    
    def Xws(self):
        return self.Xws_
    
    @property    
    def T(self):
        return self.transform(self.wkset_X)
    
    @property    
    def P(self):
        return self.components_
    
    @property
    def wkset_X(self):
        model_x = self.X_model.copy()
        return model_x




class PLS_model(PLSRegression):
    
    def __init__(self, n_components=2, *, scale=False, max_iter=500, tol=1e-06, copy=True, #):
                 deflation_mode="regression", mode="A", algorithm='nipals'):

        """  setting parameters other than these below requires a call using set_params"""
        super().__init__(
            n_components=n_components, scale=scale,
            max_iter=max_iter, tol=tol, copy=copy)
        
        self.n_components = n_components
        self.is_MeanCentered = True #Always True with standard sklearn PLSRegression
        self.scale = scale
        self.X_model = None
        self.Y_model = None
        self.SSX_ = np.asarray([])
        self.S2X_ = np.asarray([])
        self.is_fitted = False
 
        
    def fit(self, X, Y):
        self.X_model = X
        self.Y_model = Y        
        super().fit(self.wkset_X, self.wkset_Y)
        
        self.SSX_ = np.asarray([])
        self.S2X_ = np.asarray([])
        self.is_fitted = True

#    @property    
#    def n_components(self):
#        return self.n_components 
               
    def not_fitted_msg(self):
        print("Don't forget to fit the model before looking for model content")
    
    @property    
    def W(self):
        if self.is_fitted:
            return self.x_weights_.T # Loading vectors as rows
        else:
            self.not_fitted_msg()
    
    @property    
    def P(self):       
        if self.is_fitted:
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
            return self.x_mean_
        else:
            self.not_fitted_msg()
    
    @property    
    def Xws(self):
        if self.is_fitted:
            return self.x_std_
        else:
            self.not_fitted_msg()
   
    @property    
    def T(self):
        if self.is_fitted:
            return self.x_scores_
        else:
            self.not_fitted_msg()
    
    @property    
    def U(self):
        if self.is_fitted:
            return self.y_scores_ 
    
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
            
    @property
    def wkset_Y(self):
        model_y = self.Y_model.copy()
        return model_y
    
    @property
    def wkset_X(self):
        model_x = self.X_model.copy()
        return model_x
    
    
    
    def center_scale_xy(self, X, Y, scale=False):
        """ Center X, Y and scale if the scale parameter==True
        Returns
        -------
            X, Y, x_mean, y_mean, x_std, y_std
        """
        if Y.ndim == 1: # make column vector
            Y = np.mat(Y).T
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
#        preTreated_X, _ = np.asarray(self.CenterAndScale(X_model, self.Xws, self.Xavg))
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
            model_SSX[comp+1] = np.nansum(np.asarray(E)**2)
        
        return model_SSX, model_S2X
    
    def get_model_pooled_SD(self, A):
        A0 = int(self.is_MeanCentered)
        N, K = self.wkset_X.shape
#        model_SSX = self.SSX[A]
        model_degs_of_freedom = (N-A-A0)*(K-A)
        model_pooled_SD = np.sqrt(self.SSX[A]/model_degs_of_freedom)
        return model_pooled_SD
    
    
    def Obs_residual_SD(self, E, is_normalized_residual=True, n_components=-1):
        """ Get the non-normalized observation residuals aka 'DModX'        
        Arguments:
        E:            full matrix of residuals from prediction
        
        Keyword arguments:
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
        """ Get the non-normalized observation residuals aka 'DModX'        
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
            return self.transform(X_pred_set)
        else:
            self.not_fitted_msg()
    
    def Epred(self, X_pred_set):
        if self.is_fitted:
            return X_pred_set - self.inverse_transform(self.Tpred(X_pred_set))
        else:
            self.not_fitted_msg()
    
    def DModXpred(self, X_pred_set):
        if self.is_fitted:            
            E = self.Epred( X_pred_set)
            Obs_resid = self.Obs_residual_SD(E)
            return Obs_resid
        else:
            self.not_fitted_msg()
            
        
    def VarResXpred(self, X_pred_set):
        if self.is_fitted:
            return self.Var_residual_SD(self.Epred(np.asarray(X_pred_set)))
        else:
            self.not_fitted_msg()
        
        

    

    
class yo_PLS_model(sklearn.base.BaseEstimator):
    
    def __init__(self, n_components=2, center=True, scale=False, copy=True, verbose=0):
        super().__init__()
        
        self.X_model = np.asarray([])
        self.Y_model = np.asarray([])
        self.n_components = n_components
        self.is_centered = center
        self.is_scaled = scale
#        if self.is_scaled:
#            print('UV-scaling not yet implemented')

#TODO -- implement UV scaling
            
            
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
        assert np.squeeze(np.asarray(Y0)).ndim == 1, "This implemetation of YO-PLS only handles one Y-variable"
        X, self.Xavg_, self.Xws_ = self.center_scale(X0, self.is_scaled)
        Y, self.Yavg_, self.Yws_ = self.center_scale(Y0, self.is_scaled)
        
        self.X_model = np.asarray(X)
        self.Y_model = np.asarray(Y)       
        self.yorth_PLS1(self.n_components)
               
        
    @property    
    def Xavg(self):
        return self.Xavg_ 
    
    def Yavg(self):
        return self.Yavg_ 
    
    @property    
    def Xws(self):
        return self.Xws_
    
    @property    
    def Yws(self):
        return self.Yws_

    @property    
    def T(self): return self.T_
    
    @property    
    def To(self): return self.To_
    
    @property    
    def P(self): return self.P_
    
    @property    
    def Po(self): return self.Po_
    
    @property    
    def W(self): return self.W_
    
    @property    
    def Wo(self): return self.Wo_
    
    @property    
    def C(self): return self.C_
    
    @property    
    def U(self): return self.U_
    
    @property    
    def E(self): return self.E_
    
    @property    
    def f1_p(self): return self.f1_p_

    
    def p_trace(self, v_name, v, trace=True):
        if trace:
            if v.ndim == 2:
                print(v_name, v.shape, type(v[0,0]), type(v))
            elif v.ndim == 1:
                print(v_name, v.shape, type(v[0]), type(v))
            else:
                print('p_trace', v.ndim, 'number of ndims not handled')
    
    
    def center(self, X0):
        if X0.ndim == 1:
            X = np.mat(X0).T
        else:
            X = X0        
        centrum = np.mat(X.mean(axis=0))
        one_col = np.mat(np.ones_like(X[:,0]))
    #    print('center', one_col.shape, centrum.shape, X.shape)
        centered = X - np.outer(one_col,centrum)
        return centered, centrum


    def PLS1comp(self, X, y):
        trace=False
        
        w_tmp = y.T * X
        self.p_trace('w_tmp', w_tmp, trace)
        
        w = w_tmp/np.linalg.norm(w_tmp)
        self.p_trace('w', w, trace)
        if trace:
            print('w_len', np.linalg.norm(w))
        
        t = X * w.T
        self.p_trace('t', t, trace)
        
        p = t.T/(t.T*t) * X
        self.p_trace('p', p, trace)
        if trace:
            print('p_len', np.linalg.norm(p))
        
        q = y.T * t/(t.T*t)
        if trace:
            print('q', q.shape, type(q[0,0]), q)
        
        u = y * q/(q.T*q)
        self.p_trace('u', u, trace)
        
        E = X - t * p
        f = y - t * q
        
        return t, p, w, q, u, E, f
        
                             

    def yorth_PLS_loop(self, Xa, y):
        
        trace = False
        # Regular PLS part
        t, p, w, c, u, E_PLS1comp, f_PLS1comp = self.PLS1comp(Xa, y)
        
        pw_diff = p-w
        wo = pw_diff/np.linalg.norm(pw_diff)
        if trace:
            print('wo.shape', wo.shape)
        to = Xa * wo.T
        if trace:
            print('Xa.shape, to.shape', Xa.shape, to.shape)
        po_1 = (Xa.T * to)
        if trace:
            print('po_1.shape', po_1.shape)
        po = po_1/(to.T * to)
        if trace:
            print('po.shape', po.shape)
                
        E = Xa - to * po.T
        f = y
        return t, to, p, po, w, wo, c, u, E, f
    
    
    def mat_to_1Dvec(self, v):
        return np.squeeze(np.asarray(v))

        
    def yorth_PLS1(self, n_components):
        
#        if is_centered:
#            Xin, self.yorth_Xavg = self.center(self.X_model)
#            y_in, self.yorth_y_avg = self.center(self.Y_model)
#        else:
#            Xin = self.X
#            if self.y.ndim == 1:
#                y_in = self.y[:, np.newaxis]
#            else:
#                y_in = self.y
#        self.Xws_ = np.ones((self.X_model.shape[0]))
#        self.Yws_ = np.ones((1)) #as this is PLS1
        
        Xin = np.mat(self.X_model)
#        if self.Y_model.ndim == 1:
#            y_in = self.Y_model[:, np.newaxis]
#        else:
#            y_in = self.Y_model
        if self.Y_model.ndim == 1:
            y_in = np.mat(self.Y_model).T
        else:
            y_in = np.mat(self.Y_model)        
        
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
        
        for i in range(n_components-1):
            t_yorth, to, p_yorth, po, w_yorth, wo, c_yorth, u_yorth, E_yorth, f  = self.yorth_PLS_loop(E_yorth, y)
            
            To[:,i] = self.mat_to_1Dvec(to)
            Po[i,:] = self.mat_to_1Dvec(po) 
            Wo[i,:] = self.mat_to_1Dvec(wo) 
            
        t1_p, p1_p, w1_p, q1_p, u1_p, E, f1_p = self.PLS1comp(E_yorth, y)
        T[:,0] = self.mat_to_1Dvec(t1_p)
        P[0,:] = self.mat_to_1Dvec(p1_p)
        W[0,:] = self.mat_to_1Dvec(w1_p)
        C[0,:] = self.mat_to_1Dvec(q1_p)
        U[:,0] = self.mat_to_1Dvec(u1_p)
        
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
                            
        return T, To, P, Po, W, Wo, C, U, E, f1_p            

    
    
    
    