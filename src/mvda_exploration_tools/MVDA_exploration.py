#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 23:22:04 2024

@author: mats_j
"""

import sys, os, ast, time
import typing
import platform
import darkdetect

import qtpy
from qtpy import QtWidgets
from qtpy import QtGui
from qtpy import QtCore

import pickle
import numpy as np
import sklearn
import webbrowser
import pandas as pd

from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import matplotlib.pyplot
#from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar


try:
    import MVDA_exploration_tools as mv
    print('Loading development version of MVDA_exploration_tools')
except ImportError:
    import mvda_exploration_tools.MVDA_exploration_tools as mv



__version__ = '0.1.13'
version_date = '2026-04-25'


def get_number(string):
    try:
        value = float(string)
        return value
    except ValueError:
        return np.NaN
    
def is_number(string):
    try:
        value = float(string)
        return True
    except ValueError:
        return False
    
    
def datetime_stamp():
    stamp = '%4d-%02d-%02d_%02d.%02d.%02d' % time.localtime()[:6]
    return stamp

def timestamped_filename( base_filename ):
    filename = base_filename+datetime_stamp()
    return filename


def getNpIndices( booleanArray ):
    indices = np.where(booleanArray==True)[0]
    return indices


class Matplot_view(QtWidgets.QWidget):
    
    def __init__(self, is_white=True, plt_interaction=0, 
                 matPlotLabel_len=10,
                 matPlotShowTxtLabel=False,
                 scoreID_callback=None, **kwargs):
        super().__init__()
        
        self.plotUpdates = False
        self.is_white = is_white
        if self.is_white:
            self.mkr_edge_color = 'k'
        else:
            self.mkr_edge_color = 'w'
        self.matPlotShowTxtLabel = matPlotShowTxtLabel
        self.set_background()
        self.decimateIDs = 1
        self.classIDs = []
        self.matPlotLabel_len = matPlotLabel_len   
        self.R2X = np.asarray([])
        self.D_scatter = np.asarray([])
        self.input_vectors = {}
        self.is_depth_scale = False
            
        self.fig = Figure(**kwargs)
        cid = self.fig.canvas.mpl_connect('button_press_event', self.mouse_event)
        self.axes = self.fig.add_subplot(111)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setParent(self)
        self.mpl_toolbar = NavigationToolbar(self.canvas, self)
        vbox = QtWidgets.QVBoxLayout()
        vbox.addWidget(self.canvas)  # the matplotlib canvas
#        vbox.addWidget(self.View_DataName)
        vbox.addWidget(self.mpl_toolbar)
        self.setLayout(vbox)
        # self.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        # self.inverted_x_axis = False
        # self.inverted_y_axis = False
        self.plot_vectors = {}
        self.scoreID_callback = scoreID_callback
        
        if plt_interaction >= 1:
            hbox = QtWidgets.QHBoxLayout()
            vbox.addLayout(hbox)
            
            hbox1_comp = QtWidgets.QHBoxLayout()
            hbox2_comp = QtWidgets.QHBoxLayout()
            vbox_comp = QtWidgets.QVBoxLayout()
            clearButton = QtWidgets.QPushButton('Clear plot')
            clearButton.clicked.connect(self.clear)
            hbox1_comp.addWidget(clearButton)
            # self.autoClearCheckBox = QtWidgets.QCheckBox('Auto clear')
            # self.autoClearCheckBox.setCheckState(QtCore.Qt.Checked) 
            # hbox1_comp.addWidget(self.autoClearCheckBox)
            vbox_comp.addLayout(hbox1_comp)
            hbox.addLayout(vbox_comp)
            
        if plt_interaction >= 2:
            prevButton = QtWidgets.QPushButton('Prev.')
            nextButton = QtWidgets.QPushButton('Next')
            prevButton.clicked.connect(self.prev_comp)
            nextButton.clicked.connect(self.next_comp)
            hbox1_comp.addWidget(prevButton)
            hbox1_comp.addWidget(nextButton)
            
            self.input_select_combobox = QtWidgets.QComboBox()
            self.input_select_combobox.currentIndexChanged.connect(self.switch_input_vectors)
            self.input_select_combobox.setToolTip('Select type of scores or loadings')              
            self.x_axis_spin = QtWidgets.QSpinBox()
            self.x_axis_spin.setMinimum(1)
            self.x_axis_spin.setPrefix("x_comp ")
            self.x_axis_spin.valueChanged.connect(self.axis_change)
            self.y_axis_spin = QtWidgets.QSpinBox()
            self.y_axis_spin.setMinimum(1)
            self.y_axis_spin.setValue(2)
            self.y_axis_spin.setPrefix("y_comp ") 
            self.y_axis_spin.valueChanged.connect(self.axis_change)
            hbox2_comp.addWidget(self.input_select_combobox)
            hbox2_comp.addWidget(self.x_axis_spin)
            hbox2_comp.addWidget(self.y_axis_spin)
            vbox_comp.addLayout(hbox2_comp)
            
            hbox1_lbl = QtWidgets.QHBoxLayout()
            hbox2_lbl = QtWidgets.QHBoxLayout()
            vbox_lbl = QtWidgets.QVBoxLayout()
            
            self.labelsType_combobox = QtWidgets.QComboBox()
            self.labelsType_combobox.addItems(['No labels', 'Text labels', 'alt Text labels', 'Numeric labels', 'Class ID labels'])
            self.labelsType_combobox.setToolTip('Select type of labels for the projected points') 
            self.labelsType_combobox.currentIndexChanged.connect(self.axis_change)
            self.labels_decimation_spin = QtWidgets.QSpinBox()
            self.labels_decimation_spin.setMinimum(1)
            self.labels_decimation_spin.setMaximum(100000)
            self.labels_decimation_spin.setPrefix("#labels 1/")
            self.labels_decimation_spin.valueChanged.connect(self.axis_change)
            self.labels_len_spin = QtWidgets.QSpinBox()
            self.labels_len_spin.setMinimum(1)
            self.labels_len_spin.setMaximum(256)
            self.labels_len_spin.setValue(self.matPlotLabel_len)
            self.labels_len_spin.setPrefix("lbl len ")
            self.labels_len_spin.valueChanged.connect(self.axis_change)

            self.lineplot_chkbox = QtWidgets.QCheckBox('show as lines')
            self.lineplot_chkbox.setCheckState(QtCore.Qt.Unchecked)
            self.lineplot_chkbox.stateChanged.connect(self.lineplot_stateChange)
            self.num_lines_spin = QtWidgets.QSpinBox()
            self.num_lines_spin.setMinimum(1)
            self.num_lines_spin.setPrefix("# lines ")
            self.num_lines_spin.setValue(1)
            self.num_lines_spin.valueChanged.connect(self.lineplot_stateChange)
            
            self.addline_chkbox = QtWidgets.QCheckBox('add line')
            self.addline_chkbox.setCheckState(QtCore.Qt.Unchecked)
            self.addline_chkbox.stateChanged.connect(self.addline_stateChange)
            
            hbox1_lbl.addWidget(self.lineplot_chkbox)
            hbox1_lbl.addWidget(self.num_lines_spin)
            hbox2_lbl.addWidget(self.addline_chkbox)
            hbox1_lbl.addWidget(self.labelsType_combobox)
            hbox2_lbl.addWidget(self.labels_decimation_spin)
            hbox2_lbl.addWidget(self.labels_len_spin)
            vbox_lbl.addLayout(hbox1_lbl)
            vbox_lbl.addLayout(hbox2_lbl)
            hbox.addLayout(vbox_lbl)
            
            vbox_sc_tseries = QtWidgets.QVBoxLayout()
                    
            self.x_axis_scaleType_combobox = QtWidgets.QComboBox()
            self.x_axis_scaleType_combobox.addItems(['indexed x scale', 'time scale (s)', 'time scale (min)'])
            self.x_axis_scaleType_combobox.setToolTip('Select type of scale for one component plots')        
            self.x_axis_scaleType_combobox.currentIndexChanged.connect(self.axis_change)
            
            self.tseries_combo = QtWidgets.QComboBox()
            self.tseries_combo.setToolTip('Select individual time series for plotting')  
            self.tseries_combo.currentIndexChanged.connect(self.axis_change)

            vbox_sc_tseries.addWidget(self.x_axis_scaleType_combobox)
            vbox_sc_tseries.addWidget(self.tseries_combo)
            hbox.addLayout(vbox_sc_tseries)


    def plot(self, x_scale, y_vector, label='', is_inverted=False, **kwargs):    
        if label:
            self.axes.plot(x_scale, y_vector, label=label, **kwargs)
            self.axes.legend()
        else:
            if y_vector.shape[0] != x_scale.shape[0]:                
                self.axes.plot(x_scale, y_vector.T, **kwargs)
            else:
                self.axes.plot(x_scale, y_vector, **kwargs)
        self.canvas.draw()

    def is_inverted_x(self, xlabel):
        is_inverted = 'wavenumber' in xlabel.lower() 
        return is_inverted
    
    def set_inverted_x_axis(self):
        if not self.axes.xaxis_inverted():
            self.axes.invert_xaxis()
            
    def set_uninverted_x_axis(self):
        if self.axes.xaxis_inverted():
            self.axes.invert_xaxis()
            
    def set_inverted_y_axis(self):
        if not self.axes.yaxis_inverted():
            self.axes.invert_yaxis()
            
    def set_uninverted_y_axis(self):
        if self.axes.xaxis_inverted():
            self.axes.invert_xaxis()
        
    def set_xlabel(self, label, set_inverted=False):
        self.axes.set_xlabel(label)
        if not set_inverted:
            if self.is_inverted_x(label):
                self.set_inverted_x_axis()
            else:
                self.set_uninverted_x_axis()
        elif set_inverted:
            self.set_inverted_x_axis()        
        self.canvas.draw()
        
    def set_ylabel(self, label, set_inverted=False):
        self.axes.set_ylabel(label)
        if set_inverted:
            self.set_inverted_y_axis()
        else:
            self.set_uninverted_y_axis()       
        self.canvas.draw()
        
    def set_title(self, title):
        self.axes.set_title(title)
        self.canvas.draw()


    def set_background(self):        
        if self.is_white:
            matplotlib.pyplot.style.use('default')
        else:
            matplotlib.pyplot.style.use('dark_background')
            
    def tight_layout(self):        
        self.fig.tight_layout()
    
    
    def clear(self):
        self.axes.cla()  # Clear the canvas.
        self.set_uninverted_x_axis()
        self.set_uninverted_y_axis() 
        self.canvas.draw()
        

    def prev_comp(self):
        current_x = self.x_axis_spin.value()
        self.x_axis_spin.setValue(current_x-1)
        current_y = self.y_axis_spin.value()
        self.y_axis_spin.setValue(current_y-1)
        self.axis_change()


    def next_comp(self):
        current_x = self.x_axis_spin.value()
        self.x_axis_spin.setValue(current_x+1)
        current_y = self.y_axis_spin.value()
        self.y_axis_spin.setValue(current_y+1)
        self.axis_change()

        
    def replot_same_comp(self):         
        current_x = self.x_axis_spin.value()
        self.x_axis_spin.setValue(current_x)
        current_y = self.y_axis_spin.value()
        self.y_axis_spin.setValue(current_y)
        self.axis_change()
        
        
    def lineplot_stateChange(self):
        self.axis_change()
        
    def addline_stateChange(self):
        self.axis_change()            
        
    def scatter_plot(self, v1, v2=np.asarray([]), v3=np.asarray([]), 
                     pointIDs=[], decimateIDs=1, pointIDs_max_len=10, 
                     fontsize:float=10.0, markersize=10, **kwargs):
        """Plot a scatter plot with text annotations for each plotted coordinate present in pointIDs
           An empty pointIDs gives numbered coordinates, fontsize=0 switches off any coordinate annotations
        """                        
#        self.set_default_kwargs(kwargs)
        v1a0 = np.asarray(v1)
        v2a0 = np.asarray(v2)
        v3a0 = np.asarray(v3) 
                        
        if v3a0.size:
            v1a = np.atleast_1d(np.squeeze(np.asarray(v1a0)))
            v2a = np.atleast_1d(np.squeeze(np.asarray(v2a0)))
            v3a = np.atleast_1d(np.squeeze(np.asarray(v3a0)))
            self.axes.plot( v1a,  v2a,  v3a, 'o') #, **kwargs) 
        elif v2a0.size:
            v1a = np.atleast_1d(np.squeeze(np.asarray(v1a0)))
            v2a = np.atleast_1d(np.squeeze(np.asarray(v2a0)))
            self.axes.plot(v1a, v2a, 'o', **kwargs )
            self.axes.axvline(0.0, color='0.5', linewidth=0.5, linestyle='--', **kwargs )
            self.axes.axhline(0.0, color='0.5', linewidth=0.5, linestyle='--', **kwargs )
            
            # if self.inverted_y_axis:
            #     self.axes.invert_yaxis()
            txt_size = float(fontsize) 
            """
            10 is default size,
            txt_size has to be a float otherwise it has no effect in the annotation"""
            
#            txt_offset = 1.2*np.ceil(kwargs['markersize']/2)
            txt_offset = 1.2*np.ceil(markersize/2)
            pointIDs_lst = list(pointIDs)
            if pointIDs_lst:
                for i, txt in enumerate(pointIDs_lst):
                    if i % decimateIDs == 0:
                        txt_len = min(len(txt), pointIDs_max_len)
                        txt_shown = txt[0:txt_len]
                        self.axes.annotate(txt_shown, (v1a[i], v2a[i]), textcoords='offset points', xytext=(txt_offset,txt_offset))
            elif (v1a.ndim == 1) and (v2a.ndim == 1) and txt_size > 0:
                for i in range(v1a.shape[0]):
                    if i % decimateIDs == 0:
                        if np.any(np.isclose(v1a[:i], v1a[i]*np.ones_like(v1a[:i]), atol=0.1)) and np.any(np.isclose(v2a[:i], v2a[i]*np.ones_like(v2a[:i]), atol=0.1)):
                            self.axes.annotate(str(i), (v1a[i], v2a[i]), textcoords='offset points', xytext=(txt_offset,-txt_offset-txt_size), fontsize=txt_size)
                        else:
                            self.axes.annotate(str(i), (v1a[i], v2a[i]), textcoords='offset points', xytext=(txt_offset,txt_offset), fontsize=txt_size) 
            else:
                """Could not find a way annotate with more than 1D vectors"""
                pass                
        else:
            self.axes.plot(v1, 'o') #, **kwargs)
            self.axes.axvline(0.0, color='0.5', linewidth=0.5, linestyle='--', **kwargs )
            self.axes.axhline(0.0, color='0.5', linewidth=0.5, linestyle='--', **kwargs )
            
        self.canvas.draw()

        
    def set_scatter_plot_input(self, inp, plot_title='', label_prefix='', 
                               pointIDs=[], alt_pointIDs=[],
                               decimateIDs=1, 
                               time_scale= np.asarray([]), 
                               time_series_ix= np.asarray([]),
                               time_series_classes= [],
                               R2X = np.asarray([]),
                               is_depth_scale = False):
        self.plotUpdates = False
        trace = False
        self.input_select_combobox.clear()
        self.input_vectors = inp
        if isinstance(inp, dict):
            for entry_num, entry in enumerate(inp):
                self.input_select_combobox.addItem(entry)
                if entry_num == 0:
                    self.D_scatter = inp[entry]['vectors']
                    self.axlabel_prefix = inp[entry]['labels_prefix']
                    self.plot_title = inp[entry]['plot_title']
        else:
            self.D_scatter = inp
            self.axlabel_prefix = label_prefix
            self.plot_title = plot_title
                
        if trace:
            print('set_scatter_plot_input', self.D_scatter.shape, type(self.D_scatter[0,0]))  
            print('pointIDs', pointIDs) 
            print()
            print('self.input_vectors', self.input_vectors)
        #self.axlabel_prefix = label_prefix
        self.R2X = R2X
        self.point_lbls = pointIDs
        self.alt_point_lbls = alt_pointIDs
        self.time_scale = time_scale 
        self.time_series_ix = time_series_ix
        if time_series_classes:
            self.classIDs = time_series_classes
        else:
            self.classIDs = []
        self.x_axis_spin.setMaximum(self.D_scatter.shape[1])
        self.y_axis_spin.setMaximum(self.D_scatter.shape[1])
        self.x_axis_spin.setValue(1)
        if self.D_scatter.shape[1] == 1:            
            self.y_axis_spin.setValue(1)
        else:
            self.y_axis_spin.setValue(2)
        self.tseries_combo.clear()
        self.tseries_combo.addItem('all')
        self.tseries_combo.setCurrentIndex(0)
        if time_series_ix.size:
            unique_ts_value_strs = list(map(str, sorted(list(set(time_series_ix)))))
            self.tseries_combo.addItems(unique_ts_value_strs)

        if decimateIDs > 1: 
            self.labels_decimation_spin.setValue( decimateIDs )
        self.is_depth_scale = is_depth_scale
        if self.is_depth_scale:
            self.lineplot_chkbox.hide()
            self.num_lines_spin.hide()
        if self.matPlotShowTxtLabel:
            self.labelsType_combobox.setCurrentText('Text labels')  
           
        else:
            self.lineplot_chkbox.show()
            self.num_lines_spin.show()
            
#        if time_series_classes:
#            self.classIDs = time_series_classes
#        else:
#            self.classIDs = []
        self.plotUpdates = True # Needed to not prematurely update the plot before all data is present 
        self.clear()
        self.plot_selected_scatter()


    def switch_input_vectors(self):
        entry = self.input_select_combobox.currentText()
        if entry:
            self.D_scatter = self.input_vectors[entry]['vectors']
            self.axlabel_prefix = self.input_vectors[entry]['labels_prefix']
            self.plot_title = self.input_vectors[entry]['plot_title']
        
        self.x_axis_spin.setMaximum(self.D_scatter.shape[1])
        self.y_axis_spin.setMaximum(self.D_scatter.shape[1])
        self.x_axis_spin.setValue(1)
        if self.D_scatter.shape[1] == 1:            
            self.y_axis_spin.setValue(1)
        else:
            self.y_axis_spin.setValue(2)
        self.re_plot()
        

    def plot_selected_scatter(self):
        trace = False
        comp1 = self.x_axis_spin.value()
        comp2 = self.y_axis_spin.value()
        if not self.D_scatter.size:
            self.showMsgBox("Please enter some data and make a model")
        else:            
            if self.R2X.size:
                R2X_suffix_1 = ',  R2X['+str(comp1)+']='+ str(np.round(self.R2X[comp1-1], 4))
                R2X_suffix_2 = ',  R2X['+str(comp2)+']='+ str(np.round(self.R2X[comp2-1], 4))
            else:
                R2X_suffix_1 = ''
                R2X_suffix_2 = ''
            
            if self.labelsType_combobox.currentText() == 'Text labels':
                current_pointIDs = self.point_lbls
                current_fontsize = 10.0
            elif self.labelsType_combobox.currentText() == 'alt Text labels':
                current_pointIDs = self.alt_point_lbls
                if current_pointIDs:
                    current_fontsize = 10.0
                else:
                    current_fontsize = 0.0
            elif self.labelsType_combobox.currentText() == 'Numeric labels':
                current_pointIDs = []
                current_fontsize = 10.0
            elif self.labelsType_combobox.currentText() == 'Class ID labels':
                current_pointIDs = self.classIDs
                current_fontsize = 10.0                
            else: # No labels
                current_pointIDs = []
                current_fontsize = 0.0 
    
    
            if self.tseries_combo.currentText() == 'all':
                t_series_bool = np.ones((self.D_scatter.shape[0]), dtype=np.bool_)
                self.set_title(self.plot_title)
            elif self.tseries_combo.currentText() == '':
                t_series_bool = np.ones((self.D_scatter.shape[0]), dtype=np.bool_)
            else:
                t_series_bool = self.time_series_ix == int(self.tseries_combo.currentText())
                t_series_ID_ix = np.where(t_series_bool)[0][0]
                self.set_title(self.plot_title+' '+self.point_lbls[t_series_ID_ix])               
            
            if comp1 == comp2:
                if trace:
                    print('comp1 == comp2')                        
                if self.is_depth_scale:
                    if trace:
                        print('is_depth_scale == True')
                    if self.x_axis_scaleType_combobox.currentText() == 'indexed x scale' and self.time_scale.size:
                        x_scale0 = np.arange(self.D_scatter.shape[0])+1
                        x_scale = x_scale0[t_series_bool]
                        self.set_ylabel('observation index', set_inverted=True)
                    else:                    
                        x_scale = self.time_scale[t_series_bool]
                        self.set_ylabel('depth (m)', set_inverted=True)
     
                    if isinstance(self.axlabel_prefix, list):
                        self.set_xlabel(self.axlabel_prefix[comp1-1] + R2X_suffix_1)
                    else:
                        self.set_xlabel(self.axlabel_prefix + str(comp1) + R2X_suffix_1)                           
                
                else:
                    if trace:
                        print('is_depth_scale == False')
                    if self.x_axis_scaleType_combobox.currentText() == 'time scale (s)' and self.time_scale.size:
                        x_scale = self.time_scale[t_series_bool]
                        self.set_xlabel('time (s)' + R2X_suffix_1)
                    elif self.x_axis_scaleType_combobox.currentText() == 'time scale (min)' and self.time_scale.size:
                        x_scale = self.time_scale[t_series_bool]/60
                        self.set_xlabel('time (min)' + R2X_suffix_1)
                    else:
                        x_scale0 = np.arange(self.D_scatter.shape[0])+1
                        x_scale = x_scale0[t_series_bool]
                        self.set_xlabel('observation index' + R2X_suffix_1)
                    if isinstance(self.axlabel_prefix, list):
                        self.set_ylabel(self.axlabel_prefix[comp1-1])
                    else:
                        self.set_ylabel(self.axlabel_prefix + str(comp1))
                        
                    
                if trace:
                    print('current_pointIDs', type(current_pointIDs))
                    print(current_pointIDs)
                
                
                t_series_ix = getNpIndices(t_series_bool)
                primary_time_windowed_pointIDs = np.asarray(self.point_lbls)[t_series_ix]
                self.plot_vectors['time_windowed_pointIDs'] = primary_time_windowed_pointIDs
                if current_pointIDs:                
                    # t_series_ix = getNpIndices(t_series_bool)
                    # primary_time_windowed_pointIDs = np.asarray(self.point_lbls)[t_series_ix]
                    # self.plot_vectors['time_windowed_pointIDs'] = primary_time_windowed_pointIDs
                    time_windowed_pointIDs = np.asarray(current_pointIDs)[t_series_ix]
                else:
                    time_windowed_pointIDs = []
                    # self.plot_vectors.pop('time_windowed_pointIDs', None)
                                            
                if self.is_depth_scale:
                    if trace:
                        print('is_depth_scale == True')
                    self.plot_vectors['x'] = self.D_scatter[:, comp1-1][t_series_bool]
                    self.plot_vectors['y'] = x_scale
                    
                    if self.addline_chkbox.isChecked():
                        self.plot(self.D_scatter[:, comp1-1][t_series_bool],
                                  x_scale, linewidth=0.5)
                        
                    self.scatter_plot(self.D_scatter[:, comp1-1][t_series_bool],
                                      x_scale,
                                      pointIDs=time_windowed_pointIDs,
                                      decimateIDs=self.labels_decimation_spin.value(),
                                      pointIDs_max_len=self.labels_len_spin.value(),
                                      fontsize=current_fontsize,
                                      markeredgecolor= self.mkr_edge_color)
                
                else: # not depth scale
                    if trace:
                        print('is_depth_scale == False')
                    self.plot_vectors['x'] = x_scale
                    self.plot_vectors['y'] = self.D_scatter[:, comp1-1][t_series_bool]
                    
                    if self.lineplot_chkbox.isChecked():
                        self.plot(self.plot_vectors['x'], 
                                  self.plot_vectors['y'], linewidth=0.5)
                        
                        num_of_lines = self.num_lines_spin.value()
                        available_lines = self.D_scatter.shape[1] - comp1
                        for added_line in range(min((available_lines, num_of_lines-1))):
                        #if self.D_scatter.shape[1] > comp1:
                            self.plot(self.plot_vectors['x'], 
                                      self.D_scatter[:, added_line+comp1][t_series_bool], linewidth=0.5, label='comp '+str(added_line+comp1+1))
                        
                    else:
                        if self.addline_chkbox.isChecked():
                            self.plot(self.plot_vectors['x'], 
                                      self.plot_vectors['y'], linewidth=0.5, color='orange')
                        self.scatter_plot(self.plot_vectors['x'], 
                                          self.plot_vectors['y'],
                                          pointIDs=time_windowed_pointIDs,
                                          decimateIDs=self.labels_decimation_spin.value(),
                                          pointIDs_max_len=self.labels_len_spin.value(),
                                          fontsize=current_fontsize,
                                          markeredgecolor= self.mkr_edge_color)
                            
            else: # comp1 != comp2
                if trace:
                    print('comp1 != comp2')
                self.plot_vectors['x'] = self.D_scatter[:, comp1-1]
                self.plot_vectors['y'] = self.D_scatter[:, comp2-1]
                self.plot_vectors.pop('time_windowed_pointIDs', None)

                if self.classIDs:
                    if trace:
                        print('self.classIDs == True', 'element type', type(self.classIDs[0]))
                        print('self.classIDs', self.classIDs)
                    unique_classIDs = sorted(list(set(self.classIDs))) #TODO fix problem with numeric classIDs sorted in wrong order when taken as text 
                    all_classIDs_ix = np.asarray([], dtype=np.int_)
    
                    for classID in unique_classIDs:
                        classIDs_ix = getNpIndices(np.asarray(self.classIDs) == classID)
                        all_classIDs_ix = np.concatenate((all_classIDs_ix, classIDs_ix), axis=0)
                        if trace:
                            print('all_classIDs_ix', all_classIDs_ix.shape, type(all_classIDs_ix), 'added', classID )
    
                    for classID in unique_classIDs:
                        classIDs_ix = getNpIndices(np.asarray(self.classIDs) == classID)
                        if current_pointIDs:
                            selected_pointIDs = np.asarray(current_pointIDs, dtype=np.str_)[classIDs_ix]
                        else:
                            selected_pointIDs = []
                            
                        if trace:
                            print('self.D_scatter.shape', self.D_scatter.shape)
                            print('classIDs_ix.shape', classIDs_ix.shape)
                            print('comp1, comp2', comp1, comp2)
                        if self.addline_chkbox.isChecked():
                            self.plot(self.D_scatter[classIDs_ix, comp1-1], 
                                      self.D_scatter[classIDs_ix, comp2-1], linewidth=0.5, color='orange')                        
                        
                        self.scatter_plot(self.D_scatter[classIDs_ix, comp1-1], 
                                          self.D_scatter[classIDs_ix, comp2-1],
                                          pointIDs=selected_pointIDs,
                                          decimateIDs=self.labels_decimation_spin.value(),
                                          pointIDs_max_len=self.labels_len_spin.value(),
                                          fontsize=current_fontsize,
                                          markeredgecolor= self.mkr_edge_color)                    
                else:
                    if trace:
                        print('self.classIDs == False')
                    if self.addline_chkbox.isChecked():
                        self.plot(self.D_scatter[:, comp1-1], 
                                  self.D_scatter[:, comp2-1], linewidth=0.5, color='orange') 
                        
                    self.scatter_plot(self.D_scatter[:, comp1-1], 
                                      self.D_scatter[:, comp2-1], 
                                      pointIDs=current_pointIDs,
                                      decimateIDs=self.labels_decimation_spin.value(),
                                      pointIDs_max_len=self.labels_len_spin.value(),
                                      fontsize=current_fontsize,
                                      markeredgecolor= self.mkr_edge_color)
    
                self.set_title(self.plot_title)                
                if isinstance(self.axlabel_prefix, list):
                    self.set_xlabel(self.axlabel_prefix[comp1-1] + R2X_suffix_1 + R2X_suffix_2)
                else:
                    self.set_xlabel(self.axlabel_prefix + str(comp1) + R2X_suffix_1 + R2X_suffix_2)
                
                if isinstance(self.axlabel_prefix, list):
                    self.set_ylabel(self.axlabel_prefix[comp2-1])
                else:
                    self.set_ylabel(self.axlabel_prefix + str(comp2))

        
    def axis_change(self):
        if self.plotUpdates:
            self.clear()
            self.plot_selected_scatter()
        
    def re_plot(self):
        if self.plotUpdates:
            self.clear()
            self.plot_selected_scatter()

        
    def R2_Q2_bars(self, R2_in, Q2_in, width=0.35, **kwargs):
        R2 = np.atleast_1d(np.squeeze(np.asarray(R2_in, dtype=np.float64)))
        Q2 = np.atleast_1d(np.squeeze(np.asarray(Q2_in, dtype=np.float64)))
        bar_ix = np.arange(R2.shape[0])+1
#        viridis_colors = matplotlib.cm.viridis(np.linspace(0,1,10))
        R2_color = 'g'
        Q2_color = 'b'
        self.axes.bar(bar_ix-width/2, R2, width=width, color=R2_color, label='R2', **kwargs )
        if Q2.shape == R2.shape:
            self.axes.bar(bar_ix+width/2, Q2, width=width, color=Q2_color, label='Q2', **kwargs )        
        self.axes.legend()
        
    def get_x_axis_index(self):
        return self.x_axis_spin.value()
    
    def get_y_axis_index(self):
        return self.y_axis_spin.value()
    
    def get_PlotLabel_len(self):
        return self.labels_len_spin.value()
    
    def showMsgBox(self, msg):
        msgBox = QtWidgets.QMessageBox()
        msgBox.setText(msg)
        msgBox.exec_()    
    
    def mouse_event(self, event):
        trace = False
        if trace:
            print('x: {} and y: {}'.format(event.xdata, event.ydata))
        x_max = np.max(np.abs(self.plot_vectors['x'])) 
        y_max = np.max(np.abs(self.plot_vectors['y'])) # to balance x and y numerical magnitudes
        dx_sq = (self.plot_vectors['x']/x_max - event.xdata/x_max) ** 2
        dy_sq = (self.plot_vectors['y']/y_max - event.ydata/y_max) ** 2
        dx_dy_sumsqrt = np.sqrt(dx_sq+dy_sq)
        pt_ix = np.argmin(dx_dy_sumsqrt)
        if trace:
            print('np.argmin(dx_dy_sum)', pt_ix)
            print('x y', self.plot_vectors['x'][pt_ix], self.plot_vectors['y'][pt_ix])
            #print('argmin -1', self.plot_vectors['x'][pt_ix-1], self.plot_vectors['y'][pt_ix-1], self.point_lbls[pt_ix-1])
            #print('argmin +1', self.plot_vectors['x'][pt_ix+1], self.plot_vectors['y'][pt_ix+1], self.point_lbls[pt_ix+1])
            print('self.point_lbls[pt_ix]', self.point_lbls[pt_ix])
            print()
        if self.scoreID_callback:
            if 'time_windowed_pointIDs' in self.plot_vectors:
                if trace:
                    print('Using', 'time_windowed_pointIDs')
                self.scoreID_callback(self.plot_vectors['time_windowed_pointIDs'][pt_ix])
            else:
                self.scoreID_callback(self.point_lbls[pt_ix])
    

    # def plot_peak_marks(self, pk_apex_ix, peak_coords):                
    #     self.scatter_plot(self.plot_vectors['x'][pk_apex_ix], 
    #                       self.plot_vectors['y'][pk_apex_ix],
    #                       markeredgecolor= self.mkr_edge_color)
        
    #     for pk_range in peak_coords:
    #         pk_ix = np.asarray(pk_range, dtype=np.int_)
    #         self.scatter_plot(self.plot_vectors['x'][pk_ix], 
    #                           self.plot_vectors['y'][pk_ix],
    #                           markeredgecolor= self.mkr_edge_color)


    # def peak_find(self, SG_params) -> np.ndarray:
    #     peak_marks, peak_coords, peak_maxima_ix, peak_mid_half_height_ix = PeakCatcher.get_peaks(self.plot_vectors['y'], 
    #                                                                                              SG_params)
        
    #     print('peak_find', 'peak_maxima_ix', peak_maxima_ix)
    #     print('peak_find', 'peak_mid_half_height_ix', peak_mid_half_height_ix)
        
    #     # self.scatter_plot(self.plot_vectors['x'][peak_mid_half_height_ix], 
    #     #                   self.plot_vectors['y'][peak_mid_half_height_ix],
    #     #                   markeredgecolor= self.mkr_edge_color)
        
    #     # for pk_range in peak_coords:
    #     #     pk_ix = np.asarray(pk_range, dtype=np.int_)
    #     #     self.scatter_plot(self.plot_vectors['x'][pk_ix], 
    #     #                       self.plot_vectors['y'][pk_ix],
    #     #                       markeredgecolor= self.mkr_edge_color)
    #     self.plot_peak_marks(peak_mid_half_height_ix, peak_coords)
        
    #     return peak_mid_half_height_ix
    
    
    # def peak_tag(self, SG_params):
    #     peak_marks, peak_coords, peak_maxima_ix, peak_mid_half_height_ix = PeakCatcher.get_peaks(self.plot_vectors['y'], 
    #                                                                                              SG_params)        
    #     print('peak_tag', 'peak_maxima_ix', peak_maxima_ix)
    #     print('peak_tag', 'peak_mid_half_height_ix', peak_mid_half_height_ix)
        
    #     self.plot_peak_marks(peak_maxima_ix, peak_coords)
        
    #     peak_tags_num = np.zeros((self.plot_vectors['y'].shape[0]), dtype=np.int_)
    #     peak_tags_one_hot = np.zeros((self.plot_vectors['y'].shape[0], 3), dtype=np.int_)
    #     peak_tags_one_hot[:, 0] = 1 
    #     for pk_range in peak_coords:
    #         pk_max_ix = pk_range[0] + np.argmax(self.plot_vectors['y'][pk_range[0]:pk_range[1]])
    #         peak_tags_num[ pk_range[0] : pk_max_ix ] = 1 #peak upslope
    #         peak_tags_num[ pk_max_ix : pk_range[1] ] = 2 #peak downslope
            
    #         peak_tags_one_hot[ pk_range[0] : pk_max_ix, 0] = 0 #peak non-baseline
    #         peak_tags_one_hot[ pk_max_ix : pk_range[1], 0] = 0 #peak non-baseline
    #         peak_tags_one_hot[ pk_range[0] : pk_max_ix, 1] = 1 #peak upslope
    #         peak_tags_one_hot[ pk_max_ix : pk_range[1], 2] = 1 #peak downslope

    #     peak_tags = []
    #     for ptag_ix in range(self.plot_vectors['y'].shape[0]):
    #         if peak_tags_num[ptag_ix] == 0:
    #              peak_tags.append('baseline')
    #         elif peak_tags_num[ptag_ix] == 1:
    #              peak_tags.append('upslope')
    #         elif peak_tags_num[ptag_ix] == 2:
    #              peak_tags.append('downslope')
    #         else:
    #              peak_tags.append('undefined')

    # return peak_tags, peak_tags_num, peak_tags_one_hot
                    

class DropableListWidget(QtWidgets.QListWidget):

    fileDropped = QtCore.Signal(list)

    def __init__(self, type, parent=None):
        super(DropableListWidget, self).__init__(parent)
        self.setAcceptDrops(True)
        self.setIconSize(QtCore.QSize(72, 72))

    def dragEnterEvent(self, event):
#        print('dragEnterEvent',event.mimeData())
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
#        print('dragMoveEvent',event.mimeData())
        if event.mimeData().hasUrls():
            event.setDropAction(QtCore.Qt.CopyAction)
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
#        print('dropEvent',event.mimeData())
        if event.mimeData().hasUrls():
            event.setDropAction(QtCore.Qt.CopyAction)
            event.accept()
            links = []
            for url in event.mimeData().urls():
                links.append(str(url.toLocalFile()))
            self.fileDropped.emit(links)
#            print('links', links)
        else:
            event.ignore()
            
class DataRepository:
    """ The DataRepository class is meant as a handler of log-files and output results in a common 
        directory structure using dates, software component names and numbers as directory names. 
        It is intended as a common results repository for more than one software, if desired. """
        
    def __init__(self):
        self.IsRepositoryReady = False
        self.CurrentLocation = 'DataRepositoryLocation.txt'
        if platform.system() == 'Linux':
            self.os_settings_file_location = os.path.join(os.path.expanduser('~'),'.data_repository' ) 
        elif platform.system() == 'Windows':
            self.os_settings_file_location = 'C:\\ProgramData\\MVA_data_repository'
        self.settings_file_path =  os.path.join(self.os_settings_file_location, self.CurrentLocation)
        # print('DataRepository self.settings_file_path ', self.settings_file_path)
        if os.path.exists(self.settings_file_path):
            self.IsRepositoryReady = True
            
    def make_dir_if_absent(self, d):
        if not os.path.exists(d):
            os.makedirs(d)
            

    def set_location(self, location_of_repository):
        self.make_dir_if_absent(os.path.dirname(self.settings_file_path)) 
        settings_file = open(self.settings_file_path, 'w')
        settings_file.write(location_of_repository)
        self.IsRepositoryReady = True
        settings_file.close()

        
    def get_location(self):
        if self.IsRepositoryReady:
            settings_file = open(self.settings_file_path, 'r')
            Repository_directory = settings_file.readline()
            settings_file.close()
            return Repository_directory
        else:
            return None
            
    def IsDefined(self):
        return self.IsRepositoryReady   

        
    def get_pos_integer(self, s):
        try:
            i = int(s)
            if i == float(s):
                return int(s)
            else:
                return 0
        except ValueError:
            return 0
        
    def get_next_numeric_directory_index(self, dir0):
        dir_content = os.listdir(dir0)
        numlist = map(self.get_pos_integer, dir_content )
        if numlist:
            dir_index = max(numlist) + 1
        else: 
            dir_index = 1
        return dir_index
        
        
    def get_new_entry_in_data_repository(self, RepositoryId, repository_dir=''):
        """
        RepositoryId: becomes directory level name below date
        repository_dir: if empty, use pre-set directory, otherwise use input directory"""
        
        if not (repository_dir or self.IsRepositoryReady):
            return None
        elif repository_dir:
            pass         
        elif self.IsRepositoryReady:
            repository_dir = self.get_location()
       
        date_stamp = '%4d-%02d-%02d' % time.localtime()[:3]
        todays_dir = os.path.join( repository_dir, date_stamp )
        data_dir =  os.path.join( todays_dir, RepositoryId )                
        if os.path.exists( data_dir ):
            dir_index = self.get_next_numeric_directory_index(data_dir)
        else: 
            dir_index = 1
        working_dir = os.path.join(data_dir, str(dir_index) )
#            os.makedirs(working_dir)
        return working_dir

        
    def get_new_entry_at_data_location(self, DataDirectory, RepositoryId):
        date_stamp = '%4d-%02d-%02d' % time.localtime()[:3]
        todays_dir = os.path.join( DataDirectory, date_stamp )
        data_dir =  os.path.join( todays_dir, RepositoryId )                
        if os.path.exists( data_dir ):
            dir_index = self.get_next_numeric_directory_index(data_dir)
        else: 
            dir_index = 1
        working_dir = os.path.join(data_dir, str(dir_index) )
        return working_dir


            
    #TODO Include check of if e.g. the network disk is alive 
    def is_existing_remote_repository(self):
        return True
            
            
class TableModel(QtCore.QAbstractTableModel):

    def __init__(self, table_data, parent=None, num_obsIds=1):
        super().__init__(parent)
        self.table_data = table_data
        self.table_data_arr = np.asarray(self.table_data)
        self.is_light = darkdetect.isLight()
        self.obsIds_ix = np.arange(num_obsIds, dtype=np.int64)

    def rowCount(self, parent: QtCore.QModelIndex = ...) -> int:
        return self.table_data.shape[0]

    def columnCount(self, parent: QtCore.QModelIndex = ...) -> int:
        return self.table_data.shape[1]

    def data(self, index: QtCore.QModelIndex, role: int = ...) -> typing.Any:
        if index.isValid():
            if role == QtCore.Qt.DisplayRole:
                value = self.table_data_arr[index.row(), index.column()]               
                return str(value)
            if role == QtCore.Qt.ForegroundRole:
                value = self.table_data_arr[index.row(), index.column()] 
                if str(value) == 'nan':           
                    return QtGui.QColor('red')
            if role == QtCore.Qt.BackgroundRole:
                # value = self.table_data_arr[index.row(), index.column()] 
                if index.column() in self.obsIds_ix:
                    if self.is_light:
                        return QtGui.QColor('lightgrey')
                    else:
                        return QtGui.QColor('grey')


    def headerData(self, section: int, orientation: QtCore.Qt.Orientation, role: int = ...) -> typing.Any:
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            return str(self.table_data.columns[section])

    def setColumn(self, col, array_items):
        """Set column data"""
        self.table_data[col] = array_items
        # Notify table, that data has been changed
        self.dataChanged.emit(QtCore.QModelIndex(), QtCore.QModelIndex())

    def getColumn(self, col):
        """Get column data"""
        return self.table_data[col]



class exploration_GUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        #main_widget = QtWidgets.QWidget()
        main_widget = QtWidgets.QSplitter()
        #hBox_layout = QtWidgets.QHBoxLayout()
        #main_widget.setLayout(hBox_layout)
        self.setWindowTitle('MVDA_exploration ver. '+__version__+', '+version_date)
        self.setCentralWidget(main_widget) # Some widget is needed here to contain the layout
        self.createStatusBar()
        self.is_light_background = True
        self.is_DataRepository = False
        self.MPL_screen_dpi = 96
        self.orig_input_selectedFilter = ''
        
        self.CurrentDataRepositoryDir = ''
        self.previous_input_location = ''
        self.previous_mat_location = ''
        self.previous_input_list_location = ''
        self.previous_output_location = ''
        self.previous_models_location = ''
        self.last_selected_preproc_methods = []
        self.matPlotLabel_len = 10
        self.matPlotShowTxtLabel = False
        self.xlsx_save_data_status = False
        self.dictionary_of_datasets = {}
        self.current_dataset = ''
        self.current_files = []
        # self.clear_data()
        self.XY_settings = {}
        
        self.model_cnt = 0
        self.model = {}
        self.PCA_components = 2
        self.NMF_components = 2
        self.PLS_components = 2
        self.YO_PLS_components = 2
        self.Y_predicted_df = None
        self.model_ID_last_prediction = ''
    
        
        self.missing_values_msg = """
        This version of cannot handle missing values in data.
        Please remove variables or observations with missing values")
        """
 
        self.OutputRepository = DataRepository()
        self.readQtSettings()
#        self.init_dockableUI()
        
        """create menu"""
        menubar = self.menuBar()
        menuFile = menubar.addMenu("File")       
        open_FileMenu = menuFile.addMenu("Open")
        open_FileAction = open_FileMenu.addAction('&Files, csv or xlsx')
        open_FileAction.triggered.connect(self.on_OpenFile)       

        open_ModelsAction = open_FileMenu.addAction('&Models')
        open_ModelsAction.triggered.connect(self.on_OpenModels)

        save_FileMenu = menuFile.addMenu("Save")
        save_modelsAction = save_FileMenu.addAction('&All models')
        save_modelsAction.triggered.connect(self.on_Save_models)
        save_resultsAction = save_FileMenu.addAction('&Prediction results')
        save_resultsAction.triggered.connect(self.on_Save_predictions)
        
        showFolderAct = QtWidgets.QAction("&Show in folder", self, shortcut="Ctrl+S",
                                          statusTip="Show the current output folder",
                                          triggered=self.Open_current_output_folder)
        menuFile.addAction(showFolderAct)
        menuFile.addSeparator()
        
        
        clear_entered_data_FileAction = menuFile.addAction("Clear entered data")
        clear_entered_data_FileAction.triggered.connect(self.clear_entered_data)
        menuFile.addSeparator()
        exit_FileAction = menuFile.addAction("Exit")
        exit_FileAction.triggered.connect(self.close)

        menubar.addMenu("Edit")
        
        actionView = menubar.addMenu("View")
        actionColor_style = actionView.addMenu("Plot style")
        dark_style_Action = actionColor_style.addAction('Set dark style')
        dark_style_Action.triggered.connect(self.set_dark_style_plots)
        light_style_Action = actionColor_style.addAction('Set light style')
        light_style_Action.triggered.connect(self.set_light_style_plots)
        # actionDPI_style = actionView.addMenu("Plot magnification (dpi)")
        # actionDPI_style.triggered.connect(self.set_plot_DPI_level)
        
#        actionSettings = menubar.addMenu("Settings")
#        actionShowDataRepos = actionSettings.addAction("&Show data repository location")
#        actionShowDataRepos.triggered.connect(self.show_data_repository)
#        actionSetDataRepos = actionSettings.addAction("Set data &repository location")
#        actionSetDataRepos.triggered.connect(self.set_data_repository)
        
        menubar.addMenu("Help")
        
        """create tab-widget"""
        tab_widget1 = QtWidgets.QTabWidget()
        self.tab_widget2 = QtWidgets.QTabWidget()
#        tab_widget.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)

        main_widget.addWidget(tab_widget1)
        main_widget.addWidget(self.tab_widget2)
        
        tab_input = QtWidgets.QWidget()
        tab_model = QtWidgets.QWidget()
        tab_settings = QtWidgets.QWidget()
        
        tab_data_table = QtWidgets.QWidget()
        tab_pre_process = QtWidgets.QWidget()
        tab_diag = QtWidgets.QWidget()
        tab_scores = QtWidgets.QWidget()
        tab_loadings = QtWidgets.QWidget()
        tab_pred_results = QtWidgets.QWidget()

        tab_widget1.addTab(tab_input, "Input")
        tab_widget1.addTab(tab_model, "Model")
        tab_widget1.addTab(tab_settings, "Settings")
        
        self.tab_widget2.addTab(tab_data_table, "Data table")
        self.tab_widget2.addTab(tab_pre_process, "Pre-process")
        self.tab_widget2.addTab(tab_diag, "Diagnostics")
        self.tab_widget2.addTab(tab_scores, "Scores")
        self.tab_widget2.addTab(tab_loadings, "Loadings")
        self.tab_widget2.addTab(tab_pred_results, "Prediction results")
     
        self.tab_widget2_tab_selector = {}
        for t_ix in range(self.tab_widget2.count()):
            tab_widget = self.tab_widget2.widget(t_ix)
            tab_text = self.tab_widget2.tabText(t_ix)           
            if tab_widget:
                self.tab_widget2_tab_selector[tab_text] = tab_widget

        self.build_input(tab_input)
        self.build_model(tab_model)
        self.build_settings(tab_settings)
        
        self.build_data_table(tab_data_table)
        self.build_tab_pre_processing(tab_pre_process)
        #self.build_tab_4(tab4)
        self.build_tab_model_diagnostics(tab_diag)
        self.build_scores(tab_scores)
        self.build_loadings(tab_loadings)
        self.build_prediction_results(tab_pred_results)
        
        
    def build_input(self, tab):
        tab_hbox = QtWidgets.QHBoxLayout(tab) 
        tab_vbox = QtWidgets.QVBoxLayout()
        #self.InpDataSetsListBox = QtWidgets.QListWidget()
        self.InpDataSetsListBox = DropableListWidget(self)
        self.InpDataSetsListBox.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        """Then enable multiple selection"""
        self.InpDataSetsListBox.setSelectionMode( QtWidgets.QAbstractItemView.ExtendedSelection)
        self.InpDataSetsListBox.itemClicked.connect(self.OnInpDataSetsListBox_ItemClicked)
        self.InpDataSetsListBox.fileDropped.connect(self.InputFilesDropped)
        tab_vbox.addWidget(self.InpDataSetsListBox)

        
        self.varTable = QtWidgets.QTableWidget(0,3)
        
        self.columnLabels = ["X", "Y", "variable"]
        self.varTable.setHorizontalHeaderLabels(self.columnLabels)
        self.varTable.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        self.varTable.resizeColumnsToContents()
        # build03_hbox = QtWidgets.QHBoxLayout()
        tab_vbox.addWidget(self.varTable, stretch=4)
        tab_hbox.addLayout(tab_vbox)
 

    def clear_variables_table(self):    
        self.varTable.setRowCount(0)

    def set_variables_table_defaults(self, current_dataset):
        trace = False
        if trace:
            print('Enter set_variables_table_defaults')
        current_row = self.varTable.rowCount()
        wkset_df = self.dictionary_of_datasets[ current_dataset ]
        last_obsID, last_obsID_ix = self.get_obsID_data_splitpoint(wkset_df)
        numVarIDs = list(wkset_df.columns)[last_obsID_ix+1:] # 1st col is primary obs ID
        for ID_ix, varID in enumerate(numVarIDs):
            chkBoxItemX = QtWidgets.QTableWidgetItem()
            chkBoxItemX.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
            chkBoxItemX.setCheckState(QtCore.Qt.Checked)
            
            chkBoxItemY = QtWidgets.QTableWidgetItem()
            chkBoxItemY.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
            chkBoxItemY.setCheckState(QtCore.Qt.Unchecked)
            if trace:
                print('    current_row ', varID)
            entryName = QtWidgets.QTableWidgetItem(varID)
            self.varTable.insertRow(current_row )
            self.varTable.setItem(current_row, 0, chkBoxItemX)
            self.varTable.setItem(current_row, 1, chkBoxItemY)
            self.varTable.setItem(current_row, 2, entryName)                                    
            current_row += 1

        
    def update_variables_table(self, current_dataset, XY_item_settings={}):

        current_row = self.varTable.rowCount()
        wkset_df = self.dictionary_of_datasets[ current_dataset ]
        last_obsID, last_obsID_ix = self.get_obsID_data_splitpoint(wkset_df)
        numVarIDs = list(wkset_df.columns)[last_obsID_ix+1:] # 1st col is primary obs ID
        
        if XY_item_settings:
            if current_dataset in XY_item_settings: # Fill with previous settings
                for ID_ix, varID in enumerate(numVarIDs):
                    chkBoxItemX = QtWidgets.QTableWidgetItem()
                    chkBoxItemX.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
                    if varID in XY_item_settings[current_dataset]['X']:
                        chkBoxItemX.setCheckState(QtCore.Qt.Checked)
                    else:
                        chkBoxItemX.setCheckState(QtCore.Qt.Unchecked)

                    chkBoxItemY = QtWidgets.QTableWidgetItem()
                    chkBoxItemY.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)                                        
                    if varID in XY_item_settings[current_dataset]['Y']:
                        chkBoxItemY.setCheckState(QtCore.Qt.Checked)
                    else:
                        chkBoxItemY.setCheckState(QtCore.Qt.Unchecked)
                    entryName = QtWidgets.QTableWidgetItem(varID)
                    self.varTable.insertRow(current_row )
                    self.varTable.setItem(current_row, 0, chkBoxItemX)
                    self.varTable.setItem(current_row, 1, chkBoxItemY)
                    self.varTable.setItem(current_row, 2, entryName)                                    
                    current_row += 1
            else:
                self.set_variables_table_defaults(current_dataset)

        else:
             self.set_variables_table_defaults(current_dataset)
            
        self.varTable.resizeColumnsToContents()
        self.varTable.resizeRowsToContents()
               
        
    def get_XY_selections(self):
        chk_X_col = 0
        chk_Y_col = 1
        varID_col = 2
        rows = self.varTable.rowCount()
        xvar_ix = []
        yvar_ix = []
        xvar_IDs = []
        yvar_IDs = []
        for row_num in range(rows):
            item_X_obj = self.varTable.item(row_num, chk_X_col)
            item_Y_obj = self.varTable.item(row_num, chk_Y_col)
            is_X = (item_X_obj.checkState() == QtCore.Qt.Checked)
            is_Y = (item_Y_obj.checkState() == QtCore.Qt.Checked)
            var_ID = self.varTable.item(row_num, varID_col).text()
            if is_X and is_Y:
                msg = var_ID+' cannot be selected for both X and Y'
                self.showMsgBox(msg)
                print(msg)
            elif is_X:                
                xvar_ix.append(row_num)
                xvar_IDs.append(var_ID)
            elif is_Y:
                yvar_ix.append(row_num)
                yvar_IDs.append(var_ID)
        return xvar_IDs, yvar_IDs


    def remember_XY_settings(self, current_dataset, xvar_IDs, yvar_IDs):
        current_settings = {}
        current_settings['X'] = xvar_IDs
        current_settings['Y'] = yvar_IDs
        self.XY_settings[current_dataset] = current_settings
        
    def clear_XY_settings_for(self, dataset ):
        self.XY_settings.pop(dataset)
        
               
        
    def build_data_table(self, tab):
        tab_hbox = QtWidgets.QHBoxLayout(tab) 
        tab_vbox = QtWidgets.QVBoxLayout()
        table_data = pd.DataFrame(data={'primary obs ID': ['observation1', 'observation2', 'observation3', 'observation4', 'observation5'],
                                        'variable1': [1, 2, 3, 4, 5], 'variable2': [1, 2, 3, 4, 5]})
        self.table = QtWidgets.QTableView()
        self.table.setModel(TableModel(table_data=table_data))
        
        
        tab_vbox.addWidget(self.table)
        tab_hbox.addLayout(tab_vbox)

        
    def build_tab_pre_processing(self, tab):
        tab_hbox = QtWidgets.QHBoxLayout(tab)        
        tab_vbox = QtWidgets.QVBoxLayout()
        tab_vbox.addWidget( QtWidgets.QLabel('Not yet implemented') )
        tab_hbox.addLayout(tab_vbox)

        
        
    def build_model(self, tab):
        tab_hbox = QtWidgets.QHBoxLayout(tab) 
        tab_vbox = QtWidgets.QVBoxLayout()               
        tab_hbox.addLayout(tab_vbox)
        
        tab3_view1_vbox = QtWidgets.QVBoxLayout()
        # self.tab3_plot_view1 = QT_graph_view(is_white=self.is_light_background, plt_interaction=2)        
        # tab3_view1_vbox.addWidget(self.tab3_plot_view1)
        # plotIndexed_button = QtWidgets.QPushButton('Plot loadings with scores components selection')
        # plotIndexed_button.clicked.connect(self.on_plotIndexed)
        # tab3_view1_vbox.addWidget(plotIndexed_button)
        # tab_hbox.addLayout(tab3_view1_vbox)
        
#        self.tab3_plot_view2 = QT_graph_view(is_white=True) 
        # self.tab3_plot_view2 = Matplot_view(is_white=self.is_light_background, 
        #                                     matPlotLabel_len=self.matPlotLabel_len,
        #                                     plt_interaction=2, 
        #                                     dpi=self.MPL_screen_dpi, scoreID_callback=self.plot_preproc_spectrum)
        
        # tab_hbox.addWidget(self.tab3_plot_view2)
        
        model_frame = QtWidgets.QFrame()
        model_frame.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        m_vbox = QtWidgets.QVBoxLayout(model_frame) 
        Data_desc = QtWidgets.QLabel('X (observations, variables) + dataset name')
        Data_desc.adjustSize()
        m_vbox.addWidget(Data_desc)
        self.X_shape = QtWidgets.QLineEdit('(0,0)')
        self.X_shape.setReadOnly(True)
        self.X_obs_1_name = QtWidgets.QLineEdit('')
        self.X_obs_1_name.setReadOnly(True)
        m_vbox.addWidget(self.X_shape)
        m_vbox.addWidget(self.X_obs_1_name)
        var_scaling_lbl = QtWidgets.QLabel('Scaling of variables:')
        scaling_types = ['Unit variance (UV)', 'Mean centered only (Ctr)']
               
        self.scalingSelector = QtWidgets.QComboBox()
        self.scalingSelector.addItems(scaling_types)
        self.scalingSelector.currentIndexChanged.connect(self.set_scaling)
        
        num_diagnostic_components_lbl = QtWidgets.QLabel('Maximum number of \ndiagnostic components:')
        self.diagnostic_max_components = QtWidgets.QSpinBox()
        self.diagnostic_max_components.setMinimum(1)
        self.diagnostic_max_components.setValue(10)
        runPCA_diag_button = QtWidgets.QPushButton('Run PCA diagnostics') 
        runPCA_diag_button.clicked.connect(self.on_PCA_Diagnostics)
        runDiagnostics_button = QtWidgets.QPushButton('Run PLS diagnostics')
        runDiagnostics_button.clicked.connect(self.on_Diagnostics)
        separator = QtWidgets.QFrame()
        separator.setFrameShape(QtWidgets.QFrame.HLine)
        separator.setLineWidth(3)

        make_actual_model_lbl = QtWidgets.QLabel('Then select parameters and make models:')
        model_types = ['PCA', 'PLS', 'YO_PLS'] # , 'NMF'
        self.ModelSelector = QtWidgets.QComboBox()
        self.ModelSelector.addItems(model_types)
        self.ModelSelector.currentIndexChanged.connect(self.set_num_components)
        
        num_components_lbl = QtWidgets.QLabel('Number of components:')
        self.num_components = QtWidgets.QSpinBox()
        self.num_components.setMinimum(1)
        self.set_num_components()
        self.num_components.valueChanged.connect(self.update_num_components)
        
        m_hbox1 = QtWidgets.QHBoxLayout()
        m_hbox1.addWidget(var_scaling_lbl)
        m_hbox1.addWidget(self.scalingSelector)
        m_vbox.addLayout(m_hbox1)
        q_hbox2 = QtWidgets.QHBoxLayout() 
        q_hbox2.addWidget(num_diagnostic_components_lbl)
        q_hbox2.addWidget(self.diagnostic_max_components)
        m_vbox.addLayout(q_hbox2)
        m_vbox.addWidget(runPCA_diag_button)
        m_vbox.addWidget(runDiagnostics_button)
        
        m_vbox.addWidget(separator)
        m_vbox.addWidget(make_actual_model_lbl)        
        m_hbox2 = QtWidgets.QHBoxLayout() 
        m_hbox2.addWidget(self.ModelSelector)
        m_hbox2.addWidget(num_components_lbl)
        m_hbox2.addWidget(self.num_components)
        m_vbox.addLayout(m_hbox2)
        makeModel_button = QtWidgets.QPushButton('Make model')
        makeModel_button.clicked.connect(self.on_MakeModel)
        makePred_button = QtWidgets.QPushButton('Make prediction')
        makePred_button.clicked.connect(self.on_MakePrediction)
        m_vbox.addWidget(makeModel_button)
        m_vbox.addWidget(makePred_button)
        tab_vbox.addWidget(model_frame)
        
        self.ModelsListBox = QtWidgets.QListWidget()
        self.ModelsListBox.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        self.ModelsListBox.itemClicked.connect(self.on_ModelsListBox_ItemClicked)
        tab_vbox.addWidget(self.ModelsListBox)
        
        self.modelPreProcessing = QtWidgets.QTextEdit()
        self.modelPreProcessing.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        tab_vbox.addWidget(self.modelPreProcessing)

        
    def build_settings(self, tab):
        tab_hbox = QtWidgets.QHBoxLayout(tab) 
        tab_vbox = QtWidgets.QVBoxLayout()               
        tab_hbox.addLayout(tab_vbox)
        self.plot_DPI_for_GUI = QtWidgets.QSpinBox()
        self.plot_DPI_for_GUI.setPrefix("plot_magnification (dpi) ")
        self.plot_DPI_for_GUI.setMinimum(1)
        self.plot_DPI_for_GUI.setMaximum(2000)
        self.plot_DPI_for_GUI.setValue(self.MPL_screen_dpi)
        self.plot_DPI_for_GUI.valueChanged.connect(self.plot_dpi_change)       
        tab_vbox.addWidget(self.plot_DPI_for_GUI)
        var_scaling_lbl = QtWidgets.QLabel('A Change in plot_magnification will\n take effect after restart of the program')
        tab_vbox.addWidget(var_scaling_lbl)
        tab_vbox.addStretch()
        
        
    def build_tab_model_diagnostics(self, tab):
        tab_hbox = QtWidgets.QHBoxLayout(tab) 
        tab_vbox = QtWidgets.QVBoxLayout()               
        tab_hbox.addLayout(tab_vbox)
        self.Diagnostics = Matplot_view(is_white=self.is_light_background, plt_interaction=1, dpi=self.MPL_screen_dpi )
        tab_hbox.addWidget(self.Diagnostics)
        
        var_scaling_lbl = QtWidgets.QLabel('Variable scaling:')
        self.diagnostics_scaling_info = QtWidgets.QLineEdit()
        self.diagnostics_scaling_info.setReadOnly(True)
        q_hbox1 = QtWidgets.QHBoxLayout()
        q_hbox1.addWidget(var_scaling_lbl)
        q_hbox1.addWidget(self.diagnostics_scaling_info)
        tab_vbox.addLayout(q_hbox1)
        tab_vbox.addStretch()
        self.set_scaling()
        
    def build_scores(self, tab):
        tab_hbox = QtWidgets.QHBoxLayout(tab) 
        self.scores_plot = Matplot_view(is_white=self.is_light_background, 
                                        matPlotLabel_len=self.matPlotLabel_len,
                                        matPlotShowTxtLabel=True,
                                        plt_interaction=2, 
                                        dpi=self.MPL_screen_dpi) #, scoreID_callback=self.plot_preproc_spectrum)
        
        tab_hbox.addWidget(self.scores_plot)
        
    def build_loadings(self, tab):
        tab_hbox = QtWidgets.QHBoxLayout(tab)
        self.loadings_plot = Matplot_view(is_white=self.is_light_background, 
                                          matPlotLabel_len=self.matPlotLabel_len,
                                          matPlotShowTxtLabel=True,
                                          plt_interaction=2, 
                                          dpi=self.MPL_screen_dpi)
        
        tab_hbox.addWidget(self.loadings_plot)
        
        
    def build_prediction_results(self, tab):
        tab_hbox = QtWidgets.QHBoxLayout(tab) 
        tab_vbox = QtWidgets.QVBoxLayout()
        table_data = pd.DataFrame(data={'primary obs ID': ['observation1', 'observation2', 'observation3', 'observation4', 'observation5'],
                                        'predicted \n values': [0, 0, 0, 0, 0], 'DModX': [3, 3, 3, 3, 3]})
        self.prediction_table = QtWidgets.QTableView()
        self.prediction_table.setModel(TableModel(table_data=table_data))
        tab_vbox.addWidget(self.prediction_table)
        tab_hbox.addLayout(tab_vbox)
        hbox_2 = QtWidgets.QHBoxLayout()
        hbox_2.addWidget(QtWidgets.QLabel('Prediction results from dataset:'))        
        self.current_dataset_for_pred = QtWidgets.QLineEdit('')
        hbox_2.addWidget(self.current_dataset_for_pred)
        saveButton = QtWidgets.QPushButton('Save results')
        saveButton.clicked.connect(self.on_Save_predictions)
        hbox_2.addWidget(saveButton)
        tab_vbox.addLayout(hbox_2)
        
        
    def get_obsID_data_splitpoint(self, table_df):
        trace = False
        obsID01_hdr = list(table_df.columns)[0]
        
        all_obsIDs = []
        for obsID in list(table_df.columns):
            if 'OBSID' in obsID:
                all_obsIDs.append(obsID)
              
        if all_obsIDs:
            last_obsID = all_obsIDs[-1]
            last_obsID_ix = list(table_df.columns).index(last_obsID)
        else:
            last_obsID = obsID01_hdr
            last_obsID_ix = 0
        if trace:
            print('last_obsID', last_obsID)
            print('last_obsID_ix', last_obsID_ix)
        return last_obsID, last_obsID_ix
        
        
    def update_table_contents(self, table_qt_obj, table_df, decimals=6):
        
        obsID01_hdr = list(table_df.columns)[0]           
        last_obsID, last_obsID_ix = self.get_obsID_data_splitpoint(table_df)
        
        numVarIDs = list(table_df.columns)[last_obsID_ix+1:]
        table_arr = np.asarray(table_df.loc[:, numVarIDs], dtype=np.float64)
        
        rounded_table_df = pd.DataFrame(table_arr.round(decimals), columns=numVarIDs)
        table_out_df = pd.concat((table_df.loc[:, obsID01_hdr:last_obsID], rounded_table_df), axis=1)
        
        table_qt_obj.setModel(TableModel(table_data=table_out_df, num_obsIds=last_obsID_ix+1))
      
    def plot_dpi_change(self):
        self.MPL_screen_dpi = self.plot_DPI_for_GUI.value()
        
        
    def update_modelling_info(self, X, obs_name_1st, dataset_name):
        self.X_shape.setText(repr(X.shape))
        self.X_obs_1_name.setText(dataset_name)
        

    def set_scaling(self):
        var_scaling = self.scalingSelector.currentText()
        self.diagnostics_scaling_info.setText(var_scaling)
               
    def set_num_components(self):
        if self.ModelSelector.currentText() == 'PCA':
            self.num_components.setValue(self.PCA_components)
        elif self.ModelSelector.currentText() == 'NMF':
            self.num_components.setValue(self.NMF_components)
        elif self.ModelSelector.currentText() == 'PLS':
            self.num_components.setValue(self.PLS_components)
        elif self.ModelSelector.currentText() == 'YO_PLS':
            self.num_components.setValue(self.YO_PLS_components)        
        
    def update_num_components(self):
        if self.ModelSelector.currentText() == 'PCA':
            self.PCA_components = self.num_components.value()
        elif self.ModelSelector.currentText() == 'NMF':
            self.NMF_components = self.num_components.value()
        elif self.ModelSelector.currentText() == 'PLS':
            self.PLS_components = self.num_components.value()
        elif self.ModelSelector.currentText() == 'YO_PLS':
            self.YO_PLS_components = self.num_components.value()
            
    def get_trainingset(self):
        verbose = 1
        D = {}
        if self.current_dataset:
            wkset_df = self.dictionary_of_datasets[ self.current_dataset ]
            xvar_IDs, yvar_IDs = self.get_XY_selections()
            self.remember_XY_settings(self.current_dataset, xvar_IDs, yvar_IDs)
            last_obsID, last_obsID_ix = self.get_obsID_data_splitpoint(wkset_df)
            Xin_df = wkset_df.iloc[:,last_obsID_ix+1:].loc[:, xvar_IDs] # 1st col is primary obs ID
            Xin = np.asarray(Xin_df, dtype=np.float64)
            D['wkset'] = Xin
            D['legend'] = self.current_dataset
            D['VarIDs_list'] = list(Xin_df.columns) 
            if verbose > 1:
                print("trainingset['VarIDs_list']", D['VarIDs_list'] )
            D['xlabel'] = 'Variables'
            D['ylabel'] = 'response'
            D['ObsIDs_list'] = list( wkset_df.iloc[:,0] )
            D['ObsIDs_list_alt'] = list( wkset_df.iloc[:,0] )
            D['additional_obsID_names'] = []
            D['DataSourceID'] = 'MVDA_exploration'
            D['is_inverted_x'] = False
            D['dataset_name'] = self.current_dataset
            self.is_same_wkset = False
        return D


    def get_Yin(self, verbose=0):
        xvar_IDs, yvar_IDs = self.get_XY_selections()
        self.remember_XY_settings(self.current_dataset, xvar_IDs, yvar_IDs)
        if yvar_IDs:
            print('get_Yin', yvar_IDs)
            dataset_df = self.dictionary_of_datasets[ self.current_dataset ]
            Yin = np.asarray(dataset_df.loc[:, yvar_IDs], dtype=np.float64)
            print('Yin', Yin.dtype)
        else:
            Yin = np.array([])
            yvar_IDs = []
        return Yin, yvar_IDs

    
    def get_UV_scaling(self):
        var_scaling = self.scalingSelector.currentText()
        if 'UV' in var_scaling:
            is_UV_scaling = True
        else:
            is_UV_scaling = False
        return is_UV_scaling
    
    
    def YO_plotting_fix(self, model_ID, model):
        if model[model_ID]['meta']['model_type'] == 'YO_PLS':
            P = np.vstack((model[model_ID]['model_obj'].P, model[model_ID]['model_obj'].Po))
            T = np.hstack((model[model_ID]['model_obj'].T, 
                           model[model_ID]['model_obj'].To))
            
            P_labels_prefix =  [model_ID + ' p 1']
            for p_orto_ix in range(model[model_ID]['model_obj'].Po.shape[0]):
                P_labels_prefix.append(model_ID + ' po '+str(p_orto_ix+1))
            
            T_labels_prefix =  ['t 1']
            for t_orto_ix in range(model[model_ID]['model_obj'].To.shape[1]):
                T_labels_prefix.append('to '+str(t_orto_ix+1))
              
        else:
            P = model[model_ID]['model_obj'].P
            T = model[model_ID]['model_obj'].T
            P_labels_prefix = model_ID + ' p '
            T_labels_prefix = 't '
        return P, T, P_labels_prefix, T_labels_prefix 
    

    def get_loadings_for_plot(self, model_ID, P, P_labels_prefix):
        verbose = 0
        if verbose > 0:
             print('P', P.shape, type(P) )
             print('P_labels_prefix', P_labels_prefix)
        M = self.model[model_ID]
        loadings = {}
        loadings['P'] = {}
        loadings['P']['vectors'] = P.T
        loadings['P']['labels_prefix'] = P_labels_prefix
        loadings['P']['plot_title'] = model_ID + ', p loadings'
        if M['meta']['model_type'] == 'PLS':
            loadings['W'] = {}
            loadings['W']['vectors'] = M['model_obj'].W.T
            loadings['W']['labels_prefix'] = model_ID + ' w '
            loadings['W']['plot_title'] = model_ID + ', PLS w loadings'
        return loadings

            
    def on_MakeModel(self):
        verbose = 1
        if verbose > 0:
            print('on_MakeModel')
        meta = {}
        model_obj = None # to not be undefined if a model is not available due to modelling limitations
        # segment_len = self.t_decimation_spin.value()
        # preProcessed = self.get_preprocessed(segment_len=segment_len, 
        #                                      is_spike_removal=self.t_decimation_spike_removal.isChecked())
        trainingset = self.get_trainingset()
        if trainingset:
            if verbose > 1:
                print("trainingset['wkset']", trainingset['wkset'].dtype)
                print(trainingset['wkset'])
            Xin = trainingset['wkset']
            if not np.isfinite(np.asarray(Xin)).all():
                self.showMsgBox(self.missing_values_msg)
            else:
                if verbose > 1:
                    print(trainingset['ObsIDs_list'][0])
                self.update_modelling_info(Xin, trainingset['ObsIDs_list'][0], trainingset['dataset_name'])
                a = self.num_components.value()
                model_ID = 'M'+str(self.model_cnt+1).zfill(2)
                
                meta['var_scaling'] = self.scalingSelector.currentText()
                if 'UV' in meta['var_scaling']:
                    is_UV_scaling = True
                else:
                    is_UV_scaling = False
                    
                meta['applied_preprocessing'] = ''
                # for key in trainingset['applied_preprocessing']:            
                #     meta['applied_preprocessing'] += '    '+key+'\n'
                # meta['applied_preprocessing'] += '    Time averaging 1/'+str(trainingset['time_decimation'])+'\n'
                # if trainingset['is_spike_removal'][0]:
                #     meta['applied_preprocessing'] += '    Time domain spike removal was active'+'\n'
                # meta['time_decimation'] = trainingset['time_decimation']
                # meta['is_spike_removal'] = trainingset['is_spike_removal']
                meta['x_scale'] = trainingset['VarIDs_list'][0]
                meta['xlabel'] = trainingset['xlabel']
                meta['obs_IDs'] = trainingset['ObsIDs_list']
                meta['var_IDs'] = trainingset['VarIDs_list']
                if 'ObsIDs_list_alt' in trainingset:
                    meta['obs_IDs_alt'] = trainingset['ObsIDs_list_alt']
                # meta['obs_ID_times'] = trainingset['ObsIDs_times']
                # meta['obs_ID_timeseries_index'] = trainingset['ObsIDs_timeseries_index']
                # meta['obs_ID_timeseries_classID'] = trainingset['ObsIDs_timeseries_classIDs']
                # if verbose > 1:
                #     print('on_MakeModel', 'obs_ID_timeseries_classID length', len(meta['obs_ID_timeseries_classID']))            
                meta['is_inverted_x'] = trainingset['is_inverted_x']
                meta['model_type'] = self.ModelSelector.currentText()
                meta['is_depth_scale'] = self.is_depth_scale
                                    
                meta['R2X'] = np.asarray([]) # empty R2X
                if meta['model_type'] == 'PCA':
                    model_obj = mv.PCA_model(n_components=a, is_scale=is_UV_scaling)
                    model_obj.fit(Xin)
                    meta['R2X'] = mv.R2X_calc(model_obj.SSX)
                    
                elif meta['model_type'] == 'PLS':
                    Yin, yvar_IDs = self.get_Yin(verbose=verbose)
                    if yvar_IDs:
                        meta['Y_VarIDs'] = yvar_IDs
                    if not Yin.size:
                        self.showMsgBox('Please select one or more Y-variables for PLS')
                    elif not np.isfinite(np.asarray(Yin)).all():
                        self.showMsgBox('Please de-select Y-variables with missing values')
                    else:
                        model_obj = mv.PLS_model(n_components=a, scale=is_UV_scaling)
                        model_obj.fit(Xin, Yin)
                        meta['R2X'] = mv.R2X_calc(model_obj.SSX)
                    
                elif meta['model_type'] == 'YO_PLS':
                    Yin, yvar_IDs = self.get_Yin(verbose=verbose)
                    if yvar_IDs:
                        meta['Y_VarIDs'] = yvar_IDs
                    if not Yin.size:
                        self.showMsgBox('Please select one or more Y-variables for YO_PLS')
                    elif np.squeeze(Yin).ndim > 1:
                        self.showMsgBox("This implementation of YO-PLS only handles one Y-variable")
                    elif not np.isfinite(np.asarray(Yin)).all():
                        self.showMsgBox('Please de-select Y-variables with missing values')
                    else:
                        print('YO_PLS', Yin.shape)
                        model_obj = mv.yo_PLS_model(n_components=a, scale=is_UV_scaling)
                        model_obj.fit(Xin, Yin)
                               
                # elif meta['model_type'] == 'NMF':
                #     model_obj = NonNegMatrixFactorization(n_components=a)
                #     model_obj.fit(Xin)
                
                else:
                    model_obj = None
                    print('Undefined model selection')
                    
        
                if model_obj:
                    var_scaling_info = ' Ctr '
                    if is_UV_scaling:
                        var_scaling_info = ' UV '
                        
                    meta['model_info'] = model_ID+var_scaling_info+self.ModelSelector.currentText()+' '+str(a)+' comp '+repr(Xin.shape)+' '+self.X_obs_1_name.text()            
        
                    M = {}
                    M['model_obj'] = model_obj
                    M['meta'] = meta
                    self.model[model_ID] = M                        
                
                    P, T, P_labels_prefix, T_labels_prefix = self.YO_plotting_fix(model_ID, self.model)
                                
                    loadings = self.get_loadings_for_plot(model_ID, P, P_labels_prefix)
                    
                    #TODO: For loadings the plot_title content is overridden inside the plotting object, fix this to allow extra title info input
                    self.loadings_plot.set_scatter_plot_input(loadings,
                                                              plot_title=model_ID+' '+meta['model_type']+', loadings, scaling:'+var_scaling_info,
                                                              # label_prefix=P_labels_prefix,
                                                              pointIDs=meta['var_IDs'],
                                                              # alt_pointIDs=meta['obs_IDs_alt'],
                                                              R2X=meta['R2X'],
                                                              is_depth_scale = meta['is_depth_scale'] )
                    if verbose > 0:
                        print('T', T.shape, type(T) )            
                    self.scores_plot.set_scatter_plot_input(T, 
                                                            plot_title=model_ID+' '+meta['model_type']+' scores, scaling:'+var_scaling_info, 
                                                            label_prefix=T_labels_prefix, 
                                                            pointIDs=meta['obs_IDs'],
                                                            alt_pointIDs=meta['obs_IDs_alt'],
                                                            # time_scale=meta['obs_ID_times'],
                                                            # time_series_ix=meta['obs_ID_timeseries_index'],
                                                            # time_series_classes=meta['obs_ID_timeseries_classID'],
                                                            R2X=meta['R2X'],
                                                            is_depth_scale = meta['is_depth_scale'])
                    
                    self.addItem_and_setItAsCurrent(self.ModelsListBox, meta['model_info'])
                    
                    self.modelPreProcessing.clear()
                    self.modelPreProcessing.insertPlainText(model_ID+' pre-processing:\n')
                    if meta['applied_preprocessing']:
                        self.modelPreProcessing.insertPlainText(meta['applied_preprocessing'])
                    else:
                        self.modelPreProcessing.insertPlainText('None')
                    self.model_cnt += 1
                self.tab_widget2.setCurrentWidget(self.tab_widget2_tab_selector["Scores"])
                
                    
            
            
    def on_MakePrediction(self):
        
        def prepend_t(n):
            return 't '+str(n).zfill(2)
        
        verbose = 1
        if verbose > 0:
            print('on_MakePrediction')
        if self.ModelsListBox.count() <= 0:
            self.showMsgBox("Please create a model before the prediction")
        else:
            item = self.ModelsListBox.currentItem()
            model_ID = item.text().split(' ')[0]
            if verbose > 0:
                print('model_ID',model_ID)
            M = self.model[model_ID]
            predictionset = self.get_trainingset()
            if predictionset:
                Xpred = predictionset['wkset']
                pred_ObsIDs = predictionset['ObsIDs_list']
                pred_ObsIDs_df = pd.DataFrame(pred_ObsIDs, columns=['Predicted \nobservations'])
                print("predictionset['dataset_name']", predictionset['dataset_name'])
                
                if M['meta']['model_type'] == 'PCA':
                    try:                
                        observation_residuals = M['model_obj'].DModXpred(Xpred)
                        DModX_df = pd.DataFrame(observation_residuals, columns=['DModX'])
                        scores = M['model_obj'].Tpred(Xpred)
                        scores_VarIDs = map(prepend_t, list(np.arange(scores.shape[1])+1))
                        self.scores_predicted_df = pd.concat((pred_ObsIDs_df, pd.DataFrame(scores, columns=scores_VarIDs),  DModX_df), axis=1)
                        self.update_table_contents(self.prediction_table, self.scores_predicted_df)
                        self.current_dataset_for_pred.setText(predictionset['dataset_name'])
                        self.model_ID_last_prediction = model_ID
                        self.last_prediction_results = self.scores_predicted_df
                        self.tab_widget2.setCurrentWidget(self.tab_widget2_tab_selector["Prediction results"])
                    except ValueError as e:
                        error_msg = str(e)
                        self.showMsgBox(error_msg)
                else: #PLS            
                    try:
                        Y_predicted = M['model_obj'].predict(Xpred)
                        if 'Y_VarIDs' in M['meta']:
                            Ypred_VarIDs = M['meta']['Y_VarIDs']
                        else:            
                            Ypred_VarIDs = map(str, list(np.arange(Y_predicted.shape[1])+1))
                        observation_residuals = M['model_obj'].DModXpred(Xpred)                   
                        DModX_df = pd.DataFrame(observation_residuals, columns=['DModX'])
                        self.Y_predicted_df = pd.concat((pred_ObsIDs_df, pd.DataFrame(Y_predicted, columns=Ypred_VarIDs), DModX_df), axis=1) 
                        self.update_table_contents(self.prediction_table, self.Y_predicted_df)
                        self.current_dataset_for_pred.setText(predictionset['dataset_name'])
                        self.model_ID_last_prediction = model_ID
                        self.last_prediction_results = self.Y_predicted_df
                        self.tab_widget2.setCurrentWidget(self.tab_widget2_tab_selector["Prediction results"])
                    except AttributeError as e:
                        error_msg = 'Prediction could not be made \n'+str(e)
                        self.showMsgBox(error_msg)
                    except ValueError as e:
                        error_msg = str(e)
                        self.showMsgBox(error_msg)

                
    def on_Save_predictions(self):
        if self.is_DataRepository:
            targetDir = self.get_New_RepositoryDir('MVDA_exploration')
        else:
            targetDir = self.previous_input_location
        self.CurrentDataRepositoryDir = targetDir
        
        if self.model_ID_last_prediction:        
            target_fname = timestamped_filename(os.path.join(targetDir, 'MVDA_pred_'+self.model_ID_last_prediction+'_')) 
            self.last_prediction_results.to_excel(target_fname+'.xlsx')
                           
            self.statusText.setText('Prediction results from '+self.model_ID_last_prediction+' was saved at '+ targetDir)
        else:
            msg = 'No prediction results available, please select a model and perform a prediction'
            self.showMsgBox( msg)
            self.statusText.setText(msg)
            
            
    def addItem_and_setItAsCurrent(self, ListWidget, itemText):
        item = QtWidgets.QListWidgetItem()
        item.setText(itemText)
        ListWidget.addItem(item)
        ListWidget.setCurrentItem(item)

        
    def on_ModelsListBox_ItemClicked(self, item):
        trace = False
        if trace:
            print('on_ModelsListBox_ItemClicked')

        model_ID = item.text().split(' ')[0]
        if trace:
            print('meta content')
            for entry in self.model[model_ID]['meta']:
                print('    ', entry)
                
        P, T, P_labels_prefix, T_labels_prefix = self.YO_plotting_fix(model_ID, self.model)
        
        loadings = self.get_loadings_for_plot(model_ID, P, P_labels_prefix)
        
        
        if trace:
            print('P', P.shape )
            
        var_scaling_info = self.model[model_ID]['meta']['var_scaling']
            
        #TODO: For loadings the plot_title content is overridden inside the plotting object, fix this to allow extra title info input
        self.loadings_plot.set_scatter_plot_input(loadings,
                                                  plot_title=model_ID+' '+self.model[model_ID]['meta']['model_type']+', loadings, scaling: '+var_scaling_info,
                                                  label_prefix=P_labels_prefix,
                                                  pointIDs=self.model[model_ID]['meta']['var_IDs'],
                                                  # alt_pointIDs=meta['obs_IDs_alt'],
                                                  R2X=self.model[model_ID]['meta']['R2X'],
                                                  is_depth_scale = self.model[model_ID]['meta']['is_depth_scale'] )
        if trace:
            print('T', T.shape )
        
        self.scores_plot.set_scatter_plot_input(T,
                                                plot_title=model_ID+' '+self.model[model_ID]['meta']['model_type']+' scores, scaling: '+var_scaling_info, 
                                                label_prefix= T_labels_prefix, 
                                                pointIDs= self.model[model_ID]['meta']['obs_IDs'],
                                                alt_pointIDs= self.model[model_ID]['meta']['obs_IDs_alt'],
                                                # time_scale= self.model[model_ID]['meta']['obs_ID_times'],
                                                # time_series_ix= self.model[model_ID]['meta']['obs_ID_timeseries_index'],
                                                # time_series_classes= self.model[model_ID]['meta']['obs_ID_timeseries_classID'],
                                                R2X= self.model[model_ID]['meta']['R2X'],
                                                is_depth_scale = self.model[model_ID]['meta']['is_depth_scale'])
        
        self.modelPreProcessing.clear()
        self.modelPreProcessing.insertPlainText(model_ID+' pre-processing:\n')
        if self.model[model_ID]['meta']['applied_preprocessing']:
            self.modelPreProcessing.insertPlainText(self.model[model_ID]['meta']['applied_preprocessing'])
        else:
            self.modelPreProcessing.insertPlainText('None')
        self.tab_widget2.setCurrentWidget(self.tab_widget2_tab_selector["Scores"])

        
    def on_PCA_Diagnostics(self):
        a = self.diagnostic_max_components.value()
        trainingset = self.get_trainingset()
        if trainingset:
            Xin = trainingset['wkset']
            if not np.isfinite(np.asarray(Xin)).all():
                self.showMsgBox(self.missing_values_msg)
            else:
                is_UV_scaling = self.get_UV_scaling()
                max_comp = min(Xin.shape[0], Xin.shape[1]) - 1
                Xcs, x_mean, x_std = mv.center_scale_x(Xin, center=True, scale=is_UV_scaling)
                T, P, eigenvalues = mv.PCA_by_randomizedSVD(Xcs, min(a, max_comp))
                x_axis = np.arange(eigenvalues.shape[0])+1
                #self.Diagnostics.tight_layout()
                self.Diagnostics.clear()
                self.Diagnostics.plot(x_axis, np.log10(eigenvalues), 
                                      markeredgecolor=self.Diagnostics.mkr_edge_color)
                self.Diagnostics.plot(x_axis, np.log10(eigenvalues), linestyle='None', marker='o', 
                                      markeredgecolor=self.Diagnostics.mkr_edge_color)
                self.Diagnostics.set_xlabel('PCA component')
                self.Diagnostics.set_ylabel('log10(eigenvalues)')
                if is_UV_scaling:
                    self.Diagnostics.set_title('Scree plot, unit variance scaled variables')
                else:
                    self.Diagnostics.set_title('Scree plot, mean centered variables')
                self.tab_widget2.setCurrentWidget(self.tab_widget2_tab_selector["Diagnostics"])
        

    def on_Diagnostics(self):
        trace = False
        a = self.diagnostic_max_components.value()
        # segment_len = self.t_decimation_spin.value()
        # preProcessed = self.get_preprocessed(segment_len=segment_len, 
        #                                      is_spike_removal=self.t_decimation_spike_removal.isChecked())
        # Xin = preProcessed['wkset']
        
        trainingset = self.get_trainingset()
        if trainingset:
            Xin = trainingset['wkset']
            if not np.isfinite(np.asarray(Xin)).all():
                self.showMsgBox(self.missing_values_msg)
            else:
                Yin, yvar_IDs = self.get_Yin( verbose=1)
                if not Yin.size:
                    self.showMsgBox('Please select one or more Y-variables for PLS')
                elif not np.all(np.isfinite(Yin)):
                    self.showMsgBox('Y-values for PLS diagnostics were not found as there were missing of infinite values present')
                else:
                    is_UV_scaling = self.get_UV_scaling()
                    R2_calc, Q2_calc = mv.evalPLS_Q2(Xin, Yin, max_comp=a, is_UV_scale=is_UV_scaling, 
                                                     CV_sections=7, shuffle=True, random_state=0, 
                                                     is_plot=False, plt_fname='', plt_dir='')           
                    print()
                    print('R2_calc', R2_calc)
                    print('Q2_calc', Q2_calc)
                    
                    self.Diagnostics.clear()
                    self.Diagnostics.R2_Q2_bars(R2_calc, Q2_calc, edgecolor = self.Diagnostics.mkr_edge_color)
                    self.Diagnostics.set_xlabel('PLS-component')
                    self.Diagnostics.set_ylabel('cumulative fractions')
                    yvar_IDs_txt = ', y='+str(yvar_IDs)
                    if is_UV_scaling:
                        self.Diagnostics.set_title('PLS diagnostics, unit variance scaled vars'+yvar_IDs_txt)
                    else:
                        self.Diagnostics.set_title('PLS diagnostics, mean centered vars'+yvar_IDs_txt)
                    self.tab_widget2.setCurrentWidget(self.tab_widget2_tab_selector["Diagnostics"])


    def createStatusBar(self):
        self.statusText = QtWidgets.QLabel("Ready")
        self.statusBar().addWidget(self.statusText, 1)
        
        
    def clear_data(self):
        self.current_dataset = ''
        self.dictionary_of_datasets = {}
        self.current_files = []
        self.InpDataSetsListBox.clear()


    

    def readQtSettings(self):
        trace = False
        if trace:
            print('readQtSettings')
        settings = QtCore.QSettings("PyQT_settings", "MVDA_exploration")
        current_s = "FileLocation/PreviousInputLocation"
        if settings.contains(current_s):
            self.previous_input_location = settings.value(current_s)
        current_s = "FileLocation/PreviousMatLocation"
        if settings.contains(current_s):
            self.previous_mat_location = settings.value(current_s)
        current_s = "FileLocation/PreviousInputFilter"
        if settings.contains(current_s):
            self.orig_input_selectedFilter = settings.value(current_s) 
        current_s = "FileLocation/PreviousInputLstLocation"
        if settings.contains(current_s):
            self.previous_input_list_location = settings.value(current_s)
        current_s = "FileLocation/PreviousOutputLocation"
        if settings.contains(current_s):
            self.previous_output_location = settings.value(current_s)
        current_s = "FileLocation/PreviousModelsLocation"    
        if settings.contains(current_s):
            self.previous_models_location = settings.value(current_s) 
        
        current_s = "plotOptions/LightBackground"
        if settings.contains(current_s):
            self.is_light_background = settings.value(current_s) == 'true'
        current_s = "plotOptions/labelLength"
        if settings.contains(current_s):
            self.matPlotLabel_len = int(get_number(settings.value(current_s)))
        current_s = "plotOptions/showTextLabel"
        if settings.contains(current_s):
            self.matPlotShowTxtLabel = bool(get_number(settings.value(current_s)))
        current_s = "plotOptions/screen_dpi"
        if settings.contains(current_s):
            self.MPL_screen_dpi = int(get_number(settings.value(current_s)))

        current_s = "preProc/Methods"
        if settings.contains(current_s):
            self.last_selected_preproc_methods = ast.literal_eval(settings.value(current_s))
            
        current_s = "modelHyperparams/PCA_components"
        if settings.contains(current_s):
            self.PCA_components = int(get_number(settings.value(current_s)))
        current_s = "modelHyperparams/NMF_components"
        if settings.contains(current_s):
            self.NMF_components = int(get_number(settings.value(current_s)))        
        current_s = "modelHyperparams/PLS_components"
        if settings.contains(current_s):
            self.PLS_components = int(get_number(settings.value(current_s)))
        current_s = "modelHyperparams/YO_PLS_components"
        if settings.contains(current_s):
            self.YO_PLS_components = int(get_number(settings.value(current_s)))
            
            
    def init_dockableUI(self):
        _DOCK_OPTS = QtWidgets.QMainWindow.AnimatedDocks
        _DOCK_OPTS |= QtWidgets.QMainWindow.AllowNestedDocks
        _DOCK_OPTS |= QtWidgets.QMainWindow.AllowTabbedDocks
        
        # self.view1_data = self.get_data_empty()
        # self.view_frame1 = dqtm.dockable_matplot_view() #self.line_select_callback)
        # self.view_frame1.imshow(self.view1_data, aspect='auto') # , self.getDrawingOptions())#, aspect='auto')
        
        # self.view_frame1 = dqtp.dockable_pyqtGraph(is_white=self.is_light_background,  
        #                                            linewidth = 2)
        
        self.setDockOptions(_DOCK_OPTS)

        self.DockView1 = QtWidgets.QDockWidget('View 1', objectName='dockobj_View_1')
        self.DockView1.setWidget(self.view_frame1)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.DockView1, QtCore.Qt.Vertical)
        self.setCorner( QtCore.Qt.BottomRightCorner, QtCore.Qt.RightDockWidgetArea )
        
        
    def on_OpenFile(self):
        caption = 'Open file'
        old_path = self.previous_input_location
        filter_mask = "All files (*.*)"
        filter_mask += ";;Comma separated values (*.csv);;XML spreadsheet file (*.xlsx)"
        filter_mask += ";;Simca compatible matlab files (*.mat);;Simca compatible matlab-v4 files (*_v4.mat)"
        #filter_mask += ";;SingleBeam SPC files (*.SingleBeam.spc);;Absorbance SPC files (*.Absorbance.spc)"
        #filter_mask += ";;Galactic SPC files (*.spc)
        #filter_mask += ";;OPUS files (*.0)"
        filenames, selectedFilter = QtWidgets.QFileDialog.getOpenFileNames(self, caption, old_path, filter_mask, 
                                                                           self.orig_input_selectedFilter)
        if filenames:
            QtCore.QCoreApplication.processEvents()
            self.previous_input_location = os.path.dirname(filenames[0])
            self.orig_input_selectedFilter = selectedFilter
            self.open_InputFiles(filenames)
        else:
            self.statusText.setText("No files were selected")
            
            
    def open_InputFiles(self, filenames):
        self.is_depth_scale = False
        if os.path.splitext(filenames[0])[1] in ['.csv', '.CSV']:
            self.open_files(filenames, extension='.csv')            
        elif os.path.splitext(filenames[0])[1] in ['.xlsx', '.XLSX']:
            self.open_files(filenames, extension='.xlsx')                        
        elif os.path.splitext(filenames[0])[1] in ['.mat', '.MAT', '_v4.mat']:
            self.open_files(filenames, extension='.mat') 
        #     self.previous_mat_location = os.path.dirname(filenames[0])
        #     self.open_mat_files(filenames, is_timeseries=True)
        # elif os.path.splitext(filenames[0])[1] in ['.0']:
        #     self.open_OPUS_files(filenames)
        # elif os.path.splitext(filenames[0])[1] in ['.spc', '.SPC']:
        #     self.open_files(filenames)
        else:
            self.statusText.setText("The file extension was not recognized")
            
    def InputFilesDropped(self, dropped_listOfFiles):
        if dropped_listOfFiles:
            self.open_InputFiles(dropped_listOfFiles)
            self.previous_input_location = os.path.dirname(dropped_listOfFiles[0])
        else:
            self.statusText.setText("No files were selected")

        
    def open_files(self, filenames, extension='.csv'):
        print('extension', extension)
        print(filenames)
        if filenames:
            for filename in filenames:
                dataset_name = os.path.basename(os.path.splitext(filename)[0])
                if extension == '.csv':
                    self.dictionary_of_datasets[dataset_name] = pd.read_csv(filename)
                elif extension == '.xlsx':
                    self.dictionary_of_datasets[dataset_name] = pd.read_excel(filename)
                elif extension == '.mat':
                    df = self.get_df_from_v4_matfile(filename)
                    if not df.empty:
                        self.dictionary_of_datasets[dataset_name] = self.get_df_from_v4_matfile(filename)
                    
            self.update_InpDataSets()
            self.is_depth_scale = False
            
            
    def get_df_from_v4_matfile(self, filename):
        obj = mv.Mat4_file(filename)
        d1 = obj.loadmat(trace=False)
        if not (('DATA' in d1) and ('VARID1' in d1)):
            df_empty = pd.DataFrame({"empty": []})
            return df_empty
        else:                
            df = pd.DataFrame(d1['DATA'], columns=d1['VARID1'])
            all_obsIDs = []
            unique_obsIDs = []
            unique_obsID_types = []
            obsdIDs_nonfloat  = []
            selected_obsID = ''
            for entry in d1: 
                if ('OBSID' in entry):
                    all_obsIDs.append(entry)
                    if (d1[entry].shape[0] == len(set(d1[entry]))):
                        unique_obsIDs.append(entry)
                        unique_obsID_types.append(str(d1[entry].dtype))

            if len(unique_obsIDs) == 1:
                selected_obsID = unique_obsIDs[0]
            elif len(unique_obsIDs) > 1:
                for nonfloat_ix, typ in enumerate(unique_obsID_types):
                    if typ != 'float64':
                        obsdIDs_nonfloat.append(nonfloat_ix)
                        
                #Take the first unique text based obsID if there is one
                if obsdIDs_nonfloat:
                    selected_obsID = unique_obsIDs[obsdIDs_nonfloat[0]]
                else:
                    selected_obsID = unique_obsIDs[0]
            else: 
                 primary_obsIDs = np.arange(d1['DATA'].shape[0], dtype=np.int64)
                 
            if selected_obsID:
                primary_obsIDs = d1[selected_obsID]
            else:
                primary_obsIDs = np.arange(d1['DATA'].shape[0], dtype=np.int64)
                
            df.insert(loc=0, column=selected_obsID, value=primary_obsIDs)
            
            for obsID in all_obsIDs:
                if obsID != selected_obsID:
                    df.insert(loc=1, column=obsID, value=d1[obsID])
            
            return df
        
        
        
    def on_OpenModels(self):
        verbose = 1
        caption = 'Open models'
        old_path = self.previous_models_location
        filter_mask = "All files (*.*);;Python pickled models (*.pkl)"
        filename, selectedFilter_ = QtWidgets.QFileDialog.getOpenFileName(self, caption, old_path, filter_mask)

        if os.path.splitext(filename)[1] in ['.pkl']:
            with open(filename, 'rb') as f:
                self.model = pickle.load(f)
                
            if verbose > 0:
                for model_ID in self.model:
                    print('model_ID', model_ID)
                    for key in self.model[model_ID]['meta']:
                        print('meta', key)
                        
            #self.apply_backwards_model_compatibility(self.model)
            self.ModelsListBox.clear()            
            for model_ID in self.model:
                self.ModelsListBox.addItem(self.model[model_ID]['meta']['model_info'])    
                self.model_cnt += 1
        
            self.previous_models_location = os.path.dirname(filename)
                
        else:
            self.showMsgBox('The opened file did not have the file extension .pkl')
            
            
    def on_Save_models(self):
        if self.is_DataRepository:
            targetDir = self.get_New_RepositoryDir('MVDA_exploration')
        else:
            targetDir = self.previous_input_location
        self.CurrentDataRepositoryDir = targetDir
        
        for model_ID in self.model:
            print('model_ID', model_ID)
            for key in self.model[model_ID]['meta']:
                print('meta', key)
        
        target_fname = timestamped_filename(os.path.join(targetDir, 'MVDA_models_'))
        with open(target_fname+'.pkl', 'wb') as f:
            pickle.dump(self.model, f)
            
        # for model_ID in self.model:
        #     scores = self.get_scores_as_SimcaMat_dict(model_ID, 
        #                                               self.model[model_ID]['model_obj'].T)
        #     loadings = self.get_loadings_as_SimcaMat_dict(model_ID, 
        #                                                   self.model[model_ID]['model_obj'].P)            
        #     m_type = '_' + self.model[model_ID]['meta']['model_type']
                            
        self.statusText.setText('Model data was saved at '+ targetDir)
        
    def Open_current_output_folder(self):
        print('Enter Open_current_output_folder')
        outDir_windowsstr = ''
        if self.CurrentDataRepositoryDir:
            outDir_windowsstr = self.CurrentDataRepositoryDir
        elif self.previous_input_location:
            outDir_windowsstr = self.previous_input_location
        if outDir_windowsstr:
            webbrowser.open(os.path.join('file://', outDir_windowsstr)) 
        
        
    def clear_entered_data(self):
        self.clear_data()               
        #self.update_InpDataSets()
        #self.tab1_plot_view.clear()
        
        
    def closeEvent(self, event):
        trace = True
        if trace:
            print('Enter  closeEvent')
        settings = QtCore.QSettings("PyQT_settings", "MVDA_exploration")
        settings.setValue("plotOptions/LightBackground", self.is_light_background)
        plotLabel_len = max(self.loadings_plot.get_PlotLabel_len(),
                            self.scores_plot.get_PlotLabel_len())
        settings.setValue("plotOptions/labelLength", plotLabel_len)
        if trace:
            print(self.MPL_screen_dpi)
        settings.setValue("plotOptions/screen_dpi", self.MPL_screen_dpi)
        # settings.setValue("plotOptions/baseline_avg", self.baseline_avg_select_spin.value() )
        if self.previous_input_location:
            settings.setValue("FileLocation/PreviousInputLocation", self.previous_input_location)
        if self.previous_mat_location:
            settings.setValue("FileLocation/PreviousMatLocation", self.previous_mat_location)
        if self.orig_input_selectedFilter:
            settings.setValue("FileLocation/PreviousInputFilter", self.orig_input_selectedFilter)
        if self.previous_input_list_location:
            settings.setValue("FileLocation/PreviousInputLstLocation", self.previous_input_list_location)
        if self.previous_output_location:
            settings.setValue("FileLocation/PreviousOutputLocation", self.previous_output_location)
        if self.previous_models_location:
            settings.setValue("FileLocation/PreviousModelsLocation", self.previous_models_location)
        if self.last_selected_preproc_methods:
            settings.setValue("preProc/Methods", repr(self.last_selected_preproc_methods))

        settings.setValue("modelHyperparams/PCA_components", self.PCA_components)
        settings.setValue("modelHyperparams/NMF_components", self.NMF_components)
        settings.setValue("modelHyperparams/PLS_components", self.PLS_components)
        settings.setValue("modelHyperparams/YO_PLS_components", self.YO_PLS_components)
        
        # settings.setValue("splitParams/dydx_window", self.dydx_window_spin.value())
        # settings.setValue("splitParams/dydx_thr", self.dydx_thr_spin.value())
        # settings.setValue("splitParams/d2ydx_window", self.d2ydx_window_spin.value())
        # settings.setValue("splitParams/dist_to_end", self.dist_to_end_spin.value())
        
            
        QtWidgets.QMainWindow.closeEvent(self, event)
        
        
    def set_dark_style_plots(self):
        self.is_light_background = False
        self.showMsgBox("Dark background in plots will be set after a restart of the program")
        

    def set_light_style_plots(self):
        self.is_light_background = True
        self.showMsgBox("Light background in plots will be set after a restart of the program")
        
    def set_plot_DPI_level(self):
        pass
        
        
    def OnInpDataSetsListBox_ItemClicked(self):
          
        current_selection = []
        spectraItems = self.InpDataSetsListBox.selectedItems()
        for item in spectraItems:
            current_selection.append(item.text())
        print('current_selection', current_selection[0])
        self.current_dataset = current_selection[0]
        self.update_table_contents(self.table, self.dictionary_of_datasets[self.current_dataset])

        self.clear_variables_table()
        self.update_variables_table(self.current_dataset, XY_item_settings=self.XY_settings)
        # The below calls have no function here, too early in the processing
        # xvar_IDs, yvar_IDs = self.get_XY_selections()
        # self.remember_XY_settings(self.current_dataset, xvar_IDs, yvar_IDs)
        self.tab_widget2.setCurrentWidget(self.tab_widget2_tab_selector["Data table"])
        
        
        
    def update_InpDataSets(self):
        self.InpDataSetsListBox.clear()
        for key in self.dictionary_of_datasets:
            self.InpDataSetsListBox.addItem(key)
            
    def showMsgBox(self, msg):
        msgBox = QtWidgets.QMessageBox()
        msgBox.setText(msg)
        msgBox.exec_()
        
        
    def show_data_repository(self):
        if self.OutputRepository.get_location():
            QtWidgets.QMessageBox.information(self,'Location', 
                                              'Current location of data repository is \n' +self.OutputRepository.get_location())
        else:
            QtWidgets.QMessageBox.information(self, 'Location', 
                                              'Current location of data repository is not set, \n Please set it by: Settings, Set data repository location')

    def set_data_repository(self):
        dialog = QtWidgets.QFileDialog()
        dialog.setFileMode(QtWidgets.QFileDialog.Directory)
        dialog.setOption(QtWidgets.QFileDialog.ShowDirsOnly, True)
        dialog.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        directory = dialog.getExistingDirectory(self, 'Choose directory for results repository', os.path.curdir)
        if directory:
            self.statusText.setText("Selected repository: %s" % directory)
            self.OutputRepository.set_location(directory)
        else:
            if self.OutputRepository.get_location():
                self.statusText.setText('Info ---- data repository was kept at: '+self.OutputRepository.get_location())
            else:
                self.statusText.setText('Error ---- data repository was not set')
        
    def get_New_RepositoryDir(self, RepositoryId):
        CurrentDataDir = self.OutputRepository.get_new_entry_in_data_repository(RepositoryId)
        if not CurrentDataDir:
            self.set_data_repository()
            CurrentDataDir = self.OutputRepository.get_new_entry_in_data_repository(RepositoryId)                        
        DataRepository.make_dir_if_absent(CurrentDataDir)
        return CurrentDataDir
            

def main():
    app = QtWidgets.QApplication(sys.argv)
    screen = exploration_GUI()
    screen.show()
    sys.exit(app.exec_())
    
        

if __name__ == '__main__':
    print()
    print('Running PyQt environment is:', qtpy.API_NAME)
    print('Running sklearn version is', sklearn.__version__  )       
    main()