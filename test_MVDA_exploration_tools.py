#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 12:48:29 2021

@author: mats_j
"""

import unittest
import numpy as np
import pandas as pd
import MVDA_exploration_tools as mv

class TestMVDA_exploration_tools(unittest.TestCase):
    
    def load_data(self):
        dataFrame = pd.read_excel('decimated_spectra.xlsx')
        named_Obs_dataFrame = dataFrame.set_index('sample')
        X = named_Obs_dataFrame.loc['fm01':'fm10' , '250':'418']
        y = named_Obs_dataFrame.loc['fm01':'fm10' , ['Met']]
        test_set = named_Obs_dataFrame.loc['fm11':'fm12' , '250':'418']
        return X, y, test_set


    def test_PCA_R2X_df(self):
        X, y, test_set = self.load_data()
        
        M1 = mv.PCA_model(n_components=3)
        M1.fit(X)
        
        R2X_expected_arr = np.asarray([0.65583533, 0.99369448, 0.99991898], dtype=np.float64) 
        R2X_expected_df = pd.DataFrame(R2X_expected_arr, columns=['R2X'], index=np.arange(len(R2X_expected_arr), dtype=int)+1)
        with self.subTest(msg='PCA model R2X from dataframe'):
             self.assertTrue(np.allclose(R2X_expected_df, M1.R2X))
        
    
    def test_PCA_DModX_df(self):
        X, y, test_set = self.load_data()
        
        M1 = mv.PCA_model(n_components=2)
        M1.fit(X)

        DModX_test_set_expected_arr = np.asarray([3.65170754, 3.71251315], dtype=np.float64)
        DModX_test_set_expected_df = pd.DataFrame(DModX_test_set_expected_arr, columns=['DModX'], index=['fm11', 'fm12'])
        with self.subTest(msg='PCA prediction DModX from dataframe'):
            DModX_test_set = M1.DModXpred(test_set)
            self.assertTrue(np.allclose(DModX_test_set_expected_df, DModX_test_set))

            
    def test_PCA_DModX_forced_arr(self):
        X, y, test_set = self.load_data()
        
        M1 = mv.PCA_model(n_components=2, force_np_type_out=True)
        M1.fit(X)
        
        DModX_test_set_expected_arr = np.asarray([3.65170754, 3.71251315], dtype=np.float64)
        with self.subTest(msg='PCA prediction DModX by forced np.array output'):
            DModX_test_set = M1.DModXpred(test_set)
            self.assertTrue(np.allclose(DModX_test_set_expected_arr, DModX_test_set))
        
        
    def test_PCA_DModX_input_arr(self):
        X, y, test_set = self.load_data()
        
        M1 = mv.PCA_model(n_components=2)
        M1.fit(np.asarray(X))
        
        DModX_test_set_expected_arr = np.asarray([3.65170754, 3.71251315], dtype=np.float64)
        with self.subTest(msg='PCA prediction DModX by forced np.array output'):
            DModX_test_set = M1.DModXpred(np.asarray(test_set))
            self.assertTrue(np.allclose(DModX_test_set_expected_arr, DModX_test_set))

    
    def test_R2_Q2(self):
        X, y, test_set = self.load_data()
        
        R2_expected = np.asarray([0.90209161, 0.99730259, 0.99894325, 0.99908862], dtype=np.float64) 
        Q2_expected = np.asarray([0.66441481, 0.99013783, 0.99404477, 0.99404441], dtype=np.float64)
        R2, Q2 = mv.evalPLS_Q2(X, y, is_plot=False)
        with self.subTest(msg='evalPLS R2'):
            self.assertTrue(np.allclose(R2_expected, R2))
        with self.subTest(msg='evalPLS Q2'):    
            self.assertTrue(np.allclose(Q2_expected, Q2))
 
            
    def test_PLS(self):
        X, y, test_set = self.load_data()
        
        M1 = mv.PLS_model(n_components=2)
        M1.fit(X, y)
        expected = np.asarray([[0.0033882 ], [0.29754326]], dtype=np.float64)
        with self.subTest(msg='PLS predict'):
            prediction1 = M1.predict(test_set)
            self.assertTrue(np.allclose(expected, prediction1))
        """ 
        looking for previously seen side effect """
        with self.subTest(msg='Same model, 2nd PLS predict'):
            prediction2 = M1.predict(test_set)
            self.assertTrue(np.allclose(expected, prediction2))
                
   
    def test_PLS_DModX_df(self):
        X, y, test_set = self.load_data()
        
        M1 = mv.PLS_model(n_components=2)
        M1.fit(X, y)

        DModX_test_set_expected_arr = np.asarray([3.67359829, 3.74572741], dtype=np.float64)
        DModX_test_set_expected_df = pd.DataFrame(DModX_test_set_expected_arr, columns=['DModX'], index=['fm11', 'fm12'])
        with self.subTest(msg='PLS prediction DModX from dataframe'):
            DModX_test_set = M1.DModXpred(test_set)
            self.assertTrue(np.allclose(DModX_test_set_expected_df, DModX_test_set))
            
            
    def test_PLS_DModX_forced_arr(self):
        X, y, test_set = self.load_data()
        
        M1 = mv.PLS_model(n_components=2, force_np_type_out=True)
        M1.fit(X, y)

        DModX_test_set_expected_arr = np.asarray([3.67359829, 3.74572741], dtype=np.float64)
        with self.subTest(msg='PLS prediction DModX by forced np.array output'):
            DModX_test_set = M1.DModXpred(test_set)
            self.assertTrue(np.allclose(DModX_test_set_expected_arr, DModX_test_set))
            
            
    def test_PLS_DModX_input_arr(self):
        X, y, test_set = self.load_data()
        
        M1 = mv.PLS_model(n_components=2, force_np_type_out=False)
        M1.fit(np.asarray(X), np.asarray(y))

        DModX_test_set_expected_arr = np.asarray([3.67359829, 3.74572741], dtype=np.float64)
        with self.subTest(msg='PLS prediction DModX from input np.array'):
            DModX_test_set = M1.DModXpred(np.asarray(test_set))
            self.assertTrue(np.allclose(DModX_test_set_expected_arr, DModX_test_set))
            
            
    def test_yo_PLS(self):
        X, y, test_set = self.load_data()
        
        M2 = mv.yo_PLS_model(n_components=2)
        M2.fit(X, y)
        
        P_expected = np.asarray([[ 0.17801156,  0.98273929, -0.01065605, -0.01768575, 
                                  -0.01471857, -0.02603979, -0.03520374]], dtype=np.float64)
    
        Po_expected = np.asarray([[0.81957925, 0.16199721, 0.09766808, 0.26189987,
                                   0.47165805, 0.27809732, 0.06612749]], dtype=np.float64)

        with self.subTest(msg='yo_PLS predictive loadings (P)'):
            self.assertTrue(np.allclose(P_expected, M2.P))
            
        with self.subTest(msg='yo_PLS y-orthogonal loadings (Po)'):
            self.assertTrue(np.allclose(Po_expected, M2.Po))
            
        T_expected = np.asarray([[-0.13002167],
                                 [-0.13114534],
                                 [-0.11970962],
                                 [ 0.00579392],
                                 [ 0.00881205],
                                 [ 0.00150221],
                                 [-0.00305661],
                                 [ 0.12597339],
                                 [ 0.12251221],
                                 [ 0.11933945]], dtype=np.float64)
    
        To_expected = np.asarray([[-1.54644851e-01],
                                  [-2.70394621e-02],
                                  [ 1.40186326e-01],
                                  [-1.47695264e-01],
                                  [ 3.20210145e-05],
                                  [ 1.28214215e-01],
                                  [ 1.02445002e-01],
                                  [-1.31375581e-01],
                                  [-2.26175284e-02],
                                  [ 1.12495122e-01]], dtype=np.float64)
    
            
        with self.subTest(msg='yo_PLS predictive scores (T)'):
            self.assertTrue(np.allclose(T_expected, M2.T))
            
        with self.subTest(msg='yo_PLS y-orthogonal scores (To)'):
            self.assertTrue(np.allclose(To_expected, M2.To))    
            
        
        
        
        
        
        
        
        