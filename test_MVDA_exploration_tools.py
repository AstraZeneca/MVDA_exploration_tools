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
        #expected2 = np.asarray([[0.0033882 ], [0.39754326]], dtype=np.float64)
        with self.subTest(msg='Same model, 2nd PLS predict'):
            prediction2 = M1.predict(test_set)
            self.assertTrue(np.allclose(expected, prediction2))
        
        DModX_test_set_expected = np.asarray([3.67359829, 3.74572741], dtype=np.float64)
        with self.subTest(msg='PLS prediction DModX'):
            DModX_test_set = M1.DModXpred(np.asarray(test_set))
            self.assertTrue(np.allclose(DModX_test_set_expected, DModX_test_set))
        
   
            
            
        
        
        
        
        
        
        
        