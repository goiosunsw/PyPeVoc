#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  ProgressDisplay.py
#  
#  An IPython-friendly progress bar
#
#
#  Copyright 2014 Andre Almeida <goios@AndreUbuntu>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  

import sys

try:
    from IPython.core.display import clear_output
    have_ipython = True
except ImportError:
    have_ipython = False

try:
    from IPython.display import display
    from ipywidgets import IntProgress, HTML, VBox
    have_ipywidgets=True
except ImportError:
    have_ipywidgets=False

def in_ipynb():
    try:
        cfg = get_ipython().config
        try:
            ipk = cfg['IPKernelApp']
            if len(ipk)==0:
                return False
        except KeyError:
            return False
        return True
    except NameError:
        return False



class Progress(object):
    def __init__(self, end=1.):
        """
        Creates a progress bar display
        """
        self.current_val = 0.0
        self.max_val = end
        if in_ipynb():
            if have_ipywidgets:
                self.label = HTML()
                self.progress = IntProgress(min=0,max=100,value=1)
                self.progress.bar_style = 'info'
                self.progressHTML = VBox([self.label, self.progress])
                display(self.progressHTML)
                self.redraw = self._redraw_ipywidgets
                self.cleanup = self._cleanup_ipywidgets
            else:
                self.redraw = self._redraw_ipython
                self.cleanup = self._cleanup_ipython
        else:
            self.redraw = self._redraw_console
            self.cleanup = self._cleanup_console


    def update(self, val):
        """
        Update the progress bar value
        """
        self.current_val = val
        self.redraw()

    def _redraw_ipywidgets(self):
        self.label.value = str(self)
        self.progress.value = self.current_val/self.max_val*100

    def _redraw_ipython(self):
        clear_output()
        print(str(self))
        sys.stdout.flush()

    def _redraw_console(self):
        print('\r'+str(self),end=" ")
        sys.stdout.flush()
        
    def __str__(self):
        pct = self.current_val/self.max_val*100
        return '%d / %d (%.2f%%)'%(self.current_val,self.max_val,pct)

    def _cleanup_console(self):
        print('\n')

    def _cleanup_ipython(self):
        pass

    def _cleanup_ipywidgets(self):
        pass

    def finish(self):
        self.update(self.max_val)
        self.cleanup()


        
