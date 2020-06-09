#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 16:53:15 2019

@author: ms
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import docstring
    
def axspan(self, xmin, xmax, ymin, ymax, **kwargs):
    """
    Add a vertical span (rectangle) across the axes.
    
    Draw a vertical span (rectangle) from `xmin` to `xmax`.  With
    the default values of `ymin` = 0 and `ymax` = 1. This always
    spans the yrange, regardless of the ylim settings, even if you
    change them, e.g., with the :meth:`set_ylim` command.  That is,
    the vertical extent is in axes coords: 0=bottom, 0.5=middle,
    1.0=top but the x location is in data coordinates.
    
    Parameters
    ----------
    xmin : scalar
        Number indicating the first X-axis coordinate of the vertical
        span rectangle in data units.
    xmax : scalar
        Number indicating the second X-axis coordinate of the vertical
        span rectangle in data units.
    ymin : scalar, optional
        Number indicating the first Y-axis coordinate of the vertical
        span rectangle in relative Y-axis units (0-1). Default to 0.
    ymax : scalar, optional
        Number indicating the second Y-axis coordinate of the vertical
        span rectangle in relative Y-axis units (0-1). Default to 1.
    
    Returns
    -------
    rectangle : matplotlib.patches.Polygon
        Vertical span (rectangle) from (xmin, ymin) to (xmax, ymax).
    
    Other Parameters
    ----------------
    **kwargs
        Optional parameters are properties of the class
        matplotlib.patches.Polygon.
    
    See Also
    --------
    axhspan : Add a horizontal span across the axes.
    
    Examples
    --------
    Draw a vertical, green, translucent rectangle from x = 1.25 to
    x = 1.55 that spans the yrange of the axes.
    
    >>> axvspan(1.25, 1.55, facecolor='g', alpha=0.5)
    
    """
    #trans = self.get_xaxis_transform(which='grid')
    
    # process the unit information
    self._process_unit_info([xmin, xmax], [ymin, ymax], kwargs=kwargs)
    
    # first we need to strip away the units
    xmin, xmax = self.convert_xunits([xmin, xmax])
    ymin, ymax = self.convert_yunits([ymin, ymax])
    
    verts = [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)]
    p = mpatches.Polygon(verts, **kwargs)
    #p.set_transform(trans)
    self.add_patch(p)
    self.autoscale_view(scaley=False)
    return p
plt.Axes.axspan = axspan

@docstring.copy_dedent(plt.Axes.axspan)
def axspan(xmin, xmax, ymin, ymax, **kwargs):
    return plt.gca().axspan(xmin, xmax, ymin, ymax, **kwargs)
plt.axspan = axspan