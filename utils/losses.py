# -*- coding: utf-8 -*-
"""losses.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1fNY7rq9yd0Z2D9LKCyMpD6gkV6Rwrsjf
"""

def ph_loss(input,target,c):
    assert input.shape == target.shape, "Input and target must have the same shape"
    return th.sqrt((input - target) ** 2 + c**2) - c