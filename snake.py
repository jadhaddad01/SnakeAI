"""
Author: Jad Haddad

flappy_bird
https://github.com/jadhaddad01/FlappyBirdAI
Flappy Bird Artificial Intelligence
Using the NEAT Genetic Neural Network Architecture to train a set of birds to play the popular game Flappy Bird. Also playable by user.

License:
-------------------------------------------------------------------------------
The MIT License (MIT)
Copyright 2017-2020 Pablo Pizarro R. @ppizarror
Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software
is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
-------------------------------------------------------------------------------
"""

# -----------------------------------------------------------------------------
# Import libraries
# -----------------------------------------------------------------------------
# Public Libraries
import pygame
import neat
import time
import os
import random
import pygame_menu
import pickle
from PIL import Image
# Utils Folder files
from utils import UI, visualize, confmodif

# -----------------------------------------------------------------------------
# Pygame and font initialization
# -----------------------------------------------------------------------------
pygame.init()
pygame.font.init()

# -----------------------------------------------------------------------------
# Constants and global variables
# -----------------------------------------------------------------------------
WIN_WIDTH = 500
WIN_HEIGHT = 800
FPS = 30

# Load Window and Menu Button
win = UI.Window(WIN_WIDTH, WIN_HEIGHT)
button2 = UI.Button(
    x = 500 - 10 - 120, # Top right under score count (SEE DISPLAY FUNCTIONS)
    y = 50, 
    w = 120,
    h = 35,
    param_options={
        'curve': 0.3,
        'text' : "Menu",
        'font_colour': (255, 255, 255),
        'background_color' : (200, 200, 200), 
        'hover_background_color' : (160, 160, 160),
        'outline_half': False
    }
)

# Generation Count and Image Display
gen = 0
neural_net_image = None