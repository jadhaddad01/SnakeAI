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
import math
import pygame
import neat
import time
import os
import random
from random import choice
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
WIN_WIDTH = 800
GAME_WIN_WIDTH = 600
GAME_WIN_HEIGHT = 600
FPS = 30

# Load Window and Menu Button
win = UI.Window(WIN_WIDTH, GAME_WIN_HEIGHT)
button2 = UI.Button(
    x = WIN_WIDTH - 10 - 120, # Top right under score count (SEE DISPLAY FUNCTIONS)
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

block_count = 0

# Load Fonts
STAT_FONT = pygame.font.SysFont("comicsans", 50)
STAT_FONT_SMALL = pygame.font.SysFont("comicsans", 30)
STAT_FONT_BIG = pygame.font.SysFont("comicsans", 100)

# -----------------------------------------------------------------------------
# Classes
# -----------------------------------------------------------------------------
class Snake:
	VEL = 15
	x = [0] # Where index 0 is the head
	y = [0]

	savedx = 0
	savedy = 0

	def __init__(self, x, y):
		self.x = [x, x, x]
		self.y = [y, y + 15, y + 30] # We want to start with a snake with length of 3
		self.direction = "Up"

	def move_right(self):
		if self.x[0] % 30 == 0 and self.y[0] % 30 == 0: # We want everything to be in the same "grid"
			self.direction = "Right"

	def move_left(self):
		if self.x[0] % 30 == 0 and self.y[0] % 30 == 0:
			self.direction = "Left"

	def move_up(self):
		if self.x[0] % 30 == 0 and self.y[0] % 30 == 0:
			self.direction = "Up"

	def move_down(self):
		if self.x[0] % 30 == 0 and self.y[0] % 30 == 0:
			self.direction = "Down"

	def move(self):

		for n in range(len(self.x) - 1, 0, -1):
			self.x[n] = self.x[n - 1]
			self.y[n] = self.y[n - 1]

		# if not self.y == 0 and self.direction == "Up": # Force snake not to hit wall
		if self.direction == "Up":
			self.y[0] = self.y[0] - self.VEL

		# if not self.y == GAME_WIN_HEIGHT - 30 and self.direction == "Down":
		if self.direction == "Down":
			self.y[0] = self.y[0] + self.VEL

		# if not self.x == GAME_WIN_WIDTH - 30 and self.direction == "Right":
		if self.direction == "Right":
			self.x[0] = self.x[0] + self.VEL

		# if not self.x == 0 and self.direction == "Left":
		if self.direction == "Left":
			self.x[0] = self.x[0] - self.VEL
			

	def wall_collision(self):
		if self.y[0] < 0 or (self.y[0] == GAME_WIN_HEIGHT - 15 and self.direction == "Down") or self.x[0] < 0 or (self.x[0] == GAME_WIN_WIDTH - 15 and self.direction == "Right"):
			return True

		return False

	def snake_collision(self):
		# Make list of box coordinates sublist
		tmp = []
		for n in range(len(self.x)):
			tmp.append([self.x[n], self.y[n]])

		return not len([list(i) for i in set(map(tuple, tmp))]) == len(tmp) # Checks for duplicate sublists

	def get_last_block(self):
		return (self.x[len(self.x) - 1], self.y[len(self.y) - 1])

	def get_coord_head(self):
		return (self.x[0], self.y[0])

	def get_body(self):
		return (self.x, self.y)

	def add_block(self, xadd, yadd):
		self.x.append(xadd)
		self.y.append(yadd)

	def dis_to_snake_or_wall(self):
		left = 0
		right = 0
		top = 0
		bottom = 0

		# we want closest block not farthest
		leftflag = True
		rightflag = True
		topflag = True
		bottomflag = True

		# Snake
		for n in range(1, len(self.x)): # Don't include head
			if self.y[n] == self.y[0]:
				if self.x[n] < self.x[0] and leftflag:
					left = self.x[0] - self.x[n]
					leftflag = False
				if self.x[n] > self.x[0] and rightflag:
					right = self.x[n] - self.x[0]
					rightflag = False

			if self.x[n] == self.x[0]:
				if self.y[n] < self.y[0] and bottomflag:
					bottom = self.y[0] - self.y[n]
					bottomflag = False
				if self.y[n] > self.y[0] and topflag:
					top = self.y[n] - self.y[0]
					topflag = False

		# Wall IF NO SNAKE
		if left == 0:
			left = self.x[0]
		if right == 0:
			right = GAME_WIN_WIDTH - self.x[0]
		if top == 0:
			top = self.y[0]
		if bottom == 0:
			bottom = GAME_WIN_HEIGHT - self.y[0]

		return (right, left, bottom, top)

	def draw(self, win):
		for n in range(len(self.x)): # x has same length as y
			pygame.draw.rect(win, (255,255,255), (self.x[n], self.y[n], 30, 30))

class Food:

	def __init__(self):
		self.x = random.randrange(0, GAME_WIN_WIDTH / 30) * 30 # We want everything to be in the same "grid"
		self.y = random.randrange(0, GAME_WIN_HEIGHT / 30) * 30

	def new(self, snake):

		not_satisfied = True
		while not_satisfied:
			self.x = random.randrange(0, GAME_WIN_WIDTH / 30) * 30
			self.y = random.randrange(0, GAME_WIN_HEIGHT / 30) * 30

			# Check if not in same position as snake body
			tmp = []
			# Make Snake Body 2D Array
			(xbody, ybody) = snake.get_body()
			for n in range(len(xbody)):
				tmp.append([xbody[n], ybody[n]])
			# Add food in array and check if two coordinates are the same
			tmp.append([self.x, self.y])
			# If two coordinates are the same we make new food again
			if len([list(i) for i in set(map(tuple, tmp))]) == len(tmp):
				not_satisfied = False

	def eaten(self, snake):
		(snakex, snakey) = snake.get_coord_head()
		if self.x == snakex and self.y == snakey:
			return True
		return False

	def distance_to_food(self, snake):
		(headx, heady) = snake.get_coord_head()

		x = (self.x - headx) ** 2
		y = (self.y - heady) ** 2

		return math.sqrt(x + y)

	def draw(self, win):
		pygame.draw.rect(win, (255,0,0), (self.x, self.y, 30, 30))


# -----------------------------------------------------------------------------
# Methods
# -----------------------------------------------------------------------------
def draw_window_human(win, snake, food, score, pregame):
	"""
	Draw game using given parameters (Human Game)
	Can draw both pregame and main game

	:return: None
	"""
	win.fill((0,0,0))

	snake.draw(win)

	food.draw(win)

	pygame.draw.line(win, (255,255,255), (GAME_WIN_WIDTH, 0), (GAME_WIN_WIDTH, GAME_WIN_HEIGHT))

	# Draw Current Score
	text = STAT_FONT.render("Score: " + str(score), 1, (255, 255, 255))
	win.blit(text, (WIN_WIDTH - 10 - text.get_width(), 10))

	if pregame:
		# Draw Transparency Over Game
		transparency_size = (WIN_WIDTH, GAME_WIN_HEIGHT)
		transparency = pygame.Surface(transparency_size)
		transparency.set_alpha(150)
		win.blit(transparency, (0,0))

		# Main Text
		text = STAT_FONT_BIG.render("Press Arrow Key", 1, (255, 255, 255))
		win.blit(text, (GAME_WIN_WIDTH/2- text.get_width()/2, GAME_WIN_HEIGHT/2 - text.get_height()))

	# Return To Menu if Menu Button Pressed / Draw menu button
	if button2.update():
		menu()

	# Update the Current Display
	pygame.display.update()

def main_human():
	"""
	Play game for user

	:return: None
	"""

	# Global Variables
	global FPS
	global block_count

	# Set Variables
	snake = Snake(GAME_WIN_WIDTH / 2, GAME_WIN_HEIGHT / 2)
	food = Food()

	# win = pygame.display.set_mode((WIN_WIDTH, GAME_WIN_HEIGHT))
	clock = pygame.time.Clock()

	score = 0

	# -------------------------------------------------------------------------
	# Game: Before the Game
	# -------------------------------------------------------------------------
	run_pregame = True
	while run_pregame:
		clock.tick(FPS) # Allow only for FPS Frames per Second
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				run = False
				pygame.quit()
				quit()

		# Start Game When Any key is pressed
		keys = pygame.key.get_pressed()
		if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
			snake.move_right()
			run_pregame = False
		if keys[pygame.K_LEFT] or keys[pygame.K_a]:
			snake.move_left()
			run_pregame = False
		if keys[pygame.K_DOWN] or keys[pygame.K_s]:
			snake.move_down()
			run_pregame = False
		if keys[pygame.K_UP] or keys[pygame.K_w]:
			snake.move_up()
			run_pregame = False

		# -------------------------------------------------------------------------
		# Draw To Screen
		# -------------------------------------------------------------------------
		draw_window_human(win, snake, food, score, True)

	# -------------------------------------------------------------------------
	# Game: Main Game
	# -------------------------------------------------------------------------
	run = True
	while run:
		clock.tick(FPS)
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				run = False
				pygame.quit()
				quit()

		# Go Right / Left / Up / Down
		keys = pygame.key.get_pressed()
		if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
			snake.move_right()
		if keys[pygame.K_LEFT] or keys[pygame.K_a]:
			snake.move_left()
		if keys[pygame.K_DOWN] or keys[pygame.K_s]:
			snake.move_down()
		if keys[pygame.K_UP] or keys[pygame.K_w]:
			snake.move_up()

		# Move To Wanted Direction
		snake.move()

		if block_count == 1:
			block_count += 1

		if block_count == 2:
			snake.add_block(xsaved, ysaved)
			block_count = 0

		if snake.wall_collision() or snake.snake_collision():
			main_human() # Go "back" to pregame

		if food.eaten(snake):
			score += 1
			
			# Create Extra Block Snake
			(xsaved, ysaved) = snake.get_last_block()
			block_count += 1

			food.new(snake)

		# -------------------------------------------------------------------------
		# Draw To Screen
		# -------------------------------------------------------------------------
		draw_window_human(win, snake, food, score, False)

def draw_window_ai(win, snake, food, score, gen):
	"""
	Draw game using given parameters (Human Game)
	Can draw both pregame and main game

	:return: None
	"""
	win.fill((0,0,0))

	snake.draw(win)

	food.draw(win)

	pygame.draw.line(win, (255,255,255), (GAME_WIN_WIDTH, 0), (GAME_WIN_WIDTH, GAME_WIN_HEIGHT))

	# Draw Current Score
	text = STAT_FONT.render("Score: " + str(score), 1, (255, 255, 255))
	win.blit(text, (WIN_WIDTH - 10 - text.get_width(), 10))

	# Draw Current Generation
	text = STAT_FONT.render("Gen: " + str(gen), 1, (255, 255, 255))
	win.blit(text, (WIN_WIDTH - 10 - text.get_width(), GAME_WIN_HEIGHT - 10 - text.get_height()))

	# Return To Menu if Menu Button Pressed
	if button2.update():
		menu()

	# Update the Current Display
	pygame.display.update()

def main_ai(genomes, config):
	"""
	Play game for user

	:return: None
	"""

	# Global Variables
	global FPS
	global block_count
	global gen


	ge = genomes[0][1]
	net = neat.nn.FeedForwardNetwork.create(ge, config)
	ge.fitness = 0

	# Fix second genome error
	genomes[1][1].fitness = -10

	# Set Variables
	snake = Snake(GAME_WIN_WIDTH / 2, GAME_WIN_HEIGHT / 2)
	food = Food()

	current_time = time.time()

	# win = pygame.display.set_mode((WIN_WIDTH, GAME_WIN_HEIGHT))
	clock = pygame.time.Clock()

	score = 0
	gen += 1

	# -------------------------------------------------------------------------
	# Game: Main Game
	# -------------------------------------------------------------------------
	run = True
	while run:
		clock.tick(FPS)
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				run = False
				pygame.quit()
				quit()

		(headx, heady) = snake.get_coord_head()
		food_distance = food.distance_to_food(snake)
		
		# SubInp: Distance to Snake OR Wall (What's Closest)
		(right, left, down, up) = snake.dis_to_snake_or_wall()

		# Inputs: Headx, Heady, Distance to Food, Subinp1, Subinp2
		outputs = net.activate((headx, heady, food_distance, right, left, down, up))

		# Go Right / Left / Up / Down
		if outputs[0] > 0.5:
			snake.move_right()
		if outputs[1] > 0.5:
			snake.move_left()
		if outputs[2] > 0.5:
			snake.move_down()
		if outputs[3] > 0.5:
			snake.move_up()

		# Move To Wanted Direction
		snake.move()

		if block_count == 1:
			block_count += 1

		if block_count == 2:
			snake.add_block(xsaved, ysaved)
			block_count = 0

		if snake.wall_collision() or snake.snake_collision():
			ge.fitness -= 2
			run = False
			break

		# We don't want the snake to go in loops
		if time.time() - current_time >= 10:
			 ge.fitness -= 2
			 run = False
			 break

		if food.eaten(snake):
			score += 1
			ge.fitness += 2
			
			# Create Extra Block Snake
			(xsaved, ysaved) = snake.get_last_block()
			block_count += 1

			current_time = time.time() # Reset timer

			food.new(snake)

		# -------------------------------------------------------------------------
		# Draw To Screen
		# -------------------------------------------------------------------------
		draw_window_ai(win, snake, food, score, gen)

def run(config_path):
	"""
	Use given configuration path and variables to start teaching the AI to play the game
	Then visualize the data with the genome containing highest fitness

	:param config_path: path to the neural 
	:type config_path: int / range[0 -> 99]

	:return: None
	"""

	global gen

	# -------------------------------------------------------------------------
	# Load Configuration
	# -------------------------------------------------------------------------
	config = neat.config.Config(
		neat.DefaultGenome,
		neat.DefaultReproduction,
		neat.DefaultSpeciesSet,
		neat.DefaultStagnation,
		config_path
	)

	# Create Population
	p = neat.Population(config)

	# Add StdOut Reporter (Displays Progress in Terminal)
	p.add_reporter(neat.StdOutReporter(True))
	stats = neat.StatisticsReporter()
	p.add_reporter(stats)

	# Run Up to [Gen. Option] Generations
	winner = p.run(main_ai, 1000) # We Save Best Genome

	# Reset Gen Count
	gen = 0

def start_AI():
	"""
	Prepare the artificial intelligence by resetting and setting values and the configuration

	:return: None
	"""

	# Global Variable
	global hs_genopt_popopt

	# -------------------------------------------------------------------------
	# Set and Run Configuration Path
	# -------------------------------------------------------------------------
	local_dir = os.path.dirname(__file__)
	config_path = os.path.join(local_dir, os.path.join("utils", "config-feedforward.txt"))
	run(config_path)

def menu():
	"""
	Menu function, that displays the Main Menu and the AI Options Menu

	:return: None
	"""

	global FPS

	# Menu Theme
	menu_theme = pygame_menu.themes.THEME_BLUE.copy()
	menu_theme.widget_font = pygame_menu.font.FONT_8BIT # Copy of blue theme with 8bit font instead


	# -------------------------------------------------------------------------
	# Create menus: Main menu
	# -------------------------------------------------------------------------
	menu = pygame_menu.Menu(
		GAME_WIN_HEIGHT, 
		WIN_WIDTH, 
		'Snake', 
		theme=menu_theme, 
		onclose=pygame_menu.events.EXIT
	)

	# Play Buttons
	menu.add_button('AI', start_AI)
	menu.add_button('YOU', main_human)

	# Spacing
	menu.add_label('')
	menu.add_label('')
	menu.add_label('')
	menu.add_label('')

	# Options and Quit
	menu.add_button('AI Options', pygame_menu.events.EXIT)
	menu.add_button('Quit', pygame_menu.events.EXIT)

	# Main Menu Loop
	menu.mainloop(win, fps_limit=FPS)

# -----------------------------------------------------------------------------
# Main Program
# -----------------------------------------------------------------------------
if __name__== "__main__":
	# Run Menu
	menu()