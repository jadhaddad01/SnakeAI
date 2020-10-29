"""
Author: Jad Haddad

flappy_bird
https://github.com/jadhaddad01/FlappyBirdAI
Snake Artificial Intelligence
Using the NEAT Genetic Neural Network Architecture to train a set of snakes to play the popular game Snake. Also playable by user.

License:
-------------------------------------------------------------------------------
MIT License

Copyright (c) 2020 Jad Haddad

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
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
    # Top right under score count (SEE DISPLAY FUNCTIONS)
    x=WIN_WIDTH - 10 - 120,
    y=50,
    w=120,
    h=35,
    param_options={
        'curve': 0.3,
        'text': "Menu",
        'font_colour': (255, 255, 255),
        'background_color': (200, 200, 200),
        'hover_background_color': (160, 160, 160),
        'outline_half': False
    }
)

# Generation Count and Image Display
gen = 0
neural_net_image = None

# To Save Human High Score, AI Options Gen. / Pop.
hs_genopt_popopt = [0, 1000, 16]  # Default if file not found

# Open hs_genopt_popopt File
try:
    with open(os.path.join("utils", "hs_genopt_popopt.txt"), "rb") as fp:            # Load Pickle
        hs_genopt_popopt = pickle.load(fp)

# If Not Found, Create a New One
except Exception as e:
    print("Saved Values File hs_genopt_popopt.txt Not Found. Defaulting to:")
    print("    - High Score:", hs_genopt_popopt[0])
    print("    - Generations (AI Options):", hs_genopt_popopt[1])
    print("    - Population (AI Options):", hs_genopt_popopt[2])

    with open(os.path.join("utils", "hs_genopt_popopt.txt"), "wb") as fp:            # Save Pickle
        pickle.dump(hs_genopt_popopt, fp)

# How many blocks there are
blocks = 16
snakes = 16

# Load Fonts
STAT_FONT = pygame.font.SysFont("comicsans", 50)
STAT_FONT_SMALL = pygame.font.SysFont("comicsans", 30)
STAT_FONT_BIG = pygame.font.SysFont("comicsans", 100)

# -----------------------------------------------------------------------------
# Classes
# -----------------------------------------------------------------------------


class Snake:
    # division of constants to fit into one block of many snakes
    ratio = 1
    vel = 15
    grid_sys = 30

    # snake body
    x = [0]
    y = [0]

    def __init__(self, x, y, wb, we, hb, he):
        global blocks

        # Calibrating constants to fit each block
        self.ratio = math.sqrt(blocks)
        self.vel = self.vel / self.ratio
        self.grid_sys = self.grid_sys / self.ratio

        """
        # Starting with 3 blocks
        self.x = [
                    (x/self.ratio),
                    (x/self.ratio),
                    (x/self.ratio)
                ]
        self.y = [
                    (y/self.ratio),
                    (y/self.ratio) + (self.grid_sys/2),
                    (y/self.ratio) + self.grid_sys
                ]
        """

        # Starting with 3 blocks
        self.x = [
                    x,
                    x,
                    x
                ]
        self.y = [
                    y,
                    y + (self.grid_sys/2),
                    y + self.grid_sys
                ]

        # snake direction
        self.direction = "Up"

        # block dimensions
        self.width_end = we
        self.height_end = he
        self.width_begin = wb
        self.height_begin = hb

    def move_right(self):
        # if self.x[0] % self.grid_sys == 0 and self.y[0] % self.grid_sys == 0 and self.direction != "Left":
        # We want everything to be in the same "grid"
        if self.x[0] % self.grid_sys == 0 and self.y[0] % self.grid_sys == 0:
            self.direction = "Right"

    def move_left(self):
        # if self.x[0] % self.grid_sys == 0 and self.y[0] % self.grid_sys == 0 and self.direction != "Right":
        if self.x[0] % self.grid_sys == 0 and self.y[0] % self.grid_sys == 0:
            self.direction = "Left"

    def move_up(self):
        # if self.x[0] % self.grid_sys == 0 and self.y[0] % self.grid_sys == 0 and self.direction != "Down":
        if self.x[0] % self.grid_sys == 0 and self.y[0] % self.grid_sys == 0:
            self.direction = "Up"

    def move_down(self):
        # if self.x[0] % self.grid_sys == 0 and self.y[0] % self.grid_sys == 0 and self.direction != "Up":
        if self.x[0] % self.grid_sys == 0 and self.y[0] % self.grid_sys == 0:
            self.direction = "Down"

    def move(self):

        for n in range(len(self.x) - 1, 0, -1):
            self.x[n] = self.x[n - 1]
            self.y[n] = self.y[n - 1]

        # if not self.y == 0 and self.direction == "Up": # Force snake not to hit wall
        if self.direction == "Up":
            self.y[0] = self.y[0] - self.vel

        # if not self.y == GAME_WIN_HEIGHT - 30 and self.direction == "Down":
        if self.direction == "Down":
            self.y[0] = self.y[0] + self.vel

        # if not self.x == GAME_WIN_WIDTH - 30 and self.direction == "Right":
        if self.direction == "Right":
            self.x[0] = self.x[0] + self.vel

        # if not self.x == 0 and self.direction == "Left":
        if self.direction == "Left":
            self.x[0] = self.x[0] - self.vel

    def wall_collision(self):
        return (
                self.y[0] < self.height_begin
                or (self.y[0] == self.height_end - self.vel and self.direction == "Down")
                or self.x[0] < self.width_begin
                or (self.x[0] == self.width_end - self.vel and self.direction == "Right")
                )

    def snake_collision(self):
        # Make list of box coordinates sublist
        tmp = []
        for n in range(len(self.x)):
            tmp.append([self.x[n], self.y[n]])

        # Checks for duplicate sublists
        return not len([list(i) for i in set(map(tuple, tmp))]) == len(tmp)

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
        left = self.width_begin
        right = self.width_end
        top = self.height_begin
        bottom = self.height_end

        # we want closest block not farthest
        leftflag = True
        rightflag = True
        topflag = True
        bottomflag = True

        # Snake
        for n in range(1, len(self.x)):  # Don't include head
            if self.y[n] == self.y[0]:
                if self.x[n] < self.x[0] and leftflag:
                    left = self.x[0] - self.x[n]
                    leftflag = False
                if self.x[n] > self.x[0] and rightflag:
                    right = self.x[n] - self.x[0]
                    rightflag = False

            if self.x[n] == self.x[0]:
                if self.y[n] < self.y[0] and topflag:
                    top = self.y[0] - self.y[n]
                    topflag = False
                if self.y[n] > self.y[0] and bottomflag:
                    bottom = self.y[n] - self.y[0]
                    bottomflag = False

        # Wall IF NO SNAKE
        if left == self.width_begin:
            left = self.x[0] - self.width_begin
        if right == self.width_end:
            right = self.width_end - self.x[0]
        if top == self.height_begin:
            top = self.y[0] - self.height_begin
        if bottom == self.height_end:
            bottom = self.height_end - self.y[0]

        return (right, left, bottom, top)

    def draw(self, win):
        for n in range(len(self.x)):  # x has same length as y
            pygame.draw.rect(
                win, (255, 255, 255), (self.x[n], self.y[n], self.grid_sys, self.grid_sys))


class Food:
    # division of constants to fit into one block of many foods
    ratio = 1
    grid_sys = 30

    def __init__(self, wb, we, hb, he):
        # Calibrating constants to fit each block
        self.ratio = math.sqrt(blocks)
        self.grid_sys = self.grid_sys / self.ratio

        # block dimensions
        self.width_end = we
        self.height_end = he
        self.width_begin = wb
        self.height_begin = hb

        self.x = random.randrange(
                                    100 * self.width_begin,
                                    100 * self.width_end,
                                    100 * self.grid_sys  # We want everything to be in the same "grid"
                                ) / 100  # Accounting float for larger block num
        self.y = random.randrange(
                                    100 * self.height_begin,
                                    100 * self.height_end,
                                    100 * self.grid_sys
                                ) / 100

    def new(self, snake):

        not_satisfied = True
        while not_satisfied:
            self.x = random.randrange(
                                        100 * self.width_begin,
                                        100 * self.width_end,
                                        100 * self.grid_sys
                                    ) / 100
            self.y = random.randrange(
                                        100 * self.height_begin,
                                        100 * self.height_end,
                                        100 * self.grid_sys
                                    ) / 100

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
        return self.x == snakex and self.y == snakey

    def distance_to_food(self, snake):
        (headx, heady) = snake.get_coord_head()

        """
        x = (self.x - headx) ** 2
        y = (self.y - heady) ** 2

        return math.sqrt(x + y)
        """

        return (self.x - headx, self.y - heady)

    def draw(self, win):
        pygame.draw.rect(win, (255, 0, 0), (self.x, self.y,
                         self.grid_sys, self.grid_sys))

# -----------------------------------------------------------------------------
# Methods
# -----------------------------------------------------------------------------
def next_square(num):
    """
    Gives the closest number in which its square root is an integer

    :return: closest squarable num
    :type: int
    """

    if (math.sqrt(num)).is_integer():
        return num
    return next_square(num + 1)


def draw_window_human(win, snake, food, score, pregame):
    """
    Draw game using given parameters (Human Game)
    Can draw both pregame and main game

    :return: None
    """

    win.fill((0, 0, 0))

    snake.draw(win)

    food.draw(win)

    # score seperator
    pygame.draw.line(win, (255, 255, 255), (GAME_WIN_WIDTH, 0),
                     (GAME_WIN_WIDTH, GAME_WIN_HEIGHT))

    # blocks seperators
    for i in range(1, int(math.sqrt(blocks))):
        pygame.draw.line(
                            win,
                            (255, 255, 255),
                            (i * (GAME_WIN_WIDTH / math.sqrt(blocks)), 0),
                            (i * (GAME_WIN_WIDTH / math.sqrt(blocks)), GAME_WIN_HEIGHT)
                        )
        pygame.draw.line(
                            win,
                            (255, 255, 255),
                            (0, i * (GAME_WIN_HEIGHT / math.sqrt(blocks))),
                            (GAME_WIN_WIDTH, i * (GAME_WIN_HEIGHT / math.sqrt(blocks)))
                        )

    # Draw Current Score
    text = STAT_FONT.render("Score: " + str(score), 1, (255, 255, 255))
    win.blit(text, (WIN_WIDTH - 10 - text.get_width(), 10))

    if pregame:
        # Draw Transparency Over Game
        transparency_size = (WIN_WIDTH, GAME_WIN_HEIGHT)
        transparency = pygame.Surface(transparency_size)
        transparency.set_alpha(150)
        win.blit(transparency, (0, 0))

        # Main Text
        text = STAT_FONT_BIG.render("Press Arrow Key", 1, (255, 255, 255))
        win.blit(text, (GAME_WIN_WIDTH/2 - text.get_width() /
                 2, GAME_WIN_HEIGHT/2 - text.get_height()))

        # Saved High Score
        text = STAT_FONT.render("High Score: " + str(hs_genopt_popopt[0]), 1, (255, 0, 0))
        win.blit(text, (GAME_WIN_WIDTH/2- text.get_width()/2, GAME_WIN_HEIGHT/2 + 100))

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
    global blocks
    global snakes

    # reset addition counter to 0
    block_count = 0

    # reset block and snakes to 1
    blocks = 1
    snakes = 1

    xsaved = 0
    ysaved = 0

    # Set Variables
    snake = Snake(
                    GAME_WIN_WIDTH / 2,
                    GAME_WIN_HEIGHT / 2,
                    0,
                    GAME_WIN_WIDTH,
                    0,
                    GAME_WIN_HEIGHT
                )
    food = Food(
                    0,
                    GAME_WIN_WIDTH,
                    0,
                    GAME_WIN_HEIGHT
                )

    # win = pygame.display.set_mode((WIN_WIDTH, GAME_WIN_HEIGHT))
    clock = pygame.time.Clock()

    score = 0

    # -------------------------------------------------------------------------
    # Game: Before the Game
    # -------------------------------------------------------------------------
    run_pregame = True
    while run_pregame:
        clock.tick(FPS)  # Allow only for FPS Frames per Second
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
        """
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            snake.move_down()
            run_pregame = False
        """
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
                # Before Quitting, Save New HighScore [If New Highscore]
                if(score > hs_genopt_popopt[0]):
                    with open(os.path.join("utils", "hs_genopt_popopt.txt"), "wb") as fp:            # Save Pickle
                        pickle.dump(hs_genopt_popopt, fp)
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

        snake.move()

        if snake.wall_collision() or snake.snake_collision():
            if(score > hs_genopt_popopt[0]):
                hs_genopt_popopt[0] = score
                with open(os.path.join("utils", "hs_genopt_popopt.txt"), "wb") as fp:            # Save Pickle
                    pickle.dump(hs_genopt_popopt, fp)
            main_human() # Go "back" to pregame

        if block_count == 1:
            block_count += 1

        if block_count == 2:
            snake.add_block(xsaved, ysaved)
            block_count = 0
        
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
    Draw game using given parameters (AI Game)

    :return: None
    """
    win.fill((0,0,0))

    for i in range(len(snake)):

        snake[i].draw(win)

        food[i].draw(win)

    # score seperator
    pygame.draw.line(win, (255,255,255), (GAME_WIN_WIDTH, 0), (GAME_WIN_WIDTH, GAME_WIN_HEIGHT))

    # blocks seperators
    for i in range(1, int(math.sqrt(blocks))):
        pygame.draw.line(
                            win, 
                            (255,255,255), 
                            (i * (GAME_WIN_WIDTH / math.sqrt(blocks)), 0), 
                            (i * (GAME_WIN_WIDTH / math.sqrt(blocks)), GAME_WIN_HEIGHT)
                        )
        pygame.draw.line(
                            win, 
                            (255,255,255), 
                            (0, i * (GAME_WIN_HEIGHT / math.sqrt(blocks))), 
                            (GAME_WIN_WIDTH, i * (GAME_WIN_HEIGHT / math.sqrt(blocks)))
                        )


    # Draw Current Score
    text = STAT_FONT.render("Score: " + str(score), 1, (255, 255, 255))
    win.blit(text, (WIN_WIDTH - 10 - text.get_width(), 10))

    # Draw Current Generation
    text = STAT_FONT.render("Gen: " + str(gen), 1, (255, 255, 255))
    win.blit(text, (WIN_WIDTH - 10 - text.get_width(), GAME_WIN_HEIGHT - 20 - 2 * text.get_height()))

    # Draw Current Number of Snakes Alive
    text = STAT_FONT.render("Alive: " + str(len(snake)), 1, (255, 255, 255))
    win.blit(text, (WIN_WIDTH - 10 - text.get_width(), GAME_WIN_HEIGHT - 10 - text.get_height()))

    # Return To Menu if Menu Button Pressed / Draw menu button
    if button2.update():
        menu()

    # Update the Current Display
    pygame.display.update()

def main_ai(genomes, config):
    # Global Variables
    global FPS
    global gen

    global blocks
    global snakes
    global hs_genopt_popopt

    blocks = next_square(hs_genopt_popopt[2])
    snakes = hs_genopt_popopt[2]

    # block array for width and height of each block
    width_begin = []
    width_end = []
    height_begin = []
    height_end = []

    # set w / h to each block as they go left to right top to bottom (index starts at 0)
    for i in range(0, int(math.sqrt(blocks))):
        for j in range(0, int(math.sqrt(blocks))):
            width_begin.append(j * (GAME_WIN_WIDTH / math.sqrt(blocks)))
            width_end.append((j + 1) * (GAME_WIN_WIDTH / math.sqrt(blocks)))

            height_begin.append(i * (GAME_WIN_HEIGHT / math.sqrt(blocks)))
            height_end.append((i + 1) * (GAME_WIN_HEIGHT / math.sqrt(blocks)))

    # Set Variables
    snake = []
    food = []
    times = []

    block_count = []
    xsaved = []
    ysaved = []

    nets = []
    ge = []
    i = 0
    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)

        g.fitness = 0
        ge.append(g)

        snake.append(Snake(
                        (width_end[i] + width_begin[i]) / 2, 
                        (height_end[i] + height_begin[i]) / 2, 
                        width_begin[i], 
                        width_end[i], 
                        height_begin[i], 
                        height_end[i]
                    ))
        food.append(Food(
                        width_begin[i], 
                        width_end[i], 
                        height_begin[i], 
                        height_end[i]
                    ))

        times.append(time.time())
        block_count.append(0)
        xsaved.append(0)
        ysaved.append(0)

        i += 1

    clock = pygame.time.Clock()
    score = 0
    gen += 1

    run = True
    while run:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()

        if not len(snake) > 0:
            run = False

        for x, python in enumerate(snake):
            ge[x].fitness += 0.5

            (headx, heady) = python.get_coord_head()
            (food_distance_x, food_distance_y) = food[x].distance_to_food(python)
            
            # SubInp: Distance to Snake OR Wall (What's Closest)
            (right, left, down, up) = python.dis_to_snake_or_wall()

            # Inputs: Headx, Heady, Distance to Food, Subinp1, Subinp2
            outputs = net.activate((headx, heady, food_distance_x, food_distance_y, right, left, down, up))
            direc = outputs.index(max(outputs))

            # Go Right / Left / Up / Down
            if direc == 0:
                python.move_right()
            if direc == 1:
                python.move_left()
            if direc == 2:
                python.move_down()
            if direc == 3:
                python.move_up()

            python.move()

            if python.wall_collision() or python.snake_collision() or (time.time() - times[x] >= 10):
                snake.pop(x)
                food.pop(x)
                nets.pop(x)
                times.pop(x)
                ge[x].fitness -= 2
                ge.pop(x)

        for x, apple in enumerate(food):
            if block_count[x] == 1:
                block_count[x] += 1

            if block_count[x] == 2:
                snake[x].add_block(xsaved[x], ysaved[x])
                block_count[x] = 0

            if apple.eaten(snake[x]):
                ge[x].fitness += 50

                times[x] = time.time()

                (xsaved[x], ysaved[x]) = snake[x].get_last_block()
                block_count[x] += 1

                apple.new(snake[x])

        draw_window_ai(win, snake, food, score, gen)


def run(config_path):
    """
    Use given configuration path and variables to start teaching the AI to play the game
    Then visualize the data with the genome containing highest fitness

    :param config_path: path to the neural 
    :type config_path: int / range[0 -> 99]

    :return: None
    """

    # Global Variables
    global hs_genopt_popopt
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

    """ Save Population in Generation x
    x = 3
    p.add_reporter(neat.Checkpointer(x))
    """

    # Handle Generation Count of 0
    if hs_genopt_popopt[1] < 1:
        print('Generations set to 1 instead of 0.')
        hs_genopt_popopt[1] = 1

    # Save HighScore Gen. Option and Pop. Option with Pickle
    with open(os.path.join("utils", "hs_genopt_popopt.txt"), "wb") as fp:            # Save Pickle
        pickle.dump(hs_genopt_popopt, fp)

    # Run Up to [Gen. Option] Generations
    winner = p.run(main_ai, hs_genopt_popopt[1]) # We Save Best Genome

    """ Load and Run Saved Checkpoint
    gen = x
    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-' + str(x - 1))
    p.run(main_ai, 2)
    """

    # Reset Gen Count
    gen = 0

def start_AI():
    """
    Prepare the artificial intelligence by resetting and setting values and the configuration

    :return: None
    """

    # Global Variable
    global hs_genopt_popopt

    # Handle Population Count Lower than 2
    if hs_genopt_popopt[2] < 2:
        print('Population set to 2. P.S: The NN needs at least 2 genomes to function properly.')
        hs_genopt_popopt[2] = 2

    # Modify NEAT Configuration File For Population Count
    confmodif.conf_file_modify(hs_genopt_popopt[2])

    # -------------------------------------------------------------------------
    # Set and Run Configuration Path
    # -------------------------------------------------------------------------
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, os.path.join("utils", "config-feedforward.txt"))
    run(config_path)

def set_val_gen(value):
    """
    Saving generation count from options menu

    :param value: value to set
    :type value: int / range[0 -> 99]

    :return: None
    """

    # Global Variable
    global hs_genopt_popopt

    # Set Generation Count
    hs_genopt_popopt[1] = value

def set_val_pop(value):
    """
    Saving population count from options menu

    :param value: value to set
    :type value: int / range[0 -> 99]

    :return: None
    """

    # Global Variable
    global hs_genopt_popopt

    # Set Population Count
    hs_genopt_popopt[2] = value

def menu():
    """
    Menu function, that displays the Main Menu and the AI Options Menu

    :return: None
    """

    # Global Variables
    global hs_genopt_popopt
    global FPS

    # Menu Theme
    menu_theme = pygame_menu.themes.THEME_BLUE.copy()
    menu_theme.widget_font = pygame_menu.font.FONT_8BIT # Copy of blue theme with 8bit font instead

    # -------------------------------------------------------------------------
    # Create menus: AI Options menu
    # -------------------------------------------------------------------------
    options = pygame_menu.Menu(
        GAME_WIN_HEIGHT, # Height
        WIN_WIDTH, #Width
        'AI Options',
        onclose=pygame_menu.events.EXIT, # Menu close button or ESC pressed
        theme=menu_theme # Theme
    )

    # No negative values allowed
    valid_chars = ['1','2','3','4','5','6','7','8','9','0']

    # Integer Inputs
    options.add_text_input(
        'Generations : ',
        default=str(hs_genopt_popopt[1]), # Default number set to gen input of previous AI game
        input_type=pygame_menu.locals.INPUT_INT, # Integer inputs only
        valid_chars=valid_chars,
        maxchar=4,
        onchange=set_val_gen # Save input
    )
    options.add_text_input('Population : ', 
        default=str(hs_genopt_popopt[2]), 
        input_type=pygame_menu.locals.INPUT_INT, 
        valid_chars=valid_chars, 
        maxchar=2, 
        onchange=set_val_pop
    )

    # Back Button
    options.add_button('Back', pygame_menu.events.BACK)

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
    menu.add_button('AI Options', options)
    menu.add_button('Quit', pygame_menu.events.EXIT)

    # Main Menu Loop
    menu.mainloop(win, fps_limit=FPS)

# -----------------------------------------------------------------------------
# Main Program
# -----------------------------------------------------------------------------
if __name__== "__main__":
    # Run Menu
    menu()
