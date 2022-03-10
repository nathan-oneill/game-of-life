import numpy as np
import matplotlib.pyplot as plt
import re
import time
from types import FunctionType


class Life:
    ## SETUP ##
    def __init__(self, width: int, height: int, start: np.ndarray = None, startLocation: tuple[int] = None):
        """Start a game of life.
        
        Parameters
        ----------
        start
            np.array of 0 or 1.
        
        """
        self._board = np.zeros((width, height), dtype=np.uint8)
        self._w = width
        self._h = height
        self._plot = None # Display window
        if start is not None:
            self.replace(start, startLocation)
        return
    
    def add(self, section: np.ndarray, location: tuple[int] = None):
        """Add a section to the board, non-destructively.
        Top-left corner of <section> will become <location> on the board."""
        section = self._projectArray(section)
        if location is not None:
            x, y = location
        else: # put in middle
            x = (self._w - section.shape[0]) // 2
            y = (self._h - section.shape[1]) // 2
        replacement = self._board[x : (x+section.shape[0]), y : (y+section.shape[1])]
        replacement += section
        replacement[replacement > 1] = 1
        self._board[x : (x+section.shape[0]), y : (y+section.shape[1])] = section
    
    def replace(self, section: np.ndarray, location: tuple[int] = None):
        """Replace a section of the board, destructively."""
        section = self._projectArray(section)
        if location is not None:
            x, y = location
        else: # put in middle
            x = (self._w - section.shape[0]) // 2
            y = (self._h - section.shape[1]) // 2
        self._board[x : (x+section.shape[0]), y : (y+section.shape[1])] = section
    
    @staticmethod
    def _projectArray(a: np.ndarray):
        """Project an array to the appropriate dtype. Errors if any element != 0 or 1"""
        if (np.any(np.logical_and(a!=0, a!=1))):
            raise ValueError('Array must only contain 0 or 1')
        return a.astype(np.uint8) # uint4 would be preferred, but doesn't exist.
        

    ## SIMULATION ##
    def step(self):
        """Step the game forward 1 tick
        
        Notes
        -----
        An alternate (possibly more efficient) approach would be to initialise
        8 zero arrays, then assign a submatrix of them to be a submatrix of 
        the board. This possibly saves recreating an entire array on .pad()

        """
        # Get neighbour counts
        # I could have hardcoded the 'neighbour collecter' to
        # just the surrounding 8, but doing this in general is an interesting
        # challenge on its own.
        # TODO: turn this `getZeroPaddedOffset` into a function and stick into pythonutils.
        # Mind you, depending on how fast the alternate is (and how it functions), I may
        # offer that as an alternative.
        w = self._w
        h = self._h
        offsets = [
            (-1,-1), ( 0,-1), ( 1,-1),
            (-1, 0),          ( 1, 0),
            (-1, 1), ( 0, 1), ( 1, 1)
        ]

        neighbourCounts = np.zeros_like(self._board)
        for x,y in offsets:
            # Get offset rectangle in board
            sliceX = slice(max(x,0), min(x+w,w))
            sliceY = slice(max(y,0), min(y+h,h))
            shifted = self._board[sliceX,sliceY]

            # Pad with 0's
            padX = (max(-x,0), max(x,0))
            padY = (max(-y,0), max(y,0))
            shifted = np.pad(shifted, [padX,padY])

            # Add to neighbour counts
            neighbourCounts += shifted

        # Implement rules
        self._board[np.logical_and(self._board == 1, neighbourCounts == 2)] = 1 # Rule 2
        self._board[neighbourCounts <  2] = 0 # Rule 1
        self._board[neighbourCounts == 3] = 1 # Rule 2 & Rule 4
        self._board[neighbourCounts >  3] = 0 # Rule 3
        return self._board


    ## DISPLAY ##
    def print(self):
        """Print the current state to the console"""
        # Not yet sure of an efficient way to do this
        # It's also not super practical :|
        # np.set_printoptions(threshold = np.inf) is needed for large arrays,
        # and isn't particularly helpful.
        print( re.sub(r'\[|0|,|\]', ' ', str(self._board)) )
        return        
    
    def display(self):
        """Print the current state to a new window."""
        if self._plot is None:
            plt.ion()
            self._plot = plt.imshow(self._board)
        else:
            self._plot.set_data(self._board)
        plt.pause(.01) # Pause python to render
        return
    
    def play(self, n: int, dt: float = .05, freezeOnEnded: bool = False, callback: FunctionType = None):
        """Play <n> frames, separated by <dt> time.
        
        Parameters
        ----------
        callback
            Called after each `.step()`. Takes the graph as a single parameter
        """
        if callback is None:
            callback = lambda x : None
        
        self.display()
        for _ in range(n):
            self.step()
            self.display()
            callback(self._plot)
            time.sleep(.05)
        if freezeOnEnded: # Hold the python runtime until the graph is closed manually
            plt.show()
        plt.ioff()
        return


class Pattern:
    """A collection of patterns. https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life.
    If I had more time I would add an 'orientation' parameter.
    """

    ## HELPER FUNCTIONS ##
    @staticmethod
    def composeSimple(a: np.ndarray, b: np.ndarray, space: int = 0):
        """Mash the two patterns into a single array with <space> 0's in between."""
        ax, ay = a.shape
        bx, by = b.shape
        width = ax + bx + space
        height = max(ay, by)
        board = np.zeros((width, height))
        board[:ax, :ay] = a
        board[ax+space:, :by] = b
        return board
    

    ## PATERNS ##
    class Still:
        """Still patterns"""
        block = np.array(
            [[1,1],
             [1,1]
            ])
        
        beehive = np.array(
            [[0,1,1,0],
             [1,0,0,1],
             [0,1,1,0]
            ])
        
        boat = np.array(
            [[1,1,0],
             [1,0,1],
             [0,1,0]
            ])

    class Oscillator:
        blinker = np.array([[1,1,1]])

        toad = np.array(
            [[0,1,1,1],
             [1,1,1,0]
            ])

    class Spaceship:
        glider = np.array(
            [[0,0,1],
             [1,0,1],
             [0,1,1]
            ])
        
        lightship = np.array(
            [[0,1,1,0,0],
             [1,1,1,1,0],
             [1,1,0,1,1],
             [0,0,1,1,0]
            ])


if __name__ == '__main__':
    gridsize = 50

    # Create a starting board
    p = Pattern.composeSimple(Pattern.Still.block, Pattern.Spaceship.glider)
    p = Pattern.composeSimple(p, Pattern.Oscillator.toad, 3)

    L = Life(gridsize, gridsize, p)
    print(L._board)

    # Play a game
    L.play(200)

    # Add some elements in
    middle = gridsize // 2
    L.add(Pattern.Spaceship.lightship, (middle - 10, middle - 10))
    L.add(np.rot90(Pattern.Spaceship.lightship, 2), (middle + 5, middle + 5))
    L.add(Pattern.Oscillator.toad, (middle - 5, middle - 5))

    # Continue playing
    L.play(200, freezeOnEnded=True)

    # empty print useful in VSCode debugger
    print('', end='\n')