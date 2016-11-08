# -*- coding: utf-8 -*-
r"""
Runs the Tic Tac Toe GUI.

Written by David C. Stauffer in March 2016.

"""
# Build with:
# pyinstaller --onefile --windowed run_tictactoe.py

#%% Imports
import sys
from PyQt5.QtWidgets import QApplication
from dstauffman.games.tictactoe import TicTacToeGui

#%% Execution
if __name__ == '__main__':
    # Runs the GUI application
    qapp = QApplication(sys.argv)
    # instatiates the GUI
    gui = TicTacToeGui()
    gui.show()
    # exits and returns the code on close of all main application windows
    sys.exit(qapp.exec_())
