from abc import abstractmethod
from cost import *
from neighbourhood import *
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random


class Search:
    @abstractmethod
    def run(self):
        pass
