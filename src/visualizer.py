import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
COLORS = {
    'POSITIVE': '#2ecc71',   # green
    'NEGATIVE': '#e74c3c',   # red
    'NEUTRAL':  '#3498db'    # blue
}

print("visualizer.py loaded successfully!")