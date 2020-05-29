from matplotlib import pyplot as plt

from agnes.common.plotter import draw_plot

styles = ['seaborn-whitegrid', 'seaborn-paper', 'dark_background', 'ggplot']
plt.style.use(styles[0])
draw_plot("results/MuJoCo/Ant-v2_MLP", redraw=True)
