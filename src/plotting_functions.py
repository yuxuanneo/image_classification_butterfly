import seaborn as sns
import matplotlib.pyplot as plt

def plot_value_counts(df, feature_x, fig_size = (11, 5)):
    """
    Presents the value_counts method in dataframe in a bar chart 

    Args:
        df (Pandas DataFrame): df containing feature_x.
        feature_x (str): feature for value_counts to be plotted on.
        fig_size (tuple, optional): size of plot. Defaults to (11, 5).
    """
    f, ax = plt.subplots(1, 1, figsize = fig_size)

    sns.countplot(x = feature_x, data = df, ax = ax)

    ax.set_title(f"Distribution of {feature_x}")

    ax.set_ylim(0, 1.2 * df[feature_x].value_counts().max())

    #labels for the first plot, i.e. distribution of the cat variable
    for rect in ax.patches:
        height = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / 2, height + 5, 
            f"{height} \n ({round(1000*(height/df.shape[0]))/10}%)", 
            ha="center", va="bottom"
        )

    # rotate xticks for both subplots
    ax.tick_params(labelrotation=90)
    ax.set(yticklabels=[])
    
    plt.show()

def plot_cont_per_class(df, cols_to_plot, target, fig_size = (11, 5)):
    """
    Plot n boxplots, where n is the len(cols_to_plot). Each object in each boxplot corresponds to 
    each target class.

    Args:
        df (Pandas DataFrame): df containing feature_x.
        cols_to_plot (list): Features to plot boxplots on .
        target (str): target column.
        fig_size (tuple, optional): size of plot. Defaults to (11, 5).
    """
    if len(cols_to_plot) > 1: 
        fig , axs = plt.subplots(len(cols_to_plot), 1, figsize= fig_size,sharex = True, sharey = False)
        for i, col in enumerate(cols_to_plot):
            sns.boxplot(data = df, x = target, y = col, ax = axs[i])
            axs[i].grid()
        plt.tight_layout()
        
    elif len(cols_to_plot) == 1:
        sns.set(rc={'figure.figsize':fig_size})
        sns.boxplot(data = df, x = target, y = cols_to_plot[0])
    