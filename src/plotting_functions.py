import seaborn as sns
import matplotlib.pyplot as plt

def plot_value_counts(df, feature_x, fig_size = (11, 5)):
    '''
    param feature_x: str, name of feature_x to plot the graph on.
    param fig_size: tuple of figsize to use for the subplots. Default value is (11,5)
    output: 1 subplot- Single subplot showing the distribution of the feature_x in the dataset. This is simply a 
            truncated version of the earlier defined plot_cat_feature_2_features function.
    '''
    f, ax = plt.subplots(1, 1, figsize = (11,5))

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