# Built-in imports 
import sys

# External imports
from sklearn import datasets
from sklearn.preprocessing import scale # Data scaling
from sklearn import decomposition #PCA
import pandas as pd # pandas
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.express as px



def main():
    #! IRIS DATA SET
    # Load the dataset to work with
    iris = datasets.load_iris()

    # Print the input features
    print(f"The inputs variables are: {iris.feature_names}")

    # Print the output features
    print(f"The output variables are: {iris.target_names}")

    # Define the input and output variables
    X = iris.data
    Y = iris.target

    # Print the dimension of the variables
    print(f"X shape: {X.shape} Y shape: {Y.shape}")


    #! PCA ANALYSIS
    # Data scaling
    X = scale(X)

    # Perform the PCA analysis 
    pca = decomposition.PCA(n_components=3) # Here the number of PC is defined as 3
    pca.fit(X)

    # Calculate the scores values
    scores = pca.transform(X)
    scores_df = pd.DataFrame(scores, columns=['PC1', 'PC2', 'PC3'])
    print(scores_df)

    # Create the desired vector
    Y_label = []

    for i in Y:
        if i == 0:
            Y_label.append('Setosa')
        elif i == 1:
            Y_label.append('Versicolor')
        else:
            Y_label.append('Virginica')

    # Concatenate the inputs and desired vectors
    Species = pd.DataFrame(Y_label, columns=['Species'])
    df_scores = pd.concat([scores_df, Species], axis=1)
    print(df_scores)

    # Retrieve the loadings values
    loadings = pca.components_.T
    df_loadings = pd.DataFrame(
        loadings,
        columns=['PC1', 'PC2','PC3'],
        index=iris.feature_names
    )
    print(df_loadings)

    # Explained variance for each PC
    explained_variance = pca.explained_variance_ratio_
    print(f"Explained Variance for each PC: {explained_variance}")

    #! Scree Plot
    # Preparing the explained variance data
    explained_variance = np.insert(explained_variance, 0, 0)

    # Preparing the cumulative variance data
    cumulative_variance = np.cumsum(np.round(explained_variance, decimals=3))

    # Combining the dataframe
    pc_df = pd.DataFrame(['','PC1', 'PC2', 'PC3'], columns=['PC'])
    explained_variance_df = pd.DataFrame(explained_variance, columns=['Explained Variance'])
    cumulative_variance_df = pd.DataFrame(cumulative_variance, columns=['Cumulative Variance'])

    df_explained_variance = pd.concat([pc_df, explained_variance_df, cumulative_variance_df], axis=1)
    print(df_explained_variance)

    # Plot the Explained variance and the Cumulative variance
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_explained_variance['PC'],
            y=df_explained_variance['Cumulative Variance'],
            marker=dict(size=15, color="LightSeaGreen")
        ))

    fig.add_trace(
        go.Bar(
            x=df_explained_variance['PC'],
            y=df_explained_variance['Explained Variance'],
            marker=dict(color="RoyalBlue")
        ))

    fig.show()

    #! Scores Plot
    # Basic 3D Scatter Plot
    fig = px.scatter_3d(df_scores, x='PC1', y='PC2', z='PC3', color='Species')
    fig.show()

    #! Loadings Plot
    loadings_label = df_loadings.index
    # loadings_label = df_loadings.index.str.strip(' (cm)')

    fig = px.scatter_3d(df_loadings, x='PC1', y='PC2', z='PC3', text = loadings_label)

    fig.show()

if __name__ == "__main__":
    sys.exit(main())