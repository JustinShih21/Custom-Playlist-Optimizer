from gurobipy import Model, GRB, quicksum
import pandas as pd
import numpy as np
from tabulate import tabulate

# load the artists data set
artists = pd.read_csv('artists.csv')

# load the songs data set
songs = pd.read_csv('Final_Cleaned_Song_Predictions.csv')

def user_choice(songs_dict):
    valid_user_ids = [col for col in songs_dict.keys() if col.startswith("user_")]
    print("Please enter the user ID for which you want to see the ratings:")
    while True:
        user_id = input()
        if user_id in valid_user_ids:
            return user_id
        else:
            print("Invalid user ID. Please enter a valid user ID from the dataset.")

def create_playlist(songs_dict, target_user, num_songs=30):
    """
    Create a playlist of songs for a specific user using optimization.
    
    Parameters:
    songs_dict (dict): Dictionary containing songs data
    target_user (str): User ID for whom to create the playlist
    num_songs (int): Number of songs to include in the playlist (default: 30)
    
    Returns:
    list: List of selected songs with their details
    """
    ratings = songs_dict[target_user]
    track_names = songs_dict['track_name_x']
    num_tracks = len(track_names)
    artists = songs_dict['artist_name_clean_x']
    unique_artists = list(set(artists))

    # create a model to recommend songs for this playlist
    model = Model("Playlist Recommendation")

    # create a binary variable, X_i, that is 1 if song i is selected, 0 otherwise
    X = model.addVars(num_tracks, vtype=GRB.BINARY, name="X")

    # create a binary variable, Y_i, that is 1 if artist i is selected, 0 otherwise
    Y = model.addVars(len(unique_artists), vtype=GRB.BINARY, name="Y")

    # the first constraint is that there must be exactly num_songs
    model.addConstr(quicksum(X[i] for i in range(num_tracks)) == num_songs, "num_songs")

    # the second constraint is that no artist can be repeated twice
    artist_to_songs = {artist: [] for artist in unique_artists}
    for i, artist in enumerate(artists):
        artist_to_songs[artist].append(i)

    for j, artist in enumerate(unique_artists):
        model.addConstr(quicksum(X[i] for i in artist_to_songs[artist]) <= Y[j], f"artist_{artist}_link")
        model.addConstr(quicksum(X[i] for i in artist_to_songs[artist]) <= 1, f"artist_{artist}_unique")

    # the third constraint is that no songs can be repeated (this is redundant with binary variables)
    model.addConstrs((X[i] <= 1 for i in range(num_tracks)), "no_repeats")

    # the objective function is to pick the songs with the highest ratings
    model.setObjective(quicksum(X[i] * ratings[i] for i in range(num_tracks)), GRB.MAXIMIZE)

    # optimize the model
    model.optimize()

    # collect the results
    results = []
    for idx, i in enumerate(range(num_tracks)):
        if X[i].x > 0:
            results.append([len(results) + 1, track_names[i], artists[i], round(ratings[i], 2)])

    # sort the results alphabetically by artist
    results.sort(key=lambda x: x[2])

    # re-number the songs after sorting
    for idx, row in enumerate(results):
        row[0] = idx + 1

    return results

# main execution
if __name__ == "__main__":
    songs_dict = songs.to_dict(orient="list")
    
    # call the function user_choice to get the target_user that the person wants to see
    target_user = user_choice(songs_dict)
    
    # create the playlist
    recommended_songs = create_playlist(songs_dict, target_user)
    
    # print the results in a pretty table
    print(f"\nRecommended Songs for {target_user}:")
    print(tabulate(recommended_songs, headers=["#", "Song", "Artist", "Rating"], tablefmt="fancy_grid"))
    
    # Optional: saving to CSV file
    recommended_df = pd.DataFrame(recommended_songs, columns=["#", "Song", "Artist", "Rating"])
    recommended_df.to_csv(f"recommended_playlist_{target_user}.csv", index=False)
    print(f"\nPlaylist also saved to recommended_playlist_{target_user}.csv")