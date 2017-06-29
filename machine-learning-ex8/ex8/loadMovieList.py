#!/usr/bin/env python3

import numpy as np


def loadMovieList():
    #GETMOVIELIST reads the fixed movie list in movie.txt and returns a
    #cell array of the words
    #   movieList = GETMOVIELIST() reads the fixed movie list in movie.txt 
    #   and returns a cell array of the words in movieList.


    ## Read the fixed movieulary list
    with open('movie_ids.txt', encoding='ISO-8859-1') as fid:
        movies = fid.readlines()

    movieNames = []
    for movie in movies:
        items = movie.split()
        movieNames.append(' '.join(items[1:]).strip())
    return movieNames


    #end
